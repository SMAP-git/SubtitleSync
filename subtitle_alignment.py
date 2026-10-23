"""Local semantic and contextual alignment of multilingual subtitles."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# The GUI uses pythonw, which has no console stream for Hugging Face progress bars.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import onnxruntime as ort
from fastembed.text.pooled_embedding import PooledEmbedding

import sync_subtitles
import gpu_runtime
import vulkan_embeddings


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEARCH_RADIUS_MS = 45_000
MIN_SIMILARITY = 0.58
MIN_MATCH_MARGIN = 0.015
HTML_TAG = re.compile(r"</?[^>]+>")
PARENTHETICAL = re.compile(r"\([^)]*\)|\[[^\]]*\]")
SPEAKER_PREFIX = re.compile(r"^[A-Z][A-Z .'-]{0,24}:\s*")
CREDIT = re.compile(
    r"(?i)\b(tradu[cç][aã]o|legendagem|legendas|subtitles?|subtitle by|"
    r"ripadas|sincronizadas por|www\.)\b"
)
HEARING_IMPAIRED = re.compile(
    r"(?i)\b(" 
    r"music|laughter|laughing|laughs?|applause|sighs?|breath(?:ing)?|"
    r"whisper(?:ing)?|shouts?|screams?|ring(?:ing)?|phone|doorbell|"
    r"gunshots?|sirens?|chatter|crowd|indistinct|indistinctly|inaudible|"
    r"engine|television|tv|radio|playing|static|noise|alarm|typing|"
    r"footsteps?|birds?|theme"
    r")\b"
)
MUSIC_NOTES = "♪♩♬♭♯"
PROCESSING_DEVICES = ("CPU",) if gpu_runtime.CPU_ONLY else ("CPU", "NVIDIA", "AMD", "INTEL", *gpu_runtime.INTEGRATED_DEVICES)


class _DirectMLTextEmbedding(PooledEmbedding):
    @classmethod
    def add_extra_session_options(
        cls, session_options: ort.SessionOptions, extra_options: dict[str, Any]
    ) -> None:
        super().add_extra_session_options(session_options, extra_options)
        # DirectML requires sequential execution with memory patterns disabled.
        session_options.enable_mem_pattern = False
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL


def execution_provider(device: str, device_id: int = 0) -> tuple[str, dict[str, str]]:
    if device not in PROCESSING_DEVICES:
        raise ValueError(f"Unknown processing device: {device}.")
    device = gpu_runtime.device_vendor(device)
    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError("GPU adapter ID must be a non-negative integer.")
    if device == "CPU":
        return "CPUExecutionProvider", {}
    if device == "NVIDIA":
        return "CUDAExecutionProvider", {"device_id": str(device_id)}
    if sys.platform == "win32":
        return "DmlExecutionProvider", {"device_id": str(device_id)}
    if sys.platform.startswith("linux"):
        if device == "AMD":
            return "MIGraphXExecutionProvider", {"device_id": str(device_id)}
        return "OpenVINOExecutionProvider", {"device_type": f"GPU.{device_id}"}
    raise ValueError(f"{device} GPU processing is supported only on Windows and Linux.")


def validate_processing_device(device: str, device_id: int = 0) -> None:
    provider, _ = execution_provider(device, device_id)
    if provider not in ort.get_available_providers():
        raise ValueError(
            f"{device} requires {provider}, which is not installed in this Python "
            "environment. See README.md for GPU runtime and driver setup, then "
            "restart the app, or select CPU."
        )


def _load_embedding_model(device: str, device_id: int) -> PooledEmbedding:
    from managed_updates import selected_path
    validate_processing_device(device, device_id)
    provider, options = execution_provider(device, device_id)
    if gpu_runtime.device_vendor(device) == "NVIDIA" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    model_type = _DirectMLTextEmbedding if provider == "DmlExecutionProvider" else PooledEmbedding
    selected = selected_path("embedding-model", _model_cache())
    model_options = {"specific_model_path": str(selected)} if selected != _model_cache() else {}
    model = model_type(
        model_name=MODEL_NAME,
        cache_dir=str(_model_cache()),
        providers=[(provider, options)],
        **model_options,
    )
    session = model.model
    if session is None or provider not in session.get_providers():
        raise ValueError(
            f"Could not activate {provider} for {device} adapter {device_id}. "
            "Check the GPU ID, runtime and drivers, or select CPU. "
            "CPU fallback was not accepted."
        )
    session.disable_fallback()
    return model


def _alnum_length(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _hearing_impaired_score(text: str) -> int:
    raw = text.strip()
    cleaned = _clean_caption(text)
    if not cleaned:
        return 100

    score = 0
    if any(note in raw for note in MUSIC_NOTES):
        score += 100
    if raw.startswith(("[", "(")) and raw.endswith(("]", ")")):
        score += 75
    compact = cleaned.lower()
    if HEARING_IMPAIRED.search(compact):
        score += 45
    if _alnum_length(cleaned) <= 4:
        score += 20
    if re.fullmatch(r"[-–—·.]+", raw):
        score += 60
    return min(score, 100)


def _hearing_impaired_threshold(strictness: int) -> int:
    strictness = max(0, min(100, strictness))
    return 80 - round(strictness * 0.4)


def _is_hearing_impaired_caption(text: str, strictness: int) -> bool:
    return _hearing_impaired_score(text) >= _hearing_impaired_threshold(strictness)


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class DialogueWindow:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class SubtitleTrack:
    track_id: int
    label: str


@dataclass(frozen=True)
class Match:
    cue_number: int
    target_ms: int
    similarity: float
    margin: float


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    matches = list(sync_subtitles.TIMESTAMP.finditer(text))
    for index, match in enumerate(matches):
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body_lines = [
            line.strip()
            for line in text[match.end() : segment_end].splitlines()
            if line.strip()
        ]
        if body_lines and body_lines[-1].isdigit():
            body_lines.pop()
        cues.append(
            Cue(
                start_ms=sync_subtitles.timestamp_to_ms(match.group("start")),
                end_ms=sync_subtitles.timestamp_to_ms(match.group("end")),
                text=" ".join(body_lines),
            )
        )
    if not cues:
        raise ValueError("No SRT subtitle cues were found.")
    return cues


def read_srt(path: Path) -> list[Cue]:
    try:
        text, _, _ = sync_subtitles.decode_srt(path.read_bytes())
    except OSError as error:
        raise ValueError(f"Cannot read subtitle file {path}: {error}") from error
    return parse_srt(text)


def _find_tool(name: str) -> str | None:
    installed = shutil.which(name)
    if installed:
        return installed
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if root:
            candidate = Path(root) / "MKVToolNix" / f"{name}.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def list_subtitle_tracks(video: Path) -> list[SubtitleTrack]:
    mkvmerge = _find_tool("mkvmerge")
    mkvextract = _find_tool("mkvextract")
    if not mkvmerge or not mkvextract or video.suffix.lower() not in (".mkv", ".mka", ".webm"):
        return _ffmpeg_subtitle_tracks(video)

    try:
        identified = subprocess.run(
            [mkvmerge, "-J", str(video)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ValueError(f"Could not inspect video subtitle tracks: {error}") from error
    if identified.returncode != 0:
        details = identified.stderr.strip()
        raise ValueError(f"Could not inspect video subtitle tracks. {details}".strip())
    try:
        metadata = json.loads(identified.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("MKVToolNix returned invalid track information.") from error

    candidates: list[SubtitleTrack] = []
    for track in metadata.get("tracks", []):
        if track.get("type") != "subtitles":
            continue
        codec = str(track.get("codec") or "").lower()
        if "subrip" not in codec and "srt" not in codec:
            continue
        props = track.get("properties", {})
        language = str(props.get("language_ietf") or props.get("language") or "").lower()
        name = str(props.get("track_name") or "")
        track_id = track["id"]
        candidates.append(SubtitleTrack(track_id, f"{track_id}: {language or 'und'} {name}".strip()))
    return candidates


def _ffmpeg() -> str:
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise ValueError("Video inspection needs FFmpeg. Restart using the app launcher to install audio/video support.") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_subtitle_tracks(video: Path) -> list[SubtitleTrack]:
    result = subprocess.run(
        [_ffmpeg(), "-nostdin", "-hide_banner", "-i", str(video), "-map", "0:v?",
         "-map", "0:a?", "-c", "copy", "-t", "0", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if result.returncode != 0:
        raise ValueError(f"Could not inspect video subtitle tracks.\n{result.stderr.strip()}")
    candidates = []
    for line in result.stderr.splitlines():
        match = re.search(r"Stream #0:(\d+)(?:\[[^\]]+\])?(?:\(([^)]+)\))?: Subtitle: (\w+)", line)
        if match and match[3] in ("subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"):
            stream = int(match[1])
            # Negative IDs distinguish FFmpeg stream indexes from MKVToolNix track IDs.
            candidates.append(SubtitleTrack(-stream - 1, f"{stream}: {match[2] or 'und'} ({match[3]})"))
    return candidates


def extract_reference_srt(video: Path, track_id: int) -> list[Cue]:
    mkvextract = _find_tool("mkvextract")
    if track_id >= 0 and not mkvextract:
        raise ValueError("MKVToolNix could not be found. Install it or select a reference SRT.")
    with tempfile.TemporaryDirectory(prefix="subtitle-sync-") as temporary:
        extracted = Path(temporary) / "reference.srt"
        command = (
            [_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
             "-map", f"0:{-track_id - 1}", "-c:s", "srt", str(extracted)]
            if track_id < 0 else
            [mkvextract, "tracks", str(video), f"{track_id}:{extracted}"]
        )
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise ValueError(f"Could not extract the embedded reference subtitles: {error}") from error
        if result.returncode != 0:
            details = result.stderr.strip()
            raise ValueError(
                f"Could not extract the embedded reference subtitles. {details}".strip()
            )
        return read_srt(extracted)


def _clean_caption(text: str) -> str:
    text = HTML_TAG.sub(" ", text)
    text = PARENTHETICAL.sub(" ", text)
    text = SPEAKER_PREFIX.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_windows(cues: Sequence[Cue], strictness: int) -> list[DialogueWindow]:
    windows: list[DialogueWindow] = []
    for start_index, first in enumerate(cues):
        parts: list[str] = []
        for end_index in range(start_index, min(len(cues), start_index + 5)):
            cue = cues[end_index]
            if end_index > start_index and cue.start_ms - cues[end_index - 1].end_ms > 3_500:
                break
            if _is_hearing_impaired_caption(cue.text, strictness):
                break
            clean_text = _clean_caption(cue.text)
            parts.append(clean_text)
            if end_index > start_index and cue.end_ms - first.start_ms > 9_000:
                break
            phrase = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if _alnum_length(phrase) >= 2:
                windows.append(DialogueWindow(first.start_ms, cue.end_ms, phrase))
    if not windows:
        raise ValueError("The subtitles contain no usable dialogue.")
    return windows


def _window_context(window: DialogueWindow, cues: Sequence[Cue]) -> str:
    before = [
        cue for cue in cues
        if cue.end_ms <= window.start_ms and 0 <= window.start_ms - cue.end_ms <= 4000
    ][-1:]
    after = [
        cue for cue in cues
        if cue.start_ms >= window.end_ms and 0 <= cue.start_ms - window.end_ms <= 4000
    ][:1]
    return " ".join(
        [_clean_caption(cue.text) for cue in before]
        + [window.text]
        + [_clean_caption(cue.text) for cue in after]
    )


def _model_cache() -> Path:
    override = os.environ.get("SUBTITLE_SYNC_MODEL_CACHE")
    if override:
        cache = Path(override)
        cache.mkdir(parents=True, exist_ok=True)
        return cache
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = (
        Path(local_app_data)
        if sys.platform == "win32" and local_app_data
        else Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    )
    cache = root / "SubtitleSync" / "model-cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _automatic_initial_pair(
    source: Sequence[Cue], reference: Sequence[Cue],
    vectors: dict[str, np.ndarray], strictness: int,
    notify: Callable[[str], None],
) -> tuple[int, int]:
    def usable(cues: Sequence[Cue]) -> list[tuple[int, Cue]]:
        return [
            (index + 1, cue) for index, cue in enumerate(cues)
            if not CREDIT.search(_clean_caption(cue.text))
            and not _is_hearing_impaired_caption(cue.text, strictness)
            and _alnum_length(_clean_caption(cue.text)) >= 12
        ]

    left, right = usable(source), usable(reference)
    if len(left) < 3 or len(right) < 3:
        raise ValueError("Cannot confidently find an automatic initial sync point. Select matching cues manually.")
    # Bound the search memory and sample across the timeline, not just opening credits.
    sampled = np.linspace(0, len(left) - 1, min(256, len(left)), dtype=int)
    reference_vectors = np.asarray([vectors[_clean_caption(cue.text)] for _, cue in right])
    matches: list[tuple[int, int, int, float]] = []
    for position in sampled:
        number, cue = left[int(position)]
        notify(f"Finding automatic initial sync point: source cue {number}...")
        scores = reference_vectors @ vectors[_clean_caption(cue.text)]
        best = int(np.argmax(scores))
        alternatives = np.delete(scores, best)
        score = float(scores[best])
        if score < 0.68 or score - float(np.max(alternatives)) < 0.08:
            continue
        reference_number, other = right[best]
        matches.append((number, reference_number, other.start_ms - cue.start_ms, score))
    groups = [
        [other for other in matches if abs(other[2] - match[2]) <= 3000]
        for match in matches
    ]
    groups.sort(key=len, reverse=True)
    if groups:
        group = sorted(groups[0])
        increasing = all(b[1] > a[1] for a, b in zip(group, group[1:]))
        competing = any(
            len(other) >= max(3, len(group) * 0.8)
            and abs(float(np.median([item[2] for item in other]))
                    - float(np.median([item[2] for item in group]))) > 6000
            for other in groups[1:]
        )
        if (len(group) >= 3 and increasing and not competing
                and source[group[-1][0] - 1].start_ms - source[group[0][0] - 1].start_ms >= 10000):
            selected = max(group, key=lambda match: match[3])
            notify(f"Automatic initial sync point: source cue {selected[0]}, reference cue {selected[1]}; "
                   f"{len(group)} consistent dialogue matches.")
            return selected[0], selected[1]
    raise ValueError("Cannot confidently find an automatic initial sync point. Select matching cues manually.")


def analyze_subtitles(
    source: Sequence[Cue],
    reference: Sequence[Cue],
    initial_audio_ms: int,
    progress: Callable[[str], None] | None = None,
    hearing_impaired_strictness: int = 70,
    processing_device: str = "CPU",
    device_id: int = 0,
    backend: str = "ONNX",
    source_anchor: int = 1,
    reference_anchor: int | None = None,
    automatic_initial: bool = False,
) -> tuple[list[tuple[int, float]], dict[str, int | float]]:
    if not source or not reference:
        raise ValueError("Both source and reference subtitles must contain cues.")
    if initial_audio_ms < 0:
        raise ValueError("The initial audio sync time cannot be negative.")
    if source_anchor is None:
        raise ValueError("Source anchor must be a valid 1-based cue number.")
    for number, cues, label in (
        (source_anchor, source, "Source"), (reference_anchor, reference, "Reference"),
    ):
        if number is not None and (
            isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= len(cues)
        ):
            raise ValueError(f"{label} anchor must be a valid 1-based cue number.")
    for cues in (source, reference):
        dialogue = [cue for cue in cues if not CREDIT.search(_clean_caption(cue.text))]
        if any(right.start_ms < left.start_ms for left, right in zip(dialogue, dialogue[1:])):
            raise ValueError("Subtitle dialogue cues must be in chronological order.")
    anchor_time = reference[reference_anchor - 1].start_ms if reference_anchor is not None else initial_audio_ms
    initial_offset_ms = anchor_time - source[source_anchor - 1].start_ms

    notify = progress or (lambda _message: None)
    if any(right.start_ms < left.start_ms for cues in (source, reference)
           for left, right in zip(cues, cues[1:])):
        notify("Out-of-order credit cues excluded from dialogue matching; original cue order is preserved.")
    source_dialogue = [cue for cue in source
                       if not CREDIT.search(_clean_caption(cue.text))
                       and not _is_hearing_impaired_caption(cue.text, hearing_impaired_strictness)]
    reference_dialogue = [cue for cue in reference
                          if not CREDIT.search(_clean_caption(cue.text))
                          and not _is_hearing_impaired_caption(cue.text, hearing_impaired_strictness)]
    source_windows = _build_windows(source_dialogue, hearing_impaired_strictness)
    windows = _build_windows(reference_dialogue, hearing_impaired_strictness)
    if backend not in ("ONNX", "Vulkan"):
        raise ValueError(f"Unknown inference backend: {backend}.")
    notify(f"Loading the local multilingual model on {processing_device} ({backend})...")
    model = (
        vulkan_embeddings.VulkanEmbedding(device_id, notify)
        if backend == "Vulkan" else _load_embedding_model(processing_device, device_id)
    )

    source_texts = [window.text for window in source_windows]
    reference_texts = [window.text for window in windows]
    source_contexts = [_window_context(window, source_dialogue) for window in source_windows]
    reference_contexts = [_window_context(window, reference_dialogue) for window in windows]
    texts = list(dict.fromkeys(source_texts + reference_texts + source_contexts + reference_contexts))
    try:
        notify(f"Analyzing {len(source)} source cues and surrounding dialogue context...")
        embedded = []
        for index, vector in enumerate(model.embed(texts, batch_size=128)):
            if index % 128 == 0:
                notify(f"Embedding dialogue context {index + 1} / {len(texts)}...")
            embedded.append(vector)
        vectors = np.asarray(embedded, dtype=np.float32)
    finally:
        if backend == "Vulkan":
            model.close()

    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("The embedding backend returned invalid dialogue vectors.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("The embedding backend returned empty dialogue vectors.")
    vectors /= norms
    by_text = dict(zip(texts, vectors))
    if automatic_initial:
        source_anchor, reference_anchor = _automatic_initial_pair(
            source, reference, by_text, hearing_impaired_strictness, notify,
        )
        anchor_time = reference[reference_anchor - 1].start_ms
        initial_offset_ms = anchor_time - source[source_anchor - 1].start_ms
    source_vectors = np.asarray([by_text[text] for text in source_texts])
    reference_vectors = np.asarray([by_text[text] for text in reference_texts])
    source_context_vectors = np.asarray([by_text[text] for text in source_contexts])
    reference_context_vectors = np.asarray([by_text[text] for text in reference_contexts])
    source_by_start: dict[int, list[int]] = {}
    for index, window in enumerate(source_windows):
        source_by_start.setdefault(window.start_ms, []).append(index)
    starts = np.fromiter((window.start_ms for window in windows), dtype=np.int64)

    notify("Matching meaning and dialogue context across both subtitles...")
    candidates: list[Match] = []
    for cue_index, cue in enumerate(source):
        if cue_index % 32 == 0:
            notify(f"Matching dialogue cue {cue_index + 1} / {len(source)}...")
        if cue_index + 1 == source_anchor:
            continue
        clean_text = _clean_caption(cue.text)
        source_indices = source_by_start.get(cue.start_ms, [])
        if not source_indices or _alnum_length(clean_text) < 2 or CREDIT.search(clean_text):
            continue

        expected = cue.start_ms + initial_offset_ms
        distances = np.abs(starts - expected)
        radius = min(
            70_000,
            max(SEARCH_RADIUS_MS, 14_000 + round(cue.start_ms * 0.01)),
        )
        side = starts < anchor_time if cue_index + 1 < source_anchor else starts > anchor_time
        indices = np.flatnonzero((distances <= radius) & side)
        if len(indices) == 0:
            continue

        semantic = source_vectors[source_indices] @ reference_vectors[indices].T
        contextual = source_context_vectors[source_indices] @ reference_context_vectors[indices].T
        combined = 0.75 * semantic + 0.25 * contextual
        # Do not let the shared scene alone turn unrelated lines into confident matches.
        combined = np.where((semantic >= 0.40) | (contextual >= 0.85), combined, -1.0)
        similarities = np.max(combined, axis=0)
        ranks = similarities - 0.15 * distances[indices] / radius
        best_by_start: dict[int, tuple[float, float]] = {}
        for local_index, window_index in enumerate(indices):
            start = windows[int(window_index)].start_ms
            rank = float(ranks[local_index])
            similarity = float(similarities[local_index])
            if start not in best_by_start or rank > best_by_start[start][0]:
                best_by_start[start] = (rank, similarity)
        ranked = sorted(
            ((rank, similarity, start) for start, (rank, similarity) in best_by_start.items()),
            reverse=True,
        )
        if not ranked:
            continue
        best_rank, similarity, best_start = ranked[0]
        margin = best_rank - ranked[1][0] if len(ranked) > 1 else 1.0
        if similarity < MIN_SIMILARITY or (
            margin < MIN_MATCH_MARGIN and similarity < 0.80
        ):
            continue
        target_ms = best_start
        candidates.append(Match(cue_index + 1, target_ms, similarity, margin))

    if not candidates:
        raise ValueError(
            "No reliable contextual dialogue matches were found. Check that the "
            "subtitles belong to the same edition and that the manually selected cues correspond."
        )

    candidates.sort(key=lambda match: match.cue_number)
    monotonic: list[Match] = []
    last_target = anchor_time
    for match in reversed(candidates):
        if match.cue_number < source_anchor and 0 <= match.target_ms < last_target:
            monotonic.append(match)
            last_target = match.target_ms
    monotonic.reverse()
    last_target = anchor_time
    for match in candidates:
        if match.cue_number > source_anchor and match.target_ms > last_target:
            monotonic.append(match)
            last_target = match.target_ms

    if not monotonic:
        raise ValueError("The matched dialogue did not produce a usable sync timeline.")

    # A short local median suppresses isolated bad matches while retaining gradual drift.
    smoothed: list[tuple[int, int, float]] = []
    half_window = 1
    for index, match in enumerate(monotonic):
        nearby = [
            other for other in monotonic[max(0, index - half_window) : index + half_window + 1]
            if (other.cue_number < source_anchor) == (match.cue_number < source_anchor)
        ]
        offsets = [
            other.target_ms - source[other.cue_number - 1].start_ms
            for other in nearby
        ]
        offset_ms = int(np.median(np.asarray(offsets, dtype=np.int64)))
        cue = source[match.cue_number - 1]
        smoothed.append((match.cue_number, cue.start_ms + offset_ms, offset_ms / 1000))

    initial_offset = initial_offset_ms / 1000
    anchors: list[tuple[int, float]] = []
    last_video_time = -1
    for cue_number, target_ms, offset_seconds in smoothed:
        if cue_number < source_anchor and last_video_time < target_ms < anchor_time:
            anchors.append((cue_number, offset_seconds))
            last_video_time = target_ms
    anchors.append((source_anchor, initial_offset))
    last_video_time = anchor_time
    for cue_number, target_ms, offset_seconds in smoothed:
        if cue_number > source_anchor and target_ms > last_video_time:
            anchors.append((cue_number, offset_seconds))
            last_video_time = target_ms

    final_offset = anchors[-1][1]
    for cue_number, cue in reversed(list(enumerate(source, start=1))):
        if CREDIT.search(_clean_caption(cue.text)):
            if cue_number > anchors[-1][0] and cue.start_ms + round(final_offset * 1000) > last_video_time:
                anchors.append((cue_number, final_offset))
            break

    anchors.sort(key=lambda item: item[0])
    if any(right[0] <= left[0] for left, right in zip(anchors, anchors[1:])):
        raise ValueError("Automatic analysis produced invalid subtitle anchors.")
    if any(
        source[right[0] - 1].start_ms + round(right[1] * 1000)
        <= source[left[0] - 1].start_ms + round(left[1] * 1000)
        for left, right in zip(anchors, anchors[1:])
    ):
        raise ValueError(
            "Automatic matches were not in chronological order. Verify the reference SRT."
        )

    diagnostics: dict[str, int | float] = {
        "matched_cues": len(monotonic),
        "total_cues": len(source),
        "match_percent": round(100 * len(monotonic) / max(1, len(source) - 1)),
        "source_anchor": source_anchor,
        "anchor_time_ms": anchor_time,
        "reference_anchor": reference_anchor if reference_anchor is not None else 0,
        "automatic_initial": int(automatic_initial),
    }
    return anchors, diagnostics
