"""CPU speech-activity synchronization without transcription or translation."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import threading
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import speech_to_text as speech
from manual_sync import input_signatures
from sync_subtitles import TIMESTAMP, decode_srt, ms_to_timestamp, timestamp_to_ms, publish_new_file

Progress = Callable[[str], None]
CancelledError = speech.CancelledError
TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}


@dataclass(frozen=True)
class Track:
    index: int
    kind: str
    codec: str
    label: str


def _run(command: list[str], progress: Progress, cancel: threading.Event) -> str:
    try:
        return speech.run_process(
            command, progress, cancel, operation="Audio synchronization", terminate_tree=True,
        )
    except CancelledError as error:
        raise CancelledError("Audio synchronization cancelled. No output was replaced.") from error


def inspect_tracks(video: Path, progress: Progress, cancel: threading.Event) -> list[Track]:
    if not video.is_file():
        raise ValueError("Choose an existing video or audio file.")
    log = _run([
        speech.ffmpeg_executable(), "-nostdin", "-hide_banner", "-i", str(video),
        "-map", "0:a?", "-c", "copy", "-t", "0", "-f", "null", "-",
    ], progress, cancel)
    tracks = []
    for line in log.split("Output #0", 1)[0].splitlines():
        match = re.search(
            r"Stream #0:(\d+)(?:\[[^\]]+\])?(?:\(([^)]+)\))?: (Audio|Subtitle): (\w+)",
            line,
        )
        if match:
            index, language, kind, codec = match.groups()
            tracks.append(Track(
                int(index), kind, codec,
                f"{index}: {language or 'und'} ({codec})",
            ))
    if not any(track.kind == "Audio" for track in tracks):
        raise ValueError("No audio tracks were found in this file.")
    return tracks


def validate_paths(
    video: Path, source: Path | None, output: Path, protected_paths: tuple[Path, ...] = (),
) -> None:
    inputs = (video,) + ((source,) if source is not None else ())
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {path}")
    if output.suffix.lower() != ".srt":
        raise ValueError("Choose an output filename ending in .srt.")
    if not output.parent.is_dir():
        raise ValueError("The output folder does not exist.")
    for path in inputs + protected_paths:
        if output.resolve() == path.resolve() or (
            output.exists() and path.exists() and output.samefile(path)
        ):
            raise ValueError("The output must not overwrite the video, source or another selected input.")


def retime_original(original: str, aligned: str) -> tuple[str, int]:
    """Replace timestamps only; never adopt the engine's rewritten subtitle text."""
    old, new = list(TIMESTAMP.finditer(original)), list(TIMESTAMP.finditer(aligned))
    if not old or len(old) != len(new):
        raise ValueError("Synchronization removed or added cues. No output was saved.")
    import srt

    original_cues, aligned_cues = list(srt.parse(original)), list(srt.parse(aligned))
    if len(original_cues) != len(old) or len(aligned_cues) != len(new) or any(
        left.content != right.content for left, right in zip(original_cues, aligned_cues)
    ):
        raise ValueError("Synchronization changed subtitle dialogue or order. No output was saved.")
    previous_start = -1
    replacements = []
    for match in new:
        start, end = (timestamp_to_ms(match[key]) for key in ("start", "end"))
        if start < previous_start or end <= start:
            raise ValueError("Synchronization produced invalid cue timing. No output was saved.")
        previous_start = start
        replacements.append((start, end))
    timings = iter(replacements)

    def replace(match: re.Match[str]) -> str:
        start, end = next(timings)
        return ms_to_timestamp(start) + match["arrow"] + ms_to_timestamp(end)

    return TIMESTAMP.sub(replace, original), len(old)


def synchronize(
    video: Path, source: Path | None, output: Path, audio_track: int,
    subtitle_track: int | None, fix_drift: bool, progress: Progress,
    cancel: threading.Event, overwrite: bool = False, protected_paths: tuple[Path, ...] = (),
    *, fix_sections: bool = True,
) -> int:
    validate_paths(video, source, output, protected_paths)
    if audio_track < 0 or (source is None and (subtitle_track is None or subtitle_track < 0)):
        raise ValueError("Select an audio track and an external SRT or embedded text subtitle track.")
    if importlib.util.find_spec("ffsubsync") is None:
        raise ValueError(
            "ffsubsync is missing. Restart with the desktop launcher to install dependencies. "
            "If using a separate Library updates environment, restore the base environment or update its libraries."
        )
    original_output = output.stat() if output.exists() else None
    if original_output is not None and not overwrite:
        raise ValueError("Output already exists. Confirm replacement or choose a new filename.")
    inputs = (video,) + ((source,) if source is not None else ())
    signature = input_signatures(tuple(str(path) for path in inputs))
    ffmpeg = speech.ffmpeg_executable()
    with tempfile.TemporaryDirectory(prefix=".audio-sync-", dir=output.parent) as folder:
        temporary = Path(folder)
        subtitles = temporary / "source.srt"
        if source is None:
            progress("Extracting the selected text subtitle track...")
            _run([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning",
                "-copyts", "-start_at_zero", "-i", str(video), "-map", f"0:{subtitle_track}",
                "-c:s", "srt", str(subtitles),
            ], progress, cancel)
            data = subtitles.read_bytes()
        else:
            data = source.read_bytes()
        original, encoding, bom = decode_srt(data)
        import srt

        parsed = list(srt.parse(original))
        original_matches = list(TIMESTAMP.finditer(original))
        if not original_matches:
            raise ValueError("No SRT subtitle cues were found.")
        if any(
            timestamp_to_ms(match["end"]) <= timestamp_to_ms(match["start"])
            for match in original_matches
        ):
            raise ValueError("Source subtitles contain invalid cue timing. No output was saved.")
        if len(parsed) != len(original_matches):
            # Keep malformed credit timestamps opaque, matching the app's SRT parser.
            progress("Non-standard subtitle blocks detected; matching recognized cue timestamps only. Original text is preserved.")
            parsed = [
                srt.Subtitle(
                    index=index + 1,
                    start=timedelta(milliseconds=timestamp_to_ms(match["start"])),
                    end=timedelta(milliseconds=timestamp_to_ms(match["end"])),
                    content=f"SubSync cue {index + 1}",
                )
                for index, match in enumerate(original_matches)
            ]
        order = sorted(range(len(parsed)), key=lambda index: parsed[index].start)
        ordered = srt.compose([parsed[index] for index in order], reindex=True)
        retime_original(ordered, ordered)
        subtitles.write_bytes(ordered.encode("utf-8"))
        audio = temporary / "audio.wav"
        progress("Extracting selected audio on the video timeline (CPU)...")
        _run([
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-copyts", "-start_at_zero", "-i", str(video), "-map", f"0:{audio_track}",
            "-vn", "-ac", "1", "-ar", "16000", "-af", "aresample=async=1:first_pts=0",
            "-c:a", "pcm_s16le", str(audio),
        ], progress, cancel)
        result = temporary / "aligned.srt"
        command = [
            sys.executable, "-c",
            "import logging; logging.basicConfig(level=logging.INFO, format='%(message)s'); "
            "from ffsubsync.ffsubsync import main; raise SystemExit(main())",
            str(audio), "-i", str(subtitles), "-o", str(result),
            "--ffmpeg-path", ffmpeg, "--vad", "webrtc", "--encoding", "utf-8",
            "--output-encoding", "utf-8", "--max-offset-seconds", "60" if fix_sections else "600",
            "--skip-sync-on-low-quality", "--quality-max-offset-seconds", "60" if fix_sections else "600",
            "--skip-infer-framerate-ratio", "--strict",
        ]
        if fix_drift:
            command.append("--gss")
        else:
            command.append("--no-fix-framerate")
        if fix_sections:
            # A larger penalty than ffsubsync's default avoids chasing short noisy passages.
            command.extend(["--split-penalty", "10"])
        progress("Matching speech activity to subtitles on CPU; no transcription or model download...")

        def report(line: str) -> None:
            progress(line)
            speech_segments = re.search(r"total of speech segments:\s*([0-9.eE+-]+)", line)
            if speech_segments and float(speech_segments[1]) <= 0:
                raise ValueError("No speech was detected in the selected audio track. No output was saved.")
            if "low-quality alignment (" in line:
                raise ValueError("Speech matching was unreliable. No output was saved; try another track or manual synchronization.")

        _run(command, report, cancel)
        if not result.is_file():
            raise ValueError("ffsubsync did not produce synchronized subtitles. No output was saved.")
        aligned, count = retime_original(ordered, result.read_text(encoding="utf-8-sig"))
        if count != len(order):
            raise ValueError("Synchronization removed or added cues. No output was saved.")
        aligned_timings = list(TIMESTAMP.finditer(aligned))
        restored = [""] * count
        for position, original_index in enumerate(order):
            restored[original_index] = (
                aligned_timings[position]["start"] + "\n" + aligned_timings[position]["end"]
            )
        timings = iter(restored)

        def restore(match: re.Match[str]) -> str:
            start, end = next(timings).split("\n")
            return start + match["arrow"] + end

        text = TIMESTAMP.sub(restore, original)
        if fix_sections:
            original_cues = list(TIMESTAMP.finditer(original))
            aligned_cues = list(TIMESTAMP.finditer(text))
            # Compare per-cue shifts, not durations, to find discontinuities between sections.
            shifts = [
                timestamp_to_ms(new["start"]) - timestamp_to_ms(old["start"])
                for old, new in zip(original_cues, aligned_cues)
            ]
            boundaries = [
                index + 1 for index in range(1, count)
                if abs(shifts[index] - shifts[index - 1]) > 500
            ]
            if boundaries:
                progress(
                    "Section timing changes detected near cue(s): "
                    + ", ".join(str(cue) for cue in boundaries)
                    + ". Review several cues on both sides of each boundary: speech-activity "
                    "matching can place a transition a few cues early or late."
                )
        if input_signatures(tuple(str(path) for path in inputs)) != signature:
            raise ValueError("An input changed during synchronization. No output was saved.")
        if (output.stat() if output.exists() else None) != original_output:
            raise ValueError("The output changed during synchronization. It was not replaced.")
        validate_paths(video, source, output, protected_paths)
        if cancel.is_set():
            raise CancelledError("Audio synchronization cancelled. No output was replaced.")
        payload = (b"\xef\xbb\xbf" if bom else b"") + text.encode(encoding)
        result.write_bytes(payload)
        if overwrite:
            result.replace(output)
        else:
            publish_new_file(result, output)
    progress(f"Saved {count} cues. Review the beginning, middle and end in your player.")
    return count
