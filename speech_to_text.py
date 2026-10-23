"""Local Whisper.cpp transcription with independent CPU, Vulkan and CUDA runtimes."""

from __future__ import annotations

import hashlib
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gpu_runtime
import sync_subtitles
from subtitle_alignment import Cue, parse_srt
from vulkan_embeddings import download, extract_archive

APP_DIR = Path(__file__).resolve().parent
WHISPER_TAG = "b5454"
MODEL_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
Progress = Callable[[str], None]
BACKENDS = ("Recommended", "CPU") if gpu_runtime.CPU_ONLY else ("Recommended", "CPU", "Vulkan", "CUDA")


@dataclass(frozen=True)
class Model:
    size_mb: int
    sha256: str


MODELS = {
    "tiny": Model(78, "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"),
    "tiny.en": Model(78, "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f"),
    "base": Model(148, "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"),
    "base.en": Model(148, "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"),
    "small": Model(488, "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"),
    "small.en": Model(488, "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d"),
    "medium": Model(1534, "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208"),
    "medium.en": Model(1534, "cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356"),
    "large-v3": Model(3096, "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"),
    "large-v3-turbo": Model(1625, "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"),
}
# Whisper language codes; translation is deliberately not enabled.
LANGUAGES = {
    "Auto detect": "auto", "Afrikaans": "af", "Albanian": "sq", "Amharic": "am",
    "Arabic": "ar", "Armenian": "hy", "Assamese": "as", "Azerbaijani": "az",
    "Bashkir": "ba", "Basque": "eu", "Belarusian": "be", "Bengali": "bn",
    "Bosnian": "bs", "Breton": "br", "Bulgarian": "bg", "Burmese": "my",
    "Cantonese": "yue", "Catalan": "ca", "Chinese": "zh", "Croatian": "hr",
    "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en",
    "Estonian": "et", "Faroese": "fo", "Finnish": "fi", "French": "fr",
    "Galician": "gl", "Georgian": "ka", "German": "de", "Greek": "el",
    "Gujarati": "gu", "Haitian Creole": "ht", "Hausa": "ha", "Hawaiian": "haw",
    "Hebrew": "he", "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is",
    "Indonesian": "id", "Italian": "it", "Japanese": "ja", "Javanese": "jw",
    "Kannada": "kn", "Kazakh": "kk", "Khmer": "km", "Korean": "ko",
    "Lao": "lo", "Latin": "la", "Latvian": "lv", "Lingala": "ln",
    "Lithuanian": "lt", "Luxembourgish": "lb", "Macedonian": "mk",
    "Malagasy": "mg", "Malay": "ms", "Malayalam": "ml", "Maltese": "mt",
    "Maori": "mi", "Marathi": "mr", "Mongolian": "mn", "Nepali": "ne",
    "Norwegian": "no", "Nynorsk": "nn", "Occitan": "oc", "Pashto": "ps",
    "Persian": "fa", "Polish": "pl", "Portuguese": "pt", "Punjabi": "pa",
    "Romanian": "ro", "Russian": "ru", "Sanskrit": "sa", "Serbian": "sr",
    "Shona": "sn", "Sindhi": "sd", "Sinhala": "si", "Slovak": "sk",
    "Slovenian": "sl", "Somali": "so", "Spanish": "es", "Sundanese": "su",
    "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl", "Tajik": "tg",
    "Tamil": "ta", "Tatar": "tt", "Telugu": "te", "Thai": "th",
    "Tibetan": "bo", "Turkish": "tr", "Turkmen": "tk", "Ukrainian": "uk",
    "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi", "Welsh": "cy", "Yiddish": "yi",
    "Yoruba": "yo",
}
PACKAGES = {
    ("linux", "x64", "CPU"): (
        f"ggml-org/whisper.cpp/releases/download/{WHISPER_TAG}/whisper-bin-ubuntu-x64.tar.gz",
        "a72becf15d7917f990f6313867a52638b82b7f9ef237fb0c980dac56a135781c"),
    ("linux", "arm64", "CPU"): (
        f"ggml-org/whisper.cpp/releases/download/{WHISPER_TAG}/whisper-bin-ubuntu-arm64.tar.gz",
        "6b95ebfc60447df48e70ef00a73bdc3f41679ed2d01ef465827206a2ff325149"),
    ("win32", "x64", "CPU"): (
        f"ggml-org/whisper.cpp/releases/download/{WHISPER_TAG}/whisper-bin-x64.zip",
        "6ba69e3482d7826214f90a6a9c84ca07782aec1e1d0c6a7c30c994fd5d816ccb"),
    ("win32", "arm64", "CPU"): (
        f"ggml-org/whisper.cpp/releases/download/{WHISPER_TAG}/whisper-bin-win-cpu-arm64.zip",
        "28c37e7b598c3d9bbfef94f3bd67f2ee6f12b7c86f3da5edcf5e4308d615ff6d"),
    ("linux", "x64", "Vulkan"): (
        "SubtitleEdit/support-files/releases/download/whispercpp-195/whisper-vulkan-linux64.zip",
        "64d5d302b40c40d9573066ba20a5e17cffc62024f613d67c424ce93e67860344"),
    ("win32", "x64", "Vulkan"): (
        "SubtitleEdit/support-files/releases/download/whispercpp-195/whisper-vulkan-x64.zip",
        "89f2c36455939ad42e08b2107bd6b47105b6d9c1a2b0916cbb500ec27deb520a"),
    ("linux", "x64", "CUDA"): (
        "SubtitleEdit/support-files/releases/download/whispercpp-195/whisper-cuda-linux64.zip",
        "a2c71e97f2e90a449cb261bfdd969e0182c5b15e9023b777dd882d32dcefd9a2"),
    ("win32", "x64", "CUDA"): (
        f"ggml-org/whisper.cpp/releases/download/{WHISPER_TAG}/whisper-bin-win-cuda-12.4.0-x64.zip",
        "afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a"),
}


class CancelledError(RuntimeError):
    pass


def check_cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise CancelledError("Speech-to-text cancelled. No reference SRT was replaced.")


def selected_backend(device: str, choice: str) -> str:
    vendor = gpu_runtime.device_vendor(device)
    if vendor not in ("CPU", "AMD", "INTEL", "NVIDIA") or choice not in BACKENDS:
        raise ValueError("Choose a valid speech-to-text device and backend.")
    if vendor == "CPU":
        if choice not in ("Recommended", "CPU"):
            raise ValueError("Choose a GPU/APU device to use Vulkan or CUDA.")
        return "CPU"
    backend = ("CUDA" if vendor == "NVIDIA" else "Vulkan") if choice == "Recommended" else choice
    if backend == "CUDA" and vendor != "NVIDIA":
        raise ValueError("CUDA requires an NVIDIA GPU. Use Vulkan for AMD/Intel GPUs or APUs.")
    return backend


def package(backend: str) -> tuple[str, str]:
    if gpu_runtime.CPU_ONLY and backend != "CPU":
        raise ValueError("GPU speech is disabled in the CPU-only image. Use the full GPU image.")
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    value = PACKAGES.get((sys.platform, arch, backend))
    if value is None:
        raise ValueError(
            f"No bundled Whisper.cpp {backend} package for {sys.platform}/{platform.machine()}. "
            "Select CPU, or use a supported Windows/Linux x64 installation for GPU transcription."
        )
    return "https://github.com/" + value[0], value[1]


def base_runtime_directory(backend: str) -> Path:
    package(backend)
    return APP_DIR / ".speech-to-text" / f"{WHISPER_TAG}-{sys.platform}-{platform.machine().lower()}-{backend.lower()}"


def runtime_directory(backend: str) -> Path:
    from managed_updates import selected_path
    return selected_path(f"whisper-{backend.lower()}", base_runtime_directory(backend))


def executable(backend: str) -> Path:
    matches = list(runtime_directory(backend).rglob("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"))
    if len(matches) != 1:
        raise ValueError("Install the selected speech-to-text runtime first.")
    return matches[0]


def base_model_path(model: str) -> Path:
    if model not in MODELS:
        raise ValueError("Choose a supported Whisper model.")
    return APP_DIR / ".speech-to-text" / "models" / f"ggml-{model}.bin"


def model_path(model: str) -> Path:
    from managed_updates import selected_path
    return selected_path(f"whisper-model-{model}", base_model_path(model))


def environment(backend: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("WHISPER_ARG_")}
    if sys.platform.startswith("linux"):
        libraries = [str(executable(backend).parent)]
        if backend == "CUDA" and gpu_runtime.runtime_ready("NVIDIA"):
            libraries.extend(str(path) for path in sorted(
                gpu_runtime.runtime_directory("NVIDIA").glob("lib/python*/site-packages/nvidia/*/lib")
            ) if path.is_dir())
        if env.get("LD_LIBRARY_PATH"):
            libraries.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    return env


def missing_downloads(backend: str, model: str) -> list[str]:
    """Report missing cached assets; prepare still verifies model checksums."""
    _, digest = package(backend)
    directory = runtime_directory(backend)
    marker = directory / "runtime.sha256"
    missing = []
    try:
        executable(backend)
    except ValueError:
        missing.append(f"Whisper.cpp {backend} runtime")
    else:
        if directory == base_runtime_directory(backend) and (
            not marker.is_file() or marker.read_text(encoding="ascii") != digest
        ):
            missing.append(f"Whisper.cpp {backend} runtime (incomplete or outdated)")
    path = model_path(model)
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(f"{model} model (about {MODELS[model].size_mb} MB)")
    return missing


def prepare(backend: str, model: str, progress: Progress, cancel: threading.Event) -> None:
    if (
        backend == "CUDA"
        and os.environ.get("SUBTITLE_SYNC_BUNDLED_CUDA") == "0"
        and not gpu_runtime.runtime_ready("NVIDIA")
    ):
        raise ValueError(
            "This slim image downloads NVIDIA support only when requested. "
            "In the main window select NVIDIA and Recommended/ONNX, open GPU setup, "
            "and choose Install / repair GPU support (several GB). "
            "Then retry speech to text. The download persists under /config."
        )
    path = model_path(model)
    url, digest = package(backend)
    directory = runtime_directory(backend)
    marker = directory / "runtime.sha256"
    check = lambda: check_cancelled(cancel)
    check()
    try:
        executable(backend)
        if directory == base_runtime_directory(backend) and (
            not marker.is_file() or marker.read_text(encoding="ascii") != digest
        ):
            raise ValueError("Runtime installation is incomplete.")
    except ValueError:
        if directory != base_runtime_directory(backend):
            raise ValueError("Selected Whisper runtime is incomplete. Restore it in Library updates.") from None
        directory.mkdir(parents=True, exist_ok=True)
        progress(f"Downloading Whisper.cpp {backend} runtime...")
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            archive = Path(temporary) / url.rsplit("/", 1)[1]
            download(url, archive, digest, progress, check)
            check()
            extract_archive(archive, directory)
        if sys.platform != "win32":
            executable(backend).chmod(0o755)
        marker.write_text(digest, encoding="ascii")
    check()
    checksum = hashlib.sha256()
    if path.is_file():
        progress("Verifying cached Whisper model...")
        with path.open("rb") as data:
            while chunk := data.read(1024 * 1024):
                check()
                checksum.update(chunk)
    expected = MODELS[model].sha256
    if path != base_model_path(model):
        from managed_updates import ComponentManager
        expected = ComponentManager().state()["components"][f"whisper-model-{model}"]["active"]["version"]
    if checksum.hexdigest() != expected:
        if path != base_model_path(model):
            raise ValueError("Selected Whisper model checksum mismatch. Restore or update it in Library updates.")
        progress(f"Downloading {model} model (about {MODELS[model].size_mb} MB)...")
        download(
            f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{MODEL_REVISION}/{path.name}",
            path, MODELS[model].sha256, progress, check,
        )
    check()


def run_process(
    command: list[str], progress: Progress, cancel: threading.Event,
    env: dict[str, str] | None = None,
    *, operation: str = "Speech-to-text", terminate_tree: bool = False,
) -> str:
    check_cancelled(cancel)
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        start_new_session=terminate_tree and sys.platform != "win32",
    ) as process:
        events: queue.Queue[str | None] = queue.Queue()

        def read() -> None:
            try:
                for line in process.stdout:
                    events.put(line.rstrip())
            finally:
                events.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        tail: list[str] = []
        try:
            while True:
                check_cancelled(cancel)
                try:
                    line = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if line:
                    progress(line)
                    tail = (tail + [line])[-400:]
            if process.wait() != 0:
                raise ValueError(f"{operation} process failed:\n" + "\n".join(tail[-35:]))
            check_cancelled(cancel)
            return "\n".join(tail)
        finally:
            if process.poll() is None:
                if terminate_tree and sys.platform == "win32":
                    stopped = subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if stopped.returncode and process.poll() is None:
                        raise RuntimeError(f"Cannot stop {operation} process:\n{stopped.stderr}")
                elif terminate_tree:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        if process.poll() is None:
                            raise
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if terminate_tree and sys.platform != "win32":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            reader.join(timeout=5)


def ffmpeg_executable() -> str:
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise ValueError("FFmpeg is missing. Restart using the app launcher to install audio support.") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def audio_start_ms(
    ffmpeg: str, video: Path, stream: int, progress: Progress, cancel: threading.Event,
) -> int:
    log = run_process([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info",
        "-copyts", "-start_at_zero", "-i", str(video), "-map", f"0:a:{stream}",
        "-vn", "-af", "ashowinfo", "-frames:a", "1", "-f", "null", "-",
    ], progress, cancel)
    first = re.search(
        r"ashowinfo[^\n]*?\bn:0\s+pts:\S+\s+pts_time:([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        log, re.IGNORECASE,
    )
    if not first:
        raise ValueError("FFmpeg could not determine the selected audio stream's start time. No subtitles were generated.")
    return round(float(first[1]) * 1000)


def offset_srt(text: str, offset_ms: int) -> str:
    def replace(match: re.Match[str]) -> str:
        start = sync_subtitles.timestamp_to_ms(match["start"]) + offset_ms
        end = sync_subtitles.timestamp_to_ms(match["end"]) + offset_ms
        return sync_subtitles.ms_to_timestamp(start) + match["arrow"] + sync_subtitles.ms_to_timestamp(end)
    return sync_subtitles.TIMESTAMP.sub(replace, text)


def validate_options(model: str, language: str, device_id: int, audio_stream: int) -> None:
    model_path(model)
    if language not in LANGUAGES.values():
        raise ValueError("Choose a supported input language.")
    if model.endswith(".en") and language != "en":
        raise ValueError("English-only (.en) models require English. Select a multilingual model for other languages or auto detection.")
    if language == "yue" and model not in ("large-v3", "large-v3-turbo"):
        raise ValueError("The Cantonese language code requires large-v3 or large-v3-turbo.")
    for value, label in ((device_id, "GPU adapter ID"), (audio_stream, "Audio stream index")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer.")


def whisper_command(
    backend: str, model: str, language: str, device_id: int, audio: Path, output_prefix: Path,
    split_lines: bool = True,
) -> list[str]:
    command = [
        str(executable(backend)), "-m", str(model_path(model)), "-f", str(audio),
        "-l", language, "-osrt", "-of", str(output_prefix), "-pp", "-nfa",
    ]
    if split_lines:
        command += ["-ml", "84", "-sow"]
    command += ["-ng"] if backend == "CPU" else ["-dev", str(device_id)]
    return command


def repair_word_split_cues(cues: list[Cue]) -> tuple[list[Cue], int, int]:
    """Fix the zero-length/empty pieces that Whisper's -ml/-sow word splitting can emit.

    Returns (cues, merged zero-length cues, dropped empty cues). Raises if timings are reversed.
    """
    for index, cue in enumerate(cues, 1):
        if cue.end_ms < cue.start_ms:
            raise ValueError(
                f"Whisper generated invalid subtitle cues. Cue {index}: "
                f"{sync_subtitles.ms_to_timestamp(cue.start_ms)} --> "
                f"{sync_subtitles.ms_to_timestamp(cue.end_ms)}; reversed timing. "
                "No output was replaced. Try another model or check the selected audio stream."
            )
    spoken = [cue for cue in cues if cue.text.strip()]
    dropped = len(cues) - len(spoken)
    timed = [index for index, cue in enumerate(spoken) if cue.end_ms > cue.start_ms]
    if not timed:
        if spoken:
            raise ValueError(
                "Whisper generated only zero-length subtitle cues. No output was replaced. "
                "Try another model or check the selected audio stream."
            )
        return [], 0, dropped
    groups: dict[int, list[int]] = {index: [index] for index in timed}
    for index, cue in enumerate(spoken):
        if index in groups:
            continue
        before = max((item for item in timed if item < index), default=None)
        after = min((item for item in timed if item > index), default=None)
        if after is None or (
            before is not None
            and cue.start_ms - spoken[before].end_ms <= spoken[after].start_ms - cue.end_ms
        ):
            groups[before].append(index)
        else:
            groups[after].append(index)
    repaired = []
    for target in timed:
        members = [spoken[index] for index in sorted(groups[target])]
        repaired.append(Cue(
            min(cue.start_ms for cue in members),
            max(cue.end_ms for cue in members),
            " ".join(" ".join(cue.text.split()) for cue in members),
        ) if len(members) > 1 else spoken[target])
    return repaired, len(spoken) - len(timed), dropped


def render_srt(cues: list[Cue]) -> str:
    return "".join(
        f"{number}\n{sync_subtitles.ms_to_timestamp(cue.start_ms)} --> "
        f"{sync_subtitles.ms_to_timestamp(cue.end_ms)}\n{cue.text}\n\n"
        for number, cue in enumerate(cues, 1)
    )


def verify_gpu(log: str, backend: str) -> None:
    if backend == "CPU":
        return
    # whisper.cpp logs its selected inference backend, separately from enumerating devices.
    pattern = r"using\s+(?:Vulkan|CUDA)\d*\s+backend"
    if not re.search(pattern, log, re.IGNORECASE):
        raise ValueError(
            f"Whisper did not confirm active {backend} GPU inference; CPU fallback was not accepted. "
            "Check the adapter ID and driver. Select CPU explicitly to continue without GPU."
        )
    if not re.search(rf"using\s+{backend}\d*\s+backend", log, re.IGNORECASE):
        raise ValueError(f"Whisper activated a different backend instead of {backend}.")
    if backend == "Vulkan":
        active = re.search(r"using\s+Vulkan(\d+)\s+backend", log, re.IGNORECASE)
        if active:
            device = re.search(rf"ggml_vulkan:\s*{active[1]}\s*=\s*(.+)", log)
            if device and any(name in device[1].lower() for name in ("llvmpipe", "lavapipe", "swiftshader")):
                raise ValueError("Whisper selected a software Vulkan device, not a hardware GPU/APU.")


def detect_devices(backend: str, progress: Progress, cancel: threading.Event) -> list[tuple[int, str]]:
    if backend == "CPU":
        return []
    # Initializing on a tiny silent WAV makes Whisper print its own device IDs.
    prepare(backend, "tiny", progress, cancel)
    with tempfile.TemporaryDirectory(prefix="subtitle-speech-probe-") as folder:
        audio = Path(folder) / "probe.wav"
        with wave.open(str(audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 16000)
        command = whisper_command(backend, "tiny", "auto", 0, audio, Path(folder) / "probe")
        log = run_process(command, progress, cancel, environment(backend))
        verify_gpu(log, backend)
    found = []
    for line in log.splitlines():
        match = (
            re.search(r"ggml_vulkan:\s*(\d+)\s*=\s*(.+?)(?:\s*\||$)", line)
            if backend == "Vulkan" else re.search(r"Device\s+(\d+):\s*(.+?)(?:,\s*compute|$)", line)
        )
        if match:
            found.append((int(match[1]), match[2].strip()))
    if not found:
        raise ValueError("Whisper did not report device names. Inspect the log and driver; no adapter was auto-selected.")
    return found


def transcribe(
    video: Path, output: Path, model: str, language: str, device: str, backend_choice: str,
    device_id: int, audio_stream: int, progress: Progress, cancel: threading.Event,
    overwrite: bool = False, protected_paths: tuple[Path, ...] = (),
) -> int:
    backend = selected_backend(device, backend_choice)
    validate_options(model, language, device_id, audio_stream)
    if not video.is_file():
        raise ValueError("Choose a valid video/audio file.")
    if output.suffix.lower() != ".srt":
        raise ValueError("Speech-to-text output must be an .srt file.")
    if output.resolve() in {path.resolve() for path in (video, *protected_paths)}:
        raise ValueError("Generated subtitles must not overwrite any input or synchronized-output file.")
    if output.exists() and not overwrite:
        raise ValueError("Output already exists. Confirm replacement or choose a different output.")
    if not output.parent.is_dir():
        raise ValueError("The output folder does not exist.")
    ffmpeg = ffmpeg_executable()
    original = video.stat()
    previous_output = output.stat() if output.exists() else None
    progress(f"Inspecting audio stream {audio_stream} and its media-timeline start...")
    offset_ms = audio_start_ms(ffmpeg, video, audio_stream, progress, cancel)
    progress(f"Audio stream starts at {offset_ms} ms relative to media start.")
    prepare(backend, model, progress, cancel)
    # Stage beside the destination so the final rename is atomic on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".subtitle-speech-", dir=output.parent) as folder:
        audio = Path(folder) / "audio.wav"
        prefix = Path(folder) / "transcript"
        progress(f"Extracting audio stream {audio_stream} as 16 kHz mono PCM...")
        run_process([
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-y",
            "-copyts", "-start_at_zero", "-i", str(video), "-map", f"0:a:{audio_stream}",
            "-vn", "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0", "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(audio),
        ], progress, cancel)
        progress(f"Transcribing locally: {model}, language={language}, backend={backend}, adapter={device_id}...")
        logs: list[str] = []

        def report(line: str) -> None:
            if "backend" in line.lower() or "ggml_vulkan:" in line:
                logs.append(line)
            if backend != "CPU" and line.startswith("whisper_init_state:"):
                verify_gpu("\n".join(logs), backend)
            progress(line)

        result = prefix.with_suffix(".srt")
        check_cancelled(cancel)
        logs.clear()
        result.unlink(missing_ok=True)
        log = run_process(
            whisper_command(backend, model, language, device_id, audio, prefix),
            report, cancel, environment(backend),
        )
        verify_gpu("\n".join(logs) + "\n" + log, backend)
        if not result.is_file():
            raise ValueError("Whisper did not generate an SRT file. See the transcription log.")
        text = offset_srt(result.read_text(encoding="utf-8-sig"), offset_ms)
        cues = parse_srt(text)
        repaired, merged, dropped = repair_word_split_cues(cues)
        if merged or dropped:
            cues = repaired
            text = render_srt(cues)
            progress(
                f"Word-based line splitting produced {merged} zero-length and {dropped} empty cue(s). "
                "Zero-length words were joined to the nearest neighbouring line; empty cues contained "
                "no dialogue. No transcription was repeated and no timings were invented."
            )
        silence = {"[blank_audio]", "[silence]", "(silence)", "[no_speech]", "[music]", "(music)"}
        if all(cue.text.strip().lower() in silence for cue in cues):
            raise ValueError("No spoken dialogue was recognized. Check the selected audio stream and language.")
        if any(right.start_ms < left.start_ms for left, right in zip(cues, cues[1:])):
            raise ValueError("Whisper generated out-of-order subtitle cues.")
        current = video.stat()
        if (original.st_size, original.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise ValueError("The input video changed during transcription. Output was not replaced.")
        current_output = output.stat() if output.exists() else None
        if previous_output != current_output:
            raise ValueError("The output file changed during transcription. It was not replaced.")
        check_cancelled(cancel)
        # Normalize the generated file to UTF-8 before publishing it.
        result.write_text(text, encoding="utf-8")
        result.replace(output)
    return len(cues)
