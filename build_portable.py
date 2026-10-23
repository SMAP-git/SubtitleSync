"""Create clean first-launch-download releases using an explicit source allowlist."""

import hashlib
import tarfile
import zipfile
from pathlib import Path

from ui_preferences import APP_VERSION

ROOT = Path(__file__).resolve().parent
COMMON = (
    "subtitle_sync_app.pyw", "subtitle_alignment.py", "sync_subtitles.py",
    "gpu_runtime.py", "vulkan_embeddings.py", "manual_sync.py", "speech_to_text.py",
    "speech_dialog.py", "ui_preferences.py", "ui_translations.py",
    "requirements.txt", "README.md",
    "file_picker.py",
    "library_updates.py", "desktop_updates.py",
    "managed_updates.py",
    "subtitle_sync_icon.png",
    "audio_sync.py", "audio_sync_dialog.py",
    "sync_report.py",
)


def build(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for platform, launchers in (
        ("windows-x64", ("Start_Subtitle_Sync.bat", "start_windows.ps1")),
        ("linux-x64", ("Start_Subtitle_Sync.sh",)),
    ):
        name = f"SubtitleSync-{APP_VERSION}-{platform}"
        files = COMMON + launchers
        if platform == "windows-x64":
            output = destination / (name + ".zip")
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                for filename in files:
                    archive.write(ROOT / filename, f"{name}/{filename}")
                archive.writestr(f"{name}/PORTABLE", "Portable settings and model cache enabled.\n")
        else:
            import io
            output = destination / (name + ".tar.gz")
            with tarfile.open(output, "w:gz") as archive:
                for filename in files:
                    info = archive.gettarinfo(str(ROOT / filename), f"{name}/{filename}")
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if filename.endswith(".sh") else 0o644
                    with (ROOT / filename).open("rb") as source:
                        archive.addfile(info, source)
                marker = b"Portable settings and model cache enabled.\n"
                info = tarfile.TarInfo(f"{name}/PORTABLE")
                info.size, info.mode = len(marker), 0o644
                archive.addfile(info, io.BytesIO(marker))
        outputs.append(output)
    checksums = destination / "SHA256SUMS.txt"
    checksums.write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in outputs
    ), encoding="ascii")
    return outputs + [checksums]


if __name__ == "__main__":
    for artifact in build(ROOT / "releases"):
        print(artifact)
