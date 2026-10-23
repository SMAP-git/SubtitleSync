"""Desktop library checks and reversible, isolated Python environments."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

PACKAGES = ("fastembed", "imageio-ffmpeg", "onnxruntime", "ffsubsync")
ROOT = Path(__file__).resolve().parent
DAY = 24 * 60 * 60


def desktop_enabled() -> bool:
    return (
        os.environ.get("SUBTITLE_SYNC_DESKTOP_UPDATES", "1") != "0"
        and not Path("/.dockerenv").exists()
        and not Path("/run/.containerenv").exists()
    )


class LibraryManager:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.directory = root / ".library-updates" / ("windows" if os.name == "nt" else "linux")
        self.state_path = self.directory / "state.json"

    def state(self) -> dict:
        if not self.state_path.exists():
            return {"active": "base", "previous": None, "last_check": 0}
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Invalid library update settings.")
        for key in ("active", "previous"):
            value = state.get(key)
            if key == "previous" and value is None:
                continue
            if not isinstance(value, str) or (value != "base" and not re.fullmatch(r"env-[a-f0-9]{32}", value)):
                raise ValueError(f"Invalid library environment: {key}.")
        checked = state.get("last_check", 0)
        if isinstance(checked, bool) or not isinstance(checked, (int, float)) or not 0 <= checked <= time.time() + DAY:
            raise ValueError("Invalid library update check time.")
        return state

    def _save(self, state: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / f"state-{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(state) + "\n", encoding="utf-8")
            temporary.replace(self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextlib.contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "update.lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                if lock.tell() == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise RuntimeError("Another library operation is running. Try again later.") from error
            else:
                import fcntl
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError("Another library operation is running. Try again later.") from error
            try:
                yield
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def python(self, environment: str) -> Path:
        directory = (
            self.root / (".venv" if os.name == "nt" else ".venv-linux")
            if environment == "base" else self.directory / environment
        )
        return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def launch_python(self) -> Path:
        executable = self.python(self.state()["active"])
        if not executable.is_file():
            raise ValueError("Selected library environment is missing. Run the launcher with --restore-libraries.")
        return executable

    def _run(self, arguments: list[str], timeout: int = 120, strict_stderr: bool = False) -> str:
        result = subprocess.run(
            arguments, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=self.root,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            raise RuntimeError(
                f"Library command failed (exit {result.returncode}):\n"
                + result.stdout + result.stderr
            )
        if strict_stderr and result.stderr.strip():
            raise RuntimeError("Library update check did not complete cleanly:\n" + result.stderr)
        return result.stdout

    def check(self, force: bool = False) -> list[dict[str, str]] | None:
        if not desktop_enabled():
            return None
        with self._locked():
            state = self.state()
            if not force and time.time() - state.get("last_check", 0) < DAY:
                return None
            state["last_check"] = time.time()
            self._save(state)
            output = self._run([
                sys.executable, "-m", "pip", "--disable-pip-version-check",
                "list", "--outdated", "--format=json", "--timeout", "15", "--retries", "0",
            ], strict_stderr=True)
            outdated = json.loads(output)
            if not isinstance(outdated, list):
                raise ValueError("Invalid response from the package update check.")
            updates = []
            for item in outdated:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise ValueError("Invalid package in update response.")
                if item["name"].lower().replace("_", "-") not in PACKAGES:
                    continue
                if not all(isinstance(item.get(key), str) for key in ("name", "version", "latest_version")):
                    raise ValueError("Invalid package version in update response.")
                updates.append(item)
            return updates

    def install(self, progress: Callable[[str], None]) -> None:
        if not desktop_enabled():
            raise RuntimeError("Library updates are available only on desktop installations.")
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("Start the app using its desktop launcher before installing library updates.")
        with self._locked():
            state = self.state()
            if Path(sys.executable).absolute() != self.python(state["active"]).absolute():
                raise RuntimeError("Library selection has changed. Restart using the desktop launcher first.")
            name = f"env-{uuid.uuid4().hex}"
            environment = self.directory / name
            executable = self.python(name)
            try:
                progress("Creating a separate library environment...")
                self._run([sys.executable, "-m", "venv", str(environment)])
                progress("Downloading and installing libraries. This may take several minutes...")
                self._run([
                    str(executable), "-m", "pip", "--disable-pip-version-check",
                    "install", "--upgrade", "--only-binary=:all:", "--timeout", "30",
                    *PACKAGES,
                ], timeout=1800)
                progress("Validating the new environment...")
                self._run([str(executable), "-m", "pip", "check"])
                self._run([
                    str(executable), "-c",
                    "import tkinter, fastembed, onnxruntime, imageio_ffmpeg, ffsubsync, runpy; "
                    "runpy.run_path('subtitle_sync_app.pyw'); "
                    "assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()",
                ])
                state["previous"], state["active"] = state["active"], name
                self._save(state)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                if environment.exists():
                    shutil.rmtree(environment)
                raise

    def restore(self) -> None:
        if not desktop_enabled():
            raise RuntimeError("Library version switching is available only on desktop installations.")
        with self._locked():
            state = self.state()
            previous = state.get("previous")
            if previous is None:
                raise ValueError("No previous library environment is available.")
            if not self.python(previous).is_file():
                raise ValueError("The previous library environment is missing.")
            state["active"], state["previous"] = previous, state["active"]
            self._save(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("launch-python", "restore", "restore-components"))
    args = parser.parse_args()
    try:
        manager = LibraryManager()
        if args.operation == "restore":
            manager.restore()
        elif args.operation == "restore-components":
            from managed_updates import ComponentManager
            ComponentManager().restore_all()
        else:
            print(manager.launch_python())
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Library environment error: {error}\n")


if __name__ == "__main__":
    main()
