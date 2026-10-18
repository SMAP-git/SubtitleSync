"""Pinned app-local llama.cpp Vulkan runtime for multilingual text embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

import numpy as np

APP_DIR = Path(__file__).resolve().parent
RELEASE = "b11707"
PACKAGES = {
    ("linux", "x64"): ("ubuntu-vulkan-x64.tar.gz", "5c7726d41d34e4bde75d9b36b4b366a95c6ca64f5713ec787f6b973db1719504"),
    ("linux", "arm64"): ("ubuntu-vulkan-arm64.tar.gz", "acd15202fdc21d5a599da77a350d8033ece7ed8de206f22f9baa2fb37949f7aa"),
    ("win32", "x64"): ("win-vulkan-x64.zip", "9c7d17c12053bf95b03a179f9bcdf15fdeafe680a4f2845d23bc951bc18c0fa5"),
    ("win32", "arm64"): ("win-vulkan-arm64.zip", "5903c75ebc433500d6e1d896562a75decf750bda35d7bab60fc4c53041196313"),
}
MODEL_FILE = "paraphrase-multilingual-MiniLM-L12-118M-v2-F16.gguf"
MODEL_URL = (
    "https://huggingface.co/mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf/"
    f"resolve/a964c09982b6e28b13646d5f9860b8a04f4182c4/{MODEL_FILE}"
)
MODEL_SHA256 = "fa47907f8732d52ba31796d74bc87c4642b76da465d7d1f74df04ca89f3a9aab"
Progress = Callable[[str], None]


def runtime_directory() -> Path:
    return APP_DIR / ".gpu-runtimes" / f"vulkan-{sys.platform}-{platform.machine().lower()}"


def package() -> tuple[str, str]:
    if os.environ.get("SUBTITLE_SYNC_CPU_ONLY") == "1":
        raise ValueError("Vulkan is disabled in the CPU-only image. Use the full GPU image.")
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    suffix, digest = PACKAGES.get((sys.platform, arch), ("", ""))
    if not suffix:
        raise ValueError("Automatic Vulkan setup supports Windows/Linux x64 and ARM64 only.")
    return f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/llama-{RELEASE}-bin-{suffix}", digest


def executable() -> Path:
    from managed_updates import selected_path
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    matches = list(selected_path("vulkan-runtime", runtime_directory() / RELEASE).rglob(name))
    if len(matches) != 1:
        raise ValueError("Vulkan runtime is not installed. Open GPU setup and install Vulkan support.")
    return matches[0]


def ready() -> bool:
    try:
        executable()
    except ValueError:
        return False
    return model_path().is_file()


def model_path() -> Path:
    from managed_updates import selected_path
    return selected_path("vulkan-model", runtime_directory() / MODEL_FILE)


def download(
    url: str, destination: Path, digest: str, progress: Progress,
    check_cancelled: Callable[[], None] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SubtitleSync"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            size = int(response.headers.get("Content-Length", "0"))
            received = 0
            last_report = -1
            checksum = hashlib.sha256()
            while chunk := response.read(1024 * 1024):
                if check_cancelled is not None:
                    check_cancelled()
                checksum.update(chunk)
                output.write(chunk)
                received += len(chunk)
                report = received * 20 // size if size else received // (20 * 1024 * 1024)
                if report != last_report:
                    progress(f"Downloaded {received // (1024 * 1024)} MB: {destination.name}")
                    last_report = report
            if checksum.hexdigest() != digest:
                raise ValueError(f"Checksum mismatch for {destination.name}; download was not installed.")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _archive_path(directory: Path, name: str) -> Path:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in name:
        raise ValueError("Unsafe path in runtime archive.")
    target = directory.joinpath(*path.parts)
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Runtime archive path escapes installation directory.")
    return target


def extract_archive(archive: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target = _archive_path(directory, item.filename)
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item) as data, target.open("wb") as output:
                        shutil.copyfileobj(data, output)
    else:
        with tarfile.open(archive, "r:gz") as source:
            for item in source.getmembers():
                target = _archive_path(directory, item.name)
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if item.issym() or item.islnk():
                    link = str(PurePosixPath(item.name).parent / item.linkname) if item.issym() else item.linkname
                    _archive_path(directory, link)
                elif not item.isfile():
                    raise ValueError("Unsupported file type in runtime archive.")
                data = source.extractfile(item)
                if data is None:
                    raise ValueError(f"Cannot extract runtime file: {item.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with data, target.open("wb") as output:
                    shutil.copyfileobj(data, output)
                target.chmod(item.mode & 0o777)


def install(progress: Progress) -> None:
    from managed_updates import selected_path
    if (
        selected_path("vulkan-runtime", runtime_directory() / RELEASE) != runtime_directory() / RELEASE
        or model_path() != runtime_directory() / MODEL_FILE
    ):
        raise ValueError("A versioned Vulkan component is selected. Use Library updates to update or restore it.")
    url, digest = package()
    directory = runtime_directory()
    directory.mkdir(parents=True, exist_ok=True)
    progress("Downloading official llama.cpp Vulkan runtime (about 30 MB)...")
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        archive = Path(temporary) / url.rsplit("/", 1)[1]
        download(url, archive, digest, progress)
        extract_archive(archive, directory / RELEASE)
    model = directory / MODEL_FILE
    checksum = hashlib.sha256()
    if model.is_file():
        with model.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                checksum.update(chunk)
    if checksum.hexdigest() != MODEL_SHA256:
        progress("Downloading multilingual MiniLM F16 GGUF conversion (about 242 MB)...")
        download(MODEL_URL, model, MODEL_SHA256, progress)
    progress("Vulkan runtime and model installed. System drivers were not changed.")


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    if sys.platform.startswith("linux"):
        folder = str(executable().parent)
        env["LD_LIBRARY_PATH"] = folder + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    # Do not let inherited llama.cpp flags silently change our device/model setup.
    return {key: value for key, value in env.items() if not key.startswith("LLAMA_ARG_")}


def list_devices() -> list[tuple[int, str]]:
    result = subprocess.run(
        [str(executable()), "--list-devices"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30, env=_environment(),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if result.returncode:
        raise ValueError("Vulkan device detection failed:\n" + result.stderr[-6000:])
    devices = []
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        match = re.match(r"\s*Vulkan(\d+):\s*(.+)", line)
        if match and not any(
            software in match[2].lower() for software in ("llvmpipe", "lavapipe", "swiftshader")
        ):
            devices.append((int(match[1]), match[2]))
    if not devices:
        raise ValueError(
            "No hardware Vulkan device detected. Check your existing graphics driver "
            "and Vulkan loader (Linux: vulkaninfo). ROCm/CUDA are not required for Vulkan."
        )
    return devices


class VulkanEmbedding:
    def __init__(self, device_id: int, progress: Progress) -> None:
        if not ready():
            raise ValueError("Install Vulkan support using GPU setup first.")
        if device_id not in dict(list_devices()):
            raise ValueError(f"Vulkan adapter {device_id} is unavailable. Use Detect GPUs in GPU setup.")
        self.device_id = device_id
        self.progress = progress
        self.process: subprocess.Popen | None = None
        self.log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        self.token = secrets.token_urlsafe(32)
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        command = [
            str(executable()), "-m", str(model_path()),
            "--embedding", "--pooling", "mean", "--embd-normalize", "2",
            "--host", "127.0.0.1", "--port", str(port), "--api-key", self.token,
            "--device", f"Vulkan{device_id}", "-ngl", "99",
            "--ctx-size", "512", "--batch-size", "512", "--ubatch-size", "512",
            "--parallel", "1", "--no-warmup", "--no-webui",
            "-lv", "4",
        ]
        try:
            progress(f"Loading multilingual MiniLM on Vulkan adapter {device_id}...")
            self.process = subprocess.Popen(
                command, stdout=self.log, stderr=self.log, env=_environment(),
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ValueError("Vulkan inference engine exited during startup.")
                try:
                    self._request("/health", timeout=2)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise ValueError("Timed out loading the Vulkan inference engine.")
            self.log.seek(0)
            startup = self.log.read()
            if not re.search(r"offloaded [1-9]\d*/\d+ layers to GPU", startup):
                raise ValueError("The model was not offloaded to GPU; CPU fallback was not accepted.")
        except Exception as error:
            details = self.diagnostics()
            self.close()
            raise ValueError(f"Cannot start Vulkan inference: {error}\n{details}") from error

    def diagnostics(self) -> str:
        self.log.seek(0)
        return self.log.read()[-10000:]

    def _request(self, endpoint: str, data: dict | None = None, timeout: int = 120):
        request = urllib.request.Request(
            self.url + endpoint, data=json.dumps(data).encode() if data is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with self.http.open(request, timeout=timeout) as response:
            return json.load(response)

    def embed(self, texts: Iterable[str], batch_size: int = 128) -> Iterable[np.ndarray]:
        # One input per request avoids cross-sequence truncation in non-causal models.
        for index, text in enumerate(texts):
            if index % 32 == 0:
                self.progress(f"Vulkan: embedding dialogue text {index + 1}...")
            try:
                tokens = self._request(
                    "/tokenize", {"content": text, "add_special": True, "parse_special": False}
                )["tokens"]
                if len(tokens) > 128:
                    tokens = tokens[:127] + tokens[-1:]
                result = self._request("/v1/embeddings", {"input": tokens, "encoding_format": "float"})
                vector = np.asarray(result["data"][0]["embedding"], dtype=np.float32)
                if vector.shape != (384,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 0.01:
                    raise ValueError("Vulkan returned an invalid 384-dimensional embedding.")
                yield vector
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as error:
                raise ValueError(f"Vulkan embedding failed: {error}\n{self.diagnostics()}") from error

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def test_runtime(device_id: int, progress: Progress) -> str:
    model = VulkanEmbedding(device_id, progress)
    try:
        vectors = list(model.embed(["Hello, how are you?", "Ola, como estas?", "The train leaves tomorrow."]))
        similarity = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        if similarity < 0.7 or similarity <= unrelated + 0.1:
            raise ValueError("Vulkan multilingual embedding quality test failed.")
        return f"Vulkan inference passed on adapter {device_id}; translation similarity {similarity:.3f}."
    finally:
        model.close()
