"""Versioned desktop runtime/model downloads; never modifies system components."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

from library_updates import DAY, ROOT, LibraryManager, desktop_enabled

EMBEDDING_REPO = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
GGUF_REPO = "mykor/paraphrase-multilingual-MiniLM-L12-v2.gguf"
WHISPER_REPO = "ggerganov/whisper.cpp"


def selected_path(key: str, default: Path) -> Path:
    if not desktop_enabled():
        return default
    record = ComponentManager().state().get("components", {}).get(key, {})
    active = record.get("active")
    if active is None:
        return default
    path = Path(active["path"])
    if not path.exists():
        raise ValueError(f"Selected {key} version is missing. Restore its previous version in Library updates.")
    return path


def fetch_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "SubtitleSync", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def file_hash(path: Path, git_blob: bool = False) -> str:
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ComponentManager(LibraryManager):
    def __init__(self, root: Path = ROOT) -> None:
        super().__init__(root)
        self.directory = root / ".component-updates" / ("windows" if os.name == "nt" else "linux")
        self.state_path = self.directory / "state.json"

    def state(self) -> dict:
        if not self.state_path.exists():
            return {"last_check": 0, "components": {}}
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("components"), dict):
            raise ValueError("Invalid component update settings.")
        checked = state.get("last_check", 0)
        if isinstance(checked, bool) or not isinstance(checked, (float, int)) or not 0 <= checked <= time.time() + DAY:
            raise ValueError("Invalid component check time.")
        for key, record in state["components"].items():
            if not isinstance(record, dict):
                raise ValueError(f"Invalid component selection: {key}")
            for name in ("active", "previous"):
                value = record.get(name)
                if value is not None and (
                    not isinstance(value, dict) or not isinstance(value.get("path"), str)
                    or not Path(value["path"]).is_absolute() or not isinstance(value.get("version"), str)
                ):
                    raise ValueError(f"Invalid {key} {name} version.")
        return state

    def discover(self) -> list[dict]:
        import gpu_runtime as gpu
        import speech_to_text as speech
        import subtitle_alignment as alignment
        import vulkan_embeddings as vulkan

        state = self.state()
        found = []

        def add(key: str, path: Path, version: str, kind: str, **details) -> None:
            record = state["components"].get(key, {})
            active = record.get("active") or {"path": str(path.absolute()), "version": version}
            if Path(active["path"]).exists():
                found.append({
                    "key": key, "kind": kind, **details, **active,
                    "previous": record.get("previous"),
                })

        seen = set()
        if not gpu.CPU_ONLY:
            for device in ("NVIDIA", "AMD", "INTEL"):
                path = gpu.base_runtime_directory(device)
                if path in seen:
                    continue
                seen.add(path)
                if (path / "installed").is_file() or f"gpu-{path.name}" in state["components"]:
                    add(f"gpu-{path.name}", path, "installed", "gpu", device=device)
            path = vulkan.runtime_directory()
            if (path / vulkan.RELEASE).is_dir() or "vulkan-runtime" in state["components"]:
                url, _ = vulkan.package()
                add("vulkan-runtime", path / vulkan.RELEASE, vulkan.RELEASE, "archive",
                    repo="ggml-org/llama.cpp", asset=url.rsplit("/", 1)[1],
                    executable="llama-server.exe" if sys.platform == "win32" else "llama-server")
            if (path / vulkan.MODEL_FILE).is_file() or "vulkan-model" in state["components"]:
                add("vulkan-model", path / vulkan.MODEL_FILE, vulkan.MODEL_SHA256, "file",
                    repo=GGUF_REPO, filename=vulkan.MODEL_FILE)
        for backend in speech.BACKENDS:
            if backend == "Recommended":
                continue
            try:
                url, _ = speech.package(backend)
            except ValueError:
                continue
            path = speech.base_runtime_directory(backend)
            key = f"whisper-{backend.lower()}"
            if (path / "runtime.sha256").is_file() or key in state["components"]:
                parts = url.split("/")
                add(key, path, parts[-2], "archive", repo="/".join(parts[3:5]),
                    asset=parts[-1], tag_prefix="whispercpp-" if "SubtitleEdit/" in url else "",
                    executable="whisper-cli.exe" if sys.platform == "win32" else "whisper-cli")
        for model in speech.MODELS:
            key = f"whisper-model-{model}"
            path = speech.base_model_path(model)
            if path.is_file() or key in state["components"]:
                add(key, path, speech.MODELS[model].sha256, "file", repo=WHISPER_REPO,
                    filename=path.name)
        # Fastembed's normal Hugging Face cache retains immutable snapshot directories.
        cache = alignment._model_cache()
        repositories = [
            path for path in cache.glob("models--*")
            if path.name.lower() == ("models--" + EMBEDDING_REPO.replace("/", "--")).lower()
        ]
        snapshots = [path for repo in repositories for path in (repo / "snapshots").glob("*")]
        models = [path for path in snapshots if (path / "model_optimized.onnx").is_file()]
        legacy = cache / "fast-paraphrase-multilingual-MiniLM-L12-v2"
        if (legacy / "model_optimized.onnx").is_file():
            models.append(legacy)
        if models or "embedding-model" in state["components"]:
            path = max(models, key=lambda entry: entry.stat().st_mtime) if models else cache
            for repo in sorted(repositories, key=lambda entry: entry.name != "models--" + EMBEDDING_REPO.replace("/", "--")):
                reference = repo / "refs" / "main"
                if reference.is_file():
                    revision = reference.read_text(encoding="ascii").strip()
                    if re.fullmatch(r"[a-f0-9]{40}", revision):
                        cached = repo / "snapshots" / revision
                        if cached in models:
                            path = cached
                            break
            add("embedding-model", path, path.name, "snapshot", repo=EMBEDDING_REPO)
        return found

    def _candidate(self, component: dict, cache: dict) -> dict | None:
        kind = component["kind"]
        if kind == "gpu":
            python = Path(component["path"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            installed = json.loads(self._run([
                str(python), "-m", "pip", "list", "--format=json",
            ]))
            providers = [
                item["name"] for item in installed
                if item["name"].lower().replace("_", "-") in (
                    "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-migraphx",
                )
            ]
            if len(providers) != 1:
                raise ValueError("Cannot identify a single private ONNX GPU package; use GPU setup.")
            outdated = json.loads(self._run([
                str(python), "-m", "pip", "--disable-pip-version-check", "list", "--outdated",
                "--format=json", "--timeout", "15", "--retries", "0",
            ], strict_stderr=True))
            managed = [
                item for item in outdated if item["name"].lower().replace("_", "-") in (
                    providers[0].lower().replace("_", "-"), "fastembed",
                ) or item["name"].lower().startswith("nvidia-")
            ]
            component["version"] = ", ".join(
                f"{item['name']} {item['version']}" for item in installed
                if item["name"] in providers or item["name"].lower().startswith("nvidia-")
            )
            if not managed:
                return None
            if component["device"] == "AMD" and sys.platform != "win32":
                raise ValueError("A newer private MIGraphX package is listed; its OS/ROCm-specific wheel requires manual GPU setup.")
            return {
                "version": ", ".join(f"{item['name']} {item['latest_version']}" for item in managed),
                "package": providers[0],
            }
        if kind == "archive":
            repo = component["repo"]
            if repo not in cache:
                url = f"https://api.github.com/repos/{repo}/releases"
                cache[repo] = fetch_json(url + "?per_page=100")
            prefix = component.get("tag_prefix", "")
            releases = [
                release for release in cache[repo]
                if not release.get("draft") and (
                    not release.get("prerelease") or (
                        repo in ("ggml-org/llama.cpp", "ggml-org/whisper.cpp")
                        and re.fullmatch(r"b\d+", release["tag_name"])
                    )
                )
                and release["tag_name"].startswith(prefix)
            ]
            asset_name = component["asset"]
            suffix = re.sub(r"llama-b\d+-bin-", "", asset_name)
            for release in releases:
                assets = [
                    asset for asset in release["assets"]
                    if (asset["name"] == asset_name or (
                        repo == "ggml-org/llama.cpp" and asset["name"].endswith("-bin-" + suffix)
                    ))
                ]
                if not assets:
                    continue
                if release["tag_name"] == component["version"]:
                    return None
                if re.fullmatch(r"b\d+", component["version"]) and re.fullmatch(r"b\d+", release["tag_name"]):
                    if int(release["tag_name"][1:]) <= int(component["version"][1:]):
                        return None
                asset = assets[0]
                digest = asset.get("digest") or ""
                if not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", digest):
                    raise ValueError(f"{release['tag_name']} has no verifiable SHA-256 asset digest; update not offered.")
                return {
                    "version": release["tag_name"], "url": asset["browser_download_url"],
                    "digest": digest[7:], "filename": asset["name"],
                }
            raise ValueError("No stable upstream release asset matches this installed platform/backend.")
        repo = component["repo"]
        if repo not in cache:
            cache[repo] = fetch_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
        info = cache[repo]
        revision = info["sha"]
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Invalid upstream model revision.")
        if kind == "snapshot":
            if component["version"] == revision:
                return None
            files = [
                item for item in info["siblings"]
                if item["rfilename"].endswith((".json", ".txt", ".model", ".onnx_data"))
                or item["rfilename"] == "model_optimized.onnx"
            ]
            if not any(item["rfilename"] == "model_optimized.onnx" for item in files):
                raise ValueError("The upstream revision no longer contains the expected embedding model.")
            return {"version": revision, "repo": repo, "files": files}
        sibling = next((item for item in info["siblings"] if item["rfilename"] == component["filename"]), None)
        digest = sibling.get("lfs", {}).get("sha256") if sibling else None
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Model metadata has no verifiable SHA-256 checksum.")
        current = component["version"]
        if not re.fullmatch(r"[a-f0-9]{64}", current):
            current = file_hash(Path(component["path"]))
        if current == digest:
            return None
        return {
            "version": digest, "revision": revision, "digest": digest, "filename": component["filename"],
            "url": f"https://huggingface.co/{repo}/resolve/{revision}/{component['filename']}",
        }

    def check(self, force: bool = False) -> list[dict] | None:
        if not desktop_enabled():
            return None
        with self._locked():
            state = self.state()
            if not force and time.time() - state["last_check"] < DAY:
                return None
            state["last_check"] = time.time()
            self._save(state)
            components = self.discover()
            cache = {}
            for component in components:
                try:
                    component["candidate"] = self._candidate(component, cache)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
                    component["error"] = f"Could not check {component['key']}: {error}"
            return components

    def install_component(self, component: dict, progress: Callable[[str], None]) -> None:
        if not desktop_enabled():
            raise RuntimeError("Component updates are desktop-only.")
        key = component["key"]
        candidate = component.get("candidate")
        if not candidate:
            raise ValueError("Check this component for updates first.")
        with self._locked():
            state = self.state()
            current = next((item for item in self.discover() if item["key"] == key), None)
            if current is None or current["path"] != component["path"]:
                raise ValueError("Component selection changed. Check for updates again.")
            directory = self.directory / ("version-" + uuid.uuid4().hex)
            directory.mkdir()
            try:
                kind = component["kind"]
                progress(f"Installing {key} separately; current files are retained...")
                if kind == "gpu":
                    self._run([sys.executable, "-m", "venv", str(directory)])
                    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                    package = candidate["package"]
                    if package.lower().replace("_", "-") == "onnxruntime-gpu":
                        package += "[cuda,cudnn]"
                    self._run([
                        str(python), "-m", "pip", "install", "--only-binary=:all:",
                        "fastembed", "imageio-ffmpeg", package,
                    ], timeout=1800)
                    # fastembed pulls the CPU distribution; GPU workers must contain one ORT distribution.
                    self._run([str(python), "-m", "pip", "uninstall", "-y", "onnxruntime"])
                    self._run([str(python), "-m", "pip", "install", "--force-reinstall", "--only-binary=:all:", package], timeout=1800)
                    provider = (
                        "CUDAExecutionProvider" if component["device"] == "NVIDIA" else
                        "DmlExecutionProvider" if sys.platform == "win32" else "OpenVINOExecutionProvider"
                    )
                    self._run([str(python), "-c",
                               "import fastembed, onnxruntime; "
                               f"assert {provider!r} in onnxruntime.get_available_providers()"])
                    (directory / "installed").write_text(package, encoding="utf-8")
                    target = directory
                elif kind == "snapshot":
                    target = self._snapshot(component, candidate, directory, progress)
                else:
                    from vulkan_embeddings import download, extract_archive
                    target = directory / candidate["filename"]
                    if Path(candidate["filename"]).name != candidate["filename"]:
                        raise ValueError("Invalid upstream download filename.")
                    download(candidate["url"], target, candidate["digest"], progress)
                    if kind == "archive":
                        extract_archive(target, directory / "runtime")
                        target.unlink()
                        target = directory / "runtime"
                        matches = list(target.rglob(component["executable"]))
                        if len(matches) != 1:
                            raise ValueError("Updated archive does not contain exactly one expected executable.")
                        if os.name != "nt":
                            matches[0].chmod(0o755)
                        self._run([str(matches[0]), "--version"])
                    else:
                        self._validate_model(component, target)
                active = {"path": str(target), "version": candidate["version"]}
                state["components"][key] = {
                    "active": active,
                    "previous": {"path": component["path"], "version": component["version"]},
                }
                self._save(state)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError):
                shutil.rmtree(directory)
                raise

    def _snapshot(self, component: dict, candidate: dict, directory: Path, progress: Callable[[str], None]) -> Path:
        from vulkan_embeddings import download
        target = directory / "model"
        target.mkdir()
        for item in candidate["files"]:
            path = target / item["rfilename"]
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError("Model metadata path escapes download directory.")
            digest = item.get("lfs", {}).get("sha256") or item.get("blobId")
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", digest):
                raise ValueError(f"No checksum for model file {item['rfilename']}.")
            url = (
                f"https://huggingface.co/{component['repo']}/resolve/{candidate['version']}/"
                + urllib.parse.quote(item["rfilename"], safe="/")
            )
            progress(f"Downloading embedding model file: {item['rfilename']}")
            if len(digest) == 64:
                download(url, path, digest, progress)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                request = urllib.request.Request(url, headers={"User-Agent": "SubtitleSync"})
                with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if file_hash(path, git_blob=len(digest) == 40) != digest:
                raise ValueError(f"Model file checksum mismatch: {item['rfilename']}")
        self._run([
            sys.executable, "-c",
            "from fastembed.text.pooled_embedding import PooledEmbedding; import numpy as np; "
            "m=PooledEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', "
            f"specific_model_path={str(target)!r}, providers=['CPUExecutionProvider']); "
            "v=np.asarray(list(m.embed(['model test']))); assert v.shape == (1,384) and np.isfinite(v).all()",
        ])
        return target

    def _validate_model(self, component: dict, target: Path) -> None:
        if component["key"] == "vulkan-model":
            with target.open("rb") as source:
                if source.read(4) != b"GGUF":
                    raise ValueError("Downloaded model is not a GGUF file.")
        else:
            with target.open("rb") as source:
                if source.read(4) != b"lmgg":
                    raise ValueError("Downloaded model is not a Whisper GGML file.")

    def restore_component(self, key: str) -> None:
        if not desktop_enabled():
            raise RuntimeError("Component version switching is desktop-only.")
        with self._locked():
            state = self.state()
            record = state["components"].get(key, {})
            previous = record.get("previous")
            if previous is None or not Path(previous["path"]).exists():
                raise ValueError("No previous installed version is available for this component.")
            record["active"], record["previous"] = previous, record["active"]
            self._save(state)

    def restore_all(self) -> None:
        if not desktop_enabled():
            raise RuntimeError("Component version switching is desktop-only.")
        with self._locked():
            state = self.state()
            records = [record for record in state["components"].values() if record.get("previous")]
            if not records:
                raise ValueError("No previous component versions are available.")
            for record in records:
                if not Path(record["previous"]["path"]).exists():
                    raise ValueError("A previous component is missing; no selections were changed.")
            for record in records:
                record["active"], record["previous"] = record["previous"], record["active"]
            self._save(state)
