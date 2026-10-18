import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gpu_runtime as gpu
import managed_updates as updates
import speech_to_text as speech
import subtitle_alignment as alignment
import vulkan_embeddings as vulkan


class ManagedUpdateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manager = updates.ComponentManager(self.root)
        for target, name, value in (
            (updates, "desktop_enabled", True),
            (gpu, "APP_DIR", self.root), (speech, "APP_DIR", self.root),
            (vulkan, "APP_DIR", self.root), (gpu, "CPU_ONLY", False),
        ):
            replacement = patch.object(target, name, return_value=value) if name == "desktop_enabled" else patch.object(target, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        cache = patch.object(alignment, "_model_cache", return_value=self.root / "cache")
        cache.start()
        self.addCleanup(cache.stop)

    def create(self, path, data=b"installed"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def model(self):
        path = self.create(speech.base_model_path("tiny"), b"lmggold")
        return {
            "key": "whisper-model-tiny", "kind": "file", "path": str(path),
            "version": hashlib.sha256(path.read_bytes()).hexdigest(),
            "repo": updates.WHISPER_REPO, "filename": path.name, "previous": None,
        }

    def test_discover_only_installed_models_and_runtimes(self):
        self.assertEqual(self.manager.discover(), [])
        self.create(speech.base_model_path("tiny"))
        self.create(speech.base_runtime_directory("CPU") / "runtime.sha256")
        self.create(gpu.base_runtime_directory("NVIDIA") / "installed")
        self.create(vulkan.runtime_directory() / vulkan.RELEASE / "llama-server")
        self.create(vulkan.runtime_directory() / vulkan.MODEL_FILE)
        snapshot = self.root / "cache" / ("models--" + updates.EMBEDDING_REPO.replace("/", "--")) / "snapshots" / ("a" * 40)
        self.create(snapshot / "model_optimized.onnx")
        keys = {item["key"] for item in self.manager.discover()}
        self.assertEqual(keys, {
            "whisper-model-tiny", "whisper-cpu", "gpu-" + gpu.base_runtime_directory("NVIDIA").name,
            "vulkan-runtime", "vulkan-model", "embedding-model",
        })

    def test_daily_cadence_manual_override_and_independent_failures(self):
        components = [self.model(), {**self.model(), "key": "bad-component"}]
        with (
            patch.object(self.manager, "discover", return_value=components),
            patch.object(self.manager, "_candidate", side_effect=[{"version": "new"}, OSError("offline")]) as candidate,
        ):
            result = self.manager.check()
            self.assertEqual(result[0]["candidate"]["version"], "new")
            self.assertIn("offline", result[1]["error"])
            self.assertIsNone(self.manager.check())
            self.assertEqual(candidate.call_count, 2)
        with patch.object(self.manager, "discover", return_value=[]):
            self.assertEqual(self.manager.check(force=True), [])

    def test_container_has_no_discovery_network_or_disk_changes(self):
        with patch.object(updates, "desktop_enabled", return_value=False), patch.object(self.manager, "discover") as discover:
            self.assertIsNone(self.manager.check(force=True))
            discover.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "desktop-only"):
                self.manager.install_component({}, Mock())
            with self.assertRaisesRegex(RuntimeError, "desktop-only"):
                self.manager.restore_component("x")
            default = self.root / "base"
            self.assertEqual(updates.selected_path("x", default), default)
        self.assertFalse(self.manager.directory.exists())

    def test_github_filters_stable_matching_asset_and_requires_digest(self):
        component = {
            "kind": "archive", "repo": "ggml-org/llama.cpp", "asset": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz",
            "version": "b1",
        }
        release = {
            "tag_name": "b2", "assets": [{"name": "llama-b2-bin-ubuntu-vulkan-x64.tar.gz",
                                       "digest": "sha256:" + "a" * 64, "browser_download_url": "https://github.com/download"}],
        }
        with patch.object(updates, "fetch_json", return_value=[{**release, "prerelease": True}, release]):
            candidate = self.manager._candidate(component, {})
        self.assertEqual(candidate["version"], "b2")
        self.assertEqual(candidate["digest"], "a" * 64)
        release["assets"][0]["digest"] = None
        with patch.object(updates, "fetch_json", return_value=[release]):
            with self.assertRaisesRegex(ValueError, "no verifiable"):
                self.manager._candidate(component, {})

    def test_whisper_support_files_tags_do_not_use_unrelated_releases(self):
        component = {
            "kind": "archive", "repo": "SubtitleEdit/support-files", "tag_prefix": "whispercpp-",
            "asset": "whisper-vulkan-linux64.zip", "version": "whispercpp-195",
        }
        asset = {"name": component["asset"], "digest": "sha256:" + "a" * 64, "browser_download_url": "https://github.com/file"}
        releases = [{"tag_name": "other-project", "assets": [asset]},
                    {"tag_name": "whispercpp-196", "assets": [asset]}]
        with patch.object(updates, "fetch_json", return_value=releases):
            self.assertEqual(self.manager._candidate(component, {})["version"], "whispercpp-196")

    def test_hugging_face_file_checks_use_content_not_repository_date(self):
        component = self.model()
        metadata = {"sha": "a" * 40, "siblings": [
            {"rfilename": component["filename"], "lfs": {"sha256": component["version"]}},
        ]}
        with patch.object(updates, "fetch_json", return_value=metadata):
            self.assertIsNone(self.manager._candidate(component, {}))
        metadata["siblings"][0]["lfs"]["sha256"] = "b" * 64
        with patch.object(updates, "fetch_json", return_value=metadata):
            candidate = self.manager._candidate(component, {})
        self.assertIn("/resolve/" + "a" * 40 + "/", candidate["url"])
        self.assertEqual(candidate["digest"], "b" * 64)

    def test_side_by_side_file_update_restore_and_runtime_lookup(self):
        component = self.model()
        data = b"lmggnew-model"
        component["candidate"] = {
            "version": hashlib.sha256(data).hexdigest(), "filename": "ggml-tiny.bin",
            "url": "https://huggingface.co/file", "digest": hashlib.sha256(data).hexdigest(),
        }

        def download(url, path, digest, progress):
            path.write_bytes(data)

        with patch.object(vulkan, "download", side_effect=download):
            self.manager.install_component(component, Mock())
        self.assertEqual(Path(component["path"]).read_bytes(), b"lmggold")
        active = self.manager.state()["components"][component["key"]]["active"]
        self.assertEqual(Path(active["path"]).read_bytes(), data)
        with patch.object(updates, "ComponentManager", return_value=self.manager):
            self.assertEqual(speech.model_path("tiny"), Path(active["path"]))
            self.manager.restore_component(component["key"])
            self.assertEqual(speech.model_path("tiny"), Path(component["path"]))
            self.manager.restore_component(component["key"])
            self.assertEqual(speech.model_path("tiny"), Path(active["path"]))

    def test_bad_model_or_download_timeout_keeps_selection_and_original(self):
        component = self.model()
        component["candidate"] = {"version": "b" * 64, "filename": "ggml-tiny.bin", "url": "https://huggingface.co/file", "digest": "b" * 64}
        for failure in (None, subprocess.TimeoutExpired("download", 10)):
            with self.subTest(failure=failure):
                def download(url, path, digest, progress):
                    if failure:
                        raise failure
                    path.write_bytes(b"not a model")
                with patch.object(vulkan, "download", side_effect=download):
                    with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                        self.manager.install_component(component, Mock())
                self.assertEqual(self.manager.state()["components"], {})
                self.assertEqual(list(self.manager.directory.glob("version-*")), [])
                self.assertEqual(Path(component["path"]).read_bytes(), b"lmggold")

    def test_gpu_check_runs_in_private_python_and_includes_cuda_packages(self):
        component = {"kind": "gpu", "device": "NVIDIA", "path": str(self.root / "gpu")}
        installed = [{"name": "onnxruntime-gpu", "version": "1.0"}, {"name": "nvidia-cudnn-cu12", "version": "1.0"}]
        outdated = [{**item, "latest_version": "2.0"} for item in installed]
        with patch.object(self.manager, "_run", side_effect=[json.dumps(installed), json.dumps(outdated)]) as run:
            candidate = self.manager._candidate(component, {})
        self.assertEqual(candidate["package"], "onnxruntime-gpu")
        self.assertIn("nvidia-cudnn-cu12 2.0", candidate["version"])
        self.assertTrue(run.call_args_list[0].args[0][0].startswith(str(self.root / "gpu")))

    def test_all_runtime_lookups_honor_selections(self):
        state = self.manager.state()
        gpu_key = "gpu-" + gpu.base_runtime_directory("NVIDIA").name
        paths = {}
        for key in (gpu_key, "vulkan-runtime", "vulkan-model", "whisper-cpu", "embedding-model"):
            path = self.root / key
            path.mkdir()
            paths[key] = path
            state["components"][key] = {"active": {"path": str(path), "version": "new"}, "previous": None}
        self.create(paths["vulkan-runtime"] / ("llama-server.exe" if sys.platform == "win32" else "llama-server"))
        self.create(paths["whisper-cpu"] / ("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"))
        self.manager._save(state)
        with patch.object(updates, "ComponentManager", return_value=self.manager):
            self.assertEqual(gpu.runtime_directory("NVIDIA"), paths[gpu_key])
            self.assertEqual(vulkan.executable().parent, paths["vulkan-runtime"])
            self.assertEqual(vulkan.model_path(), paths["vulkan-model"])
            self.assertEqual(speech.executable("CPU").parent, paths["whisper-cpu"])
            self.assertEqual(updates.selected_path("embedding-model", self.root), paths["embedding-model"])

    def test_missing_selected_component_errors_instead_of_silent_fallback(self):
        self.manager._save({"last_check": 0, "components": {
            "vulkan-model": {"active": {"path": str(self.root / "missing"), "version": "old"}, "previous": None},
        }})
        with patch.object(updates, "ComponentManager", return_value=self.manager):
            with self.assertRaisesRegex(ValueError, "Restore"):
                vulkan.model_path()

    def test_archive_update_is_validated_before_selection(self):
        path = self.create(speech.base_runtime_directory("CPU") / "runtime.sha256").parent
        component = next(item for item in self.manager.discover() if item["key"] == "whisper-cpu")
        component["candidate"] = {
            "version": "b9999", "url": "https://github.com/runtime.zip", "filename": "runtime.zip", "digest": "a" * 64,
        }

        def extract(archive, target):
            self.create(target / ("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"))

        with (
            patch.object(vulkan, "download", side_effect=lambda url, path, digest, progress: path.touch()),
            patch.object(vulkan, "extract_archive", side_effect=extract),
            patch.object(self.manager, "_run", return_value="version") as run,
        ):
            self.manager.install_component(component, Mock())
        self.assertEqual(run.call_args.args[0][-1], "--version")
        self.assertTrue(path.exists())
        selected = self.manager.state()["components"]["whisper-cpu"]["active"]
        with patch.object(updates, "ComponentManager", return_value=self.manager):
            self.assertEqual(speech.executable("CPU").parent, Path(selected["path"]))
            # Pinned initial-install markers must not overwrite a selected newer release.
            with patch.object(speech, "model_path", return_value=self.create(self.root / "model.bin")):
                self.assertEqual(speech.missing_downloads("CPU", "tiny"), [])

    def test_snapshot_checks_each_file_and_validation_shape(self):
        component = {"repo": updates.EMBEDDING_REPO}
        config = b'{"model_type":"bert"}'
        weights = b"onnx weights"
        config_digest = hashlib.sha1(f"blob {len(config)}\0".encode() + config).hexdigest()
        candidate = {"version": "a" * 40, "files": [
            {"rfilename": "config.json", "blobId": config_digest},
            {"rfilename": "model_optimized.onnx", "lfs": {"sha256": hashlib.sha256(weights).hexdigest()}},
        ]}
        directory = self.root / "snapshot"
        directory.mkdir()
        with (
            patch.object(updates.urllib.request, "urlopen", return_value=io.BytesIO(config)),
            patch.object(vulkan, "download", side_effect=lambda url, path, digest, progress: path.write_bytes(weights)),
            patch.object(self.manager, "_run", return_value="") as run,
        ):
            target = self.manager._snapshot(component, candidate, directory, Mock())
        self.assertEqual((target / "config.json").read_bytes(), config)
        self.assertIn("v.shape == (1,384)", run.call_args.args[0][-1])

    def test_snapshot_rejects_path_escape_before_download(self):
        candidate = {"version": "a" * 40, "files": [{"rfilename": "../../escape", "blobId": "a" * 40}]}
        directory = self.root / "snapshot"
        directory.mkdir()
        with patch.object(updates.urllib.request, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.manager._snapshot({"repo": updates.EMBEDDING_REPO}, candidate, directory, Mock())
            download.assert_not_called()

    def test_gpu_update_checks_required_provider_and_retains_previous_environment(self):
        base = gpu.base_runtime_directory("NVIDIA")
        self.create(base / "installed")
        component = next(item for item in self.manager.discover() if item["kind"] == "gpu")
        component["candidate"] = {"version": "2.0", "package": "onnxruntime-gpu"}
        with patch.object(self.manager, "_run", return_value="") as run:
            self.manager.install_component(component, Mock())
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("onnxruntime-gpu[cuda,cudnn]", commands[1])
        self.assertIn("CUDAExecutionProvider", commands[-1][-1])
        self.assertTrue((base / "installed").is_file())
        self.assertEqual(
            self.manager.state()["components"][component["key"]]["previous"]["path"], str(base),
        )

    def test_bulk_recovery_does_not_change_anything_if_a_previous_version_is_missing(self):
        good = self.create(self.root / "previous")
        state = {"last_check": 0, "components": {
            "good": {"active": {"path": str(self.root), "version": "new"},
                     "previous": {"path": str(good), "version": "old"}},
            "bad": {"active": {"path": str(self.root), "version": "new"},
                    "previous": {"path": str(self.root / "missing"), "version": "old"}},
        }}
        self.manager._save(state)
        with self.assertRaisesRegex(ValueError, "no selections were changed"):
            self.manager.restore_all()
        self.assertEqual(self.manager.state(), state)
        state["components"].pop("bad")
        self.manager._save(state)
        self.manager.restore_all()
        self.assertEqual(self.manager.state()["components"]["good"]["active"]["path"], str(good))


if __name__ == "__main__":
    unittest.main()
