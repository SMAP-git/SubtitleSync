import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import subtitle_alignment as alignment
import vulkan_embeddings as vulkan


class VulkanTests(unittest.TestCase):
    def test_pinned_platform_packages(self):
        for system in ("linux", "win32"):
            for machine in ("x86_64", "AMD64", "aarch64", "ARM64"):
                with (
                    patch.object(vulkan.sys, "platform", system),
                    patch.object(vulkan.platform, "machine", return_value=machine),
                ):
                    url, digest = vulkan.package()
                    self.assertIn(f"/{vulkan.RELEASE}/", url)
                    self.assertIn("vulkan", url)
                    self.assertEqual(len(digest), 64)
        with patch.object(vulkan.platform, "machine", return_value="unsupported"):
            with self.assertRaises(ValueError):
                vulkan.package()

    def test_download_checksum_failure_leaves_no_partial_or_installed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.gguf"
            response = io.BytesIO(b"bad content")
            response.headers = {"Content-Length": "11"}
            with patch.object(vulkan.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "Checksum"):
                    vulkan.download("https://example.test/model", target, "0" * 64, Mock())
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".gguf.part").exists())

    def test_download_success(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "runtime.zip"
            response = io.BytesIO(b"verified")
            response.headers = {}
            with patch.object(vulkan.urllib.request, "urlopen", return_value=response):
                vulkan.download(
                    "https://example.test/package", target,
                    hashlib.sha256(b"verified").hexdigest(), Mock(),
                )
            self.assertEqual(target.read_bytes(), b"verified")

    def test_archives_cannot_escape_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../escape", "/absolute", "C:/escape", "..\\escape"):
                archive = root / "unsafe.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr(name, "bad")
                with self.assertRaises(ValueError):
                    vulkan.extract_archive(archive, root / "install")
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                link = tarfile.TarInfo("link")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                output.addfile(link)
            with self.assertRaises(ValueError):
                vulkan.extract_archive(archive, root / "install")

    def test_archive_extracts_regular_files_and_safe_library_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "runtime.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                item = tarfile.TarInfo("bin/lib.so.1")
                item.size = 7
                item.mode = 0o755
                output.addfile(item, io.BytesIO(b"library"))
                link = tarfile.TarInfo("bin/lib.so")
                link.type = tarfile.SYMTYPE
                link.linkname = "lib.so.1"
                output.addfile(link)
            vulkan.extract_archive(archive, root / "install")
            self.assertEqual((root / "install/bin/lib.so").read_bytes(), b"library")
            self.assertFalse((root / "install/bin/lib.so").is_symlink())

    def test_device_detection_excludes_software_renderers(self):
        output = (
            "Available devices:\n Vulkan0: AMD Radeon RX 560 Series (4096 MiB)\n"
            " Vulkan1: Intel UHD Graphics\n Vulkan2: llvmpipe (LLVM)\n"
        )
        with (
            patch.object(vulkan, "executable", return_value=Path("/runtime/llama-server")),
            patch.object(vulkan.subprocess, "run", return_value=Mock(returncode=0, stdout=output, stderr="")),
        ):
            self.assertEqual(vulkan.list_devices(), [
                (0, "AMD Radeon RX 560 Series (4096 MiB)"), (1, "Intel UHD Graphics")
            ])
        with (
            patch.object(vulkan, "executable", return_value=Path("/runtime/llama-server")),
            patch.object(vulkan.subprocess, "run", return_value=Mock(returncode=0, stdout="", stderr="")),
        ):
            with self.assertRaisesRegex(ValueError, "No hardware Vulkan"):
                vulkan.list_devices()

    def test_vectors_and_token_limit_are_validated(self):
        model = object.__new__(vulkan.VulkanEmbedding)
        model.progress = Mock()
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1
        model._request = Mock(side_effect=[
            {"tokens": list(range(200))}, {"data": [{"embedding": vector.tolist()}]},
        ])
        result = list(model.embed(["Long dialogue"]))
        self.assertEqual(result[0].shape, (384,))
        tokens = model._request.call_args.args[1]["input"]
        self.assertEqual(len(tokens), 128)
        self.assertEqual(tokens[-1], 199)
        for bad in ([1], [0] * 384, [float("nan")] * 384):
            model._request = Mock(side_effect=[{"tokens": [1, 2]}, {"data": [{"embedding": bad}]}])
            model.diagnostics = Mock(return_value="diagnostics")
            with self.assertRaisesRegex(ValueError, "invalid 384-dimensional"):
                list(model.embed(["test"]))

    def test_invalid_adapter_is_rejected_before_startup(self):
        with (
            patch.object(vulkan, "ready", return_value=True),
            patch.object(vulkan, "list_devices", return_value=[(0, "AMD RX 560")]),
            patch.object(vulkan.subprocess, "Popen") as process,
        ):
            with self.assertRaisesRegex(ValueError, "adapter 5"):
                vulkan.VulkanEmbedding(5, Mock())
            process.assert_not_called()

    def test_startup_rejects_cpu_fallback_and_terminates_its_process(self):
        process = Mock()
        process.poll.return_value = None
        with (
            patch.object(vulkan, "ready", return_value=True),
            patch.object(vulkan, "list_devices", return_value=[(0, "AMD RX 560")]),
            patch.object(vulkan, "executable", return_value=Path("/runtime/llama-server")),
            patch.object(vulkan.subprocess, "Popen", return_value=process) as start,
            patch.object(vulkan.VulkanEmbedding, "_request", return_value={"status": "ok"}),
        ):
            with self.assertRaisesRegex(ValueError, "CPU fallback was not accepted"):
                vulkan.VulkanEmbedding(0, Mock())
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
            args = start.call_args.args[0]
            self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
            self.assertEqual(args[args.index("--device") + 1], "Vulkan0")
            self.assertTrue(args[args.index("--api-key") + 1])

    def test_vulkan_cleanup_on_embedding_failure(self):
        cues = [alignment.Cue(1000, 2000, "Dialogue here")]
        with (
            patch.object(alignment.vulkan_embeddings, "VulkanEmbedding") as constructor,
            patch.object(alignment, "_load_embedding_model") as onnx,
        ):
            constructor.return_value.embed.side_effect = ValueError("inference error")
            with self.assertRaisesRegex(ValueError, "inference error"):
                alignment.analyze_subtitles(cues, cues, 1000, processing_device="AMD", backend="Vulkan")
            constructor.return_value.close.assert_called_once()
            onnx.assert_not_called()


if __name__ == "__main__":
    unittest.main()
