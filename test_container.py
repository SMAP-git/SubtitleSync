"""Container packaging checks; no daemon or GPU is needed for these tests."""

import subprocess
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from build_portable import COMMON

ROOT = Path(__file__).resolve().parent


class ContainerPackagingTests(unittest.TestCase):
    def test_cpu_variant_has_no_gpu_configuration(self):
        root = ET.parse(ROOT / "unraid/subtitle-sync-cpu.xml").getroot()
        self.assertEqual(root.findtext("Repository"), "subtitle-sync:1.1.1-cpu")
        self.assertNotIn("--gpus", root.findtext("ExtraParams"))
        self.assertNotIn("--device", root.findtext("ExtraParams"))
        self.assertFalse(any(n.attrib["Type"] == "Device" for n in root.findall("Config")))
        code = """
import gpu_runtime as g, subtitle_alignment as a, speech_to_text as s, vulkan_embeddings as v
assert a.PROCESSING_DEVICES == ('CPU',)
assert s.BACKENDS == ('Recommended', 'CPU')
for action in (lambda: g.runtime_directory('NVIDIA'), lambda: s.package('CUDA'), v.package):
    try:
        action()
    except ValueError as error:
        assert 'CPU-only' in str(error)
    else:
        raise AssertionError('GPU operation accepted')
"""
        import os
        import sys
        subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            env={**os.environ, "SUBTITLE_SYNC_CPU_ONLY": "1"}, check=True,
        )

    def test_image_build_supports_full_and_cpu_dependencies(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("ARG WITH_CUDA=1", dockerfile)
        self.assertIn('case "$WITH_CUDA"', dockerfile)
        self.assertIn('0) ;;', dockerfile)
        self.assertFalse((ROOT / "unraid/subtitle-sync-slim.xml").exists())
        self.assertFalse((ROOT / "docker/compose.slim.yml").exists())

    def test_context_is_an_explicit_application_allowlist(self):
        rules = (ROOT / ".dockerignore").read_text().splitlines()
        self.assertEqual(rules[0], "**")
        allowed = {line[1:] for line in rules if line.startswith("!")}
        expected = set(COMMON) | {
            "Dockerfile", "docker/", "docker/startapp.sh", "docker/main-window-selection.xml",
        }
        self.assertEqual(allowed, expected)
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertNotIn("COPY . ", dockerfile)
        for filename in COMMON:
            self.assertIn(filename, dockerfile)
        for filename in allowed - {"docker/"}:
            self.assertTrue((ROOT / filename).is_file(), filename)

    def test_only_main_window_is_maximized_without_decorations(self):
        # The base image's init script reads each tag with awk /Name/, /Class/ etc., so
        # GroupName/GroupClass must not appear or they would be mismatched.
        selection = (ROOT / "docker/main-window-selection.xml").read_text().split()
        self.assertEqual(selection, ["<Class>Subsync</Class>", "<Type>normal</Type>"])
        self.assertIn(
            "docker/main-window-selection.xml /etc/openbox/main-window-selection.xml",
            (ROOT / "Dockerfile").read_text(),
        )

    def test_container_runtime_and_persistence_contract(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        for requirement in (
            "python3-tk", "mesa-vulkan-drivers", "libvulkan1",
            "mkvtoolnix", "LANG=C.UTF-8", "LC_ALL=C.UTF-8",
            "onnxruntime-gpu[cuda,cudnn]", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
            "SUBTITLE_SYNC_SETTINGS_DIR=/config/settings",
            "SUBTITLE_SYNC_MODEL_CACHE=/config/models",
            "ln -s /config/gpu-runtimes .gpu-runtimes",
            "ln -s /config/speech-to-text .speech-to-text",
        ):
            self.assertIn(requirement, dockerfile)
        script = ROOT / "docker/startapp.sh"
        subprocess.run(["sh", "-n", str(script)], check=True)
        text = script.read_text()
        self.assertIn("exec /opt/venv/bin/python", text)
        self.assertNotIn("pip install", text)
        self.assertIn("site-packages/nvidia/*/lib", text)
        self.assertIn("export LANG=C.UTF-8 LC_ALL=C.UTF-8", text)

    def test_unraid_defaults_work_without_gpu(self):
        root = ET.parse(ROOT / "unraid/subtitle-sync.xml").getroot()
        self.assertEqual(root.findtext("Repository"), "subtitle-sync:1.1.1")
        self.assertEqual(root.findtext("Privileged"), "false")
        self.assertNotIn("--gpus", root.findtext("ExtraParams"))
        configs = {node.attrib["Target"]: node for node in root.findall("Config")}
        self.assertNotIn("/dev/dri", configs)
        for node in root.findall("Config"):
            if node.attrib["Type"] == "Device":
                self.assertTrue((node.text or "").strip(), "Unraid emits --device='' for blank devices")
        self.assertIn("--device=/dev/dri:/dev/dri", root.findtext("Overview"))
        self.assertEqual(configs["/config"].text, "/mnt/user/appdata/subtitle-sync")
        self.assertEqual(configs["USER_ID"].text, "99")
        self.assertEqual(configs["GROUP_ID"].text, "100")
        self.assertEqual(
            configs["NVIDIA_DRIVER_CAPABILITIES"].text, "compute,utility,graphics"
        )
        self.assertNotIn("5900", configs)


if __name__ == "__main__":
    unittest.main()
