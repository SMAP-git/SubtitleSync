import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import onnxruntime as ort

import gpu_runtime as runtime


class GpuRuntimeTests(unittest.TestCase):
    def test_cancel_terminates_private_worker(self):
        cancel = threading.Event()
        native_popen = subprocess.Popen
        children = []

        def start(*args, **kwargs):
            process = native_popen(
                [sys.executable, "-c", "import sys,time; sys.stdin.read(); time.sleep(30)"],
                **kwargs,
            )
            children.append(process)
            return process

        timer = threading.Timer(0.3, cancel.set)
        timer.start()
        try:
            with (
                patch.object(runtime, "runtime_python", return_value=Path(sys.executable)),
                patch.object(runtime.subprocess, "Popen", side_effect=start),
            ):
                with self.assertRaises(runtime.CancelledError):
                    runtime.run_worker("NVIDIA", 0, {"operation": "test"}, Mock(), cancel=cancel)
            self.assertIsNotNone(children[0].poll())
        finally:
            timer.cancel()
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait()

    def test_legacy_amd_models_and_apus_are_listed(self):
        for model in ("Radeon RX 560", "Radeon RX 560 Series", "Radeon RX 580",
                      "Radeon RX 480", "Radeon RX Vega 56", "Radeon RX 5700 XT"):
            self.assertIn(model, runtime.AMD_GPU_MODELS)
        self.assertIn("Ryzen 5 5600G (Radeon Vega 7)", runtime.AMD_APU_MODELS)
        self.assertTrue(set(runtime.AMD_APU_MODELS).issubset(runtime.AMD_GPU_MODELS))
        self.assertEqual(len(runtime.AMD_GPU_MODELS), len(set(runtime.AMD_GPU_MODELS)))

    def test_rx560_warning_does_not_confuse_rx5600(self):
        for setup in runtime.AMD_SETUP_OPTIONS:
            for model in ("Radeon RX 560", "Radeon RX 560 Series"):
                text, _ = runtime.amd_linux_guidance(setup, model, "Ubuntu", "24.04")
                self.assertIn("RX 560 / Polaris", text)
                self.assertIn("Use CPU", text)
                self.assertIn("not a supported target", text)
            text, _ = runtime.amd_linux_guidance(setup, "Radeon RX 5600 XT", "Ubuntu", "24.04")
            self.assertNotIn("RX 560 / Polaris", text)

    def test_integrated_devices_share_vendor_runtime(self):
        for system in ("win32", "linux"):
            with patch.object(runtime.sys, "platform", system):
                for integrated, vendor in runtime.INTEGRATED_DEVICES.items():
                    self.assertEqual(runtime.runtime_directory(integrated), runtime.runtime_directory(vendor))
                    self.assertEqual(runtime.runtime_python(integrated), runtime.runtime_python(vendor))
                    if vendor == "AMD" and system == "linux":
                        with self.assertRaisesRegex(ValueError, "MIGraphX"):
                            runtime.runtime_package(integrated)
                    else:
                        self.assertEqual(runtime.runtime_package(integrated), runtime.runtime_package(vendor))

    def test_linux_distribution_detection(self):
        for release, expected in (
            ({"ID": "ubuntu", "VERSION_ID": "24.04"}, ("Ubuntu", "24.04")),
            ({"ID": "linuxmint", "VERSION_ID": "22.1"}, ("Linux Mint", "22.1")),
            ({"ID": "custom", "NAME": "Custom Linux", "VERSION_ID": "1"}, ("Custom Linux", "1")),
            ({"ID": "arch"}, ("Arch Linux", "")),
        ):
            with self.subTest(release=release):
                with patch.object(runtime.platform, "freedesktop_os_release", return_value=release):
                    self.assertEqual(runtime.linux_distribution(), expected)

    def test_amd_guidance_requires_complete_selection(self):
        for selection in (
            ("wrong", "Radeon RX 7900 XT", "Ubuntu", "24.04"),
            (runtime.AMD_SETUP_OPTIONS[0], "", "Ubuntu", "24.04"),
            (runtime.AMD_SETUP_OPTIONS[0], "GPU", "", "24.04"),
            (runtime.AMD_SETUP_OPTIONS[1], "GPU", "Ubuntu", " "),
        ):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                runtime.amd_linux_guidance(*selection)

    def test_amd_stages_provide_different_guidance_and_official_links(self):
        with patch.object(runtime.sys, "platform", "linux"):
            driver, driver_url = runtime.amd_linux_guidance(
                runtime.AMD_SETUP_OPTIONS[0], "Radeon RX 7900 XTX", "Ubuntu", "24.04"
            )
            inference, inference_url = runtime.amd_linux_guidance(
                runtime.AMD_SETUP_OPTIONS[1], "Instinct MI300X", "Red Hat Enterprise Linux", "9.6"
            )
        self.assertEqual(driver_url, runtime.AMD_ROCM_URL)
        self.assertEqual(inference_url, runtime.MIGRAPHX_URL)
        self.assertIn("Radeon RX 7900 XTX; Ubuntu 24.04", driver)
        self.assertIn("apt/deb", driver)
        self.assertIn("Radeon GPU or Ryzen APU", driver)
        self.assertIn("alone does not supply", driver)
        self.assertIn("dnf/rpm", inference)
        self.assertIn("Instinct GPU", inference)
        self.assertIn("MIGraphXExecutionProvider", inference)
        self.assertIn("bin/python", inference)
        self.assertIn("Test GPU", inference)
        for text in (driver, inference):
            self.assertIn("Compatibility is NOT verified", text)

    def test_unlisted_models_and_distributions_are_not_claimed_supported(self):
        text, _ = runtime.amd_linux_guidance(
            runtime.AMD_SETUP_OPTIONS[0], "My unlisted GPU", "My Linux", "rolling"
        )
        self.assertIn("No distribution-specific installation path", text)
        self.assertIn("Confirm the exact AMD GPU family", text)
        self.assertIn("use CPU", text)
        suse, _ = runtime.amd_linux_guidance(
            runtime.AMD_SETUP_OPTIONS[0], "Radeon PRO W7900", "openSUSE", "Tumbleweed"
        )
        self.assertIn("SLES support does not imply openSUSE support", suse)

    def test_platform_packages_and_isolation(self):
        with patch.object(runtime.sys, "platform", "win32"):
            self.assertEqual(runtime.runtime_package("AMD"), "onnxruntime-directml")
            self.assertEqual(runtime.runtime_package("INTEL"), "onnxruntime-directml")
            self.assertEqual(runtime.runtime_directory("AMD"), runtime.runtime_directory("INTEL"))
            self.assertEqual(runtime.runtime_python("AMD").name, "python.exe")
        with patch.object(runtime.sys, "platform", "linux"):
            self.assertEqual(runtime.runtime_package("INTEL"), "onnxruntime-openvino")
            with patch.object(runtime.sys, "version_info", (3, 12, 0)):
                self.assertEqual(runtime.runtime_package("NVIDIA"), "onnxruntime-gpu[cuda,cudnn]==1.26.0")
            with patch.object(runtime.sys, "version_info", (3, 10, 0)):
                self.assertEqual(runtime.runtime_package("NVIDIA"), "onnxruntime-gpu[cuda,cudnn]==1.23.2")
            self.assertNotEqual(runtime.runtime_directory("INTEL"), runtime.runtime_directory("NVIDIA"))
            with self.assertRaisesRegex(ValueError, "ROCm/MIGraphX"):
                runtime.runtime_package("AMD")
        with self.assertRaises(ValueError):
            runtime.runtime_directory("../../bad")

    def test_installer_never_modifies_running_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(runtime, "APP_DIR", Path(directory)),
                patch.object(runtime.sys, "platform", "linux"),
                patch.object(runtime.venv, "EnvBuilder") as builder,
                patch.object(runtime, "run_command") as command,
            ):
                def create(path):
                    path.mkdir(parents=True)
                builder.return_value.create.side_effect = create
                runtime.install_runtime("NVIDIA", Mock())
                commands = [call.args[0] for call in command.call_args_list]
                self.assertEqual(len(commands), 3)
                for args in commands:
                    self.assertEqual(args[0], str(runtime.runtime_python("NVIDIA")))
                    self.assertNotEqual(args[0], sys.executable)
                self.assertIn("uninstall", commands[1])
                self.assertIn("onnxruntime", commands[1])
                self.assertIn("--only-binary=:all:", commands[2])
                self.assertIn("[cuda,cudnn]", commands[2][-1])
                self.assertTrue((runtime.runtime_directory("NVIDIA") / "installed").is_file())

    def test_install_failure_does_not_mark_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(runtime, "APP_DIR", Path(directory)),
                patch.object(runtime.venv, "EnvBuilder") as builder,
                patch.object(runtime, "run_command", side_effect=ValueError("no wheel")),
            ):
                builder.return_value.create.side_effect = lambda path: path.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, "no wheel"):
                    runtime.install_runtime("NVIDIA", Mock())
                self.assertFalse(runtime.runtime_ready("NVIDIA"))

    def test_manual_amd_is_not_automatically_installed(self):
        with (
            patch.object(runtime.sys, "platform", "linux"),
            patch.object(runtime.venv, "EnvBuilder") as builder,
        ):
            with self.assertRaisesRegex(ValueError, "MIGraphX"):
                runtime.install_runtime("AMD", Mock())
            builder.assert_not_called()

    def test_failed_test_never_registers_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(runtime, "APP_DIR", Path(directory)),
                patch.object(runtime, "run_worker", side_effect=ValueError("driver missing")),
            ):
                with self.assertRaisesRegex(ValueError, "driver missing"):
                    runtime.test_runtime("NVIDIA", 0, Mock())
                self.assertFalse((runtime.runtime_directory("NVIDIA") / "installed").exists())

    def test_successful_manual_test_registers_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(runtime, "APP_DIR", Path(directory)),
                patch.object(runtime, "run_worker", return_value={"message": "passed"}),
            ):
                python = runtime.runtime_python("AMD")
                python.parent.mkdir(parents=True)
                python.touch()
                self.assertEqual(runtime.test_runtime("AMD", 0, Mock()), "passed")
                self.assertTrue((runtime.runtime_directory("AMD") / "installed").is_file())

    def test_preinstalled_provider_uses_current_python_without_private_environment(self):
        process = Mock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO('{"result": {"message": "passed"}}\n')
        process.wait.return_value = 0
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runtime, "APP_DIR", Path(directory)),
            patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
            patch.object(runtime.subprocess, "Popen", return_value=process) as popen,
        ):
            self.assertEqual(runtime.test_runtime("NVIDIA", 0, Mock()), "passed")
            self.assertEqual(popen.call_args.args[0][0], sys.executable)
            self.assertFalse(runtime.runtime_directory("NVIDIA").exists())

    def test_missing_private_and_main_provider_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runtime, "APP_DIR", Path(directory)),
            patch.object(ort, "get_available_providers", return_value=["CPUExecutionProvider"]),
            patch.object(runtime.subprocess, "Popen") as popen,
        ):
            with self.assertRaisesRegex(ValueError, "GPU runtime is not installed"):
                runtime.run_worker("NVIDIA", 0, {"operation": "test"}, Mock())
            popen.assert_not_called()

    def test_worker_protocol_success_and_errors(self):
        for events, code, expected in (
            ([{"progress": "loading"}, {"result": {"anchors": [[1, 0]]}}], 0, True),
            ([{"error": "missing driver"}], 1, False),
            ([], 1, False),
        ):
            process = Mock()
            process.stdin = io.StringIO()
            process.stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in events))
            process.wait.return_value = code
            process.__enter__ = Mock(return_value=process)
            process.__exit__ = Mock(return_value=False)
            with (
                self.subTest(events=events),
                patch.object(runtime, "runtime_python", return_value=Path(sys.executable)),
                patch.object(runtime.subprocess, "Popen", return_value=process) as popen,
            ):
                progress = Mock()
                if expected:
                    self.assertEqual(
                        runtime.run_worker("NVIDIA", 2, {"operation": "test"}, progress),
                        {"anchors": [[1, 0]]},
                    )
                    progress.assert_called_once_with("loading")
                else:
                    with self.assertRaisesRegex(ValueError, "GPU test/analysis failed"):
                        runtime.run_worker("NVIDIA", 2, {"operation": "test"}, progress)
                self.assertEqual(popen.call_args.args[0][-2:], ["NVIDIA", "2"])

    def test_child_worker_rejects_invalid_device_without_model_download(self):
        result = subprocess.run(
            [sys.executable, str(Path(runtime.__file__)), "--worker", "INVALID", "0"],
            input=json.dumps({"operation": "test"}), capture_output=True, text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 1)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertIn("Unknown processing device", events[-1]["error"])


if __name__ == "__main__":
    unittest.main()
