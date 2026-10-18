import importlib.machinery
import importlib.util
import os
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import onnxruntime as ort

import subtitle_alignment as alignment


def load_gui():
    loader = importlib.machinery.SourceFileLoader(
        "subtitle_sync_app", str(Path(__file__).with_name("subtitle_sync_app.pyw"))
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("Could not load the GUI module.")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    class OfflineDesktopUpdates(module.DesktopUpdates):
        def schedule_check(self):
            pass

    module.DesktopUpdates = OfflineDesktopUpdates
    module._test_settings_directory = tempfile.TemporaryDirectory()
    module.settings_path = lambda: Path(module._test_settings_directory.name) / "settings.json"
    return module


class ProcessingDeviceTests(unittest.TestCase):
    def test_integrated_devices_use_vendor_provider_and_selected_adapter(self):
        for system in ("win32", "linux"):
            with patch.object(alignment.sys, "platform", system):
                for integrated, vendor in alignment.gpu_runtime.INTEGRATED_DEVICES.items():
                    self.assertIn(integrated, alignment.PROCESSING_DEVICES)
                    self.assertEqual(
                        alignment.execution_provider(integrated, 2),
                        alignment.execution_provider(vendor, 2),
                    )

    def test_platform_provider_mapping(self):
        for platform in ("win32", "linux"):
            with patch.object(alignment.sys, "platform", platform):
                self.assertEqual(
                    alignment.execution_provider("CPU"), ("CPUExecutionProvider", {})
                )
                self.assertEqual(
                    alignment.execution_provider("NVIDIA", 2),
                    ("CUDAExecutionProvider", {"device_id": "2"}),
                )
        with patch.object(alignment.sys, "platform", "win32"):
            for device in ("AMD", "INTEL"):
                self.assertEqual(
                    alignment.execution_provider(device, 1),
                    ("DmlExecutionProvider", {"device_id": "1"}),
                )
        with patch.object(alignment.sys, "platform", "linux"):
            self.assertEqual(
                alignment.execution_provider("AMD", 2),
                ("MIGraphXExecutionProvider", {"device_id": "2"}),
            )
            self.assertEqual(
                alignment.execution_provider("INTEL", 2),
                ("OpenVINOExecutionProvider", {"device_type": "GPU.2"}),
            )

    def test_invalid_selection(self):
        for device_id in (-1, True, 1.5, "0"):
            with self.subTest(device_id=device_id), self.assertRaises(ValueError):
                alignment.execution_provider("NVIDIA", device_id)
        with self.assertRaisesRegex(ValueError, "Unknown processing device"):
            alignment.execution_provider("unknown")
        with patch.object(alignment.sys, "platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "Windows and Linux"):
                alignment.execution_provider("AMD")

    def test_missing_runtime_fails_before_loading_model(self):
        with (
            patch.object(ort, "get_available_providers", return_value=["CPUExecutionProvider"]),
            patch.object(alignment, "PooledEmbedding") as constructor,
        ):
            with self.assertRaisesRegex(ValueError, "CUDAExecutionProvider"):
                alignment._load_embedding_model("NVIDIA", 0)
            constructor.assert_not_called()

    def test_explicit_providers_and_fallback_disabled(self):
        for device, provider, options in (
            ("CPU", "CPUExecutionProvider", {}),
            ("NVIDIA", "CUDAExecutionProvider", {"device_id": "1"}),
            ("AMD", "MIGraphXExecutionProvider", {"device_id": "1"}),
            ("INTEL", "OpenVINOExecutionProvider", {"device_type": "GPU.1"}),
        ):
            session = Mock()
            session.get_providers.return_value = [provider, "CPUExecutionProvider"]
            model = SimpleNamespace(model=session)
            with (
                self.subTest(device=device),
                patch.object(alignment.sys, "platform", "linux"),
                patch.object(ort, "get_available_providers", return_value=[provider]),
                patch.object(alignment, "_model_cache", return_value=Path("cache")),
                patch.object(alignment, "PooledEmbedding", return_value=model) as constructor,
            ):
                self.assertIs(alignment._load_embedding_model(device, 1), model)
                self.assertEqual(constructor.call_args.kwargs["providers"], [(provider, options)])
                session.disable_fallback.assert_called_once()

    def test_silent_cpu_fallback_is_rejected(self):
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        with (
            patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
            patch.object(alignment, "_model_cache", return_value=Path("cache")),
            patch.object(alignment, "PooledEmbedding", return_value=SimpleNamespace(model=session)),
        ):
            with self.assertRaisesRegex(ValueError, "CPU fallback was not accepted"):
                alignment._load_embedding_model("NVIDIA", 0)
            session.disable_fallback.assert_not_called()

    def test_directml_options_and_model_selection(self):
        options = ort.SessionOptions()
        alignment._DirectMLTextEmbedding.add_extra_session_options(options, {})
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        session = Mock()
        session.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
        with (
            patch.object(alignment.sys, "platform", "win32"),
            patch.object(ort, "get_available_providers", return_value=["DmlExecutionProvider"]),
            patch.object(alignment, "_model_cache", return_value=Path("cache")),
            patch.object(
                alignment, "_DirectMLTextEmbedding", return_value=SimpleNamespace(model=session)
            ) as constructor,
        ):
            alignment._load_embedding_model("AMD", 3)
            self.assertEqual(
                constructor.call_args.kwargs["providers"],
                [("DmlExecutionProvider", {"device_id": "3"})],
            )
            session.disable_fallback.assert_called_once()

    def test_linux_cache_uses_xdg_not_windows_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(alignment.sys, "platform", "linux"),
                patch.dict(
                    os.environ,
                    {"XDG_DATA_HOME": directory, "LOCALAPPDATA": "unused"},
                    clear=True,
                ),
            ):
                self.assertEqual(
                    alignment._model_cache(), Path(directory) / "SubtitleSync" / "model-cache"
                )

    def test_analysis_passes_device_and_preserves_alignment(self):
        cues = [
            alignment.Cue(index * 60_000, index * 60_000 + 2000, f"Dialogue number {index}")
            for index in range(4)
        ]
        windows = [
            alignment.DialogueWindow(cue.start_ms, cue.end_ms, cue.text) for cue in cues
        ]
        model = Mock()
        model.embed.side_effect = [np.eye(4), np.eye(4)]
        with (
            patch.object(alignment, "_load_embedding_model", return_value=model) as load,
            patch.object(alignment, "_build_windows", return_value=windows),
        ):
            anchors, diagnostics = alignment.analyze_subtitles(
                cues, cues, 0, processing_device="INTEL", device_id=2
            )
        load.assert_called_once_with("INTEL", 2)
        self.assertEqual(anchors, [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)])
        self.assertEqual(diagnostics["matched_cues"], 3)


@unittest.skipUnless(os.environ.get("DISPLAY") or os.name == "nt", "Needs a GUI display")
class GuiDeviceTests(unittest.TestCase):
    def setUp(self):
        self.gui = load_gui()
        self.root = tk.Tk()
        self.app = self.gui.SubtitleSyncApp(self.root)

    def tearDown(self):
        for identifier in self.root.tk.call("after", "info"):
            self.root.after_cancel(identifier)
        self.root.destroy()

    def test_integrated_setup_uses_correct_vendor_and_apu_list(self):
        self.app.processing_backend.set("ONNX (advanced)")
        for integrated in self.gui.gpu_runtime.INTEGRATED_DEVICES:
            self.app.processing_device.set(integrated)
            self.app._update_device_display()
            self.assertIn("Integrated GPU selected", self.app.device_help.get())
            self.assertEqual(str(self.app.device_id_entry.cget("state")), "normal")
            self.app._open_gpu_setup()
            self.root.update()
            window = next(
                child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel)
            )
            self.assertIn(integrated, window.title())

            def descendants(widget):
                result = []
                for child in widget.winfo_children():
                    result.append(child)
                    result.extend(descendants(child))
                return result

            widgets = descendants(window)
            if integrated == "AMD APU (integrated)":
                selectors = [child for child in widgets if child.winfo_class() == "TCombobox"]
                if alignment.sys.platform.startswith("linux"):
                    self.assertEqual(tuple(selectors[1].cget("values")), self.gui.gpu_runtime.AMD_APU_MODELS)
            window.destroy()

    def test_device_controls_and_layout(self):
        self.root.update()
        self.assertEqual(self.app.processing_device.get(), "CPU")
        self.assertEqual(str(self.app.device_id_entry.cget("state")), "disabled")
        self.assertEqual(
            tuple(self.app.device_selector.cget("values")), alignment.PROCESSING_DEVICES
        )
        self.assertLessEqual(
            self.app.sync_button.winfo_rooty() + self.app.sync_button.winfo_height(),
            self.root.winfo_rooty() + self.root.winfo_height(),
        )
        self.app.processing_device.set("NVIDIA")
        self.app._update_device_display()
        self.assertEqual(str(self.app.device_id_entry.cget("state")), "normal")
        self.assertIn("CUDAExecutionProvider", self.app.device_help.get())
        self.app._finish()
        self.assertEqual(str(self.app.device_selector.cget("state")), "readonly")

    def test_worker_forwards_device_and_writes_srt(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\nHello there\n\n"
        diagnostics = {"matched_cues": 1, "match_percent": 100}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.srt"
            english = Path(directory) / "english.srt"
            output = Path(directory) / "output.srt"
            source.write_text(text, encoding="utf-8")
            english.write_text(text, encoding="utf-8")
            with patch.object(
                alignment, "analyze_subtitles", return_value=([(1, 1.0)], diagnostics)
            ) as analyze:
                self.app._sync_worker(None, source, english, output, 2000, 70, "NVIDIA", 1)
            self.assertEqual(analyze.call_args.kwargs["processing_device"], "NVIDIA")
            self.assertEqual(analyze.call_args.kwargs["device_id"], 1)
            self.assertIn("00:00:02,000 --> 00:00:03,000", output.read_text())
            self.assertEqual(source.read_text(), text)
            messages = []
            while not self.app.messages.empty():
                messages.append(self.app.messages.get_nowait()[0])
            self.assertIn("success", messages)
            self.assertNotIn("error", messages)

    def test_start_sync_passes_selected_device_to_thread(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\nHello there\n\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.srt"
            source.write_text(text, encoding="utf-8")
            self.app.portuguese_path.set(str(source))
            self.app.english_path.set(str(source))
            self.app.output_path.set(str(Path(directory) / "output.srt"))
            self.app.initial_time.set("00:00:01.000")
            cues = tuple(alignment.read_srt(source))
            self.app.sync_selection = self.gui.SyncSelection(
                cues, cues, 1, 1, self.gui.input_signatures((str(source), str(source)))
            )
            self.app.processing_device.set("NVIDIA")
            self.app.device_id.set("2")
            with (
                patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
                patch.object(self.gui.threading, "Thread") as thread,
            ):
                self.app._start_sync()
            self.assertEqual(thread.call_args.kwargs["args"][-2:], ("NVIDIA", 2))
            thread.return_value.start.assert_called_once()
            self.assertTrue(self.app.busy)
            self.assertEqual(str(self.app.device_selector.cget("state")), "disabled")
            self.assertEqual(str(self.app.device_id_entry.cget("state")), "disabled")
            self.app._finish()

    def test_worker_reports_failure_without_writing_output(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\nHello there\n\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.srt"
            output = Path(directory) / "output.srt"
            source.write_text(text, encoding="utf-8")
            with patch.object(
                alignment, "analyze_subtitles", side_effect=ValueError("GPU initialization failed")
            ):
                self.app._sync_worker(None, source, source, output, 1000, 70, "AMD", 0)
            self.assertFalse(output.exists())
            messages = []
            while not self.app.messages.empty():
                messages.append(self.app.messages.get_nowait())
            self.assertIn(("error", "GPU initialization failed"), messages)
            self.assertNotIn("success", [kind for kind, _ in messages])

    def test_private_runtime_analysis_writes_output(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\nHello there\n\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.srt"
            output = Path(directory) / "output.srt"
            source.write_text(text, encoding="utf-8")
            with (
                patch.object(self.gui.gpu_runtime, "runtime_ready", return_value=True),
                patch.object(
                    self.gui.gpu_runtime, "run_worker",
                    return_value={"anchors": [[1, 1.0]], "diagnostics": {"matched_cues": 1}},
                ) as run,
                patch.object(alignment, "analyze_subtitles") as local,
            ):
                self.app._sync_worker(None, source, source, output, 2000, 70, "NVIDIA", 2)
            local.assert_not_called()
            self.assertEqual(run.call_args.args[:2], ("NVIDIA", 2))
            self.assertEqual(run.call_args.args[2]["source"][0]["text"], "Hello there")
            self.assertIn("00:00:02,000 --> 00:00:03,000", output.read_text())

    def test_gpu_setup_dialog_has_guides_install_and_test_actions(self):
        self.app.processing_device.set("NVIDIA")
        self.app._open_gpu_setup()
        self.root.update()
        windows = [child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel)]
        self.assertEqual(len(windows), 1)

        def buttons(widget):
            found = []
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    found.append(child)
                found.extend(buttons(child))
            return found

        controls = {button.cget("text"): button for button in buttons(windows[0])}
        self.assertIn("1. Official driver guide", controls)
        self.assertIn("2. Install / repair GPU support", controls)
        self.assertIn("3. Test GPU", controls)
        with patch.object(self.gui.webbrowser, "open", return_value=True) as browser:
            controls["1. Official driver guide"].invoke()
            self.assertIn("nvidia.com", browser.call_args.args[0])
        controls["Close"].invoke()

    def test_gpu_setup_installs_tests_and_restores_controls(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                self.app.processing_device.set("NVIDIA")
                self.app._open_gpu_setup()
                window = next(
                    child for child in self.root.winfo_children()
                    if isinstance(child, tk.Toplevel)
                )

                def find(widget, text):
                    for child in widget.winfo_children():
                        if child.winfo_class() == "TButton" and child.cget("text") == text:
                            return child
                        found = find(child, text)
                        if found is not None:
                            return found
                    return None

                button = find(window, "2. Install / repair GPU support")
                with (
                    patch.object(self.gui.messagebox, "askyesno", return_value=True),
                    patch.object(self.gui.messagebox, "showinfo") as success,
                    patch.object(self.gui.messagebox, "showerror") as error,
                    patch.object(self.gui.gpu_runtime, "install_runtime") as install,
                    patch.object(
                        self.gui.gpu_runtime, "test_runtime", return_value="GPU test passed",
                        side_effect=ValueError("driver missing") if failure else None,
                    ) as test,
                ):
                    button.invoke()
                    self.assertTrue(self.app.setup_busy)
                    deadline = time.monotonic() + 3
                    while self.app.setup_busy and time.monotonic() < deadline:
                        self.root.update()
                        time.sleep(0.01)
                    self.assertFalse(self.app.setup_busy)
                    install.assert_called_once()
                    test.assert_called_once()
                    self.assertEqual(str(self.app.sync_button.cget("state")), "normal")
                    self.assertEqual(str(self.app.setup_button.cget("state")), "normal")
                    if failure:
                        error.assert_called_once()
                        success.assert_not_called()
                    else:
                        success.assert_called_once()
                        error.assert_not_called()
                window.destroy()

    def test_amd_linux_setup_selection_updates_guidance_and_links(self):
        self.app.processing_backend.set("ONNX (advanced)")
        with (
            patch.object(self.gui.gpu_runtime.sys, "platform", "linux"),
            patch.object(
                self.gui.gpu_runtime, "linux_distribution", return_value=("Ubuntu", "24.04")
            ),
        ):
            self.app.processing_device.set("AMD")
            self.app._open_gpu_setup()
            self.root.update()
            window = next(
                child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel)
            )

            def descendants(widget):
                result = []
                for child in widget.winfo_children():
                    result.append(child)
                    result.extend(descendants(child))
                return result

            widgets = descendants(window)
            selectors = [child for child in widgets if child.winfo_class() == "TCombobox"]
            buttons = {
                child.cget("text"): child for child in widgets if child.winfo_class() == "TButton"
            }
            texts = [child for child in widgets if isinstance(child, tk.Text)]
            self.assertEqual(len(selectors), 4)
            stage, gpu, distribution, version = selectors
            self.assertEqual(distribution.get(), "Ubuntu")
            self.assertEqual(version.get(), "24.04")
            guide = buttons["Open selected setup guide"]
            self.assertEqual(str(guide.cget("state")), "disabled")
            self.assertEqual(
                str(buttons["2. Install / repair GPU support"].cget("state")), "disabled"
            )
            gpu.set("Radeon RX 7900 XTX")
            gpu.event_generate("<<ComboboxSelected>>")
            self.root.update()
            self.assertEqual(str(guide.cget("state")), "normal")
            self.assertIn("Radeon RX 7900 XTX; Ubuntu 24.04", texts[0].get("1.0", "end"))
            with patch.object(self.gui.webbrowser, "open", return_value=True) as browser:
                guide.invoke()
                browser.assert_called_with(self.gui.gpu_runtime.AMD_ROCM_URL)
                stage.set(self.gui.gpu_runtime.AMD_SETUP_OPTIONS[1])
                stage.event_generate("<<ComboboxSelected>>")
                self.root.update()
                guide.invoke()
                browser.assert_called_with(self.gui.gpu_runtime.MIGRAPHX_URL)
                self.assertIn("MIGraphXExecutionProvider", texts[0].get("1.0", "end"))
                distribution.set("My Linux")
                version.set("rolling")
                gpu.set("My unlisted GPU")
                gpu.event_generate("<<ComboboxSelected>>")
                self.root.update()
                self.assertIn("My unlisted GPU; My Linux rolling", texts[0].get("1.0", "end"))
                self.assertIn("use CPU", texts[0].get("1.0", "end"))
                version.set("")
                version.event_generate("<FocusOut>")
                self.root.update()
                self.assertEqual(str(guide.cget("state")), "disabled")
            buttons["Close"].invoke()

    def test_recommended_backend_and_explicit_overrides(self):
        for device, expected in (
            ("CPU", "ONNX"), ("NVIDIA", "ONNX"), ("AMD", "Vulkan"),
            ("INTEL", "Vulkan"), ("AMD APU (integrated)", "Vulkan"),
            ("INTEL integrated GPU", "Vulkan"),
        ):
            self.app.processing_device.set(device)
            self.app.processing_backend.set("Recommended")
            self.assertEqual(self.app._selected_backend(), expected)
            self.app._update_device_display()
            if expected == "Vulkan":
                self.assertIn("Vulkan", self.app.device_help.get())
                self.assertIn("without ROCm/CUDA", self.app.device_help.get())
            self.app.processing_backend.set("ONNX (advanced)")
            self.assertEqual(self.app._selected_backend(), "ONNX")
            self.app.processing_backend.set("Vulkan")
            self.assertEqual(self.app._selected_backend(), "ONNX" if device == "CPU" else "Vulkan")

    def test_vulkan_setup_detection_selection_and_test(self):
        self.app.processing_device.set("AMD")
        self.app._open_gpu_setup()
        self.root.update()
        window = next(w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel))
        self.assertEqual(window.title(), "Vulkan GPU / APU setup")
        frame = window.winfo_children()[0]
        actions = frame.winfo_children()[-1]
        controls = {child.cget("text"): child for child in actions.winfo_children()}
        selector = next(child for child in frame.winfo_children() if child.winfo_class() == "TCombobox")
        with (
            patch.object(self.gui.vulkan_embeddings, "list_devices", return_value=[
                (0, "AMD Radeon RX 560"), (1, "Intel UHD Graphics")
            ]),
            patch.object(self.gui.vulkan_embeddings, "test_runtime", return_value="Vulkan test passed") as test,
        ):
            controls["Detect GPUs"].invoke()
            deadline = time.monotonic() + 3
            while self.app.setup_busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
            self.assertFalse(self.app.setup_busy)
            self.assertIn("AMD Radeon RX 560", selector.get())
            selector.current(1)
            selector.event_generate("<<ComboboxSelected>>")
            self.root.update()
            self.assertEqual(self.app.device_id.get(), "1")
            controls["Test GPU"].invoke()
            deadline = time.monotonic() + 3
            while self.app.setup_busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
            self.assertFalse(self.app.setup_busy)
            self.assertEqual(test.call_args.args[0], 1)
            self.assertEqual(str(self.app.backend_selector.cget("state")), "readonly")
        controls["Close"].invoke()

    def test_vulkan_worker_does_not_use_onnx_environment(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\nHello there\n\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.srt"
            output = Path(directory) / "output.srt"
            source.write_text(text, encoding="utf-8")
            with (
                patch.object(alignment, "analyze_subtitles", return_value=([(1, 1.0)], {})) as analyze,
                patch.object(self.gui.gpu_runtime, "run_worker") as onnx_worker,
            ):
                self.app._sync_worker(None, source, source, output, 2000, 70, "AMD", 0, "Vulkan")
            self.assertEqual(analyze.call_args.kwargs["backend"], "Vulkan")
            onnx_worker.assert_not_called()
            self.assertIn("00:00:02,000 --> 00:00:03,000", output.read_text())


if __name__ == "__main__":
    unittest.main()
