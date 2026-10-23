import hashlib
import os
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import speech_dialog
import speech_to_text as speech
import subtitle_alignment as alignment
from manual_sync import input_signatures
from test_subtitle_devices import load_gui


SRT = "1\n00:00:02,000 --> 00:00:04,000\nThe journey begins.\n\n"
GPU_LOG = (
    "ggml_vulkan: 2 = AMD Radeon RX 560 Series | fp16: 0\n"
    "whisper_backend_init_gpu: using Vulkan2 backend\n"
)


class SpeechConfigurationTests(unittest.TestCase):
    def test_recommended_backends_for_all_devices_and_apus(self):
        for device, expected in (
            ("CPU", "CPU"), ("NVIDIA", "CUDA"), ("AMD", "Vulkan"),
            ("INTEL", "Vulkan"), ("AMD APU (integrated)", "Vulkan"),
            ("INTEL integrated GPU", "Vulkan"),
        ):
            self.assertEqual(speech.selected_backend(device, "Recommended"), expected)
        self.assertEqual(speech.selected_backend("NVIDIA", "Vulkan"), "Vulkan")
        self.assertEqual(speech.selected_backend("AMD", "CPU"), "CPU")

    def test_invalid_backend_choices_are_explicit_errors(self):
        for device, backend in (("CPU", "CUDA"), ("CPU", "Vulkan"), ("AMD", "CUDA"), ("unknown", "CPU"), ("INTEL", "ONNX")):
            with self.subTest(device=device, backend=backend), self.assertRaises(ValueError):
                speech.selected_backend(device, backend)

    def test_language_model_and_index_validation(self):
        for language in ("auto", "ja", "pt", "ar", "fr", "en"):
            speech.validate_options("base", language, 0, 1)
        speech.validate_options("tiny.en", "en", 0, 0)
        speech.validate_options("large-v3", "yue", 0, 0)
        for model, language, adapter, stream in (
            ("tiny.en", "auto", 0, 0), ("base.en", "pt", 0, 0),
            ("base", "yue", 0, 0), ("tiny", "xx", 0, 0), ("unknown", "en", 0, 0),
            ("tiny", "en", -1, 0), ("tiny", "en", 0, -1),
            ("tiny", "en", True, 0), ("tiny", "en", 0, "0"),
        ):
            with self.subTest(model=model, language=language, adapter=adapter, stream=stream), self.assertRaises(ValueError):
                speech.validate_options(model, language, adapter, stream)

    def test_platform_downloads_are_pinned_and_hashed(self):
        for system in ("linux", "win32"):
            with patch.object(speech.sys, "platform", system), patch.object(speech.platform, "machine", return_value="AMD64"):
                for backend in ("CPU", "CUDA", "Vulkan"):
                    url, digest = speech.package(backend)
                    self.assertTrue(url.startswith("https://github.com/"))
                    self.assertTrue("b5454" in url or "whispercpp-195" in url)
                    self.assertRegex(digest, r"^[0-9a-f]{64}$")
        with patch.object(speech.platform, "machine", return_value="aarch64"):
            speech.package("CPU")
            with self.assertRaisesRegex(ValueError, "No bundled"):
                speech.package("Vulkan")

    def test_command_has_no_translation_and_preserves_spaced_paths(self):
        with patch.object(speech, "executable", return_value=Path("/app with spaces/whisper-cli")):
            command = speech.whisper_command("Vulkan", "small", "pt", 2, Path("/a b/audio.wav"), Path("/out b/transcript"))
            self.assertEqual(command[0], "/app with spaces/whisper-cli")
            self.assertEqual(command[command.index("-l") + 1], "pt")
            self.assertEqual(command[command.index("-dev") + 1], "2")
            self.assertIn("/a b/audio.wav", command)
            self.assertIn("-osrt", command)
            self.assertIn("-nfa", command)
            self.assertEqual(command[command.index("-ml") + 1], "84")
            self.assertIn("-sow", command)
            self.assertNotIn("-tr", command)
            self.assertNotIn("-ng", command)
            cpu = speech.whisper_command("CPU", "tiny", "en", 0, Path("a.wav"), Path("out"))
            self.assertIn("-ng", cpu)
            self.assertNotIn("-dev", cpu)

    def test_gpu_verification_rejects_loaded_only_wrong_backend_and_software(self):
        speech.verify_gpu(GPU_LOG, "Vulkan")
        speech.verify_gpu("whisper_backend_init_gpu: using CUDA0 backend", "CUDA")
        for log, backend in (
            ("load_backend: loaded Vulkan backend", "Vulkan"),
            ("whisper_backend_init_gpu: no GPU found", "CUDA"),
            (GPU_LOG, "CUDA"),
            (GPU_LOG.replace("AMD Radeon RX 560 Series", "llvmpipe"), "Vulkan"),
        ):
            with self.subTest(log=log), self.assertRaises(ValueError):
                speech.verify_gpu(log, backend)

    def test_environment_cannot_override_requested_whisper_options(self):
        with (
            patch.dict(os.environ, {"WHISPER_ARG_MODEL": "wrong.bin", "WHISPER_ARG_DEVICE": "99"}),
            patch.object(speech, "executable", return_value=Path("/speech-runtime/whisper-cli")),
        ):
            env = speech.environment("CPU")
        self.assertNotIn("WHISPER_ARG_DEVICE", env)
        self.assertNotIn("WHISPER_ARG_MODEL", env)
        if sys.platform == "linux":
            self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/speech-runtime"))

    def test_cached_models_are_rehashed_and_reused(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(speech, "APP_DIR", Path(folder)):
            exe = speech.runtime_directory("CPU") / ("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli")
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"runtime")
            (exe.parent / "runtime.sha256").write_text(speech.package("CPU")[1], encoding="ascii")
            path = speech.model_path("tiny")
            path.parent.mkdir()
            path.write_bytes(b"verified model")
            model = speech.Model(1, hashlib.sha256(path.read_bytes()).hexdigest())
            with patch.dict(speech.MODELS, {"tiny": model}), patch.object(speech, "download") as download:
                speech.prepare("CPU", "tiny", Mock(), threading.Event())
                download.assert_not_called()
                path.write_bytes(b"damaged model")
                speech.prepare("CPU", "tiny", Mock(), threading.Event())
                self.assertEqual(download.call_count, 1)

    def test_device_detection_uses_whisper_names_and_ids(self):
        with (
            patch.object(speech, "prepare"),
            patch.object(speech, "executable", return_value=Path("/whisper-cli")),
            patch.object(speech, "run_process", return_value=GPU_LOG),
        ):
            self.assertEqual(
                speech.detect_devices("Vulkan", Mock(), threading.Event()),
                [(2, "AMD Radeon RX 560 Series")],
            )
        self.assertEqual(speech.detect_devices("CPU", Mock(), threading.Event()), [])


class SpeechProcessTests(unittest.TestCase):
    def test_ffmpeg_discovers_supported_subtitle_languages_and_ignores_images(self):
        from types import SimpleNamespace
        metadata = (
            "Stream #0:2(eng): Subtitle: subrip\n"
            "Stream #0:4[0x5](jpn): Subtitle: mov_text (tx3g)\n"
            "Stream #0:5(fra): Subtitle: hdmv_pgs_subtitle\n"
        )
        with (
            patch.object(alignment, "_find_tool", return_value=None),
            patch.object(alignment, "_ffmpeg", return_value="ffmpeg"),
            patch.object(alignment.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr=metadata)),
        ):
            tracks = alignment.list_subtitle_tracks(Path("video.mp4"))
        self.assertEqual([track.track_id for track in tracks], [-3, -5])
        self.assertIn("eng", tracks[0].label)
        self.assertIn("jpn", tracks[1].label)

    def test_ffmpeg_inspection_errors_are_not_treated_as_missing_subtitles(self):
        from types import SimpleNamespace
        with (
            patch.object(alignment, "_find_tool", return_value=None),
            patch.object(alignment, "_ffmpeg", return_value="ffmpeg"),
            patch.object(alignment.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr="Invalid video")),
            self.assertRaisesRegex(ValueError, "Invalid video"),
        ):
            alignment.list_subtitle_tracks(Path("video.mp4"))

    def test_process_error_exposes_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "missing GPU library"):
            speech.run_process(
                [sys.executable, "-c", "import sys; print('missing GPU library'); sys.exit(4)"],
                Mock(), threading.Event(),
            )

    def test_cancellation_terminates_running_process(self):
        cancel = threading.Event()
        started = time.monotonic()
        with self.assertRaises(speech.CancelledError):
            speech.run_process(
                [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(30)"],
                lambda _text: cancel.set(), cancel,
            )
        self.assertLess(time.monotonic() - started, 7)

    def test_cancelled_job_does_not_start_a_process(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(speech.subprocess, "Popen") as popen, self.assertRaises(speech.CancelledError):
            speech.run_process(["missing-executable"], Mock(), cancel)
        popen.assert_not_called()

    def test_missing_ffmpeg_has_actionable_error(self):
        with patch.object(speech.shutil, "which", return_value=None), patch.dict(sys.modules, {"imageio_ffmpeg": None}):
            with self.assertRaisesRegex(ValueError, "launcher"):
                speech.ffmpeg_executable()


class SlimRuntimeTests(unittest.TestCase):
    def test_cached_assets_do_not_require_download_confirmation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(speech, "APP_DIR", Path(folder)):
            self.assertEqual(len(speech.missing_downloads("CPU", "base")), 2)
            runtime = speech.runtime_directory("CPU")
            runtime.mkdir(parents=True)
            binary = runtime / ("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli")
            binary.touch()
            (runtime / "runtime.sha256").write_text(speech.package("CPU")[1])
            model = speech.model_path("base")
            model.parent.mkdir(parents=True)
            model.write_bytes(b"cached; verified by prepare")
            self.assertEqual(speech.missing_downloads("CPU", "base"), [])
            self.assertEqual(speech.missing_downloads("CPU", "tiny"), ["tiny model (about 78 MB)"])
            (runtime / "runtime.sha256").write_text("old")
            self.assertIn("incomplete or outdated", speech.missing_downloads("CPU", "base")[0])

    def test_slim_cuda_requires_explicit_setup_before_downloading_model(self):
        with (
            patch.dict(os.environ, {"SUBTITLE_SYNC_BUNDLED_CUDA": "0"}),
            patch.object(speech.gpu_runtime, "runtime_ready", return_value=False),
            patch.object(speech, "download") as download,
        ):
            with self.assertRaisesRegex(ValueError, "Install / repair GPU support"):
                speech.prepare("CUDA", "tiny", Mock(), threading.Event())
            download.assert_not_called()

    def test_speech_cuda_finds_appdata_libraries_without_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder) / "lib/python3.12/site-packages/nvidia/cublas/lib"
            library.mkdir(parents=True)
            with (
                patch.object(speech.sys, "platform", "linux"),
                patch.object(speech.gpu_runtime, "runtime_ready", return_value=True),
                patch.object(speech.gpu_runtime, "runtime_directory", return_value=Path(folder)),
                patch.object(speech, "executable", return_value=Path("/whisper/whisper-cli")),
                patch.dict(os.environ, {"LD_LIBRARY_PATH": "/host/driver"}),
            ):
                env = speech.environment("CUDA")
                self.assertEqual(env["LD_LIBRARY_PATH"], f"/whisper:{library}:/host/driver")
                self.assertNotIn(str(library), speech.environment("Vulkan")["LD_LIBRARY_PATH"])


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.video = self.folder / "video with spaces.mkv"
        self.video.write_bytes(b"video data")
        self.output = self.folder / "generated reference.srt"
        self.cancel = threading.Event()
        self.log = GPU_LOG
        self.generated = SRT
        self.commands = []
        self.after_whisper = lambda: None
        patches = (
            patch.object(speech, "prepare"),
            patch.object(speech, "executable", return_value=Path("/whisper runtime/whisper-cli")),
            patch.object(speech, "ffmpeg_executable", return_value="/ffmpeg executable"),
            patch.object(speech, "run_process", side_effect=self.run_process),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def run_process(self, command, _progress, _cancel, _env=None):
        self.commands.append(command)
        if "-of" in command:
            Path(command[command.index("-of") + 1] + ".srt").write_text(self.generated, encoding="utf-8")
            self.after_whisper()
            return self.log
        if "ashowinfo" in command:
            return "[Parsed_ashowinfo_0] n:0 pts:0 pts_time:0 fmt:s16"
        return ""

    def transcribe(self, **kwargs):
        options = dict(
            video=self.video, output=self.output, model="tiny", language="pt",
            device="AMD", backend_choice="Recommended", device_id=2, audio_stream=1,
            progress=Mock(), cancel=self.cancel,
        )
        options.update(kwargs)
        return speech.transcribe(**options)

    def test_success_atomically_saves_separate_srt_and_maps_audio_track(self):
        self.assertEqual(self.transcribe(), 1)
        self.assertEqual(self.output.read_text(), SRT)
        self.assertEqual(self.video.read_bytes(), b"video data")
        self.assertEqual(self.commands[0][self.commands[0].index("-map") + 1], "0:a:1")
        self.assertIn("-copyts", self.commands[0])
        self.assertIn("asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0", self.commands[1])
        self.assertIn("-start_at_zero", self.commands[0])
        self.assertFalse(list(self.folder.glob(".subtitle-speech-*")))

    def test_cpu_transcription_does_not_require_gpu_log(self):
        self.log = ""
        self.assertEqual(self.transcribe(device="CPU", device_id=0), 1)
        self.assertIn("-ng", self.commands[2])

    def test_audio_start_delay_is_added_to_every_generated_timestamp(self):
        with patch.object(speech, "audio_start_ms", return_value=2345):
            self.assertEqual(self.transcribe(), 1)
        cues = speech.parse_srt(self.output.read_text())
        self.assertEqual((cues[0].start_ms, cues[0].end_ms), (4345, 6345))

    def test_unknown_audio_timestamp_is_an_error_not_a_zero_default(self):
        with patch.object(speech, "run_process", return_value="No decoded frames"):
            with self.assertRaisesRegex(ValueError, "start time"):
                self.transcribe()
        self.assertFalse(self.output.exists())

    def test_no_overwrite_without_confirmation(self):
        self.output.write_text("original")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.transcribe()
        self.assertEqual(self.output.read_text(), "original")
        self.assertFalse(self.commands)

    def test_protected_paths_are_not_replaced_even_with_confirmation(self):
        self.output.write_text("source or reference")
        with self.assertRaisesRegex(ValueError, "overwrite any input"):
            self.transcribe(overwrite=True, protected_paths=(self.output,))
        self.assertEqual(self.output.read_text(), "source or reference")

    def test_gpu_fallback_does_not_replace_existing_output(self):
        self.output.write_text("original")
        self.log = "whisper_backend_init_gpu: no GPU found"
        with self.assertRaisesRegex(ValueError, "CPU fallback"):
            self.transcribe(overwrite=True)
        self.assertEqual(self.output.read_text(), "original")
        self.assertFalse(list(self.folder.glob(".subtitle-speech-*")))

    def test_gpu_failure_is_reported_before_decoding(self):
        def process(command, progress, cancel, env=None):
            if "-of" in command:
                progress("whisper_backend_init_gpu: no GPU found")
                progress("whisper_init_state: allocating buffers")
                self.fail("CPU decoding should not start")
            if "ashowinfo" in command:
                return "[Parsed_ashowinfo_0] n:0 pts:0 pts_time:0 fmt:s16"
            return ""
        with patch.object(speech, "run_process", side_effect=process):
            with self.assertRaisesRegex(ValueError, "CPU fallback"):
                self.transcribe()

    def test_invalid_empty_silent_or_reversed_srt_is_not_published(self):
        for text in (
            "", "not an SRT", SRT.replace("00:00:04,000", "00:00:01,000"),
            SRT.replace("The journey begins.", "[BLANK_AUDIO]"),
            SRT + "2\n00:00:01,000 --> 00:00:02,000\nEarlier dialogue\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.generated = text
                self.transcribe()
            self.assertFalse(self.output.exists())

    def test_zero_length_word_is_joined_to_nearest_line_in_a_single_pass(self):
        self.generated = (
            "1\n00:00:01,000 --> 00:00:02,000\nHello there,\n\n"
            "2\n00:00:02,000 --> 00:00:02,000\nfriend.\n\n"
            "3\n00:00:05,000 --> 00:00:05,000\nWhere\n\n"
            "4\n00:00:05,100 --> 00:00:06,000\nare we going?\n\n"
            "5\n00:00:07,000 --> 00:00:07,500\n\n\n"
        )
        progress = Mock()
        self.assertEqual(self.transcribe(progress=progress), 2)
        whisper = [command for command in self.commands if "-of" in command]
        self.assertEqual(len(whisper), 1)
        self.assertIn("-ml", whisper[0])
        self.assertIn("-dev", whisper[0])
        cues = speech.parse_srt(self.output.read_text())
        self.assertEqual(
            [(cue.start_ms, cue.end_ms, cue.text) for cue in cues],
            [(1000, 2000, "Hello there, friend."), (5000, 6000, "Where are we going?")],
        )
        messages = " ".join(call.args[0] for call in progress.call_args_list)
        self.assertIn("2 zero-length and 1 empty", messages)
        self.assertNotIn("Retrying", messages)

    def test_leading_zero_length_word_joins_following_line(self):
        self.generated = (
            "1\n00:00:01,000 --> 00:00:01,000\nWell,\n\n"
            "2\n00:00:01,000 --> 00:00:03,000\nlet's go.\n\n"
        )
        self.assertEqual(self.transcribe(), 1)
        cue = speech.parse_srt(self.output.read_text())[0]
        self.assertEqual((cue.start_ms, cue.end_ms, cue.text), (1000, 3000, "Well, let's go."))

    def test_valid_whisper_srt_is_published_unchanged(self):
        self.assertEqual(self.transcribe(), 1)
        self.assertEqual(self.output.read_text(), SRT)
        self.assertEqual(len([cmd for cmd in self.commands if "-of" in cmd]), 1)

    def test_reversed_or_all_zero_length_cues_are_rejected_without_rerun(self):
        for generated, message in (
            (SRT.replace("00:00:04,000", "00:00:01,000"), "Cue 1:.*reversed"),
            (SRT.replace("00:00:04,000", "00:00:02,000"), "only zero-length"),
        ):
            with self.subTest(message=message):
                self.commands.clear()
                self.generated = generated
                self.output.write_text("original")
                with self.assertRaisesRegex(ValueError, message):
                    self.transcribe(overwrite=True)
                self.assertEqual(len([cmd for cmd in self.commands if "-of" in cmd]), 1)
                self.assertEqual(self.output.read_text(), "original")

    def test_cancelled_transcription_preserves_original_output(self):
        self.output.write_text("original")
        self.after_whisper = self.cancel.set
        with self.assertRaises(speech.CancelledError):
            self.transcribe(overwrite=True)
        self.assertEqual(self.output.read_text(), "original")
        self.assertFalse(list(self.folder.glob(".subtitle-speech-*")))

    def test_input_changes_are_detected(self):
        self.after_whisper = lambda: self.video.write_bytes(b"changed input")
        with self.assertRaisesRegex(ValueError, "video changed"):
            self.transcribe()
        self.assertFalse(self.output.exists())

    def test_concurrent_output_changes_are_not_replaced(self):
        self.after_whisper = lambda: self.output.write_text("written elsewhere")
        with self.assertRaisesRegex(ValueError, "output file changed"):
            self.transcribe()
        self.assertEqual(self.output.read_text(), "written elsewhere")

    def test_ffmpeg_failure_does_not_start_whisper(self):
        with patch.object(speech, "run_process", side_effect=ValueError("No audio stream")) as run:
            with self.assertRaisesRegex(ValueError, "No audio stream"):
                self.transcribe()
        self.assertEqual(run.call_count, 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.folder.glob(".subtitle-speech-*")))


@unittest.skipUnless(os.environ.get("DISPLAY") or sys.platform == "win32", "Tk display required")
class AutomaticReferenceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.gui = load_gui()
        self.app = self.gui.SubtitleSyncApp(self.root)
        self.temporary = tempfile.TemporaryDirectory()
        self.video = Path(self.temporary.name) / "video.mkv"
        self.video.touch()
        self.app.video_path.set(str(self.video))

    def tearDown(self):
        for callback in self.root.tk.call("after", "info"):
            self.root.after_cancel(callback)
        self.root.destroy()
        self.temporary.cleanup()

    def wait(self):
        deadline = time.monotonic() + 3
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app.busy)

    def test_browsing_subtitle_free_video_opens_speech_without_source_srt(self):
        with (
            patch.object(self.gui.filedialog, "askopenfilename", return_value=str(self.video)),
            patch.object(alignment, "list_subtitle_tracks", return_value=[]),
            patch.object(self.app, "_open_speech_to_text") as speech_popup,
        ):
            self.app._choose_video()
            self.wait()
        speech_popup.assert_called_once()

    def test_external_reference_bypasses_inspection(self):
        self.app.reference_path.set("external.srt")
        with (
            patch.object(self.gui.filedialog, "askopenfilename", return_value=str(self.video)),
            patch.object(alignment, "list_subtitle_tracks") as inspect,
        ):
            self.app._choose_video()
        inspect.assert_not_called()

    def test_multiple_tracks_popup_remembers_selected_track_for_pairing(self):
        tracks = [alignment.SubtitleTrack(2, "2: English"), alignment.SubtitleTrack(7, "7: Japanese")]
        with patch.object(alignment, "list_subtitle_tracks", return_value=tracks):
            self.app._inspect_video_reference()
            self.wait()
        window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))
        combo = next(child for child in window.winfo_children() if child.winfo_class() == "TCombobox")
        combo.current(1)
        next(child for child in window.winfo_children() if child.winfo_class() == "TButton").invoke()
        self.assertEqual(self.app.embedded_track[1], 7)
        source = Path(self.temporary.name) / "source.srt"
        source.write_text(SRT)
        self.app.source_path.set(str(source))
        with (
            patch.object(alignment, "list_subtitle_tracks") as inspect,
            patch.object(alignment, "extract_reference_srt", return_value=alignment.parse_srt(SRT)) as extract,
            patch.object(self.app, "_show_pair_picker") as pair,
        ):
            self.app._open_initial_sync()
            self.wait()
        inspect.assert_not_called()
        extract.assert_called_once_with(self.video, 7)
        pair.assert_called_once()

    def test_single_track_is_selected_without_popup(self):
        with (
            patch.object(alignment, "list_subtitle_tracks", return_value=[alignment.SubtitleTrack(3, "3: French")]),
            patch.object(self.app, "_show_track_picker") as picker,
        ):
            self.app._inspect_video_reference()
            self.wait()
        picker.assert_not_called()
        self.assertEqual(self.app.embedded_track, (input_signatures((str(self.video),)), 3))

    def test_stale_inspection_cannot_open_speech(self):
        signature = input_signatures((str(self.video),))
        self.video.write_bytes(b"changed")
        with patch.object(self.app, "_open_speech_to_text") as popup:
            self.app._use_video_tracks(signature, [])
        popup.assert_not_called()

    def test_inspection_failure_is_visible_and_does_not_open_speech(self):
        with (
            patch.object(alignment, "list_subtitle_tracks", side_effect=ValueError("Cannot inspect video")),
            patch.object(self.gui.messagebox, "showerror") as error,
            patch.object(self.app, "_open_speech_to_text") as popup,
        ):
            self.app._inspect_video_reference()
            self.wait()
        error.assert_called_once()
        popup.assert_not_called()

    def test_manual_pair_action_also_opens_speech_when_video_has_no_tracks(self):
        source = Path(self.temporary.name) / "source.srt"
        source.write_text(SRT)
        self.app.source_path.set(str(source))
        with (
            patch.object(alignment, "list_subtitle_tracks", return_value=[]),
            patch.object(self.app, "_open_speech_to_text") as popup,
        ):
            self.app._open_initial_sync()
            self.wait()
        popup.assert_called_once()


@unittest.skipUnless(os.environ.get("DISPLAY") or sys.platform == "win32", "Tk display required")
class SpeechDialogTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = load_gui().SubtitleSyncApp(self.root)
        self.dialog = speech_dialog.SpeechDialog(
            self.root, "", (), lambda path: self.app.reference_path.set(str(path)),
        )

    def tearDown(self):
        if self.dialog._poll_id is not None:
            self.dialog.after_cancel(self.dialog._poll_id)
        self.dialog.destroy()
        self.root.destroy()

    def test_cached_setup_skips_prompt_and_detection_checks_tiny(self):
        self.dialog.device.set("AMD")
        self.dialog.backend.set("Vulkan")
        with (
            patch.object(speech, "missing_downloads", return_value=[]) as missing,
            patch.object(speech_dialog.messagebox, "askokcancel") as prompt,
            patch.object(speech_dialog.threading, "Thread") as thread,
        ):
            self.dialog._start("detect")
            missing.assert_called_once_with("Vulkan", "tiny")
            prompt.assert_not_called()
            thread.return_value.start.assert_called_once()
        self.dialog.progress.stop()

    def test_missing_setup_prompts_and_cancel_starts_no_worker(self):
        with (
            patch.object(speech, "missing_downloads", return_value=["base model (about 148 MB)"]),
            patch.object(speech_dialog.messagebox, "askokcancel", return_value=False) as prompt,
            patch.object(speech_dialog.threading, "Thread") as thread,
        ):
            self.dialog._start("prepare")
            self.assertIn("base model", prompt.call_args.args[1])
            self.assertNotIn("685 MB", prompt.call_args.args[1])
            thread.assert_not_called()

    def test_generation_selects_reference_and_invalidates_old_manual_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "reference.srt"
            video = Path(folder) / "video.mkv"
            video.touch()
            self.dialog.video.set(str(video))
            self.dialog.output.set(str(output))
            self.dialog.model.set("tiny")
            self.app.sync_selection = object()

            def generate(*args):
                self.assertEqual(args[0], video)
                self.assertEqual(args[1], output)
                self.assertEqual(args[3], "auto")
                output.write_text(SRT)
                return 1

            with (
                patch.object(speech_dialog.speech, "transcribe", side_effect=generate),
                patch.object(speech_dialog.messagebox, "askokcancel", return_value=True),
                patch.object(speech_dialog.messagebox, "showinfo"),
            ):
                self.dialog._start("generate")
                deadline = time.monotonic() + 5
                while self.dialog.busy and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.01)
            self.assertFalse(self.dialog.busy)
            self.assertEqual(self.app.reference_path.get(), str(output))
            self.assertIsNone(self.app.sync_selection)
            self.assertEqual(output.read_text(), SRT)
            self.assertIn("select matching cues", self.dialog.status.get())

    def test_invalid_english_only_language_fails_before_worker(self):
        self.dialog.model.set("tiny.en")
        self.dialog.language.set("Portuguese")
        with (
            patch.object(speech_dialog.messagebox, "showerror") as error,
            patch.object(speech_dialog.threading, "Thread") as worker,
        ):
            self.dialog._start("generate")
        error.assert_called_once()
        worker.assert_not_called()

    def test_cancel_and_error_leave_reference_unchanged(self):
        self.app.reference_path.set("old-reference.srt")
        with patch.object(speech_dialog.messagebox, "showerror"):
            for kind in ("cancelled", "error"):
                self.dialog.busy = True
                self.dialog.events.put((kind, "Stopped before writing"))
                self.dialog._poll()
                self.assertFalse(self.dialog.busy)
                self.assertEqual(self.app.reference_path.get(), "old-reference.srt")


if __name__ == "__main__":
    unittest.main()
