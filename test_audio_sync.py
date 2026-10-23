import math
import errno
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import Mock, patch

import audio_sync as sync
from audio_sync_dialog import AudioSyncDialog
from sync_subtitles import TIMESTAMP, ms_to_timestamp, timestamp_to_ms


SOURCE = (
    "7\r\n00:00:03,000 --> 00:00:04,000\r\n<i>Hello!</i>\r\nSecond line.\r\n\r\n"
    "12\r\n00:00:07,000 --> 00:00:08,500\r\nA different line.\r\n\r\n"
)
ALIGNED = (
    "1\n00:00:01,000 --> 00:00:02,000\n<i>Hello!</i>\nSecond line.\n\n"
    "2\n00:00:05,000 --> 00:00:06,500\nA different line.\n\n"
)


class AudioSyncTests(unittest.TestCase):
    def test_network_share_without_hardlinks_saves_audio_result(self):
        with patch("sync_subtitles.os.link", side_effect=OSError(errno.EPERM, "Operation not permitted")):
            self.assertEqual(self.run_sync(), 2)
        self.assertTrue(self.output.exists())
        self.assertEqual(self.source.read_bytes(), b"\xef\xbb\xbf" + SOURCE.encode())

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.video = self.root / "video.mkv"
        self.video.write_bytes(b"video")
        self.source = self.root / "source.srt"
        self.source.write_bytes(b"\xef\xbb\xbf" + SOURCE.encode("utf-8"))
        self.output = self.root / "output.srt"
        self.cancel = threading.Event()
        self.commands = []
        self.progress = Mock()

    def process(self, command, progress, cancel):
        self.commands.append(command)
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_text(ALIGNED, encoding="utf-8")
        elif "-c:s" in command:
            Path(command[-1]).write_text(SOURCE, encoding="utf-8")
        elif "-af" in command:
            Path(command[-1]).write_bytes(b"audio")
        return ""

    def run_sync(self, **kwargs):
        options = dict(
            video=self.video, source=self.source, output=self.output, audio_track=2,
            subtitle_track=None, fix_drift=True, progress=self.progress, cancel=self.cancel,
        )
        options.update(kwargs)
        with (
            patch.object(sync, "_run", side_effect=self.process),
            patch.object(sync.speech, "ffmpeg_executable", return_value="bundled-ffmpeg"),
        ):
            return sync.synchronize(**options)

    def test_changes_only_timestamps_preserving_encoding_bom_and_formatting(self):
        self.assertEqual(self.run_sync(), 2)
        expected = SOURCE.replace("03,000", "01,000").replace("04,000", "02,000")
        expected = expected.replace("07,000", "05,000").replace("08,500", "06,500")
        self.assertEqual(self.output.read_bytes(), b"\xef\xbb\xbf" + expected.encode("utf-8"))
        self.assertEqual(self.source.read_bytes(), b"\xef\xbb\xbf" + SOURCE.encode("utf-8"))
        self.assertEqual(self.video.read_bytes(), b"video")
        self.assertFalse(list(self.root.glob(".audio-sync-*")))
        audio = next(command for command in self.commands if "-af" in command)
        self.assertEqual(audio[audio.index("-map") + 1], "0:2")
        self.assertIn("-copyts", audio)
        self.assertIn("aresample=async=1:first_pts=0", audio)
        engine = self.commands[-1]
        self.assertEqual(engine[0], sys.executable)
        self.assertEqual(engine[engine.index("--vad") + 1], "webrtc")
        self.assertIn("--gss", engine)
        self.assertEqual(engine[engine.index("--split-penalty") + 1], "10")
        self.assertEqual(engine[engine.index("--max-offset-seconds") + 1], "60")
        self.assertIn("--skip-infer-framerate-ratio", engine)
        self.assertEqual(engine[engine.index("--ffmpeg-path") + 1], "bundled-ffmpeg")

    def test_misplaced_credit_is_sorted_only_in_private_engine_input(self):
        original = (
            "1\r\n00:00:10,000 --> 00:00:11,000\r\nDialogue\r\n\r\n"
            "99\r\n00:00:01,000 --> 00:00:02,000\r\nDownloaded From www.example.org\r\n\r\n"
        )
        self.source.write_bytes(original.encode())

        def process(command, progress, cancel):
            if "-o" in command:
                private = Path(command[command.index("-i") + 1]).read_text()
                self.assertLess(private.index("Downloaded"), private.index("Dialogue"))
                Path(command[command.index("-o") + 1]).write_text(
                    private.replace("01,000", "03,000").replace("02,000", "04,000")
                    .replace("10,000", "12,000").replace("11,000", "13,000"),
                )
            else:
                self.process(command, progress, cancel)

        with patch.object(sync, "_run", side_effect=process):
            sync.synchronize(
                self.video, self.source, self.output, 0, None, False, self.progress, self.cancel,
            )
        expected = original.replace("01,000", "03,000").replace("02,000", "04,000")
        expected = expected.replace("10,000", "12,000").replace("11,000", "13,000")
        self.assertEqual(self.output.read_bytes(), expected.encode())
        self.assertEqual(self.source.read_bytes(), original.encode())

    def test_nonstandard_credit_timestamp_remains_opaque_and_unchanged(self):
        original = (
            "1\n00:00:10,000 --> 00:00:11,000\nDialogue\n\n"
            "9999\n00:00:0,500 --> 00:00:2,00\nwww.example.org\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nDownloaded From www.example.org\n"
        )
        self.source.write_text(original)

        def process(command, progress, cancel):
            if "-o" in command:
                text = Path(command[command.index("-i") + 1]).read_text()
                self.assertEqual(len(list(TIMESTAMP.finditer(text))), 2)
                def shift(match):
                    return (
                        ms_to_timestamp(timestamp_to_ms(match["start"]) + 2000)
                        + match["arrow"]
                        + ms_to_timestamp(timestamp_to_ms(match["end"]) + 2000)
                    )
                Path(command[command.index("-o") + 1]).write_text(
                    TIMESTAMP.sub(shift, text),
                )
            else:
                self.process(command, progress, cancel)

        with patch.object(sync, "_run", side_effect=process):
            count = sync.synchronize(
                self.video, self.source, self.output, 0, None, False, self.progress, self.cancel,
            )
        self.assertEqual(count, 2)
        result = self.output.read_text()
        self.assertIn("00:00:0,500 --> 00:00:2,00\nwww.example.org", result)
        self.assertEqual(TIMESTAMP.sub("", result), TIMESTAMP.sub("", original))

    def test_embedded_track_extraction_and_offset_only_mode(self):
        self.run_sync(source=None, subtitle_track=5, fix_drift=False, fix_sections=False)
        extract = self.commands[0]
        self.assertEqual(extract[extract.index("-map") + 1], "0:5")
        self.assertIn("--no-fix-framerate", self.commands[-1])
        self.assertNotIn("--gss", self.commands[-1])
        self.assertNotIn("--split-penalty", self.commands[-1])
        self.assertEqual(self.commands[-1][self.commands[-1].index("--max-offset-seconds") + 1], "600")

    def test_section_transitions_are_reported_for_manual_review(self):
        original_process = self.process

        def process(command, progress, cancel):
            original_process(command, progress, cancel)
            if "-o" in command:
                Path(command[command.index("-o") + 1]).write_text(
                    ALIGNED.replace("00:00:05,000", "00:00:04,000").replace("00:00:06,500", "00:00:05,500"),
                    encoding="utf-8",
                )

        with patch.object(sync, "_run", side_effect=process):
            sync.synchronize(
                self.video, self.source, self.output, 0, None, False, self.progress, self.cancel,
            )
        self.assertTrue(any(
            "near cue(s): 2" in call.args[0] and "early or late" in call.args[0]
            for call in self.progress.call_args_list
        ))

    def test_cp1252_is_preserved(self):
        self.source.write_bytes(SOURCE.replace("Hello!", "Café!").encode("cp1252"))

        def process(command, progress, cancel):
            self.process(command, progress, cancel)
            if "-o" in command:
                Path(command[command.index("-o") + 1]).write_text(
                    ALIGNED.replace("Hello!", "Café!"), encoding="utf-8",
                )

        with (
            patch.object(sync, "_run", side_effect=process),
            patch.object(sync.speech, "ffmpeg_executable", return_value="ffmpeg"),
        ):
            sync.synchronize(
                self.video, self.source, self.output, 0, None, False, self.progress, self.cancel,
            )
        self.assertIn(b"Caf\xe9!", self.output.read_bytes())

    def test_rejects_overwriting_inputs_aliases_and_protected_paths(self):
        alias = self.root / "alias.srt"
        alias.symlink_to(self.source)
        hardlink = self.root / "hardlink.srt"
        os.link(self.source, hardlink)
        for output in (self.source, alias, hardlink):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overwrite"):
                self.run_sync(output=output, overwrite=True)
        self.output.write_text("other selected input")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.run_sync(overwrite=True, protected_paths=(self.output,))

    def test_existing_output_requires_confirmation(self):
        self.output.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_sync()
        self.assertEqual(self.output.read_text(), "keep")
        self.assertEqual(self.run_sync(overwrite=True), 2)

    def test_late_created_output_is_not_overwritten(self):
        original_process = self.process

        def process(command, progress, cancel):
            original_process(command, progress, cancel)
            if "-o" in command:
                self.output.write_text("created meanwhile")

        with patch.object(sync, "_run", side_effect=process), self.assertRaisesRegex(ValueError, "output changed"):
            sync.synchronize(
                self.video, self.source, self.output, 0, None, False, self.progress, self.cancel,
            )
        self.assertEqual(self.output.read_text(), "created meanwhile")

    def test_input_changes_and_cancellation_prevent_publication(self):
        original_process = self.process
        for action in ("change", "cancel"):
            self.cancel.clear()

            def process(command, progress, cancel):
                original_process(command, progress, cancel)
                if "-o" in command:
                    if action == "change":
                        self.video.write_bytes(b"changed video")
                    else:
                        cancel.set()

            expected = ValueError if action == "change" else sync.CancelledError
            with self.subTest(action=action), patch.object(sync, "_run", side_effect=process):
                with self.assertRaises(expected):
                    sync.synchronize(
                        self.video, self.source, self.output, 0, None, False, self.progress, self.cancel,
                    )
            self.assertFalse(self.output.exists())
            self.assertFalse(list(self.root.glob(".audio-sync-*")))

    def test_process_failure_leaves_existing_output_untouched(self):
        self.output.write_text("keep")
        with patch.object(sync, "_run", side_effect=ValueError("engine failed")):
            with self.assertRaisesRegex(ValueError, "engine failed"):
                sync.synchronize(
                    self.video, self.source, self.output, 0, None, False,
                    self.progress, self.cancel, overwrite=True,
                )
        self.assertEqual(self.output.read_text(), "keep")
        self.assertFalse(list(self.root.glob(".audio-sync-*")))

    def test_low_quality_alignment_is_an_error_not_an_unchanged_success(self):
        def process(command, progress, cancel):
            self.process(command, progress, cancel)
            if "-o" in command:
                progress("low-quality alignment (score -12); leaving subtitles unmodified")

        with patch.object(sync, "_run", side_effect=process):
            with self.assertRaisesRegex(ValueError, "unreliable"):
                sync.synchronize(
                    self.video, self.source, self.output, 0, None, True, self.progress, self.cancel,
                )
        self.assertFalse(self.output.exists())

    def test_cancellation_stops_engine_and_its_child_process(self):
        leaked = self.root / "child-was-not-stopped"
        child = f"import time; from pathlib import Path; time.sleep(0.5); Path({str(leaked)!r}).touch()"
        parent = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
            "print('child started', flush=True); time.sleep(30)"
        )

        def progress(line):
            if line == "child started":
                self.cancel.set()

        with self.assertRaisesRegex(sync.CancelledError, "Audio synchronization cancelled"):
            sync._run([sys.executable, "-c", parent], progress, self.cancel)
        time.sleep(0.7)
        self.assertFalse(leaked.exists())

    def test_rejects_missing_output_and_missing_dependency(self):
        with patch.object(sync.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(ValueError, "ffsubsync is missing"):
                self.run_sync()
        with patch.object(sync, "_run", return_value=""):
            with self.assertRaisesRegex(ValueError, "did not produce"):
                sync.synchronize(
                    self.video, self.source, self.output, 0, None, True, self.progress, self.cancel,
                )

    def test_rejects_changed_text_dropped_cues_and_invalid_timings(self):
        for aligned in (
            ALIGNED.replace("Hello!", "Goodbye!"),
            ALIGNED.split("2\n00:00:05")[0],
            ALIGNED.replace("00:00:02,000", "00:00:00,500"),
            ALIGNED.replace("00:00:05,000", "00:00:00,500"),
        ):
            with self.subTest(aligned=aligned), self.assertRaises(ValueError):
                sync.retime_original(SOURCE, aligned)

    def test_inspection_uses_global_stream_indices_and_ignores_output_duplicates(self):
        log = (
            "Input #0, matroska, from 'video.mkv':\n"
            "  Stream #0:1[0x2](eng): Audio: aac, 48000 Hz\n"
            "  Stream #0:3(fra): Audio: ac3, 48000 Hz\n"
            "  Stream #0:5(eng): Subtitle: subrip (default)\n"
            "  Stream #0:6(eng): Subtitle: hdmv_pgs_subtitle\n"
            "Output #0, null, to 'pipe:':\n  Stream #0:0(eng): Audio: aac\n"
        )
        with patch.object(sync, "_run", return_value=log):
            tracks = sync.inspect_tracks(self.video, self.progress, self.cancel)
        self.assertEqual([track.index for track in tracks], [1, 3, 5, 6])
        self.assertEqual(tracks[0].label, "1: eng (aac)")
        self.assertNotIn(tracks[-1].codec, sync.TEXT_CODECS)
        with patch.object(sync, "_run", return_value="no tracks"), self.assertRaisesRegex(ValueError, "No audio"):
            sync.inspect_tracks(self.video, self.progress, self.cancel)


@unittest.skipUnless(os.environ.get("DISPLAY") or os.name == "nt", "Tk display required")
class AudioSyncDialogTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.video = Path(self.folder.name) / "video.mkv"
        self.video.touch()
        self.dialog = AudioSyncDialog(self.root, str(self.video))
        self.addCleanup(lambda: self.dialog.destroy() if self.dialog.winfo_exists() else None)

    def run_operation(self, operation):
        def thread(**kwargs):
            return Mock(start=kwargs["target"])
        with patch("audio_sync_dialog.threading.Thread", side_effect=thread):
            self.dialog._start(operation)
        if self.dialog._poll_id is not None:
            self.dialog.after_cancel(self.dialog._poll_id)
        self.dialog._poll()

    def inspect(self):
        with patch.object(sync, "inspect_tracks", return_value=[
            sync.Track(1, "Audio", "aac", "1: eng (aac)"),
            sync.Track(3, "Audio", "ac3", "3: fra (ac3)"),
            sync.Track(5, "Subtitle", "subrip", "5: eng (subrip)"),
            sync.Track(6, "Subtitle", "hdmv_pgs_subtitle", "6: eng (PGS)"),
        ]):
            self.run_operation("inspect")

    def test_track_selection_and_input_change_invalidation(self):
        self.inspect()
        self.assertEqual(tuple(self.dialog.audio_combo["values"]), ("1: eng (aac)", "3: fra (ac3)"))
        self.assertEqual(tuple(self.dialog.subtitle_combo["values"]), ("5: eng (subrip)",))
        self.dialog.video.set(str(self.video.with_name("other.mkv")))
        self.assertEqual(self.dialog.tracks, [])
        self.assertEqual(self.dialog.audio.get(), "")
        self.assertEqual(self.dialog.inspected_inputs, ())

    def test_embedded_sync_passes_selected_tracks_and_restores_controls(self):
        self.inspect()
        self.dialog.audio.set("3: fra (ac3)")
        with patch.object(sync, "synchronize", return_value=2) as synchronize, patch(
            "audio_sync_dialog.messagebox.showinfo",
        ):
            self.run_operation("sync")
        self.assertIsNone(synchronize.call_args.args[1])
        self.assertEqual(synchronize.call_args.args[3:6], (3, 5, True))
        self.assertTrue(synchronize.call_args.kwargs["fix_sections"])
        self.assertFalse(self.dialog.busy)
        self.assertEqual(str(self.dialog.audio_combo["state"]), "readonly")

    def test_section_correction_can_be_disabled(self):
        self.inspect()
        self.dialog.fix_sections.set(False)
        with patch.object(sync, "synchronize", return_value=2) as synchronize, patch(
            "audio_sync_dialog.messagebox.showinfo",
        ):
            self.run_operation("sync")
        self.assertFalse(synchronize.call_args.kwargs["fix_sections"])

    def test_optional_report_contains_engine_log_and_reports_write_failure_without_losing_success(self):
        from sync_report import report_path

        self.inspect()
        self.dialog.generate_report = True
        output = Path(self.dialog.output.get())

        def synchronize(*args, **kwargs):
            args[6]("Measured audio offset: -2.5 seconds")
            output.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
            return 1

        with patch.object(sync, "synchronize", side_effect=synchronize), patch(
            "audio_sync_dialog.messagebox.showinfo",
        ) as info:
            self.run_operation("sync")
            self.assertIn("Measured audio offset: -2.5 seconds", report_path(output).read_text())
            self.assertIn(str(report_path(output)), info.call_args.args[1])
            with (
                patch("audio_sync_dialog.messagebox.askyesno", return_value=True),
                patch("audio_sync_dialog.write_report", side_effect=PermissionError("Report is read-only")),
            ):
                self.run_operation("sync")
            self.assertIn("SRT saved, but the report could not be saved", info.call_args.args[1])
            self.assertIn("Report is read-only", info.call_args.args[1])
            self.assertTrue(output.is_file())

    def test_cancelled_audio_sync_does_not_generate_report(self):
        from sync_report import report_path

        self.inspect()
        self.dialog.generate_report = True
        with patch.object(
            sync, "synchronize", side_effect=sync.CancelledError("Cancelled"),
        ), patch("audio_sync_dialog.write_report") as report:
            self.run_operation("sync")
        report.assert_not_called()
        self.assertFalse(report_path(Path(self.dialog.output.get())).exists())

    def test_success_callback_uses_saved_path_even_if_output_field_changes(self):
        self.inspect()
        callback = Mock()
        self.dialog.on_saved = callback
        output = Path(self.dialog.output.get())

        def synchronize(*_args, **_kwargs):
            return 2

        def thread(**kwargs):
            def start():
                kwargs["target"]()
                self.dialog.output.set(str(output.with_name("changed.srt")))
            return Mock(start=start)

        with (
            patch.object(sync, "synchronize", side_effect=synchronize),
            patch("audio_sync_dialog.threading.Thread", side_effect=thread),
            patch("audio_sync_dialog.messagebox.showinfo"),
        ):
            self.dialog._start("sync")
            self.dialog.after_cancel(self.dialog._poll_id)
            self.dialog._poll()
        callback.assert_called_once_with(output)

    def test_error_and_cancellation_do_not_call_success_callback(self):
        self.inspect()
        callback = Mock()
        self.dialog.on_saved = callback
        for error in (ValueError("failed"), sync.CancelledError("cancelled")):
            with (
                self.subTest(error=error),
                patch.object(sync, "synchronize", side_effect=error),
                patch("audio_sync_dialog.messagebox.showerror"),
            ):
                self.run_operation("sync")
        callback.assert_not_called()

    def test_requires_inspection_and_displays_worker_errors(self):
        with patch("audio_sync_dialog.messagebox.showerror") as error:
            self.dialog._start("sync")
            self.assertIn("Inspect tracks", error.call_args.args[1])
        self.inspect()
        with patch.object(sync, "synchronize", side_effect=ValueError("bad audio")), patch(
            "audio_sync_dialog.messagebox.showerror",
        ) as error:
            self.run_operation("sync")
            self.assertEqual(error.call_args.args[1], "bad audio")
        self.assertFalse(self.dialog.busy)

    def test_cancel_keeps_dialog_open_until_worker_finishes(self):
        self.dialog.busy = True
        with patch("audio_sync_dialog.messagebox.askyesno", return_value=True):
            self.dialog._close()
        self.assertTrue(self.dialog.cancel_event.is_set())
        self.assertTrue(self.dialog.winfo_exists())
        self.dialog.busy = False

    def test_close_button_is_available_without_window_manager_decorations(self):
        self.root.update_idletasks()
        self.assertEqual(str(self.dialog.close_button["state"]), "normal")
        self.assertFalse(self.dialog.transient())
        if self.dialog._windowingsystem == "x11":
            self.assertIn("normal", self.dialog.wm_attributes("-type"))
        self.dialog.busy = True
        with patch("audio_sync_dialog.messagebox.askyesno", return_value=True):
            self.dialog.close_button.invoke()
        self.assertTrue(self.dialog.cancel_event.is_set())
        self.assertTrue(self.dialog.winfo_exists())
        self.dialog.busy = False
        self.dialog.close_button.invoke()
        self.assertFalse(self.dialog.winfo_exists())

    def test_dynamic_status_messages_follow_selected_language_without_translating_paths(self):
        from ui_preferences import InterfacePreferences
        from ui_translations import LANGUAGES

        self.dialog.destroy()
        output = Path("/data/episode.en.audio-synced.srt")
        for language in LANGUAGES:
            self.root.interface_preferences = InterfacePreferences(self.root, language, "dark")
            dialog = AudioSyncDialog(self.root)
            try:
                self.assertEqual(dialog.close_button["text"], dialog.tr("Close"))
                initial = "Choose a video and inspect its tracks. No reference SRT is needed."
                self.assertEqual(dialog.status.get(), dialog.tr(initial))
                if language != "en":
                    self.assertNotEqual(dialog.status.get(), initial, language)
                dialog.events.put(("saved", (output, dialog.tr("Saved {count} cues to {path}").format(
                    count=42, path=output,
                ))))
                with patch("audio_sync_dialog.messagebox.showinfo") as info:
                    dialog._poll()
                self.assertEqual(info.call_args.args[0], dialog.tr("Audio synchronization complete"))
                self.assertIn(str(output), dialog.status.get())
                dialog._cancel()
                self.assertEqual(dialog.status.get(), dialog.tr("Cancelling audio synchronization..."))
            finally:
                dialog.destroy()


class AudioSyncIntegrationTests(unittest.TestCase):
    def test_local_episode_regression_against_comparison_timings(self):
        root = Path(__file__).resolve().parent
        video = root / "Veronica Mars.S03E01.Welcome Wagon.mkv"
        source = root / "Veronica Mars.S03E01.Welcome Wagon.en.unsync.srt"
        reference = root / "Veronica.Mars.S03E01.1080p.WEB-DL.AAC2.0.AVC-TrollHD.srt"
        if not all(path.is_file() for path in (video, source, reference)):
            self.skipTest("Optional local episode fixtures are not distributed with the tests.")
        from sync_subtitles import decode_srt

        original = decode_srt(source.read_bytes())[0]
        reference_cues = list(TIMESTAMP.finditer(decode_srt(reference.read_bytes())[0]))
        log = []
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "audio-synced.srt"
            # The comparison SRT is used only after synchronization, never as an input.
            count = sync.synchronize(
                video, source, output, 1, None, True, log.append, threading.Event(),
            )
            fixed = decode_srt(output.read_bytes())[0]
        self.assertEqual(count, len(reference_cues))
        self.assertEqual(TIMESTAMP.sub("<timing>", original), TIMESTAMP.sub("<timing>", fixed))
        errors = [
            abs(timestamp_to_ms(actual["start"]) - timestamp_to_ms(expected["start"]))
            for actual, expected in zip(TIMESTAMP.finditer(fixed), reference_cues)
        ]
        self.assertLessEqual(sorted(errors)[int(0.95 * (len(errors) - 1))], 300)
        self.assertGreaterEqual(sum(error <= 500 for error in errors) / count, 0.98)
        self.assertTrue(any("Section timing changes detected" in line for line in log))

    def test_real_audio_corrects_multiple_section_delays_without_reference_subtitles(self):
        randomizer = random.Random(73)
        sample_rate = 16000
        starts, lengths = [], []
        position = 2.0
        for _ in range(90):
            starts.append(position)
            length = randomizer.uniform(0.8, 2.0)
            lengths.append(length)
            position += length + randomizer.uniform(0.3, 2.2)
        samples = array("h", [0]) * round((position + 2) * sample_rate)
        for start, length in zip(starts, lengths):
            for index in range(round(length * sample_rate)):
                t = index / sample_rate
                envelope = min(1.0, t * 50, (length - t) * 50)
                value = sum(
                    math.sin(2 * math.pi * 137 * harmonic * t) / harmonic
                    for harmonic in range(1, 9)
                )
                samples[round(start * sample_rate) + index] = round(7000 * envelope * value)
        if sys.byteorder != "little":
            samples.byteswap()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio, source, output = root / "audio.wav", root / "source.srt", root / "fixed.srt"
            with wave.open(str(audio), "wb") as writer:
                writer.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
                writer.writeframes(samples.tobytes())
            source.write_text("".join(
                f"{number}\n{ms_to_timestamp(round((start + shift) * 1000))} --> "
                f"{ms_to_timestamp(round((start + length + shift) * 1000))}\n"
                f"Test dialogue {number}.\n\n"
                for number, (start, length) in enumerate(zip(starts, lengths), 1)
                for shift in ((2.6 if number <= 30 else 5.6 if number <= 60 else 8.5),)
            ))
            log = []
            self.assertEqual(sync.synchronize(
                audio, source, output, 0, None, True, log.append, threading.Event(),
            ), 90)
            results = list(TIMESTAMP.finditer(output.read_text()))
            errors = [
                abs(timestamp_to_ms(match["start"]) - start * 1000)
                for match, start in zip(results, starts)
            ]
            self.assertLess(sorted(errors)[round(len(errors) * 0.95) - 1], 300)
            for index, error in enumerate(errors):
                if abs(index - 30) > 3 and abs(index - 60) > 3:
                    self.assertLess(error, 300)
            self.assertTrue(any("Section timing changes detected" in line for line in log))

    def test_real_engine_corrects_offset_of_embedded_subtitles_on_selected_audio(self):
        sample_rate = 16000
        starts = [1.0, 4.1, 7.7, 11.2, 15.8, 19.0, 23.6, 28.1]
        lengths = [0.8, 1.5, 1.1, 2.0, 1.3, 0.9, 1.8, 1.2]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "speech-pattern.wav"
            samples = array("h", [0]) * (32 * sample_rate)
            for start, length in zip(starts, lengths):
                for index in range(round(length * sample_rate)):
                    t = index / sample_rate
                    envelope = min(1.0, t * 50, (length - t) * 50)
                    value = sum(
                        math.sin(2 * math.pi * (137 * harmonic) * t) / harmonic
                        for harmonic in range(1, 9)
                    )
                    samples[round(start * sample_rate) + index] = round(
                        7500 * envelope * (0.65 + 0.35 * math.sin(2 * math.pi * 4.7 * t)) * value,
                    )
            if sys.byteorder != "little":
                samples.byteswap()
            with wave.open(str(audio), "wb") as writer:
                writer.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
                writer.writeframes(samples.tobytes())
            for name, rate, fix_drift, delay in (
                ("offset", 1.0, False, 0.0),
                ("drift", 1.04, True, 0.0),
                ("delayed-audio", 1.0, False, 1.25),
            ):
                with self.subTest(name=name):
                    source = root / f"{name}.srt"
                    source.write_text("".join(
                        f"{number}\n{ms_to_timestamp(round(((start + delay) * rate + 2) * 1000))} --> "
                        f"{ms_to_timestamp(round(((start + delay + length) * rate + 2) * 1000))}\n"
                        f"Dialogue {number}.\n\n"
                        for number, (start, length) in enumerate(zip(starts, lengths), 1)
                    ))
                    video = root / f"{name}.mkv"
                    subprocess.run([
                        sync.speech.ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=34",
                        "-itsoffset", str(delay), "-i", str(audio),
                        "-i", str(source), "-map", "0:a", "-map", "1:a", "-map", "2:s",
                        "-c:a", "pcm_s16le", "-c:s", "srt", "-metadata:s:a:1", "language=eng",
                        "-metadata:s:s:0", "language=eng", str(video),
                    ], check=True, capture_output=True)
                    cancel = threading.Event()
                    tracks = sync.inspect_tracks(video, lambda _: None, cancel)
                    self.assertEqual([(track.index, track.kind) for track in tracks], [
                        (0, "Audio"), (1, "Audio"), (2, "Subtitle"),
                    ])
                    output = root / f"{name}.fixed.srt"
                    count = sync.synchronize(
                        video, None, output, 1, 2, fix_drift, lambda _: None, cancel,
                    )
                    self.assertEqual(count, len(starts))
                    matches = list(TIMESTAMP.finditer(output.read_text()))
                    for match, start, length in zip(matches, starts, lengths):
                        self.assertLess(
                            abs(timestamp_to_ms(match["start"]) - (start + delay) * 1000), 300,
                        )
                        self.assertLess(
                            abs(timestamp_to_ms(match["end"]) - (start + delay + length) * 1000), 300,
                        )
                    if name == "offset":
                        with self.assertRaises(ValueError):
                            sync.synchronize(
                                video, None, root / "silent.fixed.srt", 0, 2,
                                False, lambda _: None, cancel,
                            )


if __name__ == "__main__":
    unittest.main()
