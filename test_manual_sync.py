import json
import os
import queue
import stat
import tempfile
import threading
import time
import tkinter as tk
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import manual_sync
import subtitle_alignment as alignment
import sync_subtitles
from test_subtitle_devices import load_gui


def fixture(cues):
    return "\n\n".join(
        f"{number}\n{sync_subtitles.ms_to_timestamp(cue.start_ms)} --> "
        f"{sync_subtitles.ms_to_timestamp(cue.end_ms)}\n{cue.text}"
        for number, cue in enumerate(cues, 1)
    ) + "\n"


class SyncOutputTests(unittest.TestCase):
    def setUp(self):
        self.gui = load_gui()
        self.addCleanup(self.gui._test_settings_directory.cleanup)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.source = self.folder / "source.srt"
        self.source.write_bytes(b"1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        self.output = self.folder / "output.srt"
        self.app = self.gui.SubtitleSyncApp.__new__(self.gui.SubtitleSyncApp)
        self.app.messages = queue.Queue()
        self.app.sync_cancel = threading.Event()
        self.app.sync_commit_lock = threading.Lock()
        self.app.sync_committed = False

    def synchronize(self):
        with patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
            [(1, 0)], {"matched_cues": 1},
        )):
            self.app._sync_worker(None, self.source, self.source, self.output, 1000, 70)
        messages = []
        while not self.app.messages.empty():
            messages.append(self.app.messages.get())
        self.assertFalse(list(self.folder.glob(".subtitle-sync-*")))
        return messages[-1]

    @unittest.skipUnless(os.name == "posix", "POSIX file permissions")
    def test_new_and_replaced_outputs_respect_umask(self):
        for mask, expected in ((0o002, 0o664), (0o022, 0o644), (0o077, 0o600)):
            for existing in (False, True):
                with self.subTest(umask=oct(mask), existing=existing):
                    self.output.unlink(missing_ok=True)
                    if existing:
                        self.output.write_text("original", encoding="utf-8")
                        self.output.chmod(0o600)
                    self.app.sync_committed = False
                    previous = os.umask(mask)
                    try:
                        message = self.synchronize()
                    finally:
                        os.umask(previous)
                    self.assertEqual(message[0], "success")
                    self.assertTrue(self.app.sync_committed)
                    self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), expected)
                    self.assertEqual(self.output.read_bytes(), self.source.read_bytes())

    def test_cancellation_after_staging_preserves_existing_output(self):
        self.output.write_bytes(b"original")
        write_bytes = Path.write_bytes

        def cancel_after_write(path, data):
            written = write_bytes(path, data)
            self.app.sync_cancel.set()
            return written

        with patch.object(Path, "write_bytes", new=cancel_after_write):
            message = self.synchronize()
        self.assertEqual(message[0], "cancelled")
        self.assertFalse(self.app.sync_committed)
        self.assertEqual(self.output.read_bytes(), b"original")

    def test_replace_failure_is_reported_and_preserves_existing_output(self):
        self.output.write_bytes(b"original")
        with patch.object(Path, "replace", side_effect=PermissionError("Cannot replace output")):
            message = self.synchronize()
        self.assertEqual(message, ("error", "Cannot replace output"))
        self.assertFalse(self.app.sync_committed)
        self.assertEqual(self.output.read_bytes(), b"original")


class ContextMatchingTests(unittest.TestCase):
    def test_long_single_whisper_cues_are_usable_without_expanding_multi_cue_windows(self):
        cues = [
            alignment.Cue(2000, 20000, "A complete sentence from a speech recognition segment."),
            alignment.Cue(21000, 23000, "The next sentence."),
        ]
        windows = alignment._build_windows(cues, 70)
        self.assertEqual(
            windows,
            [alignment.DialogueWindow(cue.start_ms, cue.end_ms, cue.text) for cue in cues],
        )

    def test_windows_preserve_non_latin_dialogue_and_split_sentences(self):
        for text in ("你好，世界", "こんにちは", "مرحبا بالعالم", "Привет мир", "안녕하세요"):
            windows = alignment._build_windows([alignment.Cue(1000, 2000, text)], 70)
            self.assertEqual(windows[0].text, text)
        cues = [
            alignment.Cue(1000, 2000, "We should"),
            alignment.Cue(2200, 3200, "leave before dawn."),
            alignment.Cue(12000, 13000, "New scene"),
        ]
        texts = [window.text for window in alignment._build_windows(cues, 70)]
        self.assertIn("We should leave before dawn.", texts)
        self.assertNotIn("We should leave before dawn. New scene", texts)
        context = alignment._window_context(alignment.DialogueWindow(2200, 3200, "leave before dawn."), cues)
        self.assertEqual(context, "We should leave before dawn.")

    def test_context_can_prefer_paraphrase_over_literal_distractor(self):
        source = [
            alignment.Cue(0, 1000, "Manually paired opening"),
            alignment.Cue(20000, 21000, "Source idiomatic expression"),
        ]
        reference = [
            alignment.Cue(0, 1000, "Unrelated wording for selected opening"),
            alignment.Cue(20000, 21000, "Contextually correct paraphrase"),
            alignment.Cue(22000, 23000, "Literal but wrong scene"),
        ]
        left = [alignment.DialogueWindow(c.start_ms, c.end_ms, c.text) for c in source]
        right = [alignment.DialogueWindow(c.start_ms, c.end_ms, c.text) for c in reference]
        e0, e1, e2, e3 = np.eye(4)
        embeddings = {window.text: e0 for window in left + right}
        embeddings[right[1].text] = 0.65 * e0 + np.sqrt(1 - 0.65 ** 2) * e1
        embeddings[right[2].text] = 0.75 * e0 + np.sqrt(1 - 0.75 ** 2) * e2
        embeddings.update({"context:" + window.text: e3 for window in left + right})
        embeddings["context:" + right[2].text] = e2
        model = Mock()
        model.embed.side_effect = lambda texts, **_: [embeddings[text] for text in texts]
        with (
            patch.object(alignment, "_build_windows", side_effect=[left, right]),
            patch.object(alignment, "_window_context", side_effect=lambda window, _: "context:" + window.text),
            patch.object(alignment, "_load_embedding_model", return_value=model),
        ):
            anchors, diagnostics = alignment.analyze_subtitles(
                source, reference, 9999, source_anchor=1, reference_anchor=1
            )
        self.assertEqual(anchors, [(1, 0.0), (2, 0.0)])
        self.assertEqual(diagnostics["anchor_time_ms"], 0)
        embedded = model.embed.call_args.args[0]
        self.assertIn("context:" + left[1].text, embedded)
        self.assertIn("context:" + right[1].text, embedded)

    def test_middle_manual_anchor_is_exact_and_timeline_is_aligned_on_both_sides(self):
        source = [
            alignment.Cue(i * 60000 + 10000, i * 60000 + 12000, f"Distinct dialogue {i}")
            for i in range(5)
        ]
        reference = [replace(cue, start_ms=cue.start_ms + 20000, end_ms=cue.end_ms + 20000) for cue in source]
        model = Mock()
        model.embed.return_value = np.eye(5)
        with patch.object(alignment, "_load_embedding_model", return_value=model):
            anchors, diagnostics = alignment.analyze_subtitles(
                source, reference, 0, source_anchor=3, reference_anchor=3
            )
        self.assertEqual(anchors, [(i, 20.0) for i in range(1, 6)])
        self.assertEqual(diagnostics["anchor_time_ms"], reference[2].start_ms)
        output, count = sync_subtitles.sync_srt(fixture(source), anchors)
        self.assertEqual(count, 5)
        self.assertEqual(alignment.parse_srt(output), reference)

    def test_invalid_manual_indices_fail_before_inference(self):
        cues = [alignment.Cue(0, 1000, "Dialogue")]
        for left, right in ((0, 1), (2, 1), (1, 2), (True, 1), (None, 1)):
            with (
                self.subTest(left=left, right=right),
                patch.object(alignment, "_load_embedding_model") as load,
                self.assertRaises(ValueError),
            ):
                alignment.analyze_subtitles(cues, cues, 0, source_anchor=left, reference_anchor=right)
            load.assert_not_called()

    def test_manual_pair_can_have_different_source_and_reference_cue_numbers(self):
        source = [
            alignment.Cue(10000 + i * 60000, 12000 + i * 60000, f"Distinct dialogue {i}")
            for i in range(5)
        ]
        reference = [
            replace(cue, start_ms=cue.start_ms + 7000, end_ms=cue.end_ms + 7000)
            for cue in source[1:]
        ]
        model = Mock()
        model.embed.side_effect = lambda texts, **_: [
            np.eye(5)[int(text.rsplit(" ", 1)[-1])] for text in texts
        ]
        with patch.object(alignment, "_load_embedding_model", return_value=model):
            anchors, _ = alignment.analyze_subtitles(
                source, reference, 0, source_anchor=3, reference_anchor=2
            )
        self.assertIn((3, 7.0), anchors)
        self.assertTrue(all(offset == 7.0 for _, offset in anchors))
        self.assertGreaterEqual(len(anchors), 3)

    def test_all_embedded_subrip_languages_are_offered(self):
        tracks = [
            {"id": 1, "type": "subtitles", "codec": "SubRip/SRT", "properties": {"language": "jpn"}},
            {"id": 2, "type": "subtitles", "codec": "SubRip/SRT", "properties": {"language_ietf": "es", "track_name": "Spanish"}},
            {"id": 3, "type": "subtitles", "codec": "SubRip/SRT", "properties": {"language": "eng"}},
            {"id": 4, "type": "subtitles", "codec": "HDMV PGS", "properties": {"language": "fra"}},
        ]
        with (
            patch.object(alignment, "_find_tool", side_effect=lambda name: name),
            patch.object(alignment.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps({"tracks": tracks}), stderr=""
            )),
        ):
            found = alignment.list_subtitle_tracks(Path("video.mkv"))
        self.assertEqual([track.track_id for track in found], [1, 2, 3])
        self.assertIn("jpn", found[0].label)
        self.assertIn("Spanish", found[1].label)

    def test_extract_uses_explicit_reference_track_id(self):
        expected = [alignment.Cue(2000, 3000, "Hola")]

        def extract(command, **_):
            track_id, path = command[-1].split(":", 1)
            self.assertEqual(track_id, "7")
            Path(path).write_text(fixture(expected), encoding="utf-8")
            return SimpleNamespace(returncode=0, stderr="")

        with (
            patch.object(alignment, "_find_tool", return_value="mkvextract"),
            patch.object(alignment.subprocess, "run", side_effect=extract),
        ):
            self.assertEqual(alignment.extract_reference_srt(Path("video.mkv"), 7), expected)


@unittest.skipUnless(os.environ.get("DISPLAY") or os.name == "nt", "Needs a display")
class ManualSyncGuiTests(unittest.TestCase):
    def setUp(self):
        self.gui = load_gui()
        self.root = tk.Tk()
        self.app = self.gui.SubtitleSyncApp(self.root)
        self.source = tuple(
            alignment.Cue(i * 60000, i * 60000 + 2000, text)
            for i, text in enumerate(("Bonjour", "Nous partons demain", "A bientot"))
        )
        self.reference = tuple(
            alignment.Cue(i * 60000 + 5000, i * 60000 + 7000, text)
            for i, text in enumerate(("こんにちは", "明日出発します", "また会いましょう"))
        )

    def tearDown(self):
        for identifier in self.root.tk.call("after", "info"):
            self.root.after_cancel(identifier)
        self.root.destroy()

    def test_popup_requires_both_selections_and_preserves_original_cue_numbers_after_search(self):
        accept = Mock()
        dialog = manual_sync.CuePairDialog(self.root, self.source, self.reference, accept)
        self.root.update()
        self.assertEqual(str(dialog.confirm.cget("state")), "disabled")
        dialog.searches[0].set("demain")
        self.root.update()
        self.assertEqual(dialog.trees[0].get_children(), ("2",))
        dialog.trees[0].selection_set("2")
        dialog.trees[1].selection_set("3")
        self.root.update()
        self.assertEqual(str(dialog.confirm.cget("state")), "normal")
        self.assertIn("Source cue 2 = reference cue 3", dialog.summary.get())
        dialog.confirm.invoke()
        accept.assert_called_once_with(2, 3)

    def test_app_accepts_pair_and_invalidates_it_when_an_input_changes(self):
        self.app._show_pair_picker(self.source, self.reference, ())
        self.root.update()
        window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        widgets = list(descendants(window))
        trees = [widget for widget in widgets if widget.winfo_class() == "Treeview"]
        trees[0].selection_set("2")
        trees[1].selection_set("3")
        self.root.update()
        next(widget for widget in widgets if widget.winfo_class() == "TButton"
             and widget.cget("text") == "Use selected pair").invoke()
        self.assertEqual(self.app.sync_selection.source_number, 2)
        self.assertEqual(self.app.sync_selection.reference_number, 3)
        self.assertEqual(self.app.initial_time.get(), self.gui.format_time(self.reference[2].start_ms))
        self.app.english_path.set("a-different-reference.srt")
        self.assertIsNone(self.app.sync_selection)
        self.assertEqual(self.app.initial_time.get(), "")

    def test_analysis_requires_manual_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            source.write_text(fixture(self.source), encoding="utf-8")
            self.app.portuguese_path.set(str(source))
            self.app.output_path.set(str(Path(folder) / "output.srt"))
            with (
                patch.object(self.gui.messagebox, "showwarning") as warning,
                patch.object(self.gui.threading, "Thread") as worker,
            ):
                self.app._start_sync()
            warning.assert_called_once()
            worker.assert_not_called()

    def test_worker_forwards_manual_pair_to_all_backends_and_keeps_reference_unchanged(self):
        for backend, device, private in (("ONNX", "CPU", False), ("Vulkan", "AMD", False),
                                         ("ONNX", "NVIDIA", True)):
            with self.subTest(backend=backend, device=device), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "source.srt"
                reference = Path(folder) / "reference.srt"
                output = Path(folder) / "output.srt"
                source.write_text(fixture(self.source), encoding="utf-8")
                reference.write_text(fixture(self.reference), encoding="utf-8")
                selection = manual_sync.SyncSelection(
                    self.source, self.reference, 2, 2,
                    manual_sync.input_signatures((str(source), str(reference))),
                )
                with (
                    patch.object(self.gui.gpu_runtime, "runtime_ready", return_value=private),
                    patch.object(alignment, "analyze_subtitles", return_value=([(2, 5.0)], {})) as analyze,
                    patch.object(self.gui.gpu_runtime, "run_worker", return_value={
                        "anchors": [[2, 5.0]], "diagnostics": {},
                    }) as worker,
                ):
                    self.app._sync_worker(None, source, reference, output, 65000, 70, device, 0, backend, selection)
                self.assertTrue(output.is_file())
                self.assertEqual(alignment.read_srt(output)[1].start_ms, self.reference[1].start_ms)
                self.assertEqual(reference.read_text(), fixture(self.reference))
                if private:
                    self.assertEqual(worker.call_args.args[2]["source_anchor"], 2)
                    self.assertEqual(worker.call_args.args[2]["reference_anchor"], 2)
                else:
                    self.assertEqual(analyze.call_args.kwargs["source_anchor"], 2)
                    self.assertEqual(analyze.call_args.kwargs["reference_anchor"], 2)

    def test_file_changes_after_selection_abort_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            reference = Path(folder) / "reference.srt"
            output = Path(folder) / "output.srt"
            source.write_text(fixture(self.source), encoding="utf-8")
            reference.write_text(fixture(self.reference), encoding="utf-8")
            selection = manual_sync.SyncSelection(
                self.source, self.reference, 2, 2,
                manual_sync.input_signatures((str(source), str(reference))),
            )
            reference.write_text("changed externally", encoding="utf-8")
            with patch.object(alignment, "analyze_subtitles") as analyze:
                self.app._sync_worker(None, source, reference, output, 65000, 70, selection=selection)
            analyze.assert_not_called()
            self.assertFalse(output.exists())
            messages = []
            while not self.app.messages.empty():
                messages.append(self.app.messages.get_nowait())
            self.assertTrue(any(kind == "error" and "changed" in text for kind, text in messages))

    def test_loading_external_subtitles_opens_the_pair_picker(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            reference = Path(folder) / "reference.srt"
            source.write_text(fixture(self.source), encoding="utf-8")
            reference.write_text(fixture(self.reference), encoding="utf-8")
            self.app.portuguese_path.set(str(source))
            self.app.english_path.set(str(reference))
            with patch.object(self.app, "_show_pair_picker") as show:
                self.app._open_initial_sync()
                deadline = time.monotonic() + 3
                while self.app.busy and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.01)
                self.assertFalse(self.app.busy)
                show.assert_called_once()
                self.assertEqual(show.call_args.args[:2], (self.source, self.reference))

    def test_embedded_track_picker_loads_user_selected_language(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            video = Path(folder) / "video.mkv"
            source.write_text(fixture(self.source), encoding="utf-8")
            video.write_bytes(b"mock video")
            self.app.portuguese_path.set(str(source))
            self.app.video_path.set(str(video))
            with (
                patch.object(alignment, "list_subtitle_tracks", return_value=[
                    alignment.SubtitleTrack(3, "3: eng"),
                    alignment.SubtitleTrack(7, "7: jpn"),
                ]),
                patch.object(alignment, "extract_reference_srt", return_value=list(self.reference)) as extract,
                patch.object(self.app, "_show_pair_picker") as show,
            ):
                self.app._open_initial_sync()
                deadline = time.monotonic() + 3
                while self.app.busy and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.01)
                window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))
                combo = next(child for child in window.winfo_children() if child.winfo_class() == "TCombobox")
                combo.current(1)
                next(child for child in window.winfo_children() if child.winfo_class() == "TButton").invoke()
                deadline = time.monotonic() + 3
                while self.app.busy and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.01)
                extract.assert_called_once_with(video, 7)
                self.assertEqual(show.call_args.args[:2], (self.source, self.reference))

    def test_reference_cannot_be_used_as_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            reference = Path(folder) / "reference.srt"
            source.write_text(fixture(self.source), encoding="utf-8")
            reference.write_text(fixture(self.reference), encoding="utf-8")
            self.app.portuguese_path.set(str(source))
            self.app.english_path.set(str(reference))
            self.app.output_path.set(str(reference))
            self.app.sync_selection = manual_sync.SyncSelection(
                self.source, self.reference, 2, 2,
                manual_sync.input_signatures((str(source), str(reference))),
            )
            with (
                patch.object(self.gui.messagebox, "showerror") as error,
                patch.object(self.gui.threading, "Thread") as worker,
            ):
                self.app._start_sync()
            error.assert_called_once()
            worker.assert_not_called()
            self.assertEqual(reference.read_text(), fixture(self.reference))


if __name__ == "__main__":
    unittest.main()
