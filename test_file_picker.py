import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import file_picker


class FilePickerTests(unittest.TestCase):
    def setUp(self):
        self.settings = tempfile.TemporaryDirectory()
        self.addCleanup(self.settings.cleanup)
        settings = patch.object(
            file_picker, "settings_path", return_value=Path(self.settings.name) / "settings.json"
        )
        settings.start()
        self.addCleanup(settings.stop)

    def test_next_dialog_opens_in_folder_of_last_chosen_file(self):
        with tempfile.TemporaryDirectory() as directory:
            chosen = Path(directory) / "source.srt"
            chosen.touch()
            calls = []

            def native(**options):
                calls.append(options)
                return str(chosen) if len(calls) == 1 else ""

            with (
                patch.object(file_picker.sys, "platform", "win32"),
                patch.object(file_picker.filedialog, "askopenfilename", side_effect=native),
                patch.object(file_picker.filedialog, "asksaveasfilename", side_effect=native),
            ):
                self.assertEqual(file_picker.askopenfilename(title="Source"), str(chosen))
                self.assertEqual(file_picker.askopenfilename(title="Reference"), "")
                file_picker.asksaveasfilename(title="Explicit", initialdir="/elsewhere")
            self.assertNotIn("initialdir", calls[0])
            self.assertEqual(calls[1]["initialdir"], str(Path(directory).absolute()))
            self.assertEqual(calls[2]["initialdir"], "/elsewhere")
            # A cancelled dialog keeps the remembered folder.
            self.assertEqual(file_picker.last_directory(), str(Path(directory).absolute()))
            settings = (Path(self.settings.name) / "settings.json").read_text()
            self.assertIn("last_directory", settings)
        self.assertIsNone(file_picker.last_directory())

    def test_sorting_uses_actual_size_and_timestamp_and_keeps_folders_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folder").mkdir()
            for name, size, modified in (("b.srt", 2, 300), ("a.srt", 10, 100), ("video.mkv", 5, 200)):
                path = root / name
                path.write_bytes(b"x" * size)
                os.utime(path, (modified, modified))
            names = lambda key, reverse=False: [
                item[0] for item in file_picker.sorted_entries(root, ("*.srt",), key, reverse)
            ]
            self.assertEqual(names("name"), ["folder", "a.srt", "b.srt"])
            self.assertEqual(names("size"), ["folder", "b.srt", "a.srt"])
            self.assertEqual(names("modified"), ["folder", "a.srt", "b.srt"])
            self.assertEqual(names("modified", True), ["folder", "b.srt", "a.srt"])

    def test_windows_keeps_native_dialog(self):
        with (
            patch.object(file_picker.sys, "platform", "win32"),
            patch.object(file_picker.filedialog, "askopenfilename", return_value="chosen.srt") as native,
        ):
            self.assertEqual(file_picker.askopenfilename(title="Open"), "chosen.srt")
            native.assert_called_once_with(title="Open")

    @unittest.skipUnless(os.environ.get("DISPLAY"), "Tk display required")
    def test_picker_resizes_sorts_navigates_and_selects(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "file.srt").write_text("subtitle")
                dialog = file_picker.FilePicker(root, False, initialdir=directory, filetypes=[("Subtitles", "*.srt")])
                root.update()
                self.assertEqual(dialog.resizable(), (1, 1))
                self.assertFalse(dialog.transient())
                self.assertIn("normal", dialog.wm_attributes("-type"))
                self.assertEqual(dialog.grab_current(), dialog)
                before = dialog.tree.winfo_width()
                dialog.geometry("1200x750")
                root.update()
                self.assertGreater(dialog.tree.winfo_width(), before)
                dialog.sort("size")
                self.assertEqual(dialog.column, "size")
                dialog.filename.set("file.srt")
                dialog.accept()
                self.assertEqual(dialog.result, str(path / "file.srt"))
        finally:
            root.destroy()
