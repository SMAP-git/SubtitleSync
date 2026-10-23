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

    def test_mounted_locations_include_networks_disks_and_escaped_paths(self):
        mounts = (
            "1 0 8:1 / / rw - ext4 /dev/sda1 rw\n"
            "2 1 0:2 / /shares/TV\\040Shows rw - cifs //server/TV rw\n"
            "3 1 0:3 / /exports/videos rw - nfs4 server:/videos rw\n"
            "4 1 0:4 / /run/user/1000/gvfs rw - fuse.gvfsd-fuse gvfsd-fuse rw\n"
            "5 1 8:2 / /media/user/Drive rw - ext4 /dev/sdb1 rw\n"
            "6 1 0:5 / /proc rw - proc proc rw\n"
            "7 1 0:6 / /shares/TV\\040Shows rw - cifs //server/TV rw\n"
            "8 1 0:7 / /custom/smb rw - smb3 //server/files rw\n"
            "9 1 0:8 / /custom/nfs rw - nfs server:/files rw\n"
            "malformed\n"
        )
        self.assertEqual(file_picker.mounted_locations(mounts), [
            Path("/shares/TV Shows"), Path("/exports/videos"), Path("/run/user/1000/gvfs"),
            Path("/media/user/Drive"), Path("/custom/smb"), Path("/custom/nfs"),
        ])

    def test_browse_locations_discovers_desktop_mounted_shares(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            share = runtime / "gvfs" / "smb-share:server=nas,share=media"
            share.mkdir(parents=True)
            with patch.dict(os.environ, {"XDG_RUNTIME_DIR": directory}):
                self.assertIn(share, file_picker.browse_locations())

    def test_windows_native_dialog_preserves_unc_and_mapped_drive_paths(self):
        for directory in (r"\\nas\media\TV", "Z:/TV"):
            chosen = directory + "/episode.srt"
            with (
                self.subTest(directory=directory),
                patch.object(file_picker.sys, "platform", "win32"),
                patch.object(file_picker.filedialog, "askopenfilename", return_value=chosen) as native,
                patch.object(file_picker, "remember_directory") as remember,
            ):
                self.assertEqual(file_picker.askopenfilename(initialdir=directory), chosen)
                native.assert_called_once_with(initialdir=directory)
                remember.assert_called_once_with(chosen)

    @unittest.skipUnless(os.environ.get("DISPLAY"), "Tk display required")
    def test_mounted_share_navigation_selection_and_unavailable_share_error(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                share = Path(directory) / "mounted share"
                share.mkdir()
                chosen = share / "episode.srt"
                chosen.write_text("subtitle")
                with patch.object(file_picker, "browse_locations", return_value=[share]):
                    dialog = file_picker.FilePicker(root, False, initialdir=directory)
                    self.assertIn(str(share), dialog.places["values"])
                    dialog.places.set(str(share))
                    dialog.places.event_generate("<<ComboboxSelected>>")
                    root.update()
                    self.assertEqual(dialog.directory, share)
                    with patch.object(file_picker.messagebox, "showerror") as error:
                        dialog.navigate(share / "disconnected")
                        error.assert_called_once()
                    self.assertEqual(dialog.directory, share)
                    dialog.filename.set(chosen.name)
                    dialog.accept()
                    self.assertEqual(dialog.result, str(chosen))
        finally:
            root.destroy()

    @unittest.skipUnless(os.environ.get("DISPLAY"), "Tk display required")
    def test_picker_translates_dynamic_filters_and_network_error_titles(self):
        from ui_preferences import InterfacePreferences

        root = tk.Tk()
        root.withdraw()
        try:
            root.interface_preferences = InterfacePreferences(root, "pt", "dark")
            with tempfile.TemporaryDirectory() as directory:
                dialog = file_picker.FilePicker(
                    root, False, initialdir=directory, filetypes=[("SRT subtitles", "*.srt")],
                )
                try:
                    self.assertEqual(dialog.filter["values"], (dialog.tr("SRT subtitles"),))
                    with patch.object(file_picker.messagebox, "showerror") as error:
                        dialog.navigate(Path(directory) / "disconnected")
                    self.assertEqual(error.call_args.args[0], dialog.tr("Cannot open folder"))
                    self.assertNotEqual(error.call_args.args[0], "Cannot open folder")
                finally:
                    dialog.destroy()
        finally:
            root.destroy()

    @unittest.skipUnless(os.environ.get("DISPLAY"), "Tk display required")
    def test_directory_picker_selects_current_directory_and_shows_mounted_locations(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "network share").mkdir()
                (folder / "not-a-folder.srt").touch()
                with patch.object(file_picker, "browse_locations", return_value=[folder]):
                    dialog = file_picker.FilePicker(root, False, directory=True, initialdir=directory)
                self.assertEqual(len(dialog.tree.get_children()), 1)
                self.assertIn(str(folder), dialog.places["values"])
                dialog.navigate(folder / "network share")
                dialog.accept()
                self.assertEqual(dialog.result, str(folder / "network share"))
                dialog = file_picker.FilePicker(root, False, directory=True, initialdir=directory)
                key = dialog.tree.get_children()[0]
                dialog.tree.selection_set(key)
                dialog.select()
                dialog.accept()
                self.assertEqual(dialog.result, str(folder / "network share"))
        finally:
            root.destroy()

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
