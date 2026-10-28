import ast
import json
import os
import re
import string
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from tkinter import ttk
from unittest.mock import Mock, patch

import speech_dialog
import ui_preferences as preferences
import ui_translations as translations
import speech_to_text as speech
from manual_sync import CuePairDialog, SyncSelection, input_signatures
from subtitle_alignment import Cue
from test_subtitle_devices import load_gui


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


GUI_MODULES = (
    "subtitle_sync_app.pyw", "speech_dialog.py", "audio_sync_dialog.py", "manual_sync.py",
    "desktop_updates.py", "file_picker.py",
)
UNTRANSLATED_NAMES = {"SubSync", "Subtitle Sync"}
MESSAGE_BOXES = set(preferences.MESSAGE_FUNCTIONS)
STATUS_VARIABLES = {"status", "summary", "first_cue_text"}


def _call_name(function):
    if isinstance(function, ast.Attribute):
        return function.attr
    return function.id if isinstance(function, ast.Name) else ""


def interface_strings():
    """Static user-visible strings from GUI modules: captions, titles, messages and status lines."""
    found = {}
    root = Path(__file__).resolve().parent
    for module in GUI_MODULES:
        tree = ast.parse((root / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            candidates = []
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if _call_name(node.exc.func) in {"ValueError", "RuntimeError", "OSError"}:
                    candidates += node.exc.args[:1]
            if not isinstance(node, ast.Call):
                for candidate in candidates:
                    if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                        found.setdefault(candidate.value, f"{module}:{node.lineno}")
                continue
            name = _call_name(node.func)
            candidates += [keyword.value for keyword in node.keywords if keyword.arg in {"text", "label"}]
            if name in {"tr", "translate", "title", "progress"}:
                candidates += node.args[:1]
            if name in MESSAGE_BOXES:
                candidates += node.args[:2]
                candidates += [keyword.value for keyword in node.keywords if keyword.arg in {"title", "message"}]
            if name == "TranslatedVar":
                candidates += node.args[2:3]
            if name == "set" and isinstance(node.func, ast.Attribute):
                target = node.func.value
                if isinstance(target, ast.Subscript):
                    target = target.value
                if _call_name(target) in STATUS_VARIABLES | {"previews"}:
                    candidates += node.args[:1]
            if name == "put" and node.args and isinstance(node.args[0], ast.Tuple):
                items = node.args[0].elts
                if len(items) == 2 and isinstance(items[0], ast.Constant) and items[0].value == "status":
                    candidates.append(items[1])
            pending = list(candidates)
            while pending:
                value = pending.pop()
                if isinstance(value, ast.IfExp):
                    pending += [value.body, value.orelse]
                elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                    # Concatenated literal pieces must be wrapped in tr(); bare ones never match a key.
                    pending += [value.left, value.right]
                elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.setdefault(value.value, f"{module}:{node.lineno}")
    return {
        text: location for text, location in found.items()
        if any(character.isalpha() for character in text) and text not in UNTRANSLATED_NAMES
    }


class PreferencesTests(unittest.TestCase):
    def test_every_static_interface_string_is_translated_in_every_language(self):
        def catalogued(text):
            numbered = re.fullmatch(r"\d+\.\s+(.+)", text, re.S)
            return text in translations.TRANSLATIONS or bool(numbered and numbered[1] in translations.TRANSLATIONS)

        strings = interface_strings()
        self.assertGreater(len(strings), 200)
        missing = sorted(f"{location}: {text!r}" for text, location in strings.items() if not catalogued(text))
        self.assertEqual(missing, [], "Untranslated interface strings:\n" + "\n".join(missing))
        formatter = string.Formatter()
        for source, values in translations.TRANSLATIONS.items():
            fields = sorted(name for _, name, _, _ in formatter.parse(source) if name is not None)
            for value in values:
                with self.subTest(source=source[:60], value=value[:60]):
                    self.assertEqual(
                        sorted(name for _, name, _, _ in formatter.parse(value) if name is not None), fields,
                    )

    def test_catalogues_cover_the_reference_apps_nineteen_languages(self):
        self.assertEqual(set(translations.LANGUAGES), {
            "en", "de", "el", "et", "fr", "hi", "hu", "it", "ja", "ko", "nl",
            "pl", "pt-br", "pt", "ro", "ru", "tr", "zn-cn", "zn-tw",
        })
        self.assertEqual(set(translations.CATALOGS), set(translations.LANGUAGES))
        for source, values in translations.TRANSLATIONS.items():
            self.assertEqual(len(values), 18, source)
            self.assertTrue(all(isinstance(value, str) and value.strip() for value in values), source)
        for language in translations.LANGUAGES:
            text = translations.instructions(language)
            self.assertIn(translations.translate("Analyze and synchronize", language), text)
            self.assertIn(translations.translate("Save synchronized SRT", language), text)
            self.assertIn(translations.translate(
                "Example 3: Synchronize existing subtitles using the audio", language,
            ), text)
            self.assertIn(translations.translate("Synchronize to audio", language), text)

    def test_settings_roundtrip_and_missing_file_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "SubtitleSync" / "settings.json"
            self.assertEqual(preferences.load_settings(path), ("en", "system"))
            preferences.save_settings(path, "pt", "dark")
            self.assertEqual(preferences.load_settings(path), ("pt", "dark"))
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_saving_interface_preferences_preserves_processing_and_whisper(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            preferences.update_settings(path, {
                "processing": {"device": "AMD", "backend": "Vulkan", "adapter": "2"},
                "whisper": {"model": "small", "device": "INTEL"},
            })
            preferences.save_settings(path, "pt", "dark")
            saved = preferences.read_settings(path)
            self.assertEqual(saved["processing"]["adapter"], "2")
            self.assertEqual(saved["whisper"]["model"], "small")
            self.assertEqual(saved["language"], "pt")

    def test_bad_settings_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            for value in ("{bad", "[]", '{"language":[]}', '{"theme":"invalid"}', '{"language":"missing"}'):
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    preferences.load_settings(path)
            with self.assertRaises(ValueError):
                preferences.save_settings(path, "xx", "dark")

    def test_platform_configuration_locations(self):
        with patch.object(preferences.sys, "platform", "linux"), patch.dict(os.environ, {"XDG_CONFIG_HOME": "/config"}):
            self.assertEqual(preferences.settings_path(), Path("/config/SubtitleSync/settings.json"))
        with patch.object(preferences.sys, "platform", "win32"), patch.dict(os.environ, {"APPDATA": "/roaming"}):
            self.assertEqual(preferences.settings_path(), Path("/roaming/SubtitleSync/settings.json"))


class ReferenceDiscoveryTests(unittest.TestCase):
    def test_discovers_only_matching_title_language_variants(self):
        gui = load_gui()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Episode.pt-pt.srt"
            names = (
                source.name, "Episode.en.srt", "Episode.English.srt", "Episode.srt",
                "Episode.pt-pt.synced.srt", "Episode.fr.audio-synced.srt",
                "Other.en.srt", "Episode.1080p.en.srt", "Episode.eng.SRT",
            )
            for name in names:
                (root / name).touch()
            output = root / "Episode.pt-pt.synced.srt"
            self.assertEqual({path.name for path in gui.reference_candidates(source, output)}, {
                "Episode.en.srt", "Episode.English.srt", "Episode.srt", "Episode.eng.SRT",
            })
            self.assertEqual(gui.subtitle_name_parts(root / "Episode.Portuguese.srt")[0], "episode")
            self.assertEqual(gui.subtitle_name_parts(root / "Episode.en_US.srt"), ("episode", "en-us"))
            self.assertEqual(gui.subtitle_name_parts(root / "Episode.S03E01.srt"), ("episode.s03e01", ""))

    def test_excludes_source_alias_and_current_output(self):
        gui = load_gui()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "Episode.pt.srt", root / "Episode.en.srt"
            source.touch()
            output.touch()
            (root / "Episode.fr.srt").symlink_to(source)
            self.assertEqual(gui.reference_candidates(source, output), [])


class DesktopIdentityTests(unittest.TestCase):
    def test_linux_desktop_entry_has_app_name_icon_and_window_class(self):
        gui = load_gui()
        exists = Path.exists
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(gui.sys, "platform", "linux"),
            patch.dict(os.environ, {"XDG_DATA_HOME": directory}),
            patch.object(Path, "exists", lambda path: False if str(path) in (
                "/.dockerenv", "/run/.containerenv",
            ) else exists(path)),
        ):
            gui.configure_desktop_identity()
            entry = Path(directory) / "applications/SubSync.desktop"
            content = entry.read_text()
            self.assertIn("Name=SubSync\n", content)
            self.assertIn("StartupWMClass=Subsync\n", content)
            self.assertIn(f"Icon={Path(gui.__file__).parent / 'subtitle_sync_icon.png'}\n", content)
            self.assertIn("Exec=/bin/bash ", content)
            previous = entry.stat().st_mtime_ns
            gui.configure_desktop_identity()
            self.assertEqual(entry.stat().st_mtime_ns, previous)

    def test_windows_app_identity_is_set_before_window_creation(self):
        import ctypes

        gui = load_gui()
        shell = Mock()
        shell.shell32.SetCurrentProcessExplicitAppUserModelID.return_value = 0
        with (
            patch.object(gui.sys, "platform", "win32"),
            patch.object(ctypes, "windll", shell, create=True),
        ):
            gui.configure_desktop_identity()
            shell.shell32.SetCurrentProcessExplicitAppUserModelID.assert_called_once_with("SubSync.Desktop")
            shell.shell32.SetCurrentProcessExplicitAppUserModelID.return_value = -1
            with self.assertRaisesRegex(OSError, "taskbar identity"):
                gui.configure_desktop_identity()

    def test_main_creates_subsync_window_class(self):
        gui = load_gui()
        with (
            patch.object(gui, "configure_desktop_identity") as configure,
            patch.object(gui.tk, "Tk") as root,
            patch.object(gui, "SubtitleSyncApp"),
        ):
            gui.main()
        configure.assert_called_once()
        root.assert_called_once_with(className="SubSync")
        root.return_value.mainloop.assert_called_once()


@unittest.skipUnless(os.environ.get("DISPLAY") or os.name == "nt", "Tk display required")
class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "settings.json"
        self.gui = load_gui()
        self.settings_patch = patch.object(self.gui, "settings_path", return_value=self.path)
        self.settings_patch.start()
        self.system_patch = patch.object(preferences, "system_is_dark", return_value=False)
        self.system_patch.start()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = self.gui.SubtitleSyncApp(self.root)

    def tearDown(self):
        for callback in self.root.tk.call("after", "info"):
            self.root.after_cancel(callback)
        self.root.destroy()
        self.system_patch.stop()
        self.settings_patch.stop()
        self.temporary.cleanup()

    def change(self, language="en", theme="system"):
        self.app.interface_language.set(language)
        self.app.interface_theme.set(theme)
        self.app._change_interface_preferences()
        self.root.update_idletasks()

    def test_banner_has_actions_gear_and_version_on_right(self):
        banner = self.app.gear_button.master
        right = [widget for widget in banner.winfo_children() if widget.pack_info().get("side") == "right"]
        self.assertEqual(
            right, [self.app.version_badge, self.app.gear_button, self.app.cancel_sync_button,
                    self.app.sync_button, self.app.speech_button, self.app.audio_sync_button],
        )

    def test_desktop_window_name_and_icon_are_subsync(self):
        self.assertEqual(self.root.title(), "SubSync")
        self.assertEqual(self.root.iconname(), "SubSync")
        self.assertTrue(str(self.root.tk.call("tk", "appname")).startswith("SubSync"))
        self.assertGreater(self.app._window_icon.width(), 0)

    def test_file_controls_use_requested_two_row_layout(self):
        groups = (
            (self.app.video_file_group, 0, 0),
            (self.app.reference_file_group, 0, 1),
            (self.app.source_file_group, 1, 0),
            (self.app.output_file_group, 1, 1),
        )
        for widget, row, column in groups:
            with self.subTest(widget=widget):
                self.assertEqual(widget.grid_info()["row"], row)
                self.assertEqual(widget.grid_info()["column"], column)
                self.assertIs(widget.master, self.app.video_file_group.master)
        self.assertFalse(hasattr(self.app, "portuguese_path"))
        self.assertFalse(hasattr(self.app, "english_path"))
        with (
            patch.object(self.gui.filedialog, "askopenfilename", return_value="/source.srt"),
            patch.object(self.gui.subtitle_alignment, "read_srt", return_value=[Cue(0, 1000, "Test")]),
            patch.object(self.gui, "reference_candidates", return_value=[]),
        ):
            self.app._choose_source()
        self.assertEqual(self.app.source_path.get(), "/source.srt")
        with (
            patch.object(self.gui.filedialog, "askopenfilename", return_value="/reference.srt"),
            patch.object(self.gui, "reference_candidates", return_value=[]),
        ):
            self.app._choose_reference()
        self.assertEqual(self.app.reference_path.get(), "/reference.srt")

    def test_hearing_filter_is_right_of_initial_sync_point(self):
        anchor = self.app.pair_button.master
        options = anchor.master
        hearing = next(
            widget for widget in options.winfo_children()
            if isinstance(widget, ttk.LabelFrame) and widget["text"] == "Hearing-impaired filter"
        )
        self.assertEqual(hearing.grid_info()["row"], anchor.grid_info()["row"])
        self.assertEqual(hearing.grid_info()["column"], 3)
        self.assertEqual(anchor.grid_info()["column"], 2)
        self.assertEqual(self.app.folder_group.grid_info()["column"], 0)
        self.assertEqual(self.app.audio_folder_group.grid_info()["column"], 1)
        self.root.deiconify()
        self.root.update()
        self.assertLessEqual(anchor.winfo_rootx() + anchor.winfo_width(), hearing.winfo_rootx())
        scale = next(widget for widget in hearing.winfo_children() if isinstance(widget, ttk.Scale))
        scale.set(42)
        self.assertEqual(self.app.hearing_impaired_strictness.get(), 42)
        self.assertEqual(self.app.hearing_impaired_value.get(), "42 / 100")

    def test_filter_mouse_focus_and_arrow_keys_adjust_one_step_and_clamp(self):
        self.root.deiconify()
        self.root.update()
        scale = self.app.hearing_scale
        scale.event_generate("<Button-1>", x=30, y=5)
        self.root.update()
        self.assertEqual(self.root.focus_get(), scale)
        self.app.hearing_impaired_strictness.set(42)
        scale.event_generate("<Right>")
        self.root.update()
        self.assertEqual(self.app.hearing_impaired_strictness.get(), 43)
        scale.event_generate("<Left>")
        self.root.update()
        self.assertEqual(self.app.hearing_impaired_strictness.get(), 42)
        self.app.hearing_impaired_strictness.set(100)
        scale.event_generate("<Up>")
        self.assertEqual(self.app.hearing_impaired_strictness.get(), 100)
        self.app.hearing_impaired_strictness.set(0)
        scale.event_generate("<Down>")
        self.assertEqual(self.app.hearing_impaired_strictness.get(), 0)

    def test_folder_discovery_language_aliases_regions_and_generated_outputs(self):
        folder = Path(self.temporary.name)
        for name in ("Episode.Portuguese.srt", "Episode.English.srt", "Other.pt-pt.srt",
                     "Other.pt-br.srt", "Other.en.srt", "Other.pt-br.synced.srt", "Unknown.srt"):
            (folder / name).touch()
        files = self.gui.folder_subtitles(folder)
        self.assertEqual(set(files["episode"]), {"pt", "en"})
        self.assertEqual(set(files["other"]), {"pt-pt", "pt-br", "en"})
        self.assertEqual(len(files), 2)

    def test_folder_popup_displays_selected_source_reference_and_allows_manual_mode(self):
        folder = Path(self.temporary.name)
        (folder / "Episode.Portuguese.srt").touch()
        (folder / "Episode.en.srt").touch()
        with patch.object(self.gui.filedialog, "askdirectory", return_value=str(folder)):
            self.app._choose_folder()
        popup = next(widget for widget in self.root.winfo_children() if isinstance(widget, tk.Toplevel))
        combos = [widget for widget in popup.winfo_children() if isinstance(widget, ttk.Combobox)]
        self.assertEqual(set(combos[0]["values"]), {"pt", "en"})
        combos[0].set("pt")
        combos[1].set("en")
        accept = next(widget for widget in popup.winfo_children()
                      if isinstance(widget, ttk.Button) and widget["text"] == "Open")
        accept.invoke()
        self.assertEqual(self.app.folder_path.get(), str(folder))
        self.assertEqual(self.app.folder_language_text.get(), "Source: pt\nReference: en")
        self.change("pt", "dark")
        self.assertEqual(self.app.folder_language_text.get(), "Origem: pt\nReferência: en")
        self.app.automatic_initial.set("1")
        self.app.automatic_initial.set("0")
        self.assertEqual(self.app.automatic_initial.get(), "0")
        self.assertEqual(self.app.folder_group["text"], "Modo de pasta SRT")
        self.assertEqual(self.app.folder_browse.grid_info()["row"], self.app.folder_entry.grid_info()["row"])
        self.assertEqual(str(self.app.folder_entry["state"]), "normal")
        self.assertFalse(any(isinstance(widget, ttk.Button) and widget["text"] == "Limpar"
                             for widget in self.app.folder_group.winfo_children()))
        self.assertEqual(set(self.app.folder_source_combo["values"]), {"pt", "en"})
        self.app.folder_source_combo.set("en")
        self.app.folder_reference_combo.set("pt")
        self.app.folder_reference_combo.event_generate("<<ComboboxSelected>>")
        self.assertEqual((self.app.folder_source, self.app.folder_reference), ("en", "pt"))
        self.app.folder_entry.delete(0, "end")
        self.assertEqual(self.app.folder_path.get(), "")
        self.assertEqual(self.app.folder_source_combo.get(), "")
        self.assertEqual(self.app.folder_reference_combo.get(), "")
        self.assertEqual(self.app.folder_source_combo["values"], "")

    def test_typed_srt_folder_path_refreshes_languages_without_popup(self):
        folder = Path(self.temporary.name)
        for name in ("Episode.pt-br.srt", "Episode.en.srt"):
            (folder / name).touch()
        self.app.folder_entry.insert(0, str(folder))
        self.app._refresh_folder()
        self.assertEqual(set(self.app.folder_source_combo["values"]), {"en", "pt-br"})
        self.assertEqual(set(self.app.folder_reference_combo["values"]), {"en", "pt-br"})
        self.assertEqual(self.app.automatic_initial.get(), "0")

    def test_folder_browse_buttons_are_right_of_editable_paths_and_audio_sync_is_below(self):
        self.root.deiconify()
        self.root.update()
        for entry, browse in ((self.app.folder_entry, self.app.folder_browse),
                              (self.app.audio_folder_entry, self.app.audio_folder_browse)):
            self.assertEqual(str(entry["state"]), "normal")
            self.assertEqual(entry.grid_info()["row"], browse.grid_info()["row"])
            self.assertLess(entry.grid_info()["column"], browse.grid_info()["column"])
            self.assertLessEqual(entry.winfo_rootx() + entry.winfo_width(), browse.winfo_rootx())
        self.assertGreater(
            self.app.audio_folder_sync.grid_info()["row"], self.app.audio_folder_entry.grid_info()["row"],
        )
        with patch.object(self.gui.filedialog, "askdirectory", return_value=self.temporary.name):
            self.app.audio_folder_browse.invoke()
        self.assertEqual(self.app.audio_folder_path.get(), self.temporary.name)
        self.app.audio_folder_entry.delete(0, "end")
        self.assertEqual(self.app.audio_folder_path.get(), "")

    def test_audio_folder_requires_path_and_blocks_competing_work(self):
        with patch.object(self.gui.messagebox, "showerror") as error:
            self.app.audio_folder_sync.invoke()
        self.assertIn("Choose a folder first", error.call_args.args[1])
        self.app.busy = True
        with patch.object(self.gui.threading, "Thread") as thread:
            self.app.audio_folder_sync.invoke()
        thread.assert_not_called()
        self.app.busy = False

    def test_audio_folder_report_failure_keeps_successful_srt(self):
        folder = Path(self.temporary.name)
        source = folder / "Episode.pt.srt"
        video = folder / "Episode.mkv"
        output = folder / "Episode.pt.audio-synced.srt"
        content = "1\n00:00:01,000 --> 00:00:02,000\nDialogue\n"
        source.write_text(content)
        video.touch()
        events = self.gui.queue.Queue()

        def sync(*args, **kwargs):
            output.write_text(content)
            return 1

        with (
            patch.object(self.gui.audio_sync, "inspect_tracks", return_value=[
                self.gui.audio_sync.Track(0, "Audio", "aac", "0: eng (aac)"),
            ]),
            patch.object(self.gui.audio_sync, "synchronize", side_effect=sync),
            patch.object(self.gui, "write_report", side_effect=PermissionError("Report denied")),
        ):
            self.app._audio_folder_worker(video, source, output, True, events)
        self.assertEqual(list(events.queue)[-1][0], "success")
        self.assertEqual(list(events.queue)[-1][1][2]["report_error"], "Report denied")
        self.assertEqual(output.read_text(), content)

    def test_audio_folder_runs_all_matching_languages_and_aligned_srts_without_reference_analysis(self):
        folder = Path(self.temporary.name)
        content = "1\n00:00:01,000 --> 00:00:02,000\nDialogue\n"
        names = ("A.pt.srt", "A.en.srt", "A.pt.synced.srt", "B.srt", "Missing.pt.srt",
                 "Duplicate.pt.srt", "A.pt.audio-synced.srt")
        for name in names:
            (folder / name).write_text(content)
        for name in ("A.mkv", "B.mp4", "Duplicate.mkv", "Duplicate.mp4"):
            (folder / name).touch()
        self.app.audio_folder_path.set(str(folder))
        self.app.report_enabled.set("1")
        self.app.processing_device.set("NVIDIA")
        self.app.device_id.set("invalid")

        def sync(video, source, output, track, subtitle, drift, progress, cancel, **kwargs):
            self.assertEqual(track, 2)
            self.assertTrue(drift)
            self.assertTrue(kwargs["fix_sections"])
            self.assertIsNone(subtitle)
            output.write_text(content.replace("01,000", "02,000"))
            progress("Audio alignment completed")
            return 1

        with (
            patch.object(self.gui.threading, "Thread", side_effect=lambda **kwargs: Mock(start=kwargs["target"])),
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "analyze_subtitles") as analyze,
            patch.object(self.gui.audio_sync, "inspect_tracks", return_value=[
                self.gui.audio_sync.Track(2, "Audio", "aac", "2: eng (aac)"),
            ]),
            patch.object(self.gui.audio_sync, "synchronize", side_effect=sync) as synchronize,
        ):
            self.app.audio_folder_sync.invoke()
        analyze.assert_not_called()
        self.assertEqual(synchronize.call_count, 3)
        rows = list(self.app.messages.queue)[-1][1]
        self.assertEqual(len(rows), 6)
        self.assertEqual([row[1] for row in rows], ["Success", "Skipped", "Success", "Success", "Failed", "Failed"])
        self.assertTrue((folder / "A.en.audio-synced.srt").is_file())
        self.assertTrue((folder / "A.pt.synced.audio-synced.srt").is_file())
        self.assertTrue((folder / "B.audio-synced.srt").is_file())
        self.assertTrue((folder / "A.en.sync.txt").is_file())
        self.assertEqual(self.app.automatic_initial.get(), "0")
        with patch.object(self.app, "_show_folder_report"):
            self.app._poll_worker()
        self.assertFalse(self.app.busy)
        self.assertEqual(str(self.app.audio_folder_sync["state"]), "normal")

    def test_audio_folder_continues_after_audio_failure_and_cancels_remaining_pairs(self):
        folder = Path(self.temporary.name)
        content = "1\n00:00:01,000 --> 00:00:02,000\nDialogue\n"
        for stem in ("A", "B", "C"):
            (folder / f"{stem}.pt.srt").write_text(content)
            (folder / f"{stem}.mkv").touch()
        self.app.audio_folder_path.set(str(folder))
        calls = []

        def sync(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("Speech matching was unreliable.")
            self.app.sync_cancel.set()
            raise self.gui.audio_sync.CancelledError("Cancelled")

        with (
            patch.object(self.gui.threading, "Thread", side_effect=lambda **kwargs: Mock(start=kwargs["target"])),
            patch.object(self.root, "after"),
            patch.object(self.gui.audio_sync, "inspect_tracks", return_value=[
                self.gui.audio_sync.Track(0, "Audio", "aac", "0: eng (aac)"),
            ]),
            patch.object(self.gui.audio_sync, "synchronize", side_effect=sync),
        ):
            self.app.audio_folder_sync.invoke()
        rows = list(self.app.messages.queue)[-1][1]
        self.assertEqual([row[1] for row in rows], ["Failed", "Cancelled", "Cancelled"])
        self.assertEqual(len(calls), 2)
        self.app.batch_running = False
        self.app._finish()

    def test_folder_batch_continues_after_failed_initial_match_and_reports_every_file(self):
        folder = Path(self.temporary.name)
        original = "1\n00:00:01,000 --> 00:00:02,000\nDialogue\n"
        for stem in ("A", "B", "C", "D"):
            (folder / f"{stem}.pt.srt").write_text(original)
            if stem != "D":
                (folder / f"{stem}.en.srt").write_text(original)
        (folder / "C.pt.synced.srt").write_text("existing result")
        self.app.folder_path.set(str(folder))
        self.app.folder_source, self.app.folder_reference = "pt", "en"
        self.app.automatic_initial.set("1")
        calls = []

        def analyze(source, reference, initial, **kwargs):
            self.assertTrue(kwargs["automatic_initial"])
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("Cannot confidently find an automatic initial sync point. Select matching cues manually.")
            return [(1, 1.)], {"matched_cues": 1, "match_percent": 100, "source_anchor": 1, "reference_anchor": 1}

        def thread(**kwargs):
            return Mock(start=kwargs["target"])

        with (
            patch.object(self.gui.threading, "Thread", side_effect=thread),
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
            patch.object(self.gui.subtitle_alignment, "analyze_subtitles", side_effect=analyze),
        ):
            self.app._start_sync()
        final = list(self.app.messages.queue)[-1]
        self.assertEqual(final[0], "batch_complete")
        self.assertEqual([row[1] for row in final[1]], ["Failed", "Success", "Skipped", "Failed"])
        self.assertFalse((folder / "A.pt.synced.srt").exists())
        self.assertIn("00:00:02,000", (folder / "B.pt.synced.srt").read_text())
        self.assertEqual((folder / "C.pt.synced.srt").read_text(), "existing result")
        with patch.object(self.app, "_show_folder_report") as report:
            self.app._poll_worker()
        report.assert_called_once_with(final[1])
        self.assertFalse(self.app.busy)
        self.assertFalse(self.app.batch_running)
        self.app._show_folder_report(final[1])
        popup = next(widget for widget in self.root.winfo_children() if isinstance(widget, tk.Toplevel))
        table = next(widget for widget in popup.winfo_children() if isinstance(widget, ttk.Treeview))
        self.assertEqual(len(table.get_children()), 4)
        popup.destroy()

    def test_folder_manual_mode_prompts_for_each_eligible_pair_and_passes_selected_anchors(self):
        folder = Path(self.temporary.name)
        content = (
            "1\n00:00:01,000 --> 00:00:02,000\nFirst dialogue\n\n"
            "2\n00:00:04,000 --> 00:00:05,000\nSecond dialogue\n"
        )
        for stem in ("A", "B", "C"):
            for language in ("pt", "en"):
                (folder / f"{stem}.{language}.srt").write_text(content)
        (folder / "C.pt.synced.srt").write_text("existing")
        self.app.folder_path.set(str(folder))
        self.app.folder_source, self.app.folder_reference = "pt", "en"
        self.app.automatic_initial.set("0")
        prompted = []
        put = self.app.messages.put

        def message(item):
            if item[0] != "folder_select_pair":
                put(item)
                return
            self.app._show_folder_pair_picker(*item[1])
            dialog = self.app.folder_pair_dialog
            prompted.append(dialog.window.title())
            self.assertTrue(self.app.busy)
            dialog.selected = [2, 1]
            dialog._confirm()

        def analyze(source, reference, initial, **kwargs):
            self.assertNotIn("automatic_initial", kwargs)
            self.assertEqual((kwargs["source_anchor"], kwargs["reference_anchor"]), (2, 1))
            return [(2, -3.)], {"matched_cues": 1, "match_percent": 100}

        with (
            patch.object(self.gui.threading, "Thread", side_effect=lambda **kwargs: Mock(start=kwargs["target"])),
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
            patch.object(self.app.messages, "put", side_effect=message),
            patch.object(self.gui.subtitle_alignment, "analyze_subtitles", side_effect=analyze) as analyze_mock,
        ):
            self.app._start_sync()
        self.assertEqual(analyze_mock.call_count, 2)
        self.assertEqual(prompted, [
            "A.pt.srt -> A.en.srt", "B.pt.srt -> B.en.srt",
        ])
        rows = list(self.app.messages.queue)[-1][1]
        self.assertEqual([row[1] for row in rows], ["Success", "Success", "Skipped"])
        self.assertEqual(self.app.automatic_initial.get(), "0")
        self.assertIsNone(self.app.folder_pair_dialog)
        self.app.batch_running = False
        self.app._finish()

    def test_folder_live_automatic_toggle_applies_to_next_file_in_both_directions(self):
        folder = Path(self.temporary.name)
        for stem in ("A", "B", "C"):
            for language in ("pt", "en"):
                (folder / f"{stem}.{language}.srt").write_text(
                    "1\n00:00:01,000 --> 00:00:02,000\nDialogue\n"
                )
        self.app.folder_path.set(str(folder))
        self.app.folder_source, self.app.folder_reference = "pt", "en"
        self.app.automatic_initial.set("1")
        self.app.final_offset_ms.set("-500")
        modes = []

        def choose(source, reference):
            left = tuple(self.gui.subtitle_alignment.read_srt(source))
            right = tuple(self.gui.subtitle_alignment.read_srt(reference))
            return SyncSelection(left, right, 1, 1, input_signatures((str(source), str(reference))))

        def worker(*args, **kwargs):
            self.assertEqual(str(self.app.automatic_initial_checkbox["state"]), "normal")
            modes.append(kwargs["automatic_initial"])
            self.assertEqual(kwargs["selection"] is None, kwargs["automatic_initial"])
            self.assertEqual(kwargs["final_offset_ms"], -500)
            self.app.automatic_initial_checkbox.invoke()
            kwargs["events"].put(("success", (args[3], 1, {})))

        with (
            patch.object(self.gui.threading, "Thread", side_effect=lambda **kwargs: Mock(start=kwargs["target"])),
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
            patch.object(self.app, "_request_folder_selection", side_effect=choose) as picker,
            patch.object(self.app, "_sync_worker", side_effect=worker),
        ):
            self.app._start_sync()
        self.assertEqual(modes, [True, False, True])
        self.assertEqual(picker.call_count, 1)
        self.app.batch_running = False
        self.app._finish()

    def test_final_offset_delays_or_advances_saved_output_and_report(self):
        source = Path(self.temporary.name) / "source.srt"
        source.write_bytes(b"\xef\xbb\xbf1\r\n00:00:01,000 --> 00:00:02,000\r\nDialogue\r\n")
        output = source.with_suffix(".synced.srt")
        for offset, expected in ((500, "00:00:02,500"), (-500, "00:00:01,500"), (0, "00:00:02,000")):
            with (
                self.subTest(offset=offset),
                patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
                    [(1, 1.)], {"matched_cues": 1},
                )),
                patch.object(self.gui, "write_report") as report,
            ):
                self.app._sync_worker(
                    None, source, source, output, 0, 70, final_offset_ms=offset, generate_report=True,
                )
                result = self.app.messages.get_nowait()
                while result[0] != "success":
                    self.assertNotEqual(result[0], "error", result)
                    result = self.app.messages.get_nowait()
                self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))
                self.assertIn(expected, output.read_text(encoding="utf-8-sig"))
                self.assertEqual(report.call_args.kwargs["settings"]["Final offset (ms)"], offset)
                self.assertIn(expected, report.call_args.kwargs["synchronized"])

    def test_final_offset_rejects_invalid_input_and_negative_result_without_overwriting(self):
        for value in ("", "1.5", "abc", "--500"):
            self.app.final_offset_ms.set(value)
            with self.assertRaises(ValueError):
                self.app._read_final_offset()
        for value, expected in (("+500", 500), (" -500 ", -500), ("0", 0)):
            self.app.final_offset_ms.set(value)
            self.assertEqual(self.app._read_final_offset(), expected)
        source = Path(self.temporary.name) / "source.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        output = source.with_suffix(".synced.srt")
        output.write_text("existing output")
        with patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
            [(1, 0.)], {"matched_cues": 1},
        )):
            self.app._sync_worker(None, source, source, output, 0, 70, final_offset_ms=-1500)
        self.assertEqual(list(self.app.messages.queue)[-1][0], "error")
        self.assertEqual(output.read_text(), "existing output")

    def test_worker_applies_timing_scale_from_local_or_private_gpu_analysis_and_reports_it(self):
        from sync_report import report_path

        source = Path(self.temporary.name) / "source.srt"
        source.write_text("1\n00:00:10,000 --> 00:00:12,000\nDialogue\n")
        output = source.with_suffix(".synced.srt")
        diagnostics = {
            "matched_cues": 1, "timing_scale": 1.25, "timing_scale_applied": 1,
            "timing_scale_matches": 20, "timing_scale_validation_error_ms": 50,
        }
        for private_gpu in (False, True):
            events = self.gui.queue.Queue()
            with (
                self.subTest(private_gpu=private_gpu),
                patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
                    [(1, 2.5)], diagnostics,
                )),
                patch.object(self.gui.gpu_runtime, "runtime_ready", return_value=private_gpu),
                patch.object(self.gui.gpu_runtime, "run_worker", return_value={
                    "anchors": [[1, 2.5]], "diagnostics": diagnostics,
                }) as worker,
            ):
                self.app._sync_worker(
                    None, source, source, output, 0, 70,
                    processing_device="NVIDIA" if private_gpu else "CPU",
                    generate_report=True, final_offset_ms=500,
                    events=events if private_gpu else None,
                )
                self.assertEqual(worker.called, private_gpu)
            messages = list((events if private_gpu else self.app.messages).queue)
            self.assertEqual(messages[-1][0], "success", messages[-1])
            self.assertIn("00:00:13,000 --> 00:00:15,500", output.read_text())
            report = report_path(output).read_text()
            self.assertIn("timing_scale: 1.25", report)
            self.assertIn("timing_scale_validation_error_ms: 50", report)
            self.assertIn("Final offset (ms): 500", report)
            output.unlink()

    def test_folder_manual_popup_close_cancels_remaining_files_without_analysis(self):
        folder = Path(self.temporary.name)
        for name in ("A.pt.srt", "A.en.srt", "B.pt.srt", "B.en.srt"):
            (folder / name).write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        self.app.folder_path.set(str(folder))
        self.app.folder_source, self.app.folder_reference = "pt", "en"
        put = self.app.messages.put

        def message(item):
            if item[0] == "folder_select_pair":
                self.app._show_folder_pair_picker(*item[1])
                self.app.folder_pair_dialog.window.destroy()
            else:
                put(item)

        with (
            patch.object(self.gui.threading, "Thread", side_effect=lambda **kwargs: Mock(start=kwargs["target"])),
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
            patch.object(self.app.messages, "put", side_effect=message),
            patch.object(self.app, "_sync_worker") as worker,
        ):
            self.app._start_sync()
        worker.assert_not_called()
        rows = list(self.app.messages.queue)[-1][1]
        self.assertEqual([row[1] for row in rows], ["Cancelled", "Cancelled"])
        self.assertFalse((folder / "A.pt.synced.srt").exists())
        self.app.batch_running = False
        self.app._finish()

    def test_folder_cancellation_closes_picker_before_batch_completion_resets_cancel_flag(self):
        source = Path(self.temporary.name) / "A.pt.srt"
        reference = source.with_name("A.en.srt")
        cues = (Cue(1000, 2000, "Dialogue"),)
        response = self.gui.queue.Queue()
        self.app.busy = self.app.batch_running = True
        self.app._show_folder_pair_picker(source, reference, cues, cues, (), response)
        window = self.app.folder_pair_dialog.window
        self.app._cancel_sync()
        self.app.messages.put(("batch_complete", []))
        with patch.object(self.app, "_show_folder_report"):
            self.app._poll_worker()
        self.assertFalse(window.winfo_exists())
        self.assertIsNone(response.get_nowait())
        self.assertFalse(self.app.busy)
        self.assertIsNone(self.app.folder_pair_dialog)

    def test_folder_cancellation_marks_remaining_files_and_does_not_run_next_worker(self):
        folder = Path(self.temporary.name)
        for name in ("A.pt.srt", "A.en.srt", "B.pt.srt", "B.en.srt"):
            (folder / name).write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        self.app.folder_path.set(str(folder))
        self.app.folder_source, self.app.folder_reference = "pt", "en"
        self.app.automatic_initial.set("1")

        def cancel(*args, **kwargs):
            self.app.sync_cancel.set()
            raise self.gui.gpu_runtime.CancelledError("Cancelled")

        with (
            patch.object(self.gui.threading, "Thread", side_effect=lambda **kwargs: Mock(start=kwargs["target"])),
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
            patch.object(self.gui.subtitle_alignment, "analyze_subtitles", side_effect=cancel) as analyze,
        ):
            self.app._start_sync()
        self.assertEqual(analyze.call_count, 1)
        rows = list(self.app.messages.queue)[-1][1]
        self.assertEqual([row[1] for row in rows], ["Cancelled", "Cancelled"])
        self.app.batch_running = False
        self.app._finish()

    def test_folder_batch_requires_unique_reference_and_video_and_does_not_guess(self):
        folder = Path(self.temporary.name)
        for name in ("A.pt.srt", "A.en.srt", "A.English.srt", "B.pt.srt", "B.en.srt",
                     "C.pt.srt", "C.en.srt", "C.mkv", "C.mp4"):
            (folder / name).write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        self.app.folder_path.set(str(folder))
        self.app.folder_source, self.app.folder_reference = "pt", "en"
        self.app.automatic_initial.set("1")
        with (
            patch.object(self.gui.threading, "Thread", side_effect=lambda **kwargs: Mock(start=kwargs["target"])),
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
            patch.object(self.app, "_sync_worker") as worker,
        ):
            self.app.audio_after_sync.set("1")
            self.app._start_sync()
        worker.assert_not_called()
        rows = list(self.app.messages.queue)[-1][1]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row[1] == "Failed" for row in rows))
        self.assertTrue(all("video" in row[3] for row in rows))
        self.app.batch_running = False
        self.app._finish()
        while not self.app.messages.empty():
            self.app.messages.get_nowait()
        self.app.audio_after_sync.set("0")
        with (
            patch.object(self.gui.threading, "Thread", side_effect=lambda **kwargs: Mock(start=kwargs["target"])),
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
            patch.object(self.gui.subtitle_alignment, "analyze_subtitles", side_effect=ValueError("Uncertain match")),
        ):
            self.app._start_sync()
        rows = list(self.app.messages.queue)[-1][1]
        self.assertEqual(rows[0][3], "Ambiguous subtitle pair.")
        self.app.batch_running = False
        self.app._finish()

    def test_folder_worker_does_not_overwrite_output_created_during_analysis(self):
        folder = Path(self.temporary.name)
        source = folder / "episode.pt.srt"
        reference = folder / "episode.en.srt"
        output = folder / "episode.pt.synced.srt"
        for path in (source, reference):
            path.write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        events = self.gui.queue.Queue()

        def analyze(*args, **kwargs):
            output.write_text("created by someone else")
            return [(1, 0.)], {"matched_cues": 1}

        with patch.object(self.gui.subtitle_alignment, "analyze_subtitles", side_effect=analyze):
            self.app._sync_worker(None, source, reference, output, 0, 70, events=events)
        self.assertEqual(list(events.queue)[-1][0], "error")
        self.assertEqual(output.read_text(), "created by someone else")

    def test_folder_worker_saves_on_share_without_hardlink_support(self):
        import errno

        folder = Path(self.temporary.name)
        source = folder / "episode.pt.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        output = folder / "episode.pt.synced.srt"
        events = self.gui.queue.Queue()
        with (
            patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
                [(1, 1.)], {"matched_cues": 1},
            )),
            patch.object(self.gui.sync_subtitles.os, "link", side_effect=OSError(errno.EPERM, "Operation not permitted")),
        ):
            self.app._sync_worker(None, source, source, output, 0, 70, events=events)
        self.assertEqual(list(events.queue)[-1][0], "success")
        self.assertIn("00:00:02,000", output.read_text())
        self.assertFalse(list(folder.glob(".subtitle-sync-*")))

    def test_report_box_is_right_of_processing_and_checkbox_is_optional_persistent_and_translated(self):
        processing = self.app.device_selector.master
        report = self.app.report_group
        self.assertIs(report.master, processing.master)
        self.assertEqual(report.grid_info()["row"], processing.grid_info()["row"])
        self.assertEqual(report.grid_info()["column"], 1)
        self.assertEqual(processing.grid_info()["column"], 0)
        self.assertEqual(self.app.report_enabled.get(), "0")
        self.app.report_checkbox.invoke()
        self.assertEqual(self.app.report_enabled.get(), "1")
        self.change("pt", "dark")
        self.assertEqual(report["text"], "Relatório e extras")
        self.assertEqual(self.app.report_checkbox["text"], "Gerar relatório de sincronização")
        self.root.deiconify()
        self.root.update()
        self.assertLessEqual(processing.winfo_rootx() + processing.winfo_width(), report.winfo_rootx())
        second = tk.Tk()
        second.withdraw()
        try:
            app = self.gui.SubtitleSyncApp(second)
            self.assertEqual(app.report_enabled.get(), "1")
        finally:
            second.destroy()
        with patch.object(self.gui, "AudioSyncDialog") as dialog, patch.object(self.root, "wait_window"):
            self.app._open_audio_sync()
        self.assertTrue(dialog.call_args.kwargs["generate_report"])

    def test_offset_controls_do_not_increase_report_box_height(self):
        self.root.update_idletasks()
        height = self.app.report_group.winfo_reqheight()
        offset_box = self.app.final_offset_entry.master
        offset_box.grid_remove()
        self.root.update_idletasks()
        self.assertEqual(self.app.report_group.winfo_reqheight(), height)
        offset_box.grid()
        self.assertLessEqual(int(self.app.device_selector["width"]), 16)
        self.assertLessEqual(int(self.app.backend_selector["width"]), 16)
        self.assertEqual(self.app.final_offset_ms.get(), "0")

    def test_reference_sync_report_is_opt_in_and_report_failure_does_not_undo_srt(self):
        from sync_report import report_path

        source = Path(self.temporary.name) / "episode.pt.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        output = source.with_suffix(".synced.srt")
        diagnostics = {"matched_cues": 1, "match_percent": 100}
        with patch.object(
            self.gui.subtitle_alignment, "analyze_subtitles",
            return_value=([(1, 1.5)], diagnostics),
        ):
            self.app._sync_worker(None, source, source, output, 1000, 70)
            self.assertFalse(report_path(output).exists())
            self.app._sync_worker(None, source, source, output, 1000, 70, generate_report=True)
            self.assertIn("start shift +1500 ms", report_path(output).read_text())
            with patch.object(self.gui, "write_report", side_effect=PermissionError("Report is read-only")):
                self.app._sync_worker(None, source, source, output, 1000, 70, generate_report=True)
        messages = list(self.app.messages.queue)
        self.assertEqual(messages[-1][0], "success")
        self.assertEqual(messages[-1][1][2]["report_error"], "Report is read-only")
        self.assertIn("00:00:02,500", output.read_text())
        while not self.app.messages.empty():
            self.app.messages.get_nowait()
        self.app.messages.put(messages[-1])
        with (
            patch.object(self.gui.messagebox, "showwarning") as warning,
            patch.object(self.gui.messagebox, "showinfo") as info,
        ):
            self.app._poll_worker()
        self.assertIn("Report is read-only", warning.call_args.args[1])
        self.assertIn("Synchronized 1 subtitle cues", info.call_args.args[1])

    def test_cancelled_sync_does_not_generate_report(self):
        from sync_report import report_path

        source = Path(self.temporary.name) / "episode.pt.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        output = source.with_suffix(".synced.srt")

        def analyze(*args, **kwargs):
            self.app.sync_cancel.set()
            return [(1, 1.5)], {"matched_cues": 1}

        with patch.object(self.gui.subtitle_alignment, "analyze_subtitles", side_effect=analyze):
            self.app._sync_worker(None, source, source, output, 1000, 70, generate_report=True)
        self.assertFalse(output.exists())
        self.assertFalse(report_path(output).exists())

    def test_audio_after_analysis_checkbox_requires_video_and_is_in_report_box(self):
        self.assertIs(self.app.audio_after_checkbox.master, self.app.report_group)
        self.assertEqual(self.app.audio_after_sync.get(), "0")
        self.app.audio_after_checkbox.invoke()
        for value in ("", "/missing/video.mkv"):
            self.app.video_path.set(value)
            with (
                patch.object(self.gui.messagebox, "showwarning") as warning,
                patch.object(self.gui.threading, "Thread") as thread,
            ):
                self.app._start_sync()
            thread.assert_not_called()
            self.assertIn("valid video file", warning.call_args.args[1])
        self.change("pt", "dark")
        self.assertEqual(self.app.audio_after_checkbox["text"], "Sincronizar com o áudio após a análise")
        second = tk.Tk()
        second.withdraw()
        try:
            app = self.gui.SubtitleSyncApp(second)
            self.assertEqual(app.audio_after_sync.get(), "1")
        finally:
            second.destroy()

    def test_automatic_initial_checkbox_is_in_anchor_box_persistent_and_translated(self):
        checkbox = self.app.automatic_initial_checkbox
        self.assertIs(checkbox.master, self.app.pair_button.master)
        self.assertEqual(self.app.automatic_initial.get(), "0")
        checkbox.invoke()
        self.assertEqual(self.app.automatic_initial.get(), "1")
        self.assertEqual(str(self.app.pair_button["state"]), "disabled")
        self.change("pt", "dark")
        self.assertEqual(checkbox["text"], "Selecionar automaticamente o ponto de sincronização inicial")
        second = tk.Tk()
        second.withdraw()
        try:
            other = self.gui.SubtitleSyncApp(second)
            self.assertEqual(other.automatic_initial.get(), "1")
        finally:
            second.destroy()
        checkbox.invoke()
        self.assertEqual(str(self.app.pair_button["state"]), "normal")

    def test_automatic_initial_starts_without_manual_selection_or_video_with_external_reference(self):
        source = Path(self.temporary.name) / "source.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        self.app.source_path.set(str(source))
        self.app.reference_path.set(str(source))
        self.app.output_path.set(str(source.with_suffix(".synced.srt")))
        self.app.automatic_initial.set("1")
        with (
            patch.object(self.gui.threading, "Thread") as thread,
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
            patch.object(self.gui.messagebox, "showwarning") as warning,
        ):
            self.app._start_sync()
        warning.assert_not_called()
        self.assertTrue(thread.call_args.kwargs["kwargs"]["automatic_initial"])
        self.assertIsNone(thread.call_args.kwargs["kwargs"]["selection"])
        self.app._finish()

    def test_automatic_initial_worker_and_result_display_preserve_original_cue_numbers(self):
        source = Path(self.temporary.name) / "source.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nFirst dialogue\n\n"
                          "2\n00:00:04,000 --> 00:00:05,000\nSecond dialogue\n")
        output = source.with_suffix(".synced.srt")
        with patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
            [(2, 1.)], {"matched_cues": 1, "match_percent": 100, "source_anchor": 2,
                         "reference_anchor": 2, "anchor_time_ms": 4000},
        )) as analyze:
            self.app._sync_worker(None, source, source, output, 0, 70, automatic_initial=True)
        self.assertTrue(analyze.call_args.kwargs["automatic_initial"])
        with patch.object(self.gui.messagebox, "showinfo"):
            self.app._poll_worker()
        self.assertEqual(self.app.sync_selection.source_number, 2)
        self.assertEqual(self.app.sync_selection.reference_number, 2)
        self.assertIn("Source cue 2", self.app.first_cue_text.get())
        self.assertEqual(self.app.initial_time.get(), self.gui.format_time(4000))

    def test_automatic_initial_uncertainty_keeps_existing_output_and_manual_mode_available(self):
        source = Path(self.temporary.name) / "source.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        output = source.with_suffix(".synced.srt")
        output.write_text("previous output")
        message = "Cannot confidently find an automatic initial sync point. Select matching cues manually."
        self.app.automatic_initial.set("1")
        self.change("pt")
        with patch.object(self.gui.subtitle_alignment, "analyze_subtitles", side_effect=ValueError(message)):
            self.app._sync_worker(None, source, source, output, 0, 70, automatic_initial=True)
        with patch.object(self.gui.messagebox, "showerror") as error:
            self.app._poll_worker()
        self.assertEqual(output.read_text(), "previous output")
        self.assertEqual(error.call_args.args[1], self.app.interface.tr(message))
        self.assertEqual(str(self.app.automatic_initial_checkbox["state"]), "normal")
        self.app.automatic_initial_checkbox.invoke()
        self.assertEqual(str(self.app.pair_button["state"]), "normal")

    def test_automatic_initial_is_forwarded_to_vulkan_and_private_gpu_workers(self):
        source = Path(self.temporary.name) / "source.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        output = source.with_suffix(".synced.srt")
        diagnostics = {"matched_cues": 1, "source_anchor": 1, "reference_anchor": 1}
        for backend in ("ONNX", "Vulkan"):
            with (
                patch.object(self.gui.gpu_runtime, "runtime_ready", return_value=True),
                patch.object(self.gui.gpu_runtime, "run_worker", return_value={
                    "anchors": [(1, 0.)], "diagnostics": diagnostics,
                }) as worker,
                patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
                    [(1, 0.)], diagnostics,
                )) as analyze,
            ):
                self.app._sync_worker(
                    None, source, source, output, 0, 70, processing_device="NVIDIA",
                    backend=backend, automatic_initial=True,
                )
            self.assertEqual(list(self.app.messages.queue)[-1][0], "success")
            if backend == "Vulkan":
                self.assertTrue(analyze.call_args.kwargs["automatic_initial"])
                worker.assert_not_called()
            else:
                self.assertTrue(worker.call_args.args[2]["automatic_initial"])
                analyze.assert_not_called()

    def test_automatic_embedded_reference_preload_does_not_open_manual_picker_or_repeat_loading(self):
        from manual_sync import input_signatures

        source = Path(self.temporary.name) / "source.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nDialogue\n")
        video = source.with_suffix(".mkv")
        video.write_bytes(b"video")
        self.app.source_path.set(str(source))
        self.app.video_path.set(str(video))
        self.app.output_path.set(str(source.with_suffix(".synced.srt")))
        self.app.automatic_initial.set("1")
        with patch.object(self.app, "_open_initial_sync") as load, patch.object(
            self.gui.subtitle_alignment, "validate_processing_device",
        ):
            self.app._start_sync()
        load.assert_called_once()
        cues = tuple(self.gui.subtitle_alignment.read_srt(source))
        self.app.messages.put(("select_pair", (cues, cues, input_signatures((str(source), str(video))))))
        with (
            patch.object(self.app, "_show_pair_picker") as picker,
            patch.object(self.app, "_open_initial_sync") as load,
            patch.object(self.gui.threading, "Thread") as thread,
            patch.object(self.root, "after"),
            patch.object(self.gui.subtitle_alignment, "validate_processing_device"),
        ):
            self.app._poll_worker()
        picker.assert_not_called()
        load.assert_not_called()
        self.assertTrue(thread.call_args.kwargs["kwargs"]["automatic_initial"])
        self.app._finish()

    def test_automatic_audio_pass_uses_aligned_source_and_applies_final_offset_after_both_stages(self):
        from sync_report import report_path

        source = Path(self.temporary.name) / "episode.pt.srt"
        original = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"
        source.write_text(original)
        output = source.with_suffix(".synced.srt")
        video = source.with_suffix(".mkv")
        video.write_bytes(b"video")
        output.write_text("previous output")
        track = self.gui.audio_sync.Track(2, "Audio", "aac", "2: eng (aac)")

        def audio_pass(video_path, staged, result, *args, **kwargs):
            self.assertEqual(video_path, video)
            self.assertIn("00:00:02,500", staged.read_text())
            self.assertEqual(output.read_text(), "previous output")
            self.assertEqual(args[:3], (2, None, True))
            result.write_text(original.replace("01,000", "03,000").replace("02,000", "04,000"))
            return 1

        with (
            patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
                [(1, 1.5)], {"matched_cues": 1, "match_percent": 100},
            )),
            patch.object(self.gui.audio_sync, "inspect_tracks", return_value=[track]),
            patch.object(self.gui.audio_sync, "synchronize", side_effect=audio_pass),
        ):
            self.app._sync_worker(
                video, source, source, output, 1000, 70, sync_to_audio=True, generate_report=True,
                final_offset_ms=500,
            )
        self.assertIn("00:00:03,500", output.read_text())
        self.assertEqual(source.read_text(), original)
        self.assertIn("start shift +2500 ms", report_path(output).read_text())
        self.assertIn("first audio track", report_path(output).read_text())
        self.assertFalse(list(output.parent.glob(".subtitle-sync-*")))

    def test_failed_or_cancelled_audio_pass_preserves_existing_output(self):
        source = Path(self.temporary.name) / "episode.pt.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        output = source.with_suffix(".synced.srt")
        video = source.with_suffix(".mkv")
        video.write_bytes(b"video")
        for error in (ValueError("No speech detected"), self.gui.audio_sync.CancelledError("Cancelled")):
            output.write_text("previous output")
            with (
                patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
                    [(1, 1.5)], {"matched_cues": 1},
                )),
                patch.object(self.gui.audio_sync, "inspect_tracks", return_value=[
                    self.gui.audio_sync.Track(0, "Audio", "aac", "0: eng (aac)"),
                ]),
                patch.object(self.gui.audio_sync, "synchronize", side_effect=error),
            ):
                self.app._sync_worker(video, source, source, output, 1000, 70, sync_to_audio=True)
            self.assertEqual(output.read_text(), "previous output")
            self.assertFalse(list(output.parent.glob(".subtitle-sync-*")))
            last = list(self.app.messages.queue)[-1]
            self.assertEqual(
                last[0], "cancelled" if isinstance(error, self.gui.audio_sync.CancelledError) else "error",
            )
            self.assertIn(str(error), last[1])

    def test_source_selection_offers_discovered_references(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Episode.pt.srt"
            reference = Path(directory) / "Episode.en.srt"
            for path in (source, reference):
                path.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n")
            with (
                patch.object(self.gui.filedialog, "askopenfilename", return_value=str(source)),
                patch.object(self.app, "_offer_reference_candidates") as offer,
            ):
                self.app._choose_source()
            offer.assert_called_once_with([reference])
            self.assertEqual(self.app.source_path.get(), str(source))

    def test_single_reference_offer_accept_decline_and_invalid_file(self):
        reference = Path(self.temporary.name) / "Episode.en.srt"
        reference.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n")
        self.app.reference_path.set("previous.srt")
        def popup():
            return next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))

        def use(window):
            next(child for child in _descendants(window) if isinstance(child, ttk.Button)
                 and child["text"] == "Use this reference").invoke()

        self.app._offer_reference_candidates([reference])
        popup().destroy()
        self.assertEqual(self.app.reference_path.get(), "previous.srt")
        self.app._offer_reference_candidates([reference])
        use(popup())
        self.assertEqual(self.app.reference_path.get(), str(reference))
        reference.unlink()
        with (
            patch.object(self.gui.messagebox, "showerror") as error,
        ):
            self.app._offer_reference_candidates([reference])
            use(popup())
            popup().destroy()
        error.assert_called_once()
        self.assertEqual(self.app.reference_path.get(), str(reference))

    def test_multiple_references_allow_choice_or_cancellation(self):
        candidates = [Path(self.temporary.name) / name for name in ("Episode.en.srt", "Episode.fr.srt")]
        for path in candidates:
            path.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n")
        self.app.reference_path.set("previous.srt")
        self.app._offer_reference_candidates(candidates)
        window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))
        window.destroy()
        self.assertEqual(self.app.reference_path.get(), "previous.srt")
        self.app._offer_reference_candidates(candidates)
        window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))
        combo = next(child for child in _descendants(window) if isinstance(child, ttk.Combobox))
        combo.current(1)
        button = next(child for child in _descendants(window) if isinstance(child, ttk.Button)
                      and child["text"] == "Use this reference")
        button.invoke()
        self.assertEqual(self.app.reference_path.get(), str(candidates[1]))

    def test_subtitle_choosers_match_all_themes_and_source_selection_sets_output(self):
        source = Path(self.temporary.name) / "Episode.pt.srt"
        source.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n")
        for theme, dark in (("light", False), ("dark", True), ("system", False), ("system", True)):
            with patch.object(preferences, "system_is_dark", return_value=dark):
                self.change(theme=theme)
                for is_source in (False, True):
                    self.app._offer_subtitle_candidates([source], source=is_source)
                    window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))
                    self.assertEqual(window["background"], self.root["background"])
                    use = next(child for child in _descendants(window) if isinstance(child, ttk.Button)
                               and str(child["text"]).startswith("Use this"))
                    self.assertEqual(use.winfo_manager(), "grid")
                    self.assertEqual(
                        self.app.interface.style.lookup("TFrame", "background"), self.root["background"],
                    )
                    window.destroy()
        self.change()
        self.app._offer_subtitle_candidates([source], source=True)
        window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))
        next(child for child in _descendants(window) if isinstance(child, ttk.Button)
             and child["text"] == "Use this source").invoke()
        self.assertEqual(self.app.source_path.get(), str(source))
        self.assertEqual(self.app.output_path.get(), str(source.with_name("Episode.pt.synced.srt")))

    def test_reference_selection_offers_sources_and_about_has_credit(self):
        source = Path(self.temporary.name) / "Episode.pt.srt"
        reference = Path(self.temporary.name) / "Episode.en.srt"
        source.touch()
        reference.touch()
        with (
            patch.object(self.gui.filedialog, "askopenfilename", return_value=str(reference)),
            patch.object(self.app, "_offer_subtitle_candidates") as offer,
        ):
            self.app._choose_reference()
        offer.assert_called_once_with([source], source=True)
        with patch.object(self.gui.messagebox, "showinfo") as info:
            self.app._show_about()
        self.assertIn("\ndeveloped by SMAP", info.call_args.args[1])

    def test_audio_sync_action_opens_with_current_inputs_and_blocks_competing_work(self):
        self.app.video_path.set("video.mkv")
        self.app.source_path.set("source.srt")
        self.app.reference_path.set("reference.srt")
        self.app.output_path.set("other-output.srt")
        with (
            patch.object(self.gui, "AudioSyncDialog") as dialog,
            patch.object(self.root, "wait_window") as wait,
        ):
            self.app.audio_sync_button.invoke()
        self.assertEqual(
            dialog.call_args.args,
            (self.root, "video.mkv", "source.srt",
             (Path("source.srt"), Path("reference.srt"), Path("other-output.srt"))),
        )
        self.app.sync_selection = Mock()
        dialog.call_args.kwargs["on_saved"](Path("/audio-result.srt"))
        self.assertEqual(self.app.reference_path.get(), "/audio-result.srt")
        self.assertIsNone(self.app.sync_selection)
        wait.assert_called_once_with(dialog.return_value)
        self.assertFalse(self.app.setup_busy)
        self.app.busy = True
        with patch.object(self.gui, "AudioSyncDialog") as dialog, patch.object(
            self.gui.messagebox, "showwarning",
        ) as warning:
            self.app._open_audio_sync()
        dialog.assert_not_called()
        warning.assert_called_once()
        self.change("pt")
        self.assertEqual(
            self.app.audio_sync_button["text"], translations.translate("Sync to audio...", "pt"),
        )

    def test_library_controls_are_desktop_only(self):
        from desktop_updates import DesktopUpdates
        for enabled in (False, True):
            self.app.library_updates.enabled = enabled
            self.app._build_settings_menu()
            help_menu = self.app.settings_menu.nametowidget(self.app.settings_menu.entrycget(2, "menu"))
            captions = [help_menu.entrycget(index, "label") for index in range(help_menu.index("end") + 1)]
            self.assertEqual("Library updates" in captions, enabled)
        controller = DesktopUpdates(self.app)
        controller.enabled = True
        controller.manager = Mock()
        controller.manager.state.return_value = {"previous": None}
        controller.open()
        self.assertIsNotNone(controller.window)
        self.assertEqual(str(controller.install_button["state"]), "disabled")
        self.assertEqual(str(controller.restore_button["state"]), "disabled")
        controller.updates = [{"name": "fastembed"}]
        controller.manager.state.return_value = {"previous": "base"}
        controller._refresh()
        self.assertEqual(str(controller.install_button["state"]), "normal")
        self.assertEqual(str(controller.restore_button["state"]), "normal")
        controller.pending_restart = True
        controller._refresh()
        self.assertEqual(str(controller.install_button["state"]), "disabled")
        controller._close()

    def test_component_update_controls_are_per_component_and_require_confirmation(self):
        from desktop_updates import DesktopUpdates
        controller = DesktopUpdates(self.app)
        controller.enabled = True
        controller.component_manager = Mock()
        component = {
            "key": "whisper-cpu", "candidate": {"version": "new"},
            "previous": {"version": "old"},
        }
        controller.components = [component, {"key": "whisper-model-base"}]
        controller.open()
        self.assertEqual(str(controller.component_install["state"]), "normal")
        self.assertEqual(str(controller.component_restore["state"]), "normal")
        with (
            patch("desktop_updates.messagebox.askyesno", return_value=False),
            patch.object(controller, "_start") as start,
        ):
            controller._change_component()
            start.assert_not_called()
        with (
            patch("desktop_updates.messagebox.askyesno", return_value=True),
            patch.object(controller, "_start") as start,
        ):
            controller._change_component()
            start.call_args.args[0]()
            controller.component_manager.install_component.assert_called_once()
            controller._change_component(True)
            start.call_args.args[0]()
            controller.component_manager.restore_component.assert_called_once_with("whisper-cpu")
        controller.component_selection.set("whisper-model-base")
        controller._refresh()
        self.assertEqual(str(controller.component_install["state"]), "disabled")
        self.assertEqual(str(controller.component_restore["state"]), "disabled")
        controller._close()
        self.assertEqual(self.app.version_badge["text"], "v" + preferences.APP_VERSION)
        self.assertEqual([self.app.settings_menu.entrycget(index, "label") for index in range(3)], ["Language", "Theme", "Help"])
        menu = self.root.nametowidget(self.app.settings_menu.entrycget(0, "menu"))
        self.assertEqual(menu.index("end"), 18)

    def test_banner_stays_visible_and_body_scrolls_on_smaller_windows(self):
        self.root.deiconify()
        self.root.geometry(f"{self.root.winfo_width()}x620")
        self.root.update()
        self.assertLessEqual(self.root.winfo_height(), self.root.winfo_screenheight())
        before = self.app.top_bar.winfo_y()
        self.app.content_canvas.yview_moveto(1.0)
        self.root.update()
        self.assertEqual(self.app.top_bar.winfo_y(), before)
        self.assertGreater(self.app.content_canvas.yview()[0], 0)
        self.assertAlmostEqual(self.app.content_canvas.yview()[1], 1.0, places=2)

    def test_all_gear_submenus_open_left_at_full_screen_width(self):
        self.root.deiconify()
        self.root.geometry(f"{self.root.winfo_screenwidth()}x620+0+0")
        self.root.update()
        self.app._open_settings_menu()
        self.root.update()
        try:
            for index in range(3):
                self.root.tk.call(self.app.settings_menu._w, "postcascade", index)
                self.root.update()
                submenu = self.root.nametowidget(self.app.settings_menu.entrycget(index, "menu"))
                self.assertTrue(submenu.winfo_ismapped())
                self.assertGreaterEqual(submenu.winfo_rootx(), 0)
                self.assertLessEqual(
                    submenu.winfo_rootx() + submenu.winfo_width(),
                    self.app.settings_menu.winfo_rootx() + 2,
                )
        finally:
            self.app.settings_menu.unpost()
            self.root.grab_release()

    def test_banner_action_colors_match_reference_in_light_and_dark(self):
        self.root.deiconify()
        self.root.update()
        for theme, secondary, primary in (
            ("light", "#596f8d", "#31557f"), ("dark", "#4c5b73", "#3a68a3"),
        ):
            self.change(theme=theme)
            self.assertEqual(self.app.speech_button["background"], secondary)
            for option in ("background", "foreground", "activebackground", "activeforeground", "disabledforeground"):
                self.assertEqual(self.app.audio_sync_button[option], self.app.speech_button[option])
            self.assertEqual(self.app.cancel_sync_button["background"], secondary)
            self.assertEqual(self.app.sync_button["background"], primary)
            self.assertEqual(self.app.sync_button["foreground"], "white")
        self.assertLess(self.app.audio_sync_button.winfo_x(), self.app.speech_button.winfo_x())
        self.assertLess(self.app.speech_button.winfo_x(), self.app.sync_button.winfo_x())
        self.assertLess(self.app.sync_button.winfo_x(), self.app.gear_button.winfo_x())
        self.assertLess(self.app.sync_button.winfo_x(), self.app.cancel_sync_button.winfo_x())
        self.assertLess(self.app.cancel_sync_button.winfo_x(), self.app.gear_button.winfo_x())
        for system_dark in (False, True):
            with patch.object(preferences, "system_is_dark", return_value=system_dark):
                self.change(theme="system")
                self.assertEqual(
                    self.app.audio_sync_button["background"], self.app.speech_button["background"],
                )

    def test_cancellation_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.srt"
            source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
            output = Path(directory) / "output.srt"
            output.write_text("original")
            self.app._cancel_sync()
            with patch.object(self.gui.subtitle_alignment, "analyze_subtitles") as analyze:
                self.app._sync_worker(None, source, source, output, 1000, 70)
            analyze.assert_not_called()
            self.assertEqual(output.read_text(), "original")
            messages = []
            while not self.app.messages.empty():
                messages.append(self.app.messages.get())
            self.assertEqual(messages[-1][0], "cancelled")
            self.assertEqual(str(self.app.cancel_sync_button["state"]), "disabled")

    def test_cancellation_during_analysis_prevents_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.srt"
            source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
            output = Path(directory) / "output.srt"

            def analyze(*args, **kwargs):
                self.app.sync_cancel.set()
                return [(1, 0)], {"matched_cues": 1}

            with patch.object(self.gui.subtitle_alignment, "analyze_subtitles", side_effect=analyze):
                self.app._sync_worker(None, source, source, output, 1000, 70)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(directory).glob(".subtitle-sync-*")))

    def test_processing_choices_restore_in_new_application_and_survive_theme_save(self):
        self.app.processing_device.set("AMD APU (integrated)")
        self.app.processing_backend.set("Vulkan")
        self.app.device_id.set("2")
        self.change("pt", "dark")
        second = tk.Tk()
        second.withdraw()
        try:
            app = self.gui.SubtitleSyncApp(second)
            self.assertEqual(app.processing_device.get(), "AMD APU (integrated)")
            self.assertEqual(app.processing_backend.get(), "Vulkan")
            self.assertEqual(app.device_id.get(), "2")
        finally:
            second.destroy()

    def test_filter_level_restores_after_restart_including_slider_fraction_and_limits(self):
        for value, expected in ((0, 0), (43.8, 43), (100, 100)):
            with self.subTest(value=value):
                self.app.hearing_impaired_strictness.set(value)
                self.change(theme="dark")
                second = tk.Tk()
                second.withdraw()
                try:
                    app = self.gui.SubtitleSyncApp(second)
                    self.assertEqual(app.hearing_impaired_strictness.get(), expected)
                    self.assertEqual(app.hearing_impaired_value.get(), f"{expected} / 100")
                    self.assertEqual(
                        preferences.read_settings(self.path)["processing"]["filter_level"], str(expected),
                    )
                finally:
                    second.destroy()

    def test_whisper_choices_restore_after_reopening_and_in_a_new_application(self):
        dialog = speech_dialog.SpeechDialog(self.root, "", (), lambda _path: None)
        expected = {
            "model": "small", "language": "Portuguese", "device": "AMD",
            "backend": "Vulkan", "adapter": "1", "audio_stream": "2",
        }
        for key, value in expected.items():
            dialog.preference_variables[key].set(value)
        dialog.destroy()
        self.change(theme="dark")
        second = tk.Tk()
        second.withdraw()
        try:
            self.gui.SubtitleSyncApp(second)
            restored = speech_dialog.SpeechDialog(second, "", (), lambda _path: None)
            try:
                self.assertEqual(
                    {key: variable.get() for key, variable in restored.preference_variables.items()}, expected,
                )
                self.assertIn(str(speech.MODELS["small"].size_mb), restored.model_info.get())
            finally:
                restored.destroy()
        finally:
            second.destroy()

    def test_partial_numeric_edits_do_not_overwrite_last_valid_processing_settings(self):
        self.app.device_id.set("3")
        self.app.device_id.set("")
        self.assertEqual(preferences.read_settings(self.path)["processing"]["adapter"], "3")
        self.app.device_id.set("4")
        self.assertEqual(preferences.read_settings(self.path)["processing"]["adapter"], "4")

    def test_processing_save_failure_is_reported(self):
        with (
            patch.object(self.gui, "persist_choices", side_effect=OSError("Read-only settings folder")),
            patch.object(self.gui.messagebox, "showerror") as error,
        ):
            self.app.device_id.set("2")
        error.assert_called_once()
    def test_open_windows_show_only_catalogued_captions(self):
        cues = (Cue(1000, 2000, "Hello"),)
        CuePairDialog(self.root, cues, cues, lambda _left, _right: None)
        self.app._show_folder_report([("/a.srt", "Success", "/a.synced.srt", "")])
        speech_dialog.SpeechDialog(self.root, "", (), lambda _path: None)
        self.gui.AudioSyncDialog(self.root)
        self.change("de")
        ignored = UNTRANSLATED_NAMES | {f"v{preferences.APP_VERSION}"}
        captions = set()
        for widget in [self.root, *_descendants(self.root)]:
            captions.update(
                value for value in (
                    getattr(widget, "_original_caption", None), getattr(widget, "_original_title", None),
                    *getattr(widget, "_original_headings", {}).values(),
                ) if value
            )
        missing = sorted(
            caption for caption in captions
            if any(character.isalpha() for character in caption) and caption not in ignored
            and caption not in translations.TRANSLATIONS and not caption.startswith("Success: ")
        )
        self.assertEqual(missing, [])
        german_speech = speech_dialog.SpeechDialog(self.root, "", (), lambda _path: None)
        self.assertTrue(german_speech.model_info.get().startswith("Modell"), german_speech.model_info.get())

    def test_all_languages_switch_live_and_restore_english_without_changing_inputs(self):
        self.app.source_path.set("/source with spaces.srt")
        self.app.reference_path.set("/reference.srt")
        self.app.processing_device.set("AMD")
        for language in translations.LANGUAGES:
            with self.subTest(language=language):
                self.change(language, "light")
                self.assertEqual(self.app.sync_button["text"], translations.translate("Analyze and synchronize", language))
                self.assertEqual(self.app.speech_button["text"], translations.translate("Speech to text...", language))
                self.assertEqual(self.app.settings_menu.entrycget(0, "label"), translations.translate("Language", language))
                self.assertEqual(self.app.processing_device.get(), "AMD")
                self.assertEqual(self.app.source_path.get(), "/source with spaces.srt")
                self.assertEqual(self.app.reference_path.get(), "/reference.srt")
        self.change()
        self.assertEqual(self.app.sync_button["text"], "Analyze and synchronize")
        self.assertIn("Example 1:", self.app.instructions_text.get("1.0", "end"))

    def test_explanatory_note_still_rewraps_when_the_window_expands(self):
        self.root.deiconify()
        self.root.update()
        note = next(
            widget for widget in self.app.content_frame.winfo_children()
            if widget.winfo_class() == "TLabel"
            and str(widget.cget("text")).startswith("Source and reference may")
        )
        before = int(note["wraplength"])
        self.root.geometry(f"{self.root.winfo_width() + 300}x{self.root.winfo_height()}")
        self.root.update()
        self.assertGreaterEqual(int(note["wraplength"]), before + 290)
        self.assertEqual(int(note["wraplength"]), note.winfo_width())

    def test_resizable_dialogs_are_maximizable_normal_windows(self):
        cue = Cue(1000, 2000, "Hello")
        speech_window = speech_dialog.SpeechDialog(self.root, "", (), lambda _path: None)
        pair = CuePairDialog(self.root, (cue,), (cue,), lambda _left, _right: None)
        try:
            for window in (speech_window, pair.window):
                with self.subTest(window=window.title()):
                    self.assertEqual(str(window.transient()), "")
                    self.assertEqual(window.resizable(), (True, True))
                    if window._windowingsystem == "x11":
                        self.assertIn("normal", window.wm_attributes("-type"))
        finally:
            pair.window.destroy()
            speech_window.destroy()

    def test_device_help_and_speech_dialog_text_rewrap_when_expanded(self):
        self.root.deiconify()
        self.root.update()
        labels = [
            widget for widget in _descendants(self.app.content_frame)
            if widget.winfo_class() == "TLabel"
            and str(widget.cget("textvariable")) == str(self.app.device_help)
        ]
        self.assertEqual(len(labels), 1)
        help_label = labels[0]
        before = help_label.winfo_width()
        self.root.geometry(f"{self.root.winfo_width() + 300}x{self.root.winfo_height()}")
        self.root.update()
        self.assertGreaterEqual(help_label.winfo_width(), before + 290)
        self.assertEqual(int(help_label["wraplength"]), help_label.winfo_width())

        dialog = speech_dialog.SpeechDialog(self.root, "", (), lambda _path: None)
        try:
            dialog.update()
            wrapped = [
                widget for widget in _descendants(dialog)
                if widget.winfo_class() == "TLabel" and int(widget["wraplength"] or 0) > 0
            ]
            self.assertGreaterEqual(len(wrapped), 4)
            widths = [widget.winfo_width() for widget in wrapped]
            dialog.geometry(f"{dialog.winfo_width() + 300}x{dialog.winfo_height()}")
            dialog.update()
            for widget, width in zip(wrapped, widths):
                self.assertGreaterEqual(widget.winfo_width(), width + 290)
                self.assertEqual(int(widget["wraplength"]), widget.winfo_width())
        finally:
            dialog.destroy()

    def test_theme_and_language_preserve_manual_selection(self):
        source = Path(self.temporary.name) / "source.srt"
        source.touch()
        self.app.source_path.set(str(source))
        cue = Cue(1000, 2000, "User dialogue, do not translate")
        selection = SyncSelection((cue,), (cue,), 1, 1, input_signatures((str(source),)))
        self.app.sync_selection = selection
        self.change("pt", "dark")
        self.assertIs(self.app.sync_selection, selection)
        self.assertEqual(self.app.interface.language, "pt")
        self.assertTrue(self.app.interface.dark)
        self.assertEqual(self.app.instructions_text["background"], "#242a33")
        self.change("en", "light")
        self.assertFalse(self.app.interface.dark)
        self.assertEqual(self.app.instructions_text["background"], "white")
        self.assertEqual(preferences.load_settings(self.path), ("en", "light"))

    def test_preferences_reload_into_a_new_window(self):
        self.change("fr", "dark")
        other = tk.Toplevel(self.root)
        # A separate Tk interpreter matches the application's normal startup.
        other.destroy()
        second = tk.Tk()
        second.withdraw()
        try:
            app = self.gui.SubtitleSyncApp(second)
            self.assertEqual(app.interface.language, "fr")
            self.assertEqual(app.interface.theme, "dark")
            self.assertEqual(app.sync_button["text"], translations.translate("Analyze and synchronize", "fr"))
        finally:
            second.destroy()

    def test_save_failure_is_visible_and_does_not_apply_unsaved_preferences(self):
        self.app.interface_language.set("pt")
        with (
            patch.object(self.gui, "save_settings", side_effect=OSError("Permission denied")),
            patch.object(self.gui.messagebox, "showerror") as error,
        ):
            self.app._change_interface_preferences()
        error.assert_called_once()
        self.assertEqual(self.app.interface.language, "en")
        self.assertEqual(self.app.interface_language.get(), "en")

    def test_about_reports_our_version_not_reference_app_version(self):
        with patch.object(self.gui.messagebox, "showinfo") as info:
            self.app._show_about()
        self.assertIn("v" + preferences.APP_VERSION, info.call_args.args[1])
        self.assertNotIn("Bulk Rename", info.call_args.args[1])

    def test_popup_controls_follow_language_and_dark_theme(self):
        self.change("pt", "dark")
        dialog = speech_dialog.SpeechDialog(self.root, "", (), lambda _path: None)
        try:
            self.assertEqual(dialog.title(), translations.translate("Speech to text - generate reference SRT", "pt"))
            self.assertEqual(dialog.cancel_button["text"], "Cancelar")
            self.assertEqual(dialog.log["background"], "#242a33")
        finally:
            dialog.destroy()
        cues = (Cue(1000, 2000, "Original dialogue"),)
        pair = CuePairDialog(self.root, cues, cues, lambda _left, _right: None)
        try:
            self.assertEqual(pair.confirm["text"], translations.translate("Use selected pair", "pt"))
            self.assertEqual(pair.trees[0].heading("text", "text"), "Diálogo")
            self.assertEqual(pair.trees[0].item("1", "values")[-1], "Original dialogue")
        finally:
            pair.window.destroy()

    def test_system_theme_uses_detected_desktop_preference(self):
        with patch.object(preferences, "system_is_dark", return_value=True):
            self.change(theme="system")
            self.assertTrue(self.app.interface.dark)
        with patch.object(preferences, "system_is_dark", return_value=False):
            self.change(theme="system")
            self.assertFalse(self.app.interface.dark)


if __name__ == "__main__":
    unittest.main()
