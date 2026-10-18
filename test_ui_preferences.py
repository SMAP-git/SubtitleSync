import json
import os
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


class PreferencesTests(unittest.TestCase):
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
            right, [self.app.version_badge, self.app.gear_button, self.app.cancel_sync_button, self.app.sync_button, self.app.speech_button],
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
            self.assertEqual(self.app.cancel_sync_button["background"], secondary)
            self.assertEqual(self.app.sync_button["background"], primary)
            self.assertEqual(self.app.sync_button["foreground"], "white")
        self.assertLess(self.app.speech_button.winfo_x(), self.app.sync_button.winfo_x())
        self.assertLess(self.app.sync_button.winfo_x(), self.app.gear_button.winfo_x())
        self.assertLess(self.app.sync_button.winfo_x(), self.app.cancel_sync_button.winfo_x())
        self.assertLess(self.app.cancel_sync_button.winfo_x(), self.app.gear_button.winfo_x())

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
    def test_all_languages_switch_live_and_restore_english_without_changing_inputs(self):
        self.app.portuguese_path.set("/source with spaces.srt")
        self.app.english_path.set("/reference.srt")
        self.app.processing_device.set("AMD")
        for language in translations.LANGUAGES:
            with self.subTest(language=language):
                self.change(language, "light")
                self.assertEqual(self.app.sync_button["text"], translations.translate("Analyze and synchronize", language))
                self.assertEqual(self.app.speech_button["text"], translations.translate("Speech to text...", language))
                self.assertEqual(self.app.settings_menu.entrycget(0, "label"), translations.translate("Language", language))
                self.assertEqual(self.app.processing_device.get(), "AMD")
                self.assertEqual(self.app.portuguese_path.get(), "/source with spaces.srt")
                self.assertEqual(self.app.english_path.get(), "/reference.srt")
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
        self.app.portuguese_path.set(str(source))
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
