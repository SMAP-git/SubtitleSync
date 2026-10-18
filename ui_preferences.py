"""Persistent interface preferences and explicit window styling/localization."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ui_translations import LANGUAGES, translate

APP_VERSION = "1.0.0"
THEMES = ("system", "light", "dark")


def make_maximizable(window: tk.Toplevel) -> None:
    """Allow maximizing; window managers often disable it for transient dialog windows."""
    window.resizable(True, True)
    if window._windowingsystem == "x11":
        window.wm_attributes("-type", "normal")


def auto_wrap(label: ttk.Label) -> ttk.Label:
    """Wrap label text at the label's current width so it follows window resizing."""
    def resize(event: tk.Event) -> None:
        if event.width > 1 and int(str(label.cget("wraplength")) or 0) != event.width:
            label.configure(wraplength=event.width)
    label.bind("<Configure>", resize, add="+")
    return label


def settings_path() -> Path:
    if os.environ.get("SUBTITLE_SYNC_SETTINGS_DIR"):
        return Path(os.environ["SUBTITLE_SYNC_SETTINGS_DIR"]) / "settings.json"
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "SubtitleSync" / "settings.json"


def load_settings(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "en", "system"
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Interface settings must be a JSON object.")
    language, theme = values.get("language", "en"), values.get("theme", "system")
    if not isinstance(language, str) or not isinstance(theme, str) or language not in LANGUAGES or theme not in THEMES:
        raise ValueError("Interface settings contain an unsupported language or theme.")
    return language, theme


def save_settings(path: Path, language: str, theme: str) -> None:
    if language not in LANGUAGES or theme not in THEMES:
        raise ValueError("Unsupported interface language or theme.")
    update_settings(path, {"language": language, "theme": theme})


def read_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Settings must be a JSON object.")
    return values


def update_settings(path: Path, changes: dict) -> None:
    values = read_settings(path)
    values.update(changes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(values) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_choices(path: Path, section: str, variables: dict[str, tk.StringVar], choices: dict[str, tuple]) -> None:
    saved = read_settings(path).get(section, {})
    if not isinstance(saved, dict):
        raise ValueError(f"Invalid {section} settings.")
    for name, value in saved.items():
        if name not in variables:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Invalid {section} setting: {name}.")
        if name in choices:
            if value not in choices[name]:
                raise ValueError(f"Unsupported {section} setting: {name}.")
        elif not value.isascii() or not value.isdigit():
            raise ValueError(f"Invalid {section} setting: {name}.")
    for name, variable in variables.items():
        if name in saved:
            variable.set(saved[name])


def persist_choices(path: Path, section: str, variables: dict[str, tk.StringVar], choices: dict[str, tuple]) -> None:
    values = {name: variable.get() for name, variable in variables.items()}
    for name, value in values.items():
        if name in choices:
            if value not in choices[name]:
                return
        elif not value.isascii() or not value.isdigit():
            return  # An adapter/stream number may be temporarily empty while being edited.
    update_settings(path, {section: values})

def system_is_dark() -> bool:
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            ) as key:
                return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
        except FileNotFoundError:
            return False
        except OSError as error:
            logging.warning("Cannot read Windows theme preference: %s", error)
            return False
    if shutil.which("gsettings"):
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                capture_output=True, text=True, timeout=1,
            )
            if result.returncode == 0:
                return "dark" in result.stdout.lower()
            logging.warning("Cannot read desktop theme preference: %s", result.stderr.strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            logging.warning("Cannot read desktop theme preference: %s", error)
    return False


class InterfacePreferences:
    def __init__(self, root: tk.Tk, language: str = "en", theme: str = "system") -> None:
        self.root = root
        self.language = language
        self.theme = theme
        self.style = ttk.Style(root)
        self.native_theme = self.style.theme_use()
        self.native_background = root.cget("background")
        self.dark = False

    def tr(self, text: str) -> str:
        return translate(text, self.language)

    def apply_theme(self) -> None:
        self.dark = self.theme == "dark" or (self.theme == "system" and system_is_dark())
        if self.theme == "system" and not self.dark:
            self.style.theme_use(self.native_theme)
            self.root.configure(background=self.native_background)
            self.root.option_add("*TCombobox*Listbox.background", "white")
            self.root.option_add("*TCombobox*Listbox.foreground", "#182230")
            self.root.option_add("*TCombobox*Listbox.selectBackground", "#c5d9f4")
            return
        self.style.theme_use("clam")
        background = "#20242b" if self.dark else "#f3f5f8"
        panel = "#2b313b" if self.dark else "#ffffff"
        foreground = "#edf0f5" if self.dark else "#182230"
        muted = "#aab5c4" if self.dark else "#5b6675"
        selected = "#385b8b" if self.dark else "#d2e2f7"
        self.root.configure(background=background)
        self.style.configure(".", background=background, foreground=foreground)
        self.style.configure("TButton", background=panel, foreground=foreground, padding=5)
        self.style.map("TButton", background=[("active", selected)], foreground=[("disabled", muted)])
        for name in ("TEntry", "TCombobox", "Treeview"):
            self.style.configure(name, fieldbackground=panel, background=panel, foreground=foreground)
            self.style.map(
                name, fieldbackground=[("readonly", panel), ("disabled", background)],
                foreground=[("disabled", muted), ("readonly", foreground)],
                selectbackground=[("!disabled", selected)],
                selectforeground=[("!disabled", foreground)],
            )
        self.style.configure("Treeview.Heading", background=background, foreground=foreground)
        self.style.map("Treeview", background=[("selected", selected)], foreground=[("selected", foreground)])
        self.style.configure("TScale", troughcolor=panel)
        self.style.configure("TProgressbar", troughcolor=panel, background="#4275ba")
        self.root.option_add("*TCombobox*Listbox.background", panel)
        self.root.option_add("*TCombobox*Listbox.foreground", foreground)
        self.root.option_add("*TCombobox*Listbox.selectBackground", selected)

    def apply_window(self, window: tk.Misc) -> None:
        """Translate static captions only; never touch paths, subtitle text or logs."""
        if isinstance(window, (tk.Tk, tk.Toplevel)):
            if not hasattr(window, "_original_title"):
                window._original_title = window.title()
            window.title(self.tr(window._original_title))
        options = window.keys()
        if "text" in options and ("textvariable" not in options or not window.cget("textvariable")):
            if not hasattr(window, "_original_caption"):
                window._original_caption = str(window.cget("text"))
            window.configure(text=self.tr(window._original_caption))
        if isinstance(window, ttk.Treeview):
            for column in window["columns"]:
                originals = getattr(window, "_original_headings", {})
                if column not in originals:
                    originals[column] = window.heading(column, "text")
                window._original_headings = originals
                window.heading(column, text=self.tr(originals[column]))
        if isinstance(window, tk.Text):
            window.configure(
                background="#242a33" if self.dark else "white",
                foreground="#edf0f5" if self.dark else "#182230",
                insertbackground="#edf0f5" if self.dark else "#182230",
                selectbackground="#385b8b" if self.dark else "#c5d9f4",
            )
        if isinstance(window, tk.Canvas):
            window.configure(background=self.root.cget("background"))
        if isinstance(window, tk.Menu):
            window.configure(
                background="#2b313b" if self.dark else "#ffffff",
                foreground="#edf0f5" if self.dark else "#182230",
                activebackground="#385b8b" if self.dark else "#d2e2f7",
                activeforeground="#ffffff" if self.dark else "#182230",
            )
        for child in window.winfo_children():
            self.apply_window(child)
