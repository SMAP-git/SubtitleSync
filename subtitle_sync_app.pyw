"""Windows and Linux GUI for context-aware multilingual subtitle alignment."""

from __future__ import annotations

import json
import queue
import re
import tempfile
import threading
import tkinter as tk
import webbrowser
from dataclasses import asdict
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

import subtitle_alignment
import sync_subtitles
import gpu_runtime
import vulkan_embeddings
import file_picker as filedialog
from desktop_updates import DesktopUpdates
from manual_sync import CuePairDialog, SyncSelection, input_signatures
from speech_dialog import SpeechDialog
from ui_preferences import (
    APP_VERSION, InterfacePreferences, load_settings, save_settings, settings_path,
    load_choices, persist_choices, auto_wrap, make_maximizable,
)
from ui_translations import LANGUAGES, instructions as translated_instructions, translate


def format_time(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_time(value: str) -> int:
    match = re.fullmatch(r"\s*(\d{1,}):([0-5]\d):([0-5]\d)[,.](\d{1,3})\s*", value)
    if not match:
        raise ValueError("Use HH:MM:SS.mmm format, for example 00:31:23.150.")
    hours, minutes, seconds, millis = match.groups()
    return (
        ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000
        + int(millis.ljust(3, "0"))
    )


class SubtitleSyncApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Subtitle Sync")
        root.geometry("780x620")
        root.minsize(720, 600)

        self.video_path = tk.StringVar()
        self.portuguese_path = tk.StringVar()
        self.english_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.initial_time = tk.StringVar()
        self.processing_device = tk.StringVar(value="CPU")
        self.processing_backend = tk.StringVar(value="Recommended")
        self.device_id = tk.StringVar(value="0")
        self.device_help = tk.StringVar()
        self.hearing_impaired_strictness = tk.IntVar(value=70)
        self.saved_filter_level = tk.StringVar(value="70")
        self.hearing_impaired_value = tk.StringVar(value="70 / 100")
        self.first_cue_text = tk.StringVar(value="Select matching dialogue cues in the two subtitles.")
        self.status = tk.StringVar(value="Choose source and reference subtitles in any supported languages.")
        self.sync_selection: SyncSelection | None = None
        self.embedded_track: tuple[tuple[tuple[str, int, int], ...], int] | None = None
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.sync_cancel = threading.Event()
        self.sync_commit_lock = threading.Lock()
        self.sync_committed = False
        self.setup_busy = False
        self.preferences_path = settings_path()
        self.library_updates = DesktopUpdates(self)
        root.preferences_path = self.preferences_path
        self.processing_variables = {
            "device": self.processing_device, "backend": self.processing_backend, "adapter": self.device_id,
            "filter_level": self.saved_filter_level,
        }
        self.processing_choices = {
            "device": subtitle_alignment.PROCESSING_DEVICES,
            "backend": ("Recommended", "ONNX (advanced)") if gpu_runtime.CPU_ONLY else ("Recommended", "Vulkan", "ONNX (advanced)"),
            "filter_level": tuple(str(level) for level in range(101)),
        }
        try:
            language, theme = load_settings(self.preferences_path)
            load_choices(self.preferences_path, "processing", self.processing_variables, self.processing_choices)
            self.hearing_impaired_strictness.set(int(self.saved_filter_level.get()))
        except (OSError, ValueError) as error:
            language, theme = "en", "system"
            root.after_idle(lambda detail=str(error): messagebox.showerror(
                "Cannot load interface settings", detail, parent=root,
            ))
        self.interface = InterfacePreferences(root, language, theme)
        root.interface_preferences = self.interface
        self.interface.apply_theme()
        self.interface_language = tk.StringVar(value=language)
        self.interface_theme = tk.StringVar(value=theme)
        self._last_interface_language = "en"

        self._build_ui()
        self.interface.apply_window(root)
        self._style_top_bar()
        self._translate_instructions()
        root.bind("<FocusIn>", self._refresh_system_theme, add="+")
        for value in (self.portuguese_path, self.english_path, self.video_path):
            value.trace_add("write", self._clear_sync_selection)
        for value in self.processing_variables.values():
            value.trace_add("write", self._save_processing_preferences)
        self.hearing_impaired_strictness.trace_add("write", self._save_filter_level)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.update_idletasks()
        width = max(780, self.top_bar.winfo_reqwidth(), self.content_frame.winfo_reqwidth() + self.content_scrollbar.winfo_reqwidth())
        desired_height = self.content_frame.winfo_reqheight() + self.top_bar.winfo_reqheight() + 40
        height = max(400, min(max(620, desired_height), root.winfo_screenheight() - 80))
        root.minsize(width, min(620, height))
        root.geometry(f"{width}x{height}")
        self.library_updates.schedule_check()

    def _build_ui(self) -> None:
        self._build_top_bar()
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self.content_canvas = tk.Canvas(body, highlightthickness=0, borderwidth=0, height=1)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.content_canvas.yview)
        self.content_scrollbar = scrollbar
        self.content_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.content_canvas.pack(side="left", fill="both", expand=True)
        frame = ttk.Frame(self.content_canvas, padding=14)
        self.content_frame = frame
        self.content_window = self.content_canvas.create_window(0, 0, window=frame, anchor="nw")
        self.content_canvas.bind("<Configure>", self._layout_content)
        frame.bind("<Configure>", self._layout_content)
        self.root.bind("<MouseWheel>", self._scroll_content, add="+")
        self.root.bind("<Button-4>", self._scroll_content, add="+")
        self.root.bind("<Button-5>", self._scroll_content, add="+")
        frame.columnconfigure(1, weight=1)

        self._file_row(
            frame, 0, "Video (optional)", self.video_path, self._choose_video, "Browse..."
        )
        self._file_row(
            frame,
            1,
            "Source SRT (to synchronize)",
            self.portuguese_path,
            self._choose_portuguese,
            "Browse...",
        )
        self._file_row(
            frame,
            2,
            "Reference SRT (optional)",
            self.english_path,
            self._choose_english,
            "Browse...",
        )
        self._file_row(
            frame, 3, "Save synchronized SRT", self.output_path, self._choose_output, "Save as..."
        )

        processing = ttk.LabelFrame(frame, text="Processing device", padding=9)
        processing.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        processing.columnconfigure(3, weight=1)
        self.device_selector = ttk.Combobox(
            processing, textvariable=self.processing_device,
            values=subtitle_alignment.PROCESSING_DEVICES, state="readonly", width=22,
        )
        self.device_selector.grid(row=0, column=0, sticky="w")
        self.device_selector.bind("<<ComboboxSelected>>", self._update_device_display)
        ttk.Label(processing, text="GPU adapter ID").grid(row=0, column=1, padx=(16, 8))
        self.device_id_entry = ttk.Entry(processing, textvariable=self.device_id, width=6)
        self.device_id_entry.grid(row=0, column=2, sticky="w")
        self.setup_button = ttk.Button(
            processing, text="GPU setup / drivers...", command=self._open_gpu_setup
        )
        self.setup_button.grid(row=0, column=3, sticky="e", padx=(12, 0))
        backend_row = ttk.Frame(processing)
        backend_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(backend_row, text="Inference backend").pack(side="left", padx=(0, 8))
        self.backend_selector = ttk.Combobox(
            backend_row, textvariable=self.processing_backend,
            values=self.processing_choices["backend"], state="readonly", width=22,
        )
        self.backend_selector.pack(side="left")
        self.backend_selector.bind("<<ComboboxSelected>>", self._update_device_display)
        auto_wrap(ttk.Label(processing, textvariable=self.device_help, wraplength=680)).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0)
        )
        self._update_device_display()

        hearing = ttk.LabelFrame(frame, text="Hearing-impaired filter", padding=9)
        hearing.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        hearing.columnconfigure(1, weight=1)
        ttk.Label(hearing, text="Relaxed").grid(row=0, column=0, sticky="w")
        scale = ttk.Scale(
            hearing,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.hearing_impaired_strictness,
            command=self._update_hearing_impaired_display,
        )
        scale.grid(row=0, column=1, sticky="ew", padx=(10, 10))
        ttk.Label(hearing, text="Strict").grid(row=0, column=2, sticky="e")
        ttk.Label(hearing, textvariable=self.hearing_impaired_value).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        self._update_hearing_impaired_display(str(self.hearing_impaired_strictness.get()))

        anchor = ttk.LabelFrame(frame, text="Initial sync point", padding=9)
        anchor.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        anchor.columnconfigure(1, weight=1)
        self.pair_button = ttk.Button(
            anchor, text="Select matching cues...", command=self._open_initial_sync
        )
        self.pair_button.grid(row=0, column=0, sticky="w")
        ttk.Entry(anchor, textvariable=self.initial_time, state="readonly", width=20).grid(
            row=0, column=1, sticky="w", padx=(10, 0)
        )
        auto_wrap(ttk.Label(anchor, textvariable=self.first_cue_text, wraplength=640)).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )

        note = (
            "Source and reference may use different languages. Leave reference SRT blank "
            "to select an embedded SubRip track, or generate one using speech to text. Select matching cues manually "
            "before analysis. Matching uses meaning and surrounding dialogue, not literal translation."
        )
        note_label = ttk.Label(frame, text=note, wraplength=680)
        note_label.grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(8, 6)
        )
        auto_wrap(note_label)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(2, 6))

        footer = ttk.Frame(frame)
        footer.grid(row=9, column=0, columnspan=3, sticky="ew")
        footer.columnconfigure(0, weight=1)
        auto_wrap(ttk.Label(footer, textvariable=self.status, wraplength=500)).grid(
            row=0, column=0, sticky="ew"
        )

        instructions = ttk.LabelFrame(
            frame, text="Instructions and examples on how to use this app", padding=8
        )
        instructions.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        frame.rowconfigure(10, weight=1)
        instructions.columnconfigure(0, weight=1)
        instructions.rowconfigure(0, weight=1)
        text = tk.Text(instructions, height=9, width=1, wrap="word", padx=6, pady=6)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(instructions, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        text.insert(
            "1.0",
            "Example 1: You have a Reference SRT\n"
            "- Select the Reference SRT and the Source SRT.\n"
            "- Choose the processing device (CPU or GPU).\n"
            "- Select the Hearing-impaired filter level.\n"
            "- Select the matching cues.\n"
            "- Choose where to save the synchronized SRT.\n"
            "- Press Analyze and synchronize.\n\n"
            "Example 2: You don't have a Reference SRT, or yours has bad timing or missing lines\n"
            "- Leave Reference SRT blank and select the Video. If it has no usable subtitle "
            "track, the Speech to text window opens automatically. Follow the process to "
            "generate an SRT file. If the video has a subtitle track that is not suitable, "
            "close the track chooser if shown, then open Speech to text manually and "
            "generate an SRT file instead. The generated SRT becomes the reference.\n"
            "- Select the Source SRT.\n"
            "- Choose the processing device (CPU or GPU).\n"
            "- Select the Hearing-impaired filter level.\n"
            "- Select the matching cues.\n"
            "- Choose where to save the synchronized SRT.\n"
            "- Press Analyze and synchronize.",
        )
        text.configure(state="disabled")
        self.instructions_text = text
        self.english_instructions = text.get("1.0", "end-1c")

    def _translate_instructions(self) -> None:
        for variable, original in (
            (self.first_cue_text, "Select matching dialogue cues in the two subtitles."),
            (self.status, "Choose source and reference subtitles in any supported languages."),
        ):
            if variable.get() == translate(original, self._last_interface_language):
                variable.set(self.interface.tr(original))
        self._last_interface_language = self.interface.language
        content = (
            self.english_instructions if self.interface.language == "en"
            else translated_instructions(self.interface.language)
        )
        self.instructions_text.configure(state="normal")
        self.instructions_text.delete("1.0", "end")
        self.instructions_text.insert("1.0", content)
        self.instructions_text.configure(state="disabled")

    def _layout_content(self, _event: tk.Event) -> None:
        canvas, frame = self.content_canvas, self.content_frame
        width = max(1, canvas.winfo_width())
        height = max(frame.winfo_reqheight(), canvas.winfo_height())
        canvas.itemconfigure(self.content_window, width=width, height=height)
        canvas.configure(scrollregion=(0, 0, width, height))

    def _scroll_content(self, event: tk.Event) -> None:
        if isinstance(event.widget, (tk.Text, ttk.Treeview, ttk.Combobox)):
            return
        if self.content_frame.winfo_height() <= self.content_canvas.winfo_height():
            return
        direction = 1 if event.num == 5 or event.delta < 0 else -1
        self.content_canvas.yview_scroll(direction * 3, "units")

    def _build_top_bar(self) -> None:
        banner = tk.Frame(self.root, background="#304b70", padx=20, pady=12)
        self.top_bar = banner
        banner.pack(fill="x")
        heading = tk.Frame(banner, background="#304b70")
        self.banner_heading = heading
        heading.pack(side="left", fill="x", expand=True)
        tk.Label(
            heading, text="Subtitle Sync", font=("TkDefaultFont", 19, "bold"),
            background="#304b70", foreground="white", anchor="w",
        ).pack(anchor="w")
        tk.Label(
            heading, text="Synchronize subtitles locally, in your language.",
            background="#304b70", foreground="#d8e4f2", anchor="w",
        ).pack(anchor="w")
        self.version_badge = tk.Label(
            banner, text=f"v{APP_VERSION}", font=("TkDefaultFont", 10, "bold"),
            background="#405f86", foreground="white", padx=12, pady=9,
        )
        self.version_badge.pack(side="right", padx=(10, 0))
        self.gear_button = tk.Button(
            banner, text="\u2699", font=("TkDefaultFont", 17), command=self._open_settings_menu,
            background="#405f86", foreground="white", activebackground="#52739d",
            activeforeground="white", relief="flat", borderwidth=0,
            padx=9, pady=2, takefocus=True,
        )
        self.gear_button.pack(side="right")
        self.cancel_sync_button = tk.Button(
            banner, text="Cancel", command=self._cancel_sync, state="disabled",
            font=("TkDefaultFont", 10), relief="flat", borderwidth=0,
            padx=10, pady=9, takefocus=True,
        )
        self.cancel_sync_button.pack(side="right", padx=(10, 10))
        self.settings_menu = tk.Menu(self.gear_button, tearoff=False)
        self.gear_button.bind(
            "<Return>", lambda _event: self._open_settings_menu(),
        )
        self.sync_button = tk.Button(
            banner, text="Analyze and synchronize", command=self._start_sync,
            font=("TkDefaultFont", 10, "bold"), relief="flat", borderwidth=0,
            padx=10, pady=9, takefocus=True,
        )
        self.sync_button.pack(side="right", padx=(10, 10))
        self.speech_button = tk.Button(
            banner, text="Speech to text...", command=self._open_speech_to_text,
            font=("TkDefaultFont", 10), relief="flat", borderwidth=0,
            padx=10, pady=9, takefocus=True,
        )
        self.speech_button.pack(side="right")
        self._build_settings_menu()

    def _open_settings_menu(self) -> None:
        self.root.update_idletasks()
        x = self.gear_button.winfo_rootx() + self.gear_button.winfo_width() - self.settings_menu.winfo_reqwidth()
        y = self.gear_button.winfo_rooty() + self.gear_button.winfo_height()
        self.settings_menu.tk_popup(max(self.root.winfo_vrootx(), x), y)

    def _position_settings_submenu(self, submenu: tk.Menu, index: int) -> None:
        if not submenu.winfo_exists() or not submenu.winfo_ismapped() or not self.settings_menu.winfo_ismapped():
            return
        screen_right = self.root.winfo_vrootx() + self.root.winfo_vrootwidth()
        near_right_edge = self.root.winfo_rootx() + self.root.winfo_width() + submenu.winfo_reqwidth() > screen_right
        if near_right_edge:
            submenu.post(
                max(self.root.winfo_vrootx(), self.settings_menu.winfo_rootx() - submenu.winfo_reqwidth()),
                self.settings_menu.winfo_rooty() + self.settings_menu.yposition(index),
            )

    def _style_top_bar(self) -> None:
        dark = self.interface.dark
        background, badge = ("#1f3250", "#2e4a72") if dark else ("#304b70", "#3d5e87")
        accent = "#3a68a3" if dark else "#31557f"
        self.top_bar.configure(background=background)
        self.banner_heading.configure(background=background)
        for child in self.banner_heading.winfo_children():
            child.configure(background=background)
        self.version_badge.configure(background=badge)
        self.gear_button.configure(background=badge)
        # Match the reference's translucent white secondary button over the banner.
        secondary = "#4c5b73" if dark else "#596f8d"
        for button, color in ((self.speech_button, secondary), (self.cancel_sync_button, secondary), (self.sync_button, accent)):
            button.configure(
                background=color, foreground="white", activebackground=badge,
                activeforeground="white", disabledforeground="#b2bdcc",
            )

    def _save_processing_preferences(self, *_args: object) -> None:
        try:
            persist_choices(self.preferences_path, "processing", self.processing_variables, self.processing_choices)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot save processing settings", str(error), parent=self.root)

    def _save_filter_level(self, *_args: object) -> None:
        level = self.hearing_impaired_strictness.get()
        self.hearing_impaired_value.set(f"{level} / 100")
        if str(level) != self.saved_filter_level.get():
            self.saved_filter_level.set(str(level))

    def _build_settings_menu(self) -> None:
        tr = self.interface.tr
        self.settings_menu.delete(0, "end")
        for child in self.settings_menu.winfo_children():
            child.destroy()
        languages = tk.Menu(self.settings_menu, tearoff=False)
        for code, name in sorted(LANGUAGES.items(), key=lambda item: item[1].casefold()):
            languages.add_radiobutton(
                label=name, variable=self.interface_language, value=code,
                command=self._change_interface_preferences,
            )
        themes = tk.Menu(self.settings_menu, tearoff=False)
        for mode, label in (("system", "Follow the system"), ("light", "Light"), ("dark", "Dark")):
            themes.add_radiobutton(
                label=tr(label), variable=self.interface_theme, value=mode,
                command=self._change_interface_preferences,
            )
        help_menu = tk.Menu(self.settings_menu, tearoff=False)
        help_menu.add_command(label=tr("About Subtitle Sync"), command=self._show_about)
        if self.library_updates.enabled:
            help_menu.add_command(label=tr("Library updates"), command=self.library_updates.open)
        self.settings_menu.add_cascade(label=tr("Language"), menu=languages)
        self.settings_menu.add_cascade(label=tr("Theme"), menu=themes)
        self.settings_menu.add_cascade(label=tr("Help"), menu=help_menu)
        for index, submenu in enumerate((languages, themes, help_menu)):
            submenu.bind(
                "<Map>",
                lambda _event, menu=submenu, number=index: self.root.after_idle(
                    lambda: self._position_settings_submenu(menu, number)
                ),
            )
        self.interface.apply_window(self.settings_menu)

    def _change_interface_preferences(self) -> None:
        language, theme = self.interface_language.get(), self.interface_theme.get()
        try:
            save_settings(self.preferences_path, language, theme)
        except (OSError, ValueError) as error:
            self.interface_language.set(self.interface.language)
            self.interface_theme.set(self.interface.theme)
            messagebox.showerror("Cannot save interface settings", str(error), parent=self.root)
            return
        self.interface.language = language
        self.interface.theme = theme
        self.interface.apply_theme()
        self.interface.apply_window(self.root)
        self._style_top_bar()
        self._translate_instructions()
        self._build_settings_menu()
        self.root.update_idletasks()
        minimum_width = max(
            780, self.top_bar.winfo_reqwidth(),
            self.content_frame.winfo_reqwidth() + self.content_scrollbar.winfo_reqwidth()
        )
        self.root.minsize(minimum_width, self.root.minsize()[1])

    def _refresh_system_theme(self, event: tk.Event) -> None:
        if event.widget == self.root and self.interface.theme == "system":
            self.interface.apply_theme()
            self.interface.apply_window(self.root)
            self._style_top_bar()

    def _show_about(self) -> None:
        messagebox.showinfo(
            self.interface.tr("About Subtitle Sync"),
            f"Subtitle Sync\nv{APP_VERSION}\n\n"
            + self.interface.tr("Synchronize subtitles locally, in your language."),
            parent=self.root,
        )

    @staticmethod
    def _file_row(
        frame: ttk.Frame,
        row: int,
        label: str,
        value: tk.StringVar,
        command: Callable[[], None],
        button: str,
    ) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frame, textvariable=value).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text=button, command=command).grid(
            row=row, column=2, sticky="e", padx=(8, 0), pady=4
        )

    def _update_hearing_impaired_display(self, value: str | None = None) -> None:
        current = int(float(value)) if value is not None else int(self.hearing_impaired_strictness.get())
        self.hearing_impaired_value.set(f"{current} / 100")

    def _update_device_display(self, _event: object = None) -> None:
        device = self.processing_device.get()
        self.device_id_entry.configure(state="disabled" if device == "CPU" else "normal")
        if self._selected_backend() == "Vulkan":
            installed = vulkan_embeddings.ready()
            self.device_help.set(
                "Vulkan (llama.cpp): "
                + ("runtime installed." if installed else "use GPU setup to download the runtime and model.")
                + " Uses your existing Vulkan graphics driver, without ROCm/CUDA. "
                "Use Detect GPUs to choose the actual GPU/APU adapter ID, then Test GPU."
            )
            return
        provider, _ = subtitle_alignment.execution_provider(device)
        private = device != "CPU" and gpu_runtime.runtime_ready(device)
        available = private or provider in subtitle_alignment.ort.get_available_providers()
        self.device_help.set(
            f"Backend: {provider}. "
            + (
                "App-local runtime installed; use Test GPU to verify."
                if private else
                "Runtime available." if available else "Use GPU setup to install the runtime and get drivers."
            )
            + (
                " GPU IDs start at 0; select the adapter ID for your card (see README.md)."
                if device != "CPU" else ""
            )
            + (
                " Integrated GPU selected: use its adapter ID, not a discrete card's. "
                "Older integrated GPUs may not support this backend."
                if device in gpu_runtime.INTEGRATED_DEVICES else ""
            )
        )

    def _selected_backend(self) -> str:
        vendor = gpu_runtime.device_vendor(self.processing_device.get())
        if vendor == "CPU":
            return "ONNX"
        choice = self.processing_backend.get()
        if choice == "Vulkan" or (choice == "Recommended" and vendor in ("AMD", "INTEL")):
            return "Vulkan"
        return "ONNX"

    def _open_speech_to_text(self) -> None:
        if self.busy or self.setup_busy:
            messagebox.showwarning("Work in progress", "Wait for current work to finish.", parent=self.root)
            return
        protected = tuple(
            Path(value.get().strip()) for value in
            (self.portuguese_path, self.english_path, self.output_path)
            if value.get().strip()
        )
        self.setup_busy = True
        try:
            dialog = SpeechDialog(
                self.root, self.video_path.get(), protected,
                lambda path: self.english_path.set(str(path)),
            )
            self.root.wait_window(dialog)
        finally:
            self.setup_busy = False

    def _close(self) -> None:
        if self.busy or self.setup_busy:
            messagebox.showwarning(
                "Work in progress", "Wait for analysis, GPU setup, speech to text, or library updates to finish before closing.",
                parent=self.root,
            )
        else:
            self.root.destroy()

    def _open_driver_page(self, url: str) -> None:
        try:
            if not webbrowser.open(url):
                raise OSError(f"Open this address in your browser:\n{url}")
        except (OSError, webbrowser.Error) as error:
            messagebox.showerror("Cannot open browser", str(error), parent=self.root)

    def _open_gpu_setup(self) -> None:
        if self.busy or self.setup_busy:
            return
        if self._selected_backend() == "Vulkan":
            self._open_vulkan_setup()
            return
        selected_device = self.processing_device.get()
        device = gpu_runtime.device_vendor(selected_device)
        if device == "CPU":
            messagebox.showinfo(
                "Select a graphics card",
                "Select NVIDIA, AMD, INTEL or an integrated GPU/APU first, then open GPU setup.",
                parent=self.root,
            )
            return
        try:
            device_id = int(self.device_id.get())
            subtitle_alignment.execution_provider(device, device_id)
        except ValueError as error:
            messagebox.showerror("Invalid GPU adapter ID", str(error), parent=self.root)
            return

        window = tk.Toplevel(self.root)
        window.title(f"{selected_device} GPU setup")
        make_maximizable(window)
        window.grab_set()
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        linux = gpu_runtime.sys.platform.startswith("linux")
        manual = linux and device == "AMD"
        text = (
            f"1. Install/update your {device} system graphics driver using the official guide below. "
            "On Linux, follow the guide for your distribution. Administrator/root access "
            "and a reboot may be required. The app never installs system drivers.\n\n"
            "2. Install the app-local GPU runtime. Downloads may be large and require internet "
            "access. Packages stay inside this app folder; CPU mode is not modified.\n\n"
            "3. Test the selected adapter. The first test may download the subtitle model. "
            "A successful test means inference works with the selected provider; individual "
            "unsupported operations may still run on CPU."
        )
        if manual:
            text = (
                "AMD Linux: choose a setup stage, your GPU and your distribution below. "
                "Both stages use official AMD software; they are not interchangeable backends. "
                "System installation remains manual. The app does not run root commands."
            )
        if selected_device in gpu_runtime.INTEGRATED_DEVICES:
            text += (
                "\n\nIntegrated GPU/APU: use the adapter ID of the GPU built into your "
                "processor. This is GPU inference, not CPU mode. Driver support varies "
                "by processor generation; having integrated graphics alone is not enough."
            )
        introduction = ttk.Frame(frame)
        introduction.grid(row=0, column=0, sticky="ew")
        introduction.columnconfigure(1, weight=1)
        auto_wrap(ttk.Label(introduction, text=text, wraplength=660)).grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        amd_guide_url = tk.StringVar(value=gpu_runtime.AMD_ROCM_URL)
        amd_controls: list[ttk.Combobox] = []
        if manual:
            setup = tk.StringVar(value=gpu_runtime.AMD_SETUP_OPTIONS[0])
            gpu = tk.StringVar()
            distribution = tk.StringVar()
            version = tk.StringVar()
            detection_note = ""
            try:
                name, release = gpu_runtime.linux_distribution()
                distribution.set(name)
                version.set(release)
            except OSError as error:
                detection_note = f"Could not detect Linux distribution: {error}. Enter it below."
            for row, label, variable, values, state in (
                (1, "Setup stage", setup, gpu_runtime.AMD_SETUP_OPTIONS, "readonly"),
                (2, "GPU model (select or type)", gpu,
                 gpu_runtime.AMD_APU_MODELS if selected_device == "AMD APU (integrated)"
                 else gpu_runtime.AMD_GPU_MODELS, "normal"),
                (3, "Distribution (select or type)", distribution, gpu_runtime.AMD_DISTRIBUTIONS, "normal"),
                (4, "Distribution release", version, (), "normal"),
            ):
                ttk.Label(introduction, text=label).grid(
                    row=row, column=0, sticky="w", padx=(0, 8), pady=3
                )
                control = ttk.Combobox(
                    introduction, textvariable=variable, values=values, state=state, width=48
                )
                control.grid(row=row, column=1, sticky="ew", pady=3)
                amd_controls.append(control)
            if detection_note:
                auto_wrap(ttk.Label(introduction, text=detection_note, wraplength=660)).grid(
                    row=5, column=0, columnspan=2, sticky="ew"
                )
            guidance = tk.Text(introduction, height=8, width=80, wrap="word", state="disabled")
            guidance.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
            guide_scroll = ttk.Scrollbar(introduction, command=guidance.yview)
            guide_scroll.grid(row=6, column=2, sticky="ns")
            guidance.configure(yscrollcommand=guide_scroll.set)

            def update_guidance(_event: object = None) -> None:
                try:
                    content, guide_url = gpu_runtime.amd_linux_guidance(
                        setup.get(), gpu.get(), distribution.get(), version.get()
                    )
                    amd_guide_url.set(guide_url)
                    selected_guide.configure(state="normal")
                except ValueError as error:
                    content = str(error)
                    selected_guide.configure(state="disabled")
                guidance.configure(state="normal")
                guidance.delete("1.0", "end")
                guidance.insert("1.0", content)
                guidance.configure(state="disabled")

            for control in amd_controls:
                control.bind("<<ComboboxSelected>>", update_guidance)
                control.bind("<KeyRelease>", update_guidance)
                control.bind("<FocusOut>", update_guidance)
        links = ttk.Frame(frame)
        links.grid(row=1, column=0, sticky="w", pady=10)
        url = gpu_runtime.LINUX_URLS[device] if linux else gpu_runtime.DRIVER_URLS[device]
        if manual:
            selected_guide = ttk.Button(
                links, text="Open selected setup guide",
                command=lambda: self._open_driver_page(amd_guide_url.get()),
            )
            selected_guide.pack(side="left")
            ttk.Button(
                links, text="Compatibility matrix",
                command=lambda: self._open_driver_page(gpu_runtime.AMD_COMPATIBILITY_URL),
            ).pack(side="left", padx=8)
            ttk.Button(
                links, text="ONNX Runtime guide",
                command=lambda: self._open_driver_page(gpu_runtime.AMD_ONNX_URL),
            ).pack(side="left")
            update_guidance()
        else:
            ttk.Button(
                links, text="1. Official driver guide",
                command=lambda: self._open_driver_page(url),
            ).pack(side="left")
        ttk.Label(frame, text="Private environment (select/copy):").grid(row=2, column=0, sticky="w")
        path = ttk.Entry(frame, width=90)
        path.insert(0, str(gpu_runtime.runtime_directory(device)))
        path.configure(state="readonly")
        path.grid(row=3, column=0, sticky="ew", pady=(4, 8))
        log = tk.Text(frame, height=5 if manual else 12, width=90, wrap="word", state="disabled")
        log.grid(row=4, column=0, sticky="nsew")
        frame.rowconfigure(4, weight=1)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=log.yview)
        scrollbar.grid(row=4, column=1, sticky="ns")
        log.configure(yscrollcommand=scrollbar.set)
        actions = ttk.Frame(frame)
        actions.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        events: queue.Queue[tuple[str, str]] = queue.Queue()

        def append(message: str) -> None:
            log.configure(state="normal")
            log.insert("end", message + "\n")
            log.see("end")
            log.configure(state="disabled")

        def close() -> None:
            if not self.setup_busy:
                window.destroy()

        def poll() -> None:
            try:
                while True:
                    kind, message = events.get_nowait()
                    append(message)
                    if kind in ("success", "error"):
                        self.setup_busy = False
                        self._finish()
                        install.configure(state="disabled" if manual else "normal")
                        test.configure(state="normal")
                        done.configure(state="normal")
                        for index, control in enumerate(amd_controls):
                            control.configure(state="readonly" if index == 0 else "normal")
                        if kind == "error":
                            messagebox.showerror("GPU setup failed", message, parent=window)
                        else:
                            messagebox.showinfo("GPU test complete", message, parent=window)
            except queue.Empty:
                pass
            if self.setup_busy:
                self.root.after(100, poll)

        def start(install_first: bool) -> None:
            if install_first and not messagebox.askyesno(
                "Download GPU runtime?",
                "Download and install GPU packages into the private environment? "
                "This may use several GB of disk space. System drivers will not be changed.",
                parent=window,
            ):
                return
            self.setup_busy = True
            self.sync_button.configure(state="disabled")
            self.setup_button.configure(state="disabled")
            self.device_selector.configure(state="disabled")
            self.backend_selector.configure(state="disabled")
            self.device_id_entry.configure(state="disabled")
            for button in (install, test, done):
                button.configure(state="disabled")
            for control in amd_controls:
                control.configure(state="disabled")

            def work() -> None:
                try:
                    progress = lambda message: events.put(("status", message))
                    if install_first:
                        gpu_runtime.install_runtime(device, progress)
                    result = gpu_runtime.test_runtime(device, device_id, progress)
                    events.put(("success", result))
                except Exception as error:
                    events.put(("error", str(error)))

            threading.Thread(target=work, daemon=True).start()
            self.root.after(100, poll)

        install = ttk.Button(
            actions, text="2. Install / repair GPU support", command=lambda: start(True),
            state="disabled" if manual else "normal",
        )
        install.pack(side="left")
        test = ttk.Button(actions, text="3. Test GPU", command=lambda: start(False))
        test.pack(side="left", padx=8)
        done = ttk.Button(actions, text="Close", command=close)
        done.pack(side="right")
        window.protocol("WM_DELETE_WINDOW", close)
        self.interface.apply_window(window)

    def _open_vulkan_setup(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Vulkan GPU / APU setup")
        make_maximizable(window)
        window.grab_set()
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)
        auto_wrap(ttk.Label(
            frame, wraplength=680,
            text=(
                "Use your existing Vulkan driver with AMD, Intel or NVIDIA GPUs/APUs. "
                "No ROCm, CUDA toolkit or driver reinstall is needed if Vulkan already works.\n\n"
                "Install the app-local llama.cpp runtime and multilingual model (about 275 MB "
                "download), detect adapters, choose yours, then test inference. "
                "Runtime/Vulkan feature compatibility still needs a successful test."
            ),
        )).pack(fill="x")
        adapter = tk.StringVar()
        devices = ttk.Combobox(frame, textvariable=adapter, state="readonly", width=90)
        devices.pack(fill="x", pady=10)
        ttk.Label(frame, text="Detected IDs apply to Vulkan only, not CUDA/DirectML.").pack(anchor="w")
        log = tk.Text(frame, width=90, height=12, wrap="word", state="disabled")
        log.pack(fill="both", expand=True, pady=10)
        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        events: queue.Queue[tuple[str, object]] = queue.Queue()
        adapter_ids: dict[str, int] = {}

        def close() -> None:
            if not self.setup_busy:
                window.destroy()

        def poll() -> None:
            try:
                while True:
                    kind, payload = events.get_nowait()
                    if kind == "devices":
                        adapter_ids.clear()
                        adapter_ids.update({f"{number}: {name}": number for number, name in payload})
                        devices.configure(values=tuple(adapter_ids))
                        selected = next(
                            (name for name, number in adapter_ids.items() if str(number) == self.device_id.get()),
                            next(iter(adapter_ids)),
                        )
                        adapter.set(selected)
                        self.device_id.set(str(adapter_ids[selected]))
                        continue
                    log.configure(state="normal")
                    log.insert("end", str(payload) + "\n")
                    log.see("end")
                    log.configure(state="disabled")
                    if kind in ("done", "error"):
                        self.setup_busy = False
                        self._finish()
                        for control in (install, detect, test, done):
                            control.configure(state="normal")
                        devices.configure(state="readonly")
                        if kind == "error":
                            messagebox.showerror("Vulkan setup failed", str(payload), parent=window)
            except queue.Empty:
                pass
            if self.setup_busy:
                self.root.after(100, poll)

        def start(operation: str) -> None:
            if operation == "install" and not messagebox.askyesno(
                "Download Vulkan support?",
                "Download the pinned official llama.cpp runtime and community GGUF conversion "
                "of the multilingual MiniLM model into this app folder? "
                "Downloads are checksum-verified. Existing system drivers and CPU mode are unchanged.",
                parent=window,
            ):
                return
            if operation == "test" and adapter.get() not in adapter_ids:
                messagebox.showerror("Select a GPU", "Use Detect GPUs and select an adapter first.", parent=window)
                return
            device_id = adapter_ids.get(adapter.get())
            self.setup_busy = True
            for control in (install, detect, test, done, devices, self.sync_button,
                            self.setup_button, self.device_selector, self.device_id_entry,
                            self.backend_selector):
                control.configure(state="disabled")

            def work() -> None:
                try:
                    progress = lambda message: events.put(("status", message))
                    if operation == "install":
                        vulkan_embeddings.install(progress)
                    if operation in ("install", "detect"):
                        found = vulkan_embeddings.list_devices()
                        events.put(("devices", found))
                        message = "Adapters detected. Select your GPU/APU and click Test GPU."
                    else:
                        message = vulkan_embeddings.test_runtime(device_id, progress)
                    events.put(("done", message))
                except Exception as error:
                    events.put(("error", str(error)))

            threading.Thread(target=work, daemon=True).start()
            self.root.after(100, poll)

        install = ttk.Button(actions, text="Install / repair Vulkan support", command=lambda: start("install"))
        install.pack(side="left")
        detect = ttk.Button(actions, text="Detect GPUs", command=lambda: start("detect"))
        detect.pack(side="left", padx=8)
        test = ttk.Button(actions, text="Test GPU", command=lambda: start("test"))
        test.pack(side="left")
        done = ttk.Button(actions, text="Close", command=close)
        done.pack(side="right")
        devices.bind("<<ComboboxSelected>>", lambda _event: self.device_id.set(str(adapter_ids[adapter.get()])))
        window.protocol("WM_DELETE_WINDOW", close)
        self.interface.apply_window(window)

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose video",
            filetypes=[
                ("Video files", "*.mkv *.mp4 *.avi *.mov *.m4v *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_path.set(path)
            if not self.english_path.get().strip():
                self._inspect_video_reference()

    def _inspect_video_reference(self) -> None:
        if self.busy or self.setup_busy or self.english_path.get().strip():
            return
        video = self.video_path.get().strip()
        try:
            signature = input_signatures((video,))
        except OSError as error:
            messagebox.showerror("Cannot read video", str(error), parent=self.root)
            return
        self.embedded_track = None
        self.busy = True
        self.progress.start(12)
        self.status.set("Checking embedded reference subtitle tracks...")

        def inspect() -> None:
            try:
                tracks = subtitle_alignment.list_subtitle_tracks(Path(video))
                if input_signatures((video,)) != signature:
                    raise ValueError("The video changed during track inspection. Choose it again.")
                self.messages.put(("video_tracks", (signature, tracks)))
            except Exception as error:
                self.messages.put(("error", str(error)))

        threading.Thread(target=inspect, daemon=True).start()
        self.root.after(150, self._poll_worker)

    def _use_video_tracks(
        self, signature: tuple[tuple[str, int, int], ...],
        tracks: list[subtitle_alignment.SubtitleTrack],
    ) -> None:
        if self.english_path.get().strip() or input_signatures((self.video_path.get().strip(),)) != signature:
            self.status.set("Inputs changed; embedded track selection was discarded.")
            return
        if not tracks:
            self.status.set("No usable embedded text subtitles found. Generate a reference using speech to text.")
            self._open_speech_to_text()
            return

        def choose(track_id: int) -> None:
            try:
                unchanged = input_signatures((self.video_path.get().strip(),)) == signature
            except OSError as error:
                messagebox.showerror("Cannot read video", str(error), parent=self.root)
                return
            if self.english_path.get().strip() or not unchanged:
                messagebox.showerror("Input changed", "Choose the video again to select its subtitle track.", parent=self.root)
                return
            self.embedded_track = (signature, track_id)
            self._clear_sync_selection()
            self.status.set("Embedded reference track selected. Select matching cues when your source SRT is ready.")

        if len(tracks) == 1:
            choose(tracks[0].track_id)
        else:
            self._show_track_picker(tracks, choose)

    def _choose_portuguese(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose source SRT (any supported language)",
            filetypes=[("SRT subtitles", "*.srt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            cues = subtitle_alignment.read_srt(Path(path))
        except ValueError as error:
            messagebox.showerror("Cannot read source SRT", str(error), parent=self.root)
            return

        self.portuguese_path.set(path)
        self.output_path.set(
            str(Path(path).with_name(f"{Path(path).stem}.synced.srt"))
        )

    def _choose_english(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose correctly timed reference SRT (any supported language)",
            filetypes=[("SRT subtitles", "*.srt"), ("All files", "*.*")],
        )
        if path:
            self.english_path.set(path)

    def _clear_sync_selection(self, *_args: object) -> None:
        self.sync_selection = None
        self.initial_time.set("")
        self.first_cue_text.set(self.interface.tr("Select matching dialogue cues in the two subtitles."))

    def _input_paths(self) -> tuple[str, ...]:
        reference = self.english_path.get().strip()
        return (self.portuguese_path.get().strip(), reference or self.video_path.get().strip())

    def _open_initial_sync(self, track_id: int | None = None) -> None:
        if self.busy or self.setup_busy:
            return
        paths = self._input_paths()
        if not all(paths):
            messagebox.showerror(
                "Choose subtitles", "Choose a source SRT and a reference SRT or video first.",
                parent=self.root,
            )
            return
        try:
            signatures = input_signatures(paths)
        except OSError as error:
            messagebox.showerror("Cannot read input files", str(error), parent=self.root)
            return
        reference_value = self.english_path.get().strip()
        if not reference_value and track_id is None and self.embedded_track is not None:
            video_signature, selected_id = self.embedded_track
            if input_signatures((paths[1],)) == video_signature:
                track_id = selected_id
        self.busy = True
        self.sync_button.configure(state="disabled")
        self.pair_button.configure(state="disabled")
        self.setup_button.configure(state="disabled")
        self.progress.start(12)
        self.status.set("Loading subtitles for manual cue selection...")

        def load() -> None:
            try:
                source = tuple(subtitle_alignment.read_srt(Path(paths[0]).expanduser()))
                if reference_value:
                    reference = tuple(subtitle_alignment.read_srt(Path(reference_value).expanduser()))
                elif track_id is None:
                    tracks = subtitle_alignment.list_subtitle_tracks(Path(paths[1]).expanduser())
                    if input_signatures(paths) != signatures:
                        raise ValueError("An input changed while loading. Select the cues again.")
                    if not tracks:
                        self.messages.put(("speech_reference", None))
                        return
                    if len(tracks) > 1:
                        self.messages.put(("select_track", tracks))
                        return
                    reference = tuple(subtitle_alignment.extract_reference_srt(
                        Path(paths[1]).expanduser(), tracks[0].track_id
                    ))
                else:
                    reference = tuple(subtitle_alignment.extract_reference_srt(
                        Path(paths[1]).expanduser(), track_id
                    ))
                if input_signatures(paths) != signatures:
                    raise ValueError("An input changed while loading. Select the cues again.")
                self.messages.put(("select_pair", (source, reference, signatures)))
            except Exception as error:
                self.messages.put(("error", str(error)))

        threading.Thread(target=load, daemon=True).start()
        self.root.after(150, self._poll_worker)

    def _show_track_picker(
        self, tracks: list[subtitle_alignment.SubtitleTrack],
        accept: Callable[[int], None] | None = None,
    ) -> None:
        try:
            signature = input_signatures((self.video_path.get().strip(),))
        except OSError as error:
            messagebox.showerror("Cannot read video", str(error), parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("Choose embedded reference subtitle track")
        window.transient(self.root)
        window.grab_set()
        ttk.Label(window, text="Choose the correctly timed reference track (any language).").pack(padx=12, pady=10)
        selected = tk.StringVar()
        combo = ttk.Combobox(window, textvariable=selected, state="readonly",
                             values=tuple(track.label for track in tracks), width=75)
        combo.pack(padx=12, pady=10)
        combo.current(0)

        def choose() -> None:
            track_id = tracks[combo.current()].track_id
            window.destroy()
            try:
                unchanged = input_signatures((self.video_path.get().strip(),)) == signature
            except OSError as error:
                messagebox.showerror("Cannot read video", str(error), parent=self.root)
                return
            if not unchanged or self.english_path.get().strip():
                messagebox.showerror("Input changed", "Choose the video again to select its subtitle track.", parent=self.root)
                return
            if accept is not None:
                accept(track_id)
            else:
                self._open_initial_sync(track_id)

        ttk.Button(window, text="Use this track", command=choose).pack(padx=12, pady=10)
        self.interface.apply_window(window)

    def _show_pair_picker(
        self, source: tuple[subtitle_alignment.Cue, ...],
        reference: tuple[subtitle_alignment.Cue, ...],
        signatures: tuple[tuple[str, int, int], ...],
    ) -> None:
        def accept(left: int, right: int) -> None:
            try:
                if input_signatures(self._input_paths()) != signatures:
                    raise ValueError("An input changed while selecting cues. Open the selector again.")
            except (OSError, ValueError) as error:
                self._clear_sync_selection()
                messagebox.showerror("Input changed", str(error), parent=self.root)
                return
            self.sync_selection = SyncSelection(source, reference, left, right, signatures)
            self.initial_time.set(format_time(reference[right - 1].start_ms))
            self.first_cue_text.set(
                f"Source cue {left}: {source[left - 1].text[:120]}\n"
                f"Reference cue {right}: {reference[right - 1].text[:120]}"
            )
            self.status.set("Manual initial sync point selected. Ready to analyze dialogue context.")

        CuePairDialog(self.root, source, reference, accept)

    def _choose_output(self) -> None:
        initial = self.output_path.get()
        path = filedialog.asksaveasfilename(
            title="Save synchronized SRT",
            defaultextension=".srt",
            initialfile=Path(initial).name if initial else "subtitles.synced.srt",
            initialdir=str(Path(initial).parent) if initial else None,
            filetypes=[("SRT subtitles", "*.srt"), ("All files", "*.*")],
        )
        if path:
            self.output_path.set(path)

    def _start_sync(self) -> None:
        if self.busy or self.setup_busy:
            return
        video_value = self.video_path.get().strip()
        video = Path(video_value).expanduser() if video_value else None
        portuguese = Path(self.portuguese_path.get()).expanduser()
        output = Path(self.output_path.get()).expanduser()
        if not portuguese.is_file():
            messagebox.showwarning(
                "Choose a source SRT", "Choose a valid source SRT file.", parent=self.root
            )
            return
        if not self.output_path.get():
            messagebox.showwarning(
                "Choose an output", "Choose where to save the synchronized SRT.", parent=self.root
            )
            return
        if portuguese.resolve() == output.resolve():
            messagebox.showerror(
                "Choose a different output",
                "The synchronized subtitle must be saved separately; the original is kept unchanged.",
                parent=self.root,
            )
            return
        if self.sync_selection is None:
            messagebox.showwarning(
                "Select the initial sync point",
                "Click Select matching cues and choose corresponding dialogue in both subtitles.",
                parent=self.root,
            )
            return
        selection = self.sync_selection
        try:
            if input_signatures(self._input_paths()) != selection.inputs:
                self._clear_sync_selection()
                raise ValueError("An input file changed. Select matching cues again.")
        except (OSError, ValueError) as error:
            messagebox.showerror("Initial sync point is outdated", str(error), parent=self.root)
            return
        initial_audio_ms = selection.reference[selection.reference_number - 1].start_ms

        processing_device = self.processing_device.get()
        backend = self._selected_backend()
        try:
            device_id = 0 if processing_device == "CPU" else int(self.device_id.get())
            subtitle_alignment.execution_provider(processing_device, device_id)
            if backend == "Vulkan":
                if not vulkan_embeddings.ready():
                    raise ValueError("Open GPU setup and install Vulkan support first.")
            elif processing_device == "CPU" or not gpu_runtime.runtime_ready(processing_device):
                subtitle_alignment.validate_processing_device(processing_device, device_id)
        except ValueError as error:
            messagebox.showerror("Processing device unavailable", str(error), parent=self.root)
            return

        english_value = self.english_path.get().strip()
        english_path = Path(english_value).expanduser() if english_value else None
        if english_path is not None and not english_path.is_file():
            messagebox.showerror(
                "Reference SRT not found",
                "Choose a valid reference SRT, or leave this field blank to use "
                "subtitles embedded in the video.",
                parent=self.root,
            )
            return
        if english_path is not None and english_path.resolve() == output.resolve():
            messagebox.showerror(
                "Choose a different output", "Output must not overwrite the reference SRT.",
                parent=self.root,
            )
            return
        if english_path is None and (video is None or not video.is_file()):
            messagebox.showwarning(
                "Choose a video",
                "Choose a valid video file when no reference SRT is provided.",
                parent=self.root,
            )
            return
        if output.exists() and not messagebox.askyesno(
            "Replace output?",
            f"This file already exists. Replace it?\n\n{output}",
            parent=self.root,
        ):
            return

        self.busy = True
        self.sync_button.configure(state="disabled")
        self.device_selector.configure(state="disabled")
        self.device_id_entry.configure(state="disabled")
        self.setup_button.configure(state="disabled")
        self.backend_selector.configure(state="disabled")
        self.progress.start(12)
        self.pair_button.configure(state="disabled")
        self.sync_cancel.clear()
        self.sync_committed = False
        self.cancel_sync_button.configure(state="normal")
        self.status.set("Starting local subtitle analysis...")
        worker = threading.Thread(
            target=self._sync_worker,
            args=(
                video,
                portuguese,
                english_path,
                output,
                initial_audio_ms,
                int(self.hearing_impaired_strictness.get()),
                processing_device,
                device_id,
            ),
            kwargs={"backend": backend, "selection": selection},
            daemon=True,
        )
        worker.start()
        self.root.after(150, self._poll_worker)

    def _cancel_sync(self) -> None:
        with self.sync_commit_lock:
            if self.sync_committed:
                return
            self.sync_cancel.set()
        self.cancel_sync_button.configure(state="disabled")
        self.status.set("Cancelling synchronization... waiting for the current inference step.")

    def _sync_worker(
        self,
        video: Path | None,
        portuguese_path: Path,
        english_path: Path | None,
        output: Path,
        initial_audio_ms: int,
        hearing_impaired_strictness: int,
        processing_device: str = "CPU",
        device_id: int = 0,
        backend: str = "ONNX",
        selection: SyncSelection | None = None,
    ) -> None:
        try:
            self.messages.put(("status", "Reading source subtitles..."))
            original_bytes = portuguese_path.read_bytes()
            portuguese_text, encoding, had_bom = sync_subtitles.decode_srt(original_bytes)
            portuguese = subtitle_alignment.parse_srt(portuguese_text)

            if selection is not None:
                if tuple(portuguese) != selection.source:
                    raise ValueError("The source subtitles changed. Select matching cues again.")
                paths = (str(portuguese_path), str(english_path or video or ""))
                if input_signatures(paths) != selection.inputs:
                    raise ValueError("An input changed. Select matching cues again.")
                english = list(selection.reference)
            elif english_path is not None:
                self.messages.put(("status", "Reading the selected reference SRT..."))
                english = subtitle_alignment.read_srt(english_path)
            else:
                raise ValueError("Select matching cues and an embedded reference track before analysis.")

            def progress(message: str) -> None:
                if self.sync_cancel.is_set():
                    raise gpu_runtime.CancelledError("Synchronization cancelled. No output was written.")
                self.messages.put(("status", message))

            progress("Starting subtitle analysis...")
            anchor_options = (
                {"source_anchor": selection.source_number, "reference_anchor": selection.reference_number}
                if selection is not None else {}
            )
            if backend == "Vulkan":
                anchors, diagnostics = subtitle_alignment.analyze_subtitles(
                    portuguese, english, initial_audio_ms, progress=progress,
                    hearing_impaired_strictness=hearing_impaired_strictness,
                    processing_device=processing_device, device_id=device_id, backend=backend,
                    **anchor_options,
                )
            elif processing_device != "CPU" and gpu_runtime.runtime_ready(processing_device):
                result = gpu_runtime.run_worker(
                    processing_device, device_id,
                    {
                        "operation": "analyze",
                        "source": [asdict(cue) for cue in portuguese],
                        "reference": [asdict(cue) for cue in english],
                        "initial_audio_ms": initial_audio_ms,
                        "strictness": hearing_impaired_strictness,
                        **anchor_options,
                    },
                    progress,
                    cancel=self.sync_cancel,
                )
                anchors, diagnostics = result["anchors"], result["diagnostics"]
            else:
                anchors, diagnostics = subtitle_alignment.analyze_subtitles(
                    portuguese, english, initial_audio_ms, progress=progress,
                    hearing_impaired_strictness=hearing_impaired_strictness,
                    processing_device=processing_device, device_id=device_id,
                    **anchor_options,
                )
            if len(portuguese) != len(sync_subtitles.TIMESTAMP.findall(portuguese_text)):
                raise ValueError("The source SRT changed during analysis. Please try again.")
            if selection is not None and input_signatures(paths) != selection.inputs:
                raise ValueError("An input changed during analysis. No output was written.")
            if english_path is not None and output.resolve() == english_path.resolve():
                raise ValueError("Output must not overwrite the reference subtitle.")
            synced_text, cue_count = sync_subtitles.sync_srt(portuguese_text, anchors)
            encoded = synced_text.encode(encoding)
            if had_bom:
                encoded = b"\xef\xbb\xbf" + encoded
            progress("Saving synchronized subtitles...")
            # A private directory keeps staging private while normal creation honors the umask.
            with tempfile.TemporaryDirectory(dir=output.parent, prefix=".subtitle-sync-") as staging_directory:
                staged_path = Path(staging_directory) / output.name
                staged_path.write_bytes(encoded)
                with self.sync_commit_lock:
                    if self.sync_cancel.is_set():
                        raise gpu_runtime.CancelledError("Synchronization cancelled. No output was written.")
                    staged_path.replace(output)
                    self.sync_committed = True
            self.messages.put(("success", (output, cue_count, diagnostics)))
        except gpu_runtime.CancelledError as error:
            self.messages.put(("cancelled", str(error)))
        except Exception as error:
            if self.sync_cancel.is_set():
                self.messages.put(("cancelled", "Synchronization cancelled. No output was written."))
            else:
                self.messages.put(("error", str(error)))

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "status":
                    if not self.sync_cancel.is_set():
                        self.status.set(str(payload))
                elif kind == "cancelled":
                    self._finish()
                    self.status.set(str(payload))
                elif kind == "select_track":
                    self._finish()
                    self._show_track_picker(payload)
                elif kind == "video_tracks":
                    self._finish()
                    try:
                        self._use_video_tracks(*payload)
                    except OSError as error:
                        messagebox.showerror("Cannot read video", str(error), parent=self.root)
                elif kind == "speech_reference":
                    self._finish()
                    if not self.english_path.get().strip():
                        self._open_speech_to_text()
                elif kind == "select_pair":
                    self._finish()
                    self._show_pair_picker(*payload)
                elif kind == "error":
                    self._finish()
                    messagebox.showerror("Synchronization failed", str(payload), parent=self.root)
                    self.status.set("Synchronization failed.")
                elif kind == "success":
                    output, cue_count, diagnostics = payload
                    self._finish()
                    self.status.set(f"Saved synchronized subtitles: {output}")
                    messagebox.showinfo(
                        "Synchronization complete",
                        f"Matched {diagnostics['matched_cues']} dialogue cues "
                        f"({diagnostics['match_percent']}%) automatically.\n"
                        f"Synchronized {cue_count} subtitle cues and saved:\n{output}\n\n"
                        "The source and reference subtitles were not changed.",
                        parent=self.root,
                    )
        except queue.Empty:
            pass

        if self.busy:
            self.root.after(150, self._poll_worker)

    def _finish(self) -> None:
        self.busy = False
        self.sync_cancel.clear()
        self.cancel_sync_button.configure(state="disabled")
        self.progress.stop()
        self.sync_button.configure(state="normal")
        self.device_selector.configure(state="readonly")
        self.backend_selector.configure(state="readonly")
        self.setup_button.configure(state="normal")
        self.pair_button.configure(state="normal")
        self._update_device_display()


def main() -> None:
    root = tk.Tk()
    SubtitleSyncApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
