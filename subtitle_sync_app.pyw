"""Windows and Linux GUI for context-aware multilingual subtitle alignment."""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import tempfile
import threading
import tkinter as tk
import webbrowser
from dataclasses import asdict
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

import subtitle_alignment
import audio_sync
import sync_subtitles
from sync_report import write_report
import gpu_runtime
import vulkan_embeddings
import file_picker as filedialog
from desktop_updates import DesktopUpdates
from manual_sync import CuePairDialog, SyncSelection, input_signatures
from speech_dialog import SpeechDialog
from speech_to_text import LANGUAGES as SPEECH_LANGUAGES
from audio_sync_dialog import AudioSyncDialog
from ui_preferences import (
    APP_VERSION, InterfacePreferences, load_settings, save_settings, settings_path,
    load_choices, persist_choices, auto_wrap, make_maximizable,
    TranslatedVar, install_message_translation,
)
from ui_translations import LANGUAGES, instructions as translated_instructions, translate


def subtitle_name_parts(path: Path) -> tuple[str, str]:
    languages = set(SPEECH_LANGUAGES) - {"Auto detect"}
    languages.update(SPEECH_LANGUAGES.values())
    languages.update((
        "eng", "por", "spa", "fra", "fre", "deu", "ger", "ita", "nld", "dut",
        "ell", "gre", "pol", "rus", "ukr", "jpn", "kor", "zho", "chi", "ara",
        "hin", "tur", "swe", "dan", "nor", "fin", "ces", "cze", "ron", "rum",
    ))
    languages.discard("auto")
    languages.update(LANGUAGES.values())
    names = "|".join(re.escape(name) for name in sorted(languages, key=len, reverse=True))
    codes = "|".join(re.escape(code) for code in sorted(set(SPEECH_LANGUAGES.values())))
    match = re.fullmatch(
        rf"(.+?)[. _-]+({names}|(?:{codes})[-_](?:[a-z]{{2}}|[a-z]{{4}})(?:[-_][a-z]{{2}})?)",
        path.stem, re.IGNORECASE,
    )
    if match:
        return match[1].casefold(), match[2].casefold().replace("_", "-")
    return path.stem.casefold(), ""


def reference_candidates(source: Path, output: Path) -> list[Path]:
    base, language = subtitle_name_parts(source)
    candidates = []
    for path in source.parent.iterdir():
        if path.suffix.lower() != ".srt" or not path.is_file():
            continue
        if path.samefile(source) or path.resolve() == output.resolve():
            continue
        candidate_base, candidate_language = subtitle_name_parts(path)
        if candidate_base == base and candidate_language != language:
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.name.casefold())


def folder_subtitles(folder: Path) -> dict[str, dict[str, list[Path]]]:
    aliases = {name.casefold(): code for name, code in SPEECH_LANGUAGES.items() if code != "auto"}
    aliases.update({
        "portuguese": "pt", "português": "pt", "english": "en", "eng": "en", "por": "pt",
        "fra": "fr", "fre": "fr", "deu": "de", "ger": "de", "spa": "es", "ita": "it",
        "nld": "nl", "dut": "nl", "pt-pt": "pt-pt", "pt-br": "pt-br",
    })
    result: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if path.suffix.lower() != ".srt" or not path.is_file():
            continue
        base, language = subtitle_name_parts(path)
        if language:
            result.setdefault(base, {}).setdefault(aliases.get(language, language), []).append(path)
    return result


def configure_desktop_identity() -> None:
    if sys.platform == "win32":
        import ctypes

        set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_app_id.argtypes = [ctypes.c_wchar_p]
        set_app_id.restype = ctypes.c_long
        if set_app_id("SubSync.Desktop") < 0:
            raise OSError("Windows could not set the SubSync taskbar identity.")
    elif sys.platform.startswith("linux") and not (
        Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
    ):
        directory = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "applications"
        app = Path(__file__).resolve().parent

        def escape(value: str) -> str:
            return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

        launcher = str(app / "Start_Subtitle_Sync.sh")
        # Exec has an additional quoting layer and treats percent signs as field codes.
        argument = "".join("\\" + char if char in '\\`"$' else char for char in launcher)
        content = (
            "[Desktop Entry]\nType=Application\nName=SubSync\n"
            f'Exec=/bin/bash "{escape(argument).replace("%", "%%")}"\n'
            f"Icon={escape(str(app / 'subtitle_sync_icon.png'))}\n"
            "StartupWMClass=Subsync\nTerminal=false\nCategories=AudioVideo;\n"
        )
        directory.mkdir(parents=True, exist_ok=True)
        entry = directory / "SubSync.desktop"
        if not entry.exists() or entry.read_text(encoding="utf-8") != content:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".desktop", delete=False) as temporary:
                staged = Path(temporary.name)
            try:
                staged.write_text(content, encoding="utf-8")
                staged.chmod(0o644)
                staged.replace(entry)
            finally:
                staged.unlink(missing_ok=True)


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
        root.title("SubSync")
        root.iconname("SubSync")
        root.tk.call("tk", "appname", "SubSync")
        self._window_icon = tk.PhotoImage(master=root, file=str(Path(__file__).with_name("subtitle_sync_icon.png")))
        root.iconphoto(True, self._window_icon)
        root.geometry("780x620")
        root.minsize(720, 600)

        self.video_path = tk.StringVar()
        self.source_path = tk.StringVar()
        self.reference_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.initial_time = tk.StringVar()
        self.processing_device = tk.StringVar(value="CPU")
        self.processing_backend = tk.StringVar(value="Recommended")
        self.device_id = tk.StringVar(value="0")
        self.device_help = tk.StringVar()
        self.report_enabled = tk.StringVar(value="0")
        self.audio_after_sync = tk.StringVar(value="0")
        self.final_offset_ms = tk.StringVar(value="0")
        self.automatic_initial = tk.StringVar(value="0")
        self.batch_automatic_initial = False
        self.folder_path = tk.StringVar()
        self.folder_language_text = tk.StringVar()
        self.folder_source = ""
        self.folder_reference = ""
        self.audio_folder_path = tk.StringVar()
        self.batch_running = False
        self.hearing_impaired_strictness = tk.IntVar(value=70)
        self.saved_filter_level = tk.StringVar(value="70")
        self.hearing_impaired_value = tk.StringVar(value="70 / 100")
        self.first_cue_text = tk.StringVar(value="Select matching dialogue cues in the two subtitles.")
        self.status = TranslatedVar(
            root, lambda text: self.interface.tr(text) if hasattr(self, "interface") else text,
            "Choose source and reference subtitles in any supported languages.",
        )
        self.sync_selection: SyncSelection | None = None
        self.folder_pair_dialog: CuePairDialog | None = None
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
            "report": self.report_enabled,
            "audio_after_sync": self.audio_after_sync,
            "automatic_initial": self.automatic_initial,
        }
        self.processing_choices = {
            "device": subtitle_alignment.PROCESSING_DEVICES,
            "backend": ("Recommended", "ONNX (advanced)") if gpu_runtime.CPU_ONLY else ("Recommended", "Vulkan", "ONNX (advanced)"),
            "filter_level": tuple(str(level) for level in range(101)),
            "report": ("0", "1"),
            "audio_after_sync": ("0", "1"),
            "automatic_initial": ("0", "1"),
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
        install_message_translation(self.interface)
        self.interface.apply_theme()
        self.interface_language = tk.StringVar(value=language)
        self.interface_theme = tk.StringVar(value=theme)
        self._last_interface_language = "en"

        self._build_ui()
        self.interface.apply_window(root)
        self._style_top_bar()
        self._translate_instructions()
        root.bind("<FocusIn>", self._refresh_system_theme, add="+")
        for value in (self.source_path, self.reference_path, self.video_path):
            value.trace_add("write", self._clear_sync_selection)
        for value in self.processing_variables.values():
            value.trace_add("write", self._save_processing_preferences)
        self.hearing_impaired_strictness.trace_add("write", self._save_filter_level)
        self.automatic_initial.trace_add("write", self._update_initial_mode)
        self.folder_path.trace_add("write", self._folder_path_changed)
        self._update_initial_mode()
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

        files = ttk.Frame(frame)
        files.grid(row=0, column=0, columnspan=3, sticky="ew")
        files.columnconfigure(0, weight=1, uniform="file-input")
        files.columnconfigure(1, weight=1, uniform="file-input")
        self.video_file_group = ttk.Frame(files)
        self.reference_file_group = ttk.Frame(files)
        self.source_file_group = ttk.Frame(files)
        self.output_file_group = ttk.Frame(files)
        for group, row, column in (
            (self.video_file_group, 0, 0), (self.reference_file_group, 0, 1),
            (self.source_file_group, 1, 0), (self.output_file_group, 1, 1),
        ):
            group.grid(row=row, column=column, sticky="ew", padx=(0, 10) if column == 0 else 0)
            group.columnconfigure(1, weight=1)
        self._file_row(
            self.video_file_group, 0, "Video (optional)", self.video_path, self._choose_video, "Browse..."
        )
        self._file_row(
            self.reference_file_group, 0, "Reference SRT (optional)",
            self.reference_path, self._choose_reference, "Browse...",
        )
        self._file_row(
            self.source_file_group, 0, "Source SRT (to synchronize)",
            self.source_path, self._choose_source, "Browse...",
        )
        self._file_row(
            self.output_file_group, 0, "Save synchronized SRT",
            self.output_path, self._choose_output, "Save as...",
        )

        processing_options = ttk.Frame(frame)
        processing_options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        processing_options.columnconfigure(0, weight=1)
        processing = ttk.LabelFrame(processing_options, text="Processing device", padding=9)
        processing.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.report_group = ttk.LabelFrame(processing_options, text="Report and extra", padding=9)
        self.report_group.grid(row=0, column=1, sticky="nsew")
        self.report_checkbox = ttk.Checkbutton(
            self.report_group, text="Generate sync report", variable=self.report_enabled,
            onvalue="1", offvalue="0",
        )
        self.report_checkbox.grid(row=0, column=0, sticky="w")
        self.audio_after_checkbox = ttk.Checkbutton(
            self.report_group, text="Sync to audio after analysis", variable=self.audio_after_sync,
            onvalue="1", offvalue="0",
        )
        self.audio_after_checkbox.grid(row=1, column=0, sticky="w", pady=(8, 0))
        auto_wrap(ttk.Label(
            self.report_group, text="Video required; uses the first audio track.", wraplength=240,
        )).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        offset_box = ttk.LabelFrame(self.report_group, text="Final offset (ms)", padding=6)
        offset_box.grid(row=0, column=1, rowspan=3, sticky="ns", padx=(10, 0))
        self.final_offset_entry = ttk.Entry(offset_box, textvariable=self.final_offset_ms, width=9)
        self.final_offset_entry.pack(anchor="w")
        ttk.Label(offset_box, text="+ delay / - advance").pack(anchor="w", pady=(4, 0))
        processing.columnconfigure(3, weight=1)
        self.device_selector = ttk.Combobox(
            processing, textvariable=self.processing_device,
            values=subtitle_alignment.PROCESSING_DEVICES, state="readonly", width=16,
        )
        self.device_selector.grid(row=0, column=0, sticky="w")
        self.device_selector.bind("<<ComboboxSelected>>", self._update_device_display)
        ttk.Label(processing, text="GPU adapter ID").grid(row=0, column=1, padx=(8, 4))
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
            values=self.processing_choices["backend"], state="readonly", width=16,
        )
        self.backend_selector.pack(side="left")
        self.backend_selector.bind("<<ComboboxSelected>>", self._update_device_display)
        auto_wrap(ttk.Label(processing, textvariable=self.device_help, wraplength=480)).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0)
        )
        self._update_device_display()

        sync_options = ttk.Frame(frame)
        sync_options.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        sync_options.columnconfigure(0, weight=1)
        sync_options.columnconfigure(1, weight=1)
        sync_options.columnconfigure(2, weight=1)
        sync_options.columnconfigure(3, weight=1)
        self.folder_group = ttk.LabelFrame(sync_options, text="SRT folder mode", padding=9)
        self.folder_group.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.folder_group.columnconfigure(0, weight=1)
        self.folder_entry = ttk.Entry(self.folder_group, textvariable=self.folder_path, width=24)
        self.folder_entry.grid(row=0, column=0, sticky="ew")
        self.folder_entry.bind("<Return>", self._refresh_folder)
        self.folder_entry.bind("<FocusOut>", self._refresh_folder)
        self.folder_browse = ttk.Button(self.folder_group, text="Browse...", command=self._choose_folder)
        self.folder_browse.grid(row=0, column=1, padx=(6, 0))
        self.folder_source_combo = ttk.Combobox(self.folder_group, state="readonly", width=12)
        self.folder_reference_combo = ttk.Combobox(self.folder_group, state="readonly", width=12)
        for row, (caption, combo) in enumerate((
            ("Source SRT (to synchronize)", self.folder_source_combo),
            ("Reference SRT", self.folder_reference_combo),
        ), 1):
            ttk.Label(self.folder_group, text=caption).grid(row=row, column=0, sticky="w", pady=(6, 0))
            combo.grid(row=row, column=1, sticky="ew", pady=(6, 0))
            combo.bind("<<ComboboxSelected>>", self._folder_languages_changed)
        self.audio_folder_group = ttk.LabelFrame(sync_options, text="Sync to audio folder mode", padding=9)
        self.audio_folder_group.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        self.audio_folder_group.columnconfigure(0, weight=1)
        self.audio_folder_entry = ttk.Entry(self.audio_folder_group, textvariable=self.audio_folder_path, width=24)
        self.audio_folder_entry.grid(row=0, column=0, sticky="ew")
        self.audio_folder_browse = ttk.Button(
            self.audio_folder_group, text="Browse...", command=self._choose_audio_folder,
        )
        self.audio_folder_browse.grid(row=0, column=1, padx=(6, 0))
        self.audio_folder_sync = ttk.Button(
            self.audio_folder_group, text="Sync", command=lambda: self._start_folder_sync(audio_only=True),
        )
        self.audio_folder_sync.grid(row=1, column=0, sticky="w", pady=(6, 0))
        hearing = ttk.LabelFrame(sync_options, text="Hearing-impaired filter", padding=9)
        hearing.grid(row=0, column=3, sticky="nsew")
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
        self.hearing_scale = scale
        scale.configure(takefocus=True)
        scale.bind("<Button-1>", lambda _event: scale.focus_set(), add="+")
        for key, delta in (("Left", -1), ("Down", -1), ("Right", 1), ("Up", 1)):
            scale.bind(f"<{key}>", lambda _event, step=delta: self._step_filter(step))
        ttk.Label(hearing, text="Strict").grid(row=0, column=2, sticky="e")
        ttk.Label(hearing, textvariable=self.hearing_impaired_value).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        self._update_hearing_impaired_display(str(self.hearing_impaired_strictness.get()))

        anchor = ttk.LabelFrame(sync_options, text="Initial sync point", padding=9)
        anchor.grid(row=0, column=2, sticky="nsew", padx=(0, 10))
        anchor.columnconfigure(1, weight=1)
        self.pair_button = ttk.Button(
            anchor, text="Select matching cues...", command=self._open_initial_sync
        )
        self.pair_button.grid(row=0, column=0, sticky="w")
        self.automatic_initial_checkbox = ttk.Checkbutton(
            anchor, text="Automatically select initial sync point", variable=self.automatic_initial,
            onvalue="1", offvalue="0",
        )
        self.automatic_initial_checkbox.grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))
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
            "- Press Analyze and synchronize.\n\n"
            "Example 3: Synchronize existing subtitles using the audio\n"
            "- Open Sync to audio in the top bar, to the left of Speech to text.\n"
            "- Choose the video/audio file and press Inspect tracks.\n"
            "- Select the correct audio track. Choose Embedded subtitles and a text subtitle "
            "track, or External SRT and your out-of-sync SRT.\n"
            "- Leave gradual drift and section-by-section correction enabled when needed.\n"
            "- Choose a separate output SRT and press Synchronize to audio. This runs on CPU; "
            "no transcription or correctly timed reference is needed.\n"
            "- Review the result and any section boundaries listed in the log. The saved SRT "
            "is automatically placed in the Reference SRT box.\n"
            "- If you only needed audio synchronization, you are done. To synchronize another "
            "Source SRT against this reference, select matching cues and press Analyze and synchronize.",
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
        self.audio_sync_button = tk.Button(
            banner, text="Sync to audio...", command=self._open_audio_sync,
            font=("TkDefaultFont", 10), relief="flat", borderwidth=0,
            padx=10, pady=9, takefocus=True,
        )
        self.audio_sync_button.pack(side="right", padx=(0, 10))
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
        for button, color in (
            (self.audio_sync_button, secondary), (self.speech_button, secondary),
            (self.cancel_sync_button, secondary), (self.sync_button, accent),
        ):
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
        self._update_folder_languages()
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
            + self.interface.tr("Synchronize subtitles locally, in your language.")
            + "\n\n" + self.interface.tr("developed by SMAP"),
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
            (self.source_path, self.reference_path, self.output_path)
            if value.get().strip()
        )
        self.setup_busy = True
        try:
            dialog = SpeechDialog(
                self.root, self.video_path.get(), protected,
                lambda path: self.reference_path.set(str(path)),
            )
            self.root.wait_window(dialog)
        finally:
            self.setup_busy = False

    def _open_audio_sync(self) -> None:
        if self.busy or self.setup_busy:
            messagebox.showwarning("Work in progress", "Wait for current work to finish.", parent=self.root)
            return
        protected = tuple(
            Path(value.get().strip()) for value in
            (self.source_path, self.reference_path, self.output_path)
            if value.get().strip()
        )
        self.setup_busy = True
        try:
            dialog = AudioSyncDialog(
                self.root, self.video_path.get(), self.source_path.get(), protected,
                on_saved=lambda path: self.reference_path.set(str(path)),
                generate_report=self.report_enabled.get() == "1",
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
        window.title(self.interface.tr("{device} GPU setup").format(device=selected_device))
        make_maximizable(window)
        window.grab_set()
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        linux = gpu_runtime.sys.platform.startswith("linux")
        manual = linux and device == "AMD"
        text = self.interface.tr(
            "1. Install/update your {device} system graphics driver using the official guide below. "
            "On Linux, follow the guide for your distribution. Administrator/root access "
            "and a reboot may be required. The app never installs system drivers.\n\n"
            "2. Install the app-local GPU runtime. Downloads may be large and require internet "
            "access. Packages stay inside this app folder; CPU mode is not modified.\n\n"
            "3. Test the selected adapter. The first test may download the subtitle model. "
            "A successful test means inference works with the selected provider; individual "
            "unsupported operations may still run on CPU."
        ).format(device=device)
        if manual:
            text = self.interface.tr(
                "AMD Linux: choose a setup stage, your GPU and your distribution below. "
                "Both stages use official AMD software; they are not interchangeable backends. "
                "System installation remains manual. The app does not run root commands."
            )
        if selected_device in gpu_runtime.INTEGRATED_DEVICES:
            text += "\n\n" + self.interface.tr(
                "Integrated GPU/APU: use the adapter ID of the GPU built into your "
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
                detection_note = self.interface.tr(
                    "Could not detect Linux distribution: {error}. Enter it below."
                ).format(error=error)
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
            if not self.reference_path.get().strip():
                self._inspect_video_reference()

    def _inspect_video_reference(self) -> None:
        if self.busy or self.setup_busy or self.reference_path.get().strip():
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
        if self.reference_path.get().strip() or input_signatures((self.video_path.get().strip(),)) != signature:
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
            if self.reference_path.get().strip() or not unchanged:
                messagebox.showerror("Input changed", "Choose the video again to select its subtitle track.", parent=self.root)
                return
            self.embedded_track = (signature, track_id)
            self._clear_sync_selection()
            self.status.set("Embedded reference track selected. Select matching cues when your source SRT is ready.")

        if len(tracks) == 1:
            choose(tracks[0].track_id)
        else:
            self._show_track_picker(tracks, choose)

    def _choose_source(self) -> None:
        path = filedialog.askopenfilename(
            title=self.interface.tr("Choose source SRT (any supported language)"),
            filetypes=[(self.interface.tr("SRT subtitles"), "*.srt"), (self.interface.tr("All files"), "*.*")],
        )
        if not path:
            return
        try:
            cues = subtitle_alignment.read_srt(Path(path))
        except ValueError as error:
            messagebox.showerror(self.interface.tr("Cannot read source SRT"), str(error), parent=self.root)
            return

        self.source_path.set(path)
        self.output_path.set(
            str(Path(path).with_name(f"{Path(path).stem}.synced.srt"))
        )
        try:
            candidates = reference_candidates(Path(path), Path(self.output_path.get()))
        except OSError as error:
            messagebox.showerror(self.interface.tr("Cannot find reference subtitles"), str(error), parent=self.root)
            return
        if candidates:
            self._offer_reference_candidates(candidates)

    def _offer_reference_candidates(self, candidates: list[Path]) -> None:
        self._offer_subtitle_candidates(candidates, source=False)

    def _offer_subtitle_candidates(self, candidates: list[Path], *, source: bool) -> None:
        selected_input = self.reference_path if source else self.source_path
        original_input = selected_input.get()
        target = self.source_path if source else self.reference_path

        def accept(path: Path) -> bool:
            if selected_input.get() != original_input:
                messagebox.showerror(
                    self.interface.tr("Input changed"), self.interface.tr("Select the input again."),
                    parent=self.root,
                )
                return False
            try:
                subtitle_alignment.read_srt(path)
            except ValueError as error:
                messagebox.showerror(
                    self.interface.tr("Cannot read source SRT" if source else "Cannot read reference SRT"),
                    str(error), parent=self.root,
                )
                return False
            target.set(str(path))
            if source:
                self.output_path.set(str(path.with_name(f"{path.stem}.synced.srt")))
            return True

        window = tk.Toplevel(self.root)
        window.title("Choose source SRT" if source else "Choose reference SRT")
        make_maximizable(window)
        window.transient(self.root)
        window.grab_set()
        window.columnconfigure(0, weight=1)
        auto_wrap(ttk.Label(
            window, text=(
                "Other language versions were found. Choose the source to synchronize, "
                "or cancel to keep the current source."
                if source else
                "Other language versions were found. Choose a correctly timed reference, "
                "or cancel to keep the current reference."
            ), wraplength=600,
        )).grid(row=0, column=0, sticky="ew", padx=12, pady=10)
        combo = ttk.Combobox(
            window, state="readonly", values=tuple(path.name for path in candidates), width=80,
        )
        combo.grid(row=1, column=0, sticky="ew", padx=12, pady=10)
        combo.current(0)

        def choose() -> None:
            if accept(candidates[combo.current()]):
                window.destroy()

        actions = ttk.Frame(window)
        actions.grid(row=2, column=0, sticky="e", padx=12, pady=10)
        ttk.Button(
            actions, text="Use this source" if source else "Use this reference", command=choose,
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Cancel", command=window.destroy).grid(row=0, column=1)
        window.bind("<Escape>", lambda _event: window.destroy())
        self.interface.apply_window(window)

    def _choose_reference(self) -> None:
        path = filedialog.askopenfilename(
            title=self.interface.tr("Choose correctly timed reference SRT (any supported language)"),
            filetypes=[(self.interface.tr("SRT subtitles"), "*.srt"), (self.interface.tr("All files"), "*.*")],
        )
        if path:
            self.reference_path.set(path)
            try:
                candidates = reference_candidates(Path(path), Path(self.output_path.get()))
            except OSError as error:
                messagebox.showerror(self.interface.tr("Cannot find source subtitles"), str(error), parent=self.root)
                return
            if candidates:
                self._offer_subtitle_candidates(candidates, source=True)

    def _step_filter(self, step: int) -> str:
        self.hearing_impaired_strictness.set(max(0, min(
            100, round(self.hearing_impaired_strictness.get()) + step,
        )))
        self._update_hearing_impaired_display(str(self.hearing_impaired_strictness.get()))
        return "break"

    def _clear_folder(self) -> None:
        if self.busy:
            return
        self.folder_path.set("")
        self.folder_source = self.folder_reference = ""
        self.folder_language_text.set("")
        self._update_initial_mode()

    def _update_folder_languages(self) -> None:
        self.folder_language_text.set(
            f"{self.interface.tr('Source')}: {self.folder_source}\n"
            f"{self.interface.tr('Reference')}: {self.folder_reference}"
            if self.folder_path.get() else ""
        )
        self.folder_source_combo.set(self.folder_source)
        self.folder_reference_combo.set(self.folder_reference)

    def _folder_path_changed(self, *_args: object) -> None:
        if not self.folder_path.get().strip():
            self.folder_source = self.folder_reference = ""
            self.folder_source_combo.configure(values=())
            self.folder_reference_combo.configure(values=())
            self._update_folder_languages()
        self._update_initial_mode()

    def _folder_languages_changed(self, _event: object = None) -> None:
        self.folder_source = self.folder_source_combo.get()
        self.folder_reference = self.folder_reference_combo.get()
        self._update_folder_languages()

    def _refresh_folder(self, _event: object = None) -> None:
        if self.busy or not self.folder_path.get().strip():
            return
        try:
            files = folder_subtitles(Path(self.folder_path.get().strip()).expanduser())
            languages = sorted({language for group in files.values() for language in group})
            for combo in (self.folder_source_combo, self.folder_reference_combo):
                combo.configure(values=languages)
            if self.folder_source not in languages:
                self.folder_source = languages[0] if languages else ""
            if self.folder_reference not in languages:
                self.folder_reference = next((code for code in languages if code != self.folder_source), "")
            self._update_folder_languages()
        except OSError as error:
            messagebox.showerror(self.interface.tr("SRT folder mode"), str(error), parent=self.root)

    def _choose_audio_folder(self) -> None:
        if self.busy or self.setup_busy:
            return
        selected = filedialog.askdirectory(
            parent=self.root, title=self.interface.tr("Sync to audio folder mode"),
        )
        if selected:
            self.audio_folder_path.set(selected)

    def _choose_folder(self) -> None:
        if self.busy or self.setup_busy:
            return
        selected = filedialog.askdirectory(parent=self.root, title=self.interface.tr("SRT folder mode"))
        if not selected:
            return
        try:
            files = folder_subtitles(Path(selected))
        except OSError as error:
            messagebox.showerror(self.interface.tr("SRT folder mode"), str(error), parent=self.root)
            return
        languages = sorted({language for group in files.values() for language in group})
        if len(languages) < 2:
            messagebox.showerror(
                self.interface.tr("SRT folder mode"),
                self.interface.tr("At least two subtitle languages are required."), parent=self.root,
            )
            return
        window = tk.Toplevel(self.root)
        window.title("Choose subtitle languages")
        window.transient(self.root)
        window.grab_set()
        source = tk.StringVar(master=window, value=(
            self.folder_source if self.folder_source in languages else languages[0]
        ))
        reference = tk.StringVar(master=window, value=(
            self.folder_reference if self.folder_reference in languages else languages[1]
        ))
        for row, (label, variable) in enumerate((
            ("Source SRT (to synchronize)", source), ("Reference SRT", reference),
        )):
            ttk.Label(window, text=label).grid(row=row, column=0, padx=12, pady=8)
            ttk.Combobox(window, textvariable=variable, values=languages, state="readonly").grid(
                row=row, column=1, padx=12, pady=8,
            )

        def accept() -> None:
            if source.get() == reference.get():
                messagebox.showerror(
                    self.interface.tr("SRT folder mode"),
                    self.interface.tr("Choose different source and reference languages."), parent=window,
                )
                return
            self.folder_source, self.folder_reference = source.get(), reference.get()
            self.folder_path.set(selected)
            self._refresh_folder()
            self._update_folder_languages()
            self._update_initial_mode()
            window.destroy()

        ttk.Button(window, text="Open", command=accept).grid(row=2, column=0, padx=12, pady=10)
        ttk.Button(window, text="Cancel", command=window.destroy).grid(row=2, column=1, padx=12, pady=10)
        self.interface.apply_window(window)

    def _start_folder_sync(self, audio_only: bool = False) -> None:
        if self.busy or self.setup_busy:
            return
        try:
            final_offset_ms = 0 if audio_only else self._read_final_offset()
            folder_value = self.audio_folder_path.get() if audio_only else self.folder_path.get()
            if not folder_value.strip():
                raise ValueError("Choose a folder first.")
            folder = Path(folder_value.strip()).expanduser()
            files = folder_subtitles(folder)
            if not audio_only and (
                not self.folder_source or not self.folder_reference or self.folder_source == self.folder_reference
            ):
                raise ValueError("Choose different source and reference languages.")
            jobs = []
            videos_by_title: dict[str, list[Path]] = {}
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm"}:
                    videos_by_title.setdefault(path.stem.casefold(), []).append(path)
            if audio_only:
                for source in sorted(folder.iterdir(), key=lambda path: path.name.casefold()):
                    if source.suffix.lower() != ".srt" or not source.is_file():
                        continue
                    if source.stem.lower().endswith(".audio-synced"):
                        continue
                    matching_source = (
                        source.with_name(source.stem[:-len(".synced")] + source.suffix)
                        if source.stem.lower().endswith(".synced") else source
                    )
                    base, _ = subtitle_name_parts(matching_source)
                    videos = videos_by_title.get(source.stem.casefold(), videos_by_title.get(base, []))
                    jobs.append((
                        source, None, source.with_suffix(".audio-synced.srt"),
                        videos[0] if len(videos) == 1 else None,
                        "" if len(videos) == 1 else "A unique matching video is required.",
                    ))
            for base, group in (files.items() if not audio_only else ()):
                for source in group.get(self.folder_source, []):
                    references = group.get(self.folder_reference, [])
                    error = ""
                    if len(group[self.folder_source]) != 1 or len(references) > 1:
                        error = "Ambiguous subtitle pair."
                    elif not references:
                        error = "No matching reference SRT."
                    videos = videos_by_title.get(base, []) if self.audio_after_sync.get() == "1" else []
                    if self.audio_after_sync.get() == "1" and len(videos) != 1:
                        error = "A unique matching video is required."
                    jobs.append((source, references[0] if len(references) == 1 else None,
                                 source.with_suffix(".synced.srt"), videos[0] if len(videos) == 1 else None, error))
            if not jobs:
                raise ValueError("No source subtitles found for the selected language.")
            device = "CPU" if audio_only else self.processing_device.get()
            adapter = 0 if device == "CPU" else int(self.device_id.get())
            backend = "ONNX" if audio_only else self._selected_backend()
            subtitle_alignment.execution_provider(device, adapter)
            if backend == "Vulkan" and not vulkan_embeddings.ready():
                raise ValueError("Open GPU setup and install Vulkan support first.")
            if not audio_only and backend != "Vulkan" and (device == "CPU" or not gpu_runtime.runtime_ready(device)):
                subtitle_alignment.validate_processing_device(device, adapter)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                self.interface.tr("Sync to audio folder mode" if audio_only else "SRT folder mode"),
                self.interface.tr(str(error)), parent=self.root,
            )
            return
        strictness = int(self.hearing_impaired_strictness.get())
        generate_report = self.report_enabled.get() == "1"
        audio_pass = self.audio_after_sync.get() == "1"
        self.busy = self.batch_running = True
        self.sync_cancel.clear()
        self.sync_committed = False
        for control in (self.sync_button, self.pair_button, self.final_offset_entry,
                        self.device_selector, self.backend_selector, self.setup_button, self.folder_browse,
                        self.folder_entry, self.folder_source_combo, self.folder_reference_combo,
                        self.audio_folder_entry, self.audio_folder_browse, self.audio_folder_sync):
            control.configure(state="disabled")
        self.cancel_sync_button.configure(state="normal")
        self.progress.start(12)

        def run() -> None:
            rows = []
            try:
                for index, (source, reference, output, video, error) in enumerate(jobs, 1):
                    self.messages.put(("status", f"{index}/{len(jobs)}: {source.name}"))
                    if self.sync_cancel.is_set():
                        rows.append((str(source), "Cancelled", str(output), ""))
                        continue
                    if error:
                        rows.append((str(source), "Failed", str(output), error))
                        continue
                    if output.exists():
                        rows.append((str(source), "Skipped", str(output), "Output already exists."))
                        continue
                    events = queue.Queue()
                    self.sync_committed = False
                    if audio_only:
                        self._audio_folder_worker(video, source, output, generate_report, events)
                    else:
                        automatic_initial = self.batch_automatic_initial
                        selection = None
                        if not automatic_initial:
                            try:
                                selection = self._request_folder_selection(source, reference)
                            except (OSError, ValueError) as error:
                                rows.append((str(source), "Failed", str(output), str(error)))
                                continue
                            if selection is None:
                                self.sync_cancel.set()
                                rows.append((str(source), "Cancelled", str(output), "Cue selection cancelled."))
                                continue
                        self._sync_worker(
                            video, source, reference, output, 0, strictness, device, adapter, backend,
                            generate_report=generate_report, sync_to_audio=audio_pass,
                            selection=selection, automatic_initial=automatic_initial, events=events,
                            final_offset_ms=final_offset_ms,
                        )
                    terminal = None
                    while not events.empty():
                        kind, payload = events.get_nowait()
                        if kind in ("success", "error", "cancelled"):
                            terminal = kind, payload
                    if terminal is None:
                        raise RuntimeError("Synchronization worker returned no result.")
                    kind, payload = terminal
                    if kind == "success":
                        _, count, diagnostics = payload
                        tr = self.interface.tr
                        detail = tr("{count} cues").format(count=count)
                        if "match_percent" in diagnostics:
                            detail += "; " + tr("{percent}% matched").format(percent=diagnostics["match_percent"])
                        if "report_path" in diagnostics:
                            detail += f"; {diagnostics['report_path']}"
                        if "report_error" in diagnostics:
                            detail += "; " + tr("Report error: {error}").format(error=diagnostics["report_error"])
                        rows.append((str(source), "Success", str(output), detail))
                    else:
                        rows.append((str(source), "Cancelled" if kind == "cancelled" else "Failed",
                                     str(output), str(payload)))
            except Exception as error:
                self.messages.put(("status", str(error)))
                for source, _, output, _, _ in jobs[len(rows):]:
                    rows.append((str(source), "Failed", str(output), str(error)))
            self.messages.put(("batch_complete", rows))

        threading.Thread(target=run, daemon=True).start()
        self.root.after(150, self._poll_worker)

    def _request_folder_selection(self, source: Path, reference: Path) -> SyncSelection | None:
        paths = (str(source), str(reference))
        signatures = input_signatures(paths)
        left = tuple(subtitle_alignment.read_srt(source))
        right = tuple(subtitle_alignment.read_srt(reference))
        if input_signatures(paths) != signatures:
            raise ValueError("An input changed while loading. Select the cues again.")
        response: queue.Queue[SyncSelection | None] = queue.Queue()
        self.messages.put(("folder_select_pair", (source, reference, left, right, signatures, response)))
        while not self.sync_cancel.is_set():
            try:
                return response.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def _show_folder_pair_picker(
        self, source_path: Path, reference_path: Path,
        source: tuple[subtitle_alignment.Cue, ...],
        reference: tuple[subtitle_alignment.Cue, ...],
        signatures: tuple[tuple[str, int, int], ...],
        response: queue.Queue[SyncSelection | None],
    ) -> None:
        if self.sync_cancel.is_set():
            response.put(None)
            return
        accepted = False

        def accept(left: int, right: int) -> None:
            nonlocal accepted
            accepted = True
            response.put(SyncSelection(source, reference, left, right, signatures))

        dialog = CuePairDialog(self.root, source, reference, accept)
        self.folder_pair_dialog = dialog
        dialog.window.title(f"{source_path.name} -> {reference_path.name}")

        def closed(event: tk.Event) -> None:
            if event.widget is dialog.window:
                self.folder_pair_dialog = None
                if not accepted:
                    self.sync_cancel.set()
                    response.put(None)

        dialog.window.bind("<Destroy>", closed, add="+")

    def _audio_folder_worker(
        self, video: Path | None, source: Path, output: Path,
        generate_report: bool, events: queue.Queue,
    ) -> None:
        log: list[str] = []

        def progress(message: str) -> None:
            log.append(message)
            self.messages.put(("status", f"{source.name}: {self.interface.tr(message)}"))

        try:
            if video is None:
                raise ValueError("A unique matching video is required.")
            original = sync_subtitles.decode_srt(source.read_bytes())[0] if generate_report else None
            tracks = audio_sync.inspect_tracks(video, progress, self.sync_cancel)
            track = next(track for track in tracks if track.kind == "Audio")
            progress(self.interface.tr("Synchronizing using the first audio track: {track}").format(track=track.label))
            count = audio_sync.synchronize(
                video, source, output, track.index, None, True, progress, self.sync_cancel,
                fix_sections=True,
            )
            self.sync_committed = True
            diagnostics: dict[str, object] = {}
            if generate_report:
                try:
                    diagnostics["report_path"] = str(write_report(
                        output, method="Audio speech-activity matching (ffsubsync)",
                        original=original, synchronized=sync_subtitles.decode_srt(output.read_bytes())[0],
                        settings={
                            "Source SRT": source, "Video": video, "Audio track": track.label,
                            "Processing device": "CPU", "Correct drift": True, "Correct sections": True,
                        },
                        log=log, protected_paths=(source, video),
                    ))
                except (OSError, ValueError, UnicodeError) as error:
                    diagnostics["report_error"] = str(error)
            events.put(("success", (output, count, diagnostics)))
        except audio_sync.CancelledError as error:
            events.put(("cancelled", str(error)))
        except Exception as error:
            events.put(("error", str(error)))

    def _show_folder_report(self, rows: list[tuple[str, str, str, str]]) -> None:
        window = tk.Toplevel(self.root)
        window.title("Folder synchronization report")
        make_maximizable(window)
        window.geometry("1100x550")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        counts = {status: sum(row[1] == status for row in rows)
                  for status in ("Success", "Failed", "Skipped", "Cancelled")}
        ttk.Label(window, text="; ".join(
            f"{self.interface.tr(status)}: {count}" for status, count in counts.items()
        )).grid(row=0, column=0, sticky="w", padx=10, pady=10)
        table = ttk.Treeview(window, columns=("source", "status", "output", "details"), show="headings")
        for name, caption, width in (
            ("source", "Source SRT (to synchronize)", 300), ("status", "Status", 90),
            ("output", "Save synchronized SRT", 300), ("details", "Details", 600),
        ):
            table.heading(name, text=caption)
            table.column(name, width=width, minwidth=80, stretch=False)
        table.grid(row=1, column=0, sticky="nsew", padx=(10, 0))
        vertical = ttk.Scrollbar(window, orient="vertical", command=table.yview)
        vertical.grid(row=1, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(window, orient="horizontal", command=table.xview)
        horizontal.grid(row=2, column=0, sticky="ew", padx=10)
        table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        for source, status, output, details in rows:
            table.insert("", "end", values=(source, self.interface.tr(status), output, self.interface.tr(details)))
        ttk.Button(window, text="Close", command=window.destroy).grid(row=3, column=0, pady=10)
        self.interface.apply_window(window)
        self.status.set(self.interface.tr("Folder synchronization report"))

    def _clear_sync_selection(self, *_args: object) -> None:
        self.sync_selection = None
        self.initial_time.set("")
        self.first_cue_text.set(self.interface.tr("Select matching dialogue cues in the two subtitles."))

    def _update_initial_mode(self, *_args: object) -> None:
        self.batch_automatic_initial = self.automatic_initial.get() == "1"
        self.pair_button.configure(
            state="disabled" if self.busy or self.automatic_initial.get() == "1" else "normal",
        )

    def _input_paths(self) -> tuple[str, ...]:
        reference = self.reference_path.get().strip()
        return (self.source_path.get().strip(), reference or self.video_path.get().strip())

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
        reference_value = self.reference_path.get().strip()
        if not reference_value and track_id is None and self.embedded_track is not None:
            video_signature, selected_id = self.embedded_track
            if input_signatures((paths[1],)) == video_signature:
                track_id = selected_id
        self.busy = True
        self.sync_button.configure(state="disabled")
        self.pair_button.configure(state="disabled")
        self.automatic_initial_checkbox.configure(state="disabled")
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
            if not unchanged or self.reference_path.get().strip():
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
                self.interface.tr("Source cue {number}: {text}").format(
                    number=left, text=source[left - 1].text[:120],
                ) + "\n" + self.interface.tr("Reference cue {number}: {text}").format(
                    number=right, text=reference[right - 1].text[:120],
                )
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
        if self.folder_path.get():
            self._start_folder_sync()
            return
        try:
            final_offset_ms = self._read_final_offset()
        except ValueError as error:
            messagebox.showerror(self.interface.tr("Report and extra"), str(error), parent=self.root)
            return
        video_value = self.video_path.get().strip()
        video = Path(video_value).expanduser() if video_value else None
        if self.audio_after_sync.get() == "1" and (video is None or not video.is_file()):
            messagebox.showwarning(
                self.interface.tr("Choose a video"),
                self.interface.tr("Choose a valid video file for automatic audio synchronization."),
                parent=self.root,
            )
            return
        source = Path(self.source_path.get()).expanduser()
        output = Path(self.output_path.get()).expanduser()
        if not source.is_file():
            messagebox.showwarning(
                "Choose a source SRT", "Choose a valid source SRT file.", parent=self.root
            )
            return
        if not self.output_path.get():
            messagebox.showwarning(
                "Choose an output", "Choose where to save the synchronized SRT.", parent=self.root
            )
            return
        if source.resolve() == output.resolve():
            messagebox.showerror(
                "Choose a different output",
                "The synchronized subtitle must be saved separately; the original is kept unchanged.",
                parent=self.root,
            )
            return
        automatic_initial = self.automatic_initial.get() == "1"
        if self.sync_selection is None and not automatic_initial:
            messagebox.showwarning(
                "Select the initial sync point",
                "Click Select matching cues and choose corresponding dialogue in both subtitles.",
                parent=self.root,
            )
            return
        selection = (
            None if automatic_initial and self.reference_path.get().strip()
            else self.sync_selection
        )
        try:
            if selection is not None and input_signatures(self._input_paths()) != selection.inputs:
                self._clear_sync_selection()
                raise ValueError("An input file changed. Select matching cues again.")
        except (OSError, ValueError) as error:
            messagebox.showerror("Initial sync point is outdated", str(error), parent=self.root)
            return
        initial_audio_ms = selection.reference[selection.reference_number - 1].start_ms if selection else 0

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

        reference_value = self.reference_path.get().strip()
        reference_path = Path(reference_value).expanduser() if reference_value else None
        if reference_path is not None and not reference_path.is_file():
            messagebox.showerror(
                "Reference SRT not found",
                "Choose a valid reference SRT, or leave this field blank to use "
                "subtitles embedded in the video.",
                parent=self.root,
            )
            return
        if reference_path is not None and reference_path.resolve() == output.resolve():
            messagebox.showerror(
                "Choose a different output", "Output must not overwrite the reference SRT.",
                parent=self.root,
            )
            return
        if reference_path is None and (video is None or not video.is_file()):
            messagebox.showwarning(
                "Choose a video",
                "Choose a valid video file when no reference SRT is provided.",
                parent=self.root,
            )
            return
        if automatic_initial and reference_path is None and selection is None:
            self._open_initial_sync()
            return
        if output.exists() and not messagebox.askyesno(
            "Replace output?",
            self.interface.tr("This file already exists. Replace it?\n\n{path}").format(path=output),
            parent=self.root,
        ):
            return

        self.busy = True
        self.sync_button.configure(state="disabled")
        self.automatic_initial_checkbox.configure(state="disabled")
        self.final_offset_entry.configure(state="disabled")
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
                source,
                reference_path,
                output,
                initial_audio_ms,
                int(self.hearing_impaired_strictness.get()),
                processing_device,
                device_id,
            ),
            kwargs={
                "backend": backend, "selection": selection,
                "generate_report": self.report_enabled.get() == "1",
                "sync_to_audio": self.audio_after_sync.get() == "1",
                "automatic_initial": automatic_initial,
                "final_offset_ms": final_offset_ms,
            },
            daemon=True,
        )
        worker.start()
        self.root.after(150, self._poll_worker)

    def _read_final_offset(self) -> int:
        value = self.final_offset_ms.get().strip()
        if not re.fullmatch(r"[+-]?[0-9]+", value):
            raise ValueError("Final offset must be a whole number of milliseconds (positive or negative).")
        return int(value)

    def _cancel_sync(self) -> None:
        with self.sync_commit_lock:
            if self.sync_committed and not self.batch_running:
                return
            self.sync_cancel.set()
        self.cancel_sync_button.configure(state="disabled")
        self.status.set("Cancelling synchronization... waiting for the current inference step.")

    def _sync_worker(
        self,
        video: Path | None,
        source_path: Path,
        reference_path: Path | None,
        output: Path,
        initial_audio_ms: int,
        hearing_impaired_strictness: int,
        processing_device: str = "CPU",
        device_id: int = 0,
        backend: str = "ONNX",
        selection: SyncSelection | None = None,
        generate_report: bool = False,
        sync_to_audio: bool = False,
        automatic_initial: bool = False,
        events: queue.Queue | None = None,
        final_offset_ms: int = 0,
    ) -> None:
        messages = self.messages if events is None else events
        try:
            analysis_inputs = input_signatures((
                str(source_path), str(reference_path or video or ""),
            ))
            if sync_to_audio and (video is None or not video.is_file()):
                raise ValueError("Choose a valid video file for automatic audio synchronization.")
            audio_inputs = (
                input_signatures(tuple(str(path) for path in (source_path, reference_path, video) if path))
                if sync_to_audio else ()
            )
            messages.put(("status", "Reading source subtitles..."))
            original_bytes = source_path.read_bytes()
            source_text, encoding, had_bom = sync_subtitles.decode_srt(original_bytes)
            source = subtitle_alignment.parse_srt(source_text)

            if selection is not None:
                if tuple(source) != selection.source:
                    raise ValueError("The source subtitles changed. Select matching cues again.")
                paths = (str(source_path), str(reference_path or video or ""))
                if input_signatures(paths) != selection.inputs:
                    raise ValueError("An input changed. Select matching cues again.")
                reference = list(selection.reference)
            elif reference_path is not None:
                messages.put(("status", "Reading the selected reference SRT..."))
                reference = subtitle_alignment.read_srt(reference_path)
            else:
                raise ValueError("Select matching cues and an embedded reference track before analysis.")

            report_log: list[str] = []

            def progress(message: str) -> None:
                if self.sync_cancel.is_set():
                    raise gpu_runtime.CancelledError("Synchronization cancelled. No output was written.")
                if generate_report:
                    report_log.append(message)
                messages.put(("status", message))

            progress("Starting subtitle analysis...")
            anchor_options = (
                {"source_anchor": selection.source_number, "reference_anchor": selection.reference_number}
                if selection is not None else {}
            )
            if automatic_initial:
                anchor_options["automatic_initial"] = True
            if backend == "Vulkan":
                anchors, diagnostics = subtitle_alignment.analyze_subtitles(
                    source, reference, initial_audio_ms, progress=progress,
                    hearing_impaired_strictness=hearing_impaired_strictness,
                    processing_device=processing_device, device_id=device_id, backend=backend,
                    **anchor_options,
                )
            elif processing_device != "CPU" and gpu_runtime.runtime_ready(processing_device):
                result = gpu_runtime.run_worker(
                    processing_device, device_id,
                    {
                        "operation": "analyze",
                        "source": [asdict(cue) for cue in source],
                        "reference": [asdict(cue) for cue in reference],
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
                    source, reference, initial_audio_ms, progress=progress,
                    hearing_impaired_strictness=hearing_impaired_strictness,
                    processing_device=processing_device, device_id=device_id,
                    **anchor_options,
                )
            if len(source) != len(sync_subtitles.TIMESTAMP.findall(source_text)):
                raise ValueError("The source SRT changed during analysis. Please try again.")
            if input_signatures((str(source_path), str(reference_path or video or ""))) != analysis_inputs:
                raise ValueError("An input changed during analysis. No output was written.")
            if selection is not None and input_signatures(paths) != selection.inputs:
                raise ValueError("An input changed during analysis. No output was written.")
            if reference_path is not None and output.resolve() == reference_path.resolve():
                raise ValueError("Output must not overwrite the reference subtitle.")
            synced_text, cue_count = sync_subtitles.sync_srt(
                source_text, anchors, timing_scale=diagnostics.get("timing_scale", 1.0),
            )
            encoded = synced_text.encode(encoding)
            if had_bom:
                encoded = b"\xef\xbb\xbf" + encoded
            progress("Saving synchronized subtitles...")
            # A private directory keeps staging private while normal creation honors the umask.
            with tempfile.TemporaryDirectory(dir=output.parent, prefix=".subtitle-sync-") as staging_directory:
                staged_path = Path(staging_directory) / output.name
                staged_path.write_bytes(encoded)
                if sync_to_audio:
                    if video is None:
                        raise ValueError("Choose a valid video file for automatic audio synchronization.")
                    audio_sync.validate_paths(
                        video, source_path, output,
                        (reference_path,) if reference_path else (),
                    )
                    tracks = audio_sync.inspect_tracks(video, progress, self.sync_cancel)
                    audio_track = next(track for track in tracks if track.kind == "Audio")
                    progress(self.interface.tr(
                    "Automatic audio synchronization using the first audio track: {track}"
                ).format(track=audio_track.label))
                    audio_result = Path(staging_directory) / "audio-result.srt"
                    audio_sync.synchronize(
                        video, staged_path, audio_result, audio_track.index, None, True,
                        progress, self.sync_cancel, protected_paths=tuple(
                            path for path in (source_path, reference_path, output) if path
                        ),
                    )
                    if input_signatures(tuple(
                        str(path) for path in (source_path, reference_path, video) if path
                    )) != audio_inputs:
                        raise ValueError("An input changed during synchronization. No output was saved.")
                    synced_text, _, _ = sync_subtitles.decode_srt(audio_result.read_bytes())
                    audio_result.replace(staged_path)
                if final_offset_ms:
                    progress(self.interface.tr("Applying final subtitle offset: {offset} ms.").format(
                        offset=f"{final_offset_ms:+d}",
                    ))
                    synced_text = sync_subtitles.shift_srt(synced_text, final_offset_ms)
                    encoded = synced_text.encode(encoding)
                    if had_bom:
                        encoded = b"\xef\xbb\xbf" + encoded
                    staged_path.write_bytes(encoded)
                with self.sync_commit_lock:
                    if self.sync_cancel.is_set():
                        raise gpu_runtime.CancelledError("Synchronization cancelled. No output was written.")
                    if events is not None:
                        sync_subtitles.publish_new_file(staged_path, output)
                    else:
                        staged_path.replace(output)
                    self.sync_committed = True
            result_diagnostics: dict[str, object] = dict(diagnostics)
            if automatic_initial:
                messages.put(("automatic_initial", (
                    source, reference, diagnostics["source_anchor"], diagnostics["reference_anchor"],
                    analysis_inputs,
                )))
            if generate_report:
                try:
                    result_diagnostics["report_path"] = str(write_report(
                        output, method=(
                            "Reference subtitle alignment followed by audio speech-activity matching"
                            if sync_to_audio else "Reference subtitle alignment"
                        ),
                        original=source_text, synchronized=synced_text,
                        settings={
                            "Source SRT": source_path, "Reference SRT": reference_path,
                            "Video": video, "Processing device": processing_device,
                            "GPU adapter ID": device_id, "Inference backend": backend,
                            "Hearing-impaired filter": hearing_impaired_strictness,
                            "Initial reference time (ms)": diagnostics.get("anchor_time_ms", initial_audio_ms),
                            "Initial source cue": diagnostics.get("source_anchor"),
                            "Initial reference cue": diagnostics.get("reference_anchor"),
                            "Automatic initial sync point": automatic_initial,
                            "Final offset (ms)": final_offset_ms,
                            "Automatic audio synchronization": sync_to_audio,
                            "Audio track": audio_track.label if sync_to_audio else None,
                        },
                        diagnostics=diagnostics, anchors=anchors, log=report_log,
                        protected_paths=tuple(path for path in (source_path, reference_path, video) if path),
                    ))
                except (OSError, ValueError, UnicodeError) as error:
                    result_diagnostics["report_error"] = str(error)
            messages.put(("success", (output, cue_count, result_diagnostics)))
        except (gpu_runtime.CancelledError, audio_sync.CancelledError) as error:
            messages.put(("cancelled", str(error)))
        except Exception as error:
            if self.sync_cancel.is_set():
                messages.put(("cancelled", "Synchronization cancelled. No output was written."))
            else:
                messages.put(("error", str(error)))

    def _poll_worker(self) -> None:
        if self.folder_pair_dialog is not None and self.sync_cancel.is_set():
            self.folder_pair_dialog.window.destroy()
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "status":
                    if not self.sync_cancel.is_set():
                        self.status.set(str(payload))
                elif kind == "cancelled":
                    self._finish()
                    self.status.set(str(payload))
                elif kind == "batch_complete":
                    self.batch_running = False
                    self._finish()
                    self._show_folder_report(payload)
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
                    if not self.reference_path.get().strip():
                        self._open_speech_to_text()
                elif kind == "select_pair":
                    self._finish()
                    if self.automatic_initial.get() == "1":
                        source, reference, signatures = payload
                        self.sync_selection = SyncSelection(source, reference, 1, 1, signatures)
                        self._start_sync()
                        return
                    else:
                        self._show_pair_picker(*payload)
                elif kind == "folder_select_pair":
                    self._show_folder_pair_picker(*payload)
                elif kind == "automatic_initial":
                    source, reference, left, right, signatures = payload
                    self.sync_selection = SyncSelection(tuple(source), tuple(reference), left, right, signatures)
                    self.initial_time.set(format_time(reference[right - 1].start_ms))
                    self.first_cue_text.set(
                        self.interface.tr("Source cue {number}: {text}").format(
                            number=left, text=source[left - 1].text[:120],
                        ) + "\n" + self.interface.tr("Reference cue {number}: {text}").format(
                            number=right, text=reference[right - 1].text[:120],
                        )
                    )
                elif kind == "error":
                    self._finish()
                    messagebox.showerror("Synchronization failed", self.interface.tr(str(payload)), parent=self.root)
                    self.status.set("Synchronization failed.")
                elif kind == "success":
                    output, cue_count, diagnostics = payload
                    self._finish()
                    self.status.set(self.interface.tr("Saved synchronized subtitles: {path}").format(path=output))
                    report_message = ""
                    if "report_path" in diagnostics:
                        report_message = "\n\n" + self.interface.tr("Report saved: {path}").format(
                            path=diagnostics["report_path"],
                        )
                    elif "report_error" in diagnostics:
                        report_message = "\n\n" + self.interface.tr(
                            "SRT saved, but the report could not be saved: {error}"
                        ).format(error=diagnostics["report_error"])
                        messagebox.showwarning(
                            self.interface.tr("Report"), report_message.strip(), parent=self.root,
                        )
                    messagebox.showinfo(
                        "Synchronization complete",
                        self.interface.tr(
                            "Matched {matched} dialogue cues ({percent}%) automatically.\n"
                            "Synchronized {count} subtitle cues and saved:\n{path}\n\n"
                            "The source and reference subtitles were not changed."
                        ).format(
                            matched=diagnostics["matched_cues"], percent=diagnostics["match_percent"],
                            count=cue_count, path=output,
                        ) + report_message,
                        parent=self.root,
                    )
        except queue.Empty:
            pass

        if self.busy:
            self.root.after(150, self._poll_worker)

    def _finish(self) -> None:
        self.busy = False
        self.automatic_initial_checkbox.configure(state="normal")
        self.final_offset_entry.configure(state="normal")
        self.sync_cancel.clear()
        self.cancel_sync_button.configure(state="disabled")
        self.progress.stop()
        self.sync_button.configure(state="normal")
        self.device_selector.configure(state="readonly")
        self.backend_selector.configure(state="readonly")
        self.setup_button.configure(state="normal")
        self.folder_browse.configure(state="normal")
        for control in (self.folder_entry, self.audio_folder_entry,
                        self.audio_folder_browse, self.audio_folder_sync):
            control.configure(state="normal")
        for combo in (self.folder_source_combo, self.folder_reference_combo):
            combo.configure(state="readonly")
        self.pair_button.configure(state="normal")
        self._update_initial_mode()
        self._update_device_display()


def main() -> None:
    identity_error = None
    try:
        configure_desktop_identity()
    except OSError as error:
        identity_error = str(error)
    root = tk.Tk(className="SubSync")
    SubtitleSyncApp(root)
    if identity_error is not None:
        root.after_idle(lambda: messagebox.showerror(
            "Cannot configure SubSync desktop identity", identity_error, parent=root,
        ))
    root.mainloop()


if __name__ == "__main__":
    main()
