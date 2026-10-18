"""Tk dialog for generating a local speech-to-text reference SRT."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import file_picker as filedialog
from typing import Callable

import speech_to_text as speech
from subtitle_alignment import PROCESSING_DEVICES
from ui_preferences import auto_wrap, make_maximizable, load_choices, persist_choices, settings_path


class SpeechDialog(tk.Toplevel):
    def __init__(
        self, parent: tk.Misc, video: str, protected_paths: tuple[Path, ...],
        on_generated: Callable[[Path], None],
    ) -> None:
        super().__init__(parent)
        self.title("Speech to text - generate reference SRT")
        make_maximizable(self)
        self.geometry("880x720")
        self.minsize(780, 650)
        self.video = tk.StringVar(value=video)
        initial = str(Path(video).with_suffix(".speech.srt")) if video else ""
        self.output = tk.StringVar(value=initial)
        self.language = tk.StringVar(value="Auto detect")
        self.model = tk.StringVar(value="base")
        self.device = tk.StringVar(value="CPU")
        self.backend = tk.StringVar(value="Recommended")
        self.device_id = tk.StringVar(value="0")
        self.audio_stream = tk.StringVar(value="0")
        self.preferences_path = getattr(parent, "preferences_path", None) or settings_path()
        self.preference_variables = {
            "language": self.language, "model": self.model, "device": self.device,
            "backend": self.backend, "adapter": self.device_id, "audio_stream": self.audio_stream,
        }
        self.preference_choices = {
            "language": tuple(speech.LANGUAGES), "model": tuple(speech.MODELS),
            "device": PROCESSING_DEVICES, "backend": speech.BACKENDS,
        }
        try:
            load_choices(self.preferences_path, "whisper", self.preference_variables, self.preference_choices)
        except (OSError, ValueError) as error:
            self.after_idle(lambda detail=str(error): messagebox.showerror(
                "Cannot load speech settings", detail, parent=self,
            ))
        self.model_info = tk.StringVar()
        self.status = tk.StringVar(value="Choose video/audio, spoken language and a Whisper model.")
        self.protected_paths = protected_paths
        self.on_generated = on_generated
        self.busy = False
        self.cancel_event = threading.Event()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.settings: list[tuple[ttk.Widget, str]] = []
        self._poll_id: str | None = None
        self._build()
        for variable in self.preference_variables.values():
            variable.trace_add("write", self._save_preferences)
        interface = getattr(parent, "interface_preferences", None)
        if interface is not None:
            interface.apply_window(self)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.grab_set()

    def _save_preferences(self, *_args: object) -> None:
        try:
            persist_choices(self.preferences_path, "whisper", self.preference_variables, self.preference_choices)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot save speech settings", str(error), parent=self)
        self._update_model()

    def _entry(self, parent: tk.Misc, value: tk.StringVar, width: int = 50) -> ttk.Entry:
        widget = ttk.Entry(parent, textvariable=value, width=width)
        self.settings.append((widget, "normal"))
        return widget

    def _combo(
        self, parent: tk.Misc, value: tk.StringVar, values: tuple[str, ...], width: int = 22,
    ) -> ttk.Combobox:
        widget = ttk.Combobox(parent, textvariable=value, values=values, state="readonly", width=width)
        self.settings.append((widget, "readonly"))
        return widget

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)
        for row, (label, value, command) in enumerate((
            ("Video / audio", self.video, self._choose_video),
            ("Generated reference SRT", self.output, self._choose_output),
        )):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            self._entry(frame, value).grid(row=row, column=1, sticky="ew", padx=8)
            button = ttk.Button(frame, text="Browse..." if row == 0 else "Save as...", command=command)
            button.grid(row=row, column=2)
            self.settings.append((button, "normal"))

        options = ttk.LabelFrame(frame, text="Whisper.cpp speech recognition (independent of subtitle matching)", padding=10)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        for row, (label, value, values) in enumerate((
            ("Spoken language", self.language, tuple(speech.LANGUAGES)),
            ("Model", self.model, tuple(speech.MODELS)),
            ("Processing device", self.device, PROCESSING_DEVICES),
            ("Backend", self.backend, speech.BACKENDS),
        )):
            ttk.Label(options, text=label).grid(row=row, column=0, sticky="w", pady=3)
            combo = self._combo(options, value, values, 28)
            combo.grid(row=row, column=1, sticky="w", padx=10)
            if row == 1:
                combo.bind("<<ComboboxSelected>>", self._update_model)
        ttk.Label(options, text="GPU adapter ID").grid(row=2, column=2, sticky="w", padx=(15, 0))
        self._entry(options, self.device_id, 5).grid(row=2, column=3, padx=10)
        ttk.Label(options, text="Audio stream index (0 = first)").grid(row=3, column=2, sticky="w", padx=(15, 0))
        self._entry(options, self.audio_stream, 5).grid(row=3, column=3, padx=10)
        auto_wrap(ttk.Label(frame, textvariable=self.model_info, wraplength=800)).grid(row=3, column=0, columnspan=3, sticky="ew")
        self._update_model()
        auto_wrap(ttk.Label(
            frame, wraplength=800,
            text=("CPU-only image: speech recognition runs on the CPU. Use the full image for GPU support.\n"
            if speech.gpu_runtime.CPU_ONLY else
            "Recommended: CPU; NVIDIA via CUDA; AMD/Intel graphics and APUs via Vulkan "
            "(might not work with all cards). Uses your installed graphics driver; no administrator access or "
            "driver installation. Adapter IDs are specific to Whisper: use Detect GPUs to see names.\n")
            +
            "Select the video's spoken language, not the source subtitle language. Output is a "
            "transcript, not an English translation. Choose the audio stream containing that language.",
        )).grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)
        actions = ttk.Frame(frame)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        for label, operation in (
            ("Download runtime / verify model", "prepare"),
            ("Detect GPUs / APUs", "detect"),
            ("Generate reference SRT", "generate"),
        ):
            button = ttk.Button(actions, text=label, command=lambda op=operation: self._start(op))
            button.pack(side="left", padx=(0, 8))
            self.settings.append((button, "normal"))
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="right")
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5)
        auto_wrap(ttk.Label(frame, textvariable=self.status, wraplength=800)).grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        log_frame = ttk.Frame(frame)
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew")
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        auto_wrap(ttk.Label(
            frame, text="After generation, review the transcript and select matching source/reference cues "
            "in the main window. Recognition can mishear speech or hallucinate during silence.",
            wraplength=800,
        )).grid(row=9, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def _update_model(self, _event: object = None) -> None:
        model = self.model.get()
        self.model_info.set(
            f"Model download: about {speech.MODELS[model].size_mb} MB; cached in .speech-to-text/models. "
            "Larger models need more RAM/VRAM. Start with tiny/base on older GPUs/APUs."
        )

    def _choose_video(self) -> None:
        chosen = filedialog.askopenfilename(parent=self, title="Choose video or audio")
        if chosen:
            self.video.set(chosen)
            self.output.set(str(Path(chosen).with_suffix(".speech.srt")))

    def _choose_output(self) -> None:
        initial = self.output.get().strip()
        chosen = filedialog.asksaveasfilename(
            parent=self, title="Save generated reference", defaultextension=".srt",
            initialfile=Path(initial).name if initial else "",
            initialdir=str(Path(initial).parent) if initial else None,
            filetypes=[("SubRip subtitles", "*.srt")],
        )
        if chosen:
            self.output.set(chosen)

    def _start(self, operation: str) -> None:
        if self.busy:
            return
        try:
            device, choice = self.device.get(), self.backend.get()
            backend = speech.selected_backend(device, choice)
            model, language = self.model.get(), speech.LANGUAGES[self.language.get()]
            adapter, stream = int(self.device_id.get()), int(self.audio_stream.get())
            speech.validate_options(model, language, adapter, stream)
            speech.package(backend)
            if operation == "detect" and backend == "CPU":
                raise ValueError("Select a GPU or APU device before detecting adapters.")
            video, output = Path(self.video.get().strip()), Path(self.output.get().strip())
            if operation == "generate" and (not self.video.get().strip() or not self.output.get().strip()):
                raise ValueError("Choose the input video/audio and generated SRT output.")
            download_model = "tiny" if operation == "detect" else model
            missing = speech.missing_downloads(backend, download_model)
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror("Speech to text", str(error), parent=self)
            return
        if missing and not messagebox.askokcancel(
            "Local speech-to-text setup",
            "The following files need downloading:\n\n"
            + "\n".join(f"- {item}" for item in missing)
            + ("\n\nThe CUDA runtime download can be up to 685 MB."
               if backend == "CUDA" and any("runtime" in item for item in missing) else "")
            + "\n\nDownloads are cached in this app's .speech-to-text folder. "
            "Existing models are checksum-verified; damaged models are downloaded again. "
            "Audio and subtitles stay on your computer.\n\nContinue?",
            parent=self,
        ):
            return
        overwrite = False
        if operation == "generate" and output.exists():
            overwrite = messagebox.askyesno(
                "Replace generated SRT?", f"Replace this existing file?\n{output}", parent=self,
            )
            if not overwrite:
                return
        self.cancel_event.clear()
        self.busy = True
        for widget, _state in self.settings:
            widget.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.set("Starting local speech recognition...")
        self.progress.start(12)

        def worker() -> None:
            progress = lambda text: self.events.put(("log", text))
            try:
                if operation == "prepare":
                    speech.prepare(backend, model, progress, self.cancel_event)
                    self.events.put(("ready", "Model checksum verified and runtime installed. Generate subtitles to test the selected device."))
                elif operation == "detect":
                    devices = speech.detect_devices(backend, progress, self.cancel_event)
                    self.events.put(("devices", devices))
                else:
                    count = speech.transcribe(
                        video, output, model, language, device, choice, adapter, stream,
                        progress, self.cancel_event, overwrite, self.protected_paths,
                    )
                    self.events.put(("generated", (output, count)))
            except speech.CancelledError as error:
                self.events.put(("cancelled", str(error)))
            except Exception as error:
                self.events.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_id = self.after(100, self._poll)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 2000:
            self.log.delete("1.0", "500.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self) -> None:
        self._poll_id = None
        for _ in range(150):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
                self.status.set(str(payload)[:220])
                continue
            self.busy = False
            self.progress.stop()
            self.cancel_button.configure(state="disabled")
            for widget, state in self.settings:
                widget.configure(state=state)
            if kind == "generated":
                path, count = payload
                self.on_generated(path)
                self.status.set(f"Saved {count} cues. This SRT is now the reference; select matching cues next.")
                self._append_log(f"Saved reference: {path}")
                messagebox.showinfo("Reference generated", self.status.get(), parent=self)
            elif kind == "devices":
                listing = "\n".join(f"{number}: {name}" for number, name in payload)
                self.status.set("Whisper adapters detected. Enter the ID for the GPU/APU you want to use.")
                self._append_log("Whisper adapter IDs:\n" + listing)
                messagebox.showinfo("Whisper GPU / APU adapters", listing, parent=self)
            else:
                self.status.set(str(payload))
                self._append_log(str(payload))
                if kind == "error":
                    messagebox.showerror("Speech to text failed", str(payload), parent=self)
            return
        if self.busy:
            self._poll_id = self.after(100, self._poll)

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status.set("Cancelling... downloads may take up to 60 seconds to respond.")
        self.cancel_button.configure(state="disabled")

    def _close(self) -> None:
        if self.busy:
            if messagebox.askyesno("Work in progress", "Cancel the current operation? Close this window after cancellation finishes.", parent=self):
                self._cancel()
            return
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.grab_release()
        self.destroy()
