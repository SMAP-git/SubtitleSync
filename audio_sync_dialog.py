"""Desktop workflow for synchronizing existing subtitles against an audio track."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

import audio_sync
import file_picker as filedialog
from manual_sync import input_signatures
from ui_preferences import auto_wrap, make_maximizable
from ui_translations import translate
from sync_report import write_report
from sync_subtitles import decode_srt


class AudioSyncDialog(tk.Toplevel):
    def __init__(
        self, parent: tk.Misc, video: str = "", source: str = "",
        protected_paths: tuple[Path, ...] = (),
        *, on_saved: Callable[[Path], None] | None = None,
        generate_report: bool = False,
    ) -> None:
        super().__init__(parent)
        self.title("Sync subtitles to audio")
        make_maximizable(self)
        self.geometry("850x640")
        self.minsize(700, 540)
        self.video = tk.StringVar(value=video)
        self.source = tk.StringVar(value=source)
        initial = source or video
        self.output = tk.StringVar(value=str(Path(initial).with_suffix(".audio-synced.srt")) if initial else "")
        self.source_kind = tk.StringVar(value="external" if source else "embedded")
        self.audio = tk.StringVar()
        self.subtitle = tk.StringVar()
        self.fix_drift = tk.BooleanVar(value=True)
        self.fix_sections = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Choose a video and inspect its tracks. No reference SRT is needed.")
        self.language = getattr(getattr(parent, "interface_preferences", None), "language", "en")
        self.status.set(self.tr(self.status.get()))
        self.protected_paths = protected_paths
        self.on_saved = on_saved
        self.generate_report = generate_report
        self.tracks: list[audio_sync.Track] = []
        self.inspected_inputs: tuple[tuple[str, int, int], ...] = ()
        self.busy = False
        self.cancel_event = threading.Event()
        self.events: queue.Queue[tuple[str, str | list[audio_sync.Track] | tuple[Path, str]]] = queue.Queue()
        self.settings: list[tuple[ttk.Widget, str]] = []
        self._poll_id: str | None = None
        self._build()
        self.video.trace_add("write", self._invalidate_tracks)
        interface = getattr(parent, "interface_preferences", None)
        if interface is not None:
            interface.apply_window(self)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.grab_set()

    def tr(self, text: str) -> str:
        return translate(text, self.language)

    def _control(self, widget: ttk.Widget, state: str = "normal") -> ttk.Widget:
        self.settings.append((widget, state))
        return widget

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(10, weight=1)
        for row, (label, variable, command) in enumerate((
            ("Video / audio", self.video, self._choose_video),
            ("Source SRT (to synchronize)", self.source, self._choose_source),
            ("Save synchronized SRT", self.output, self._choose_output),
        )):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            self._control(ttk.Entry(frame, textvariable=variable)).grid(row=row, column=1, sticky="ew", padx=8)
            self._control(ttk.Button(
                frame, text="Save as..." if row == 2 else "Browse...", command=command,
            )).grid(row=row, column=2)
        sources = ttk.Frame(frame)
        sources.grid(row=3, column=0, columnspan=3, sticky="ew", pady=8)
        for text, value in (("Embedded subtitles", "embedded"), ("External SRT", "external")):
            self._control(ttk.Radiobutton(
                sources, text=text, value=value, variable=self.source_kind,
            )).pack(side="left", padx=(0, 16))
        self._control(ttk.Button(
            sources, text="Inspect tracks", command=lambda: self._start("inspect"),
        )).pack(side="right")
        for row, (label, variable) in enumerate((
            ("Audio track", self.audio), ("Embedded text subtitle track", self.subtitle),
        ), start=4):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            combo = ttk.Combobox(frame, textvariable=variable, state="readonly")
            self._control(combo, "readonly").grid(row=row, column=1, columnspan=2, sticky="ew", padx=8)
            if row == 4:
                self.audio_combo = combo
            else:
                self.subtitle_combo = combo
        corrections = ttk.Frame(frame)
        corrections.grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        for text, variable in (
            ("Correct gradual drift / frame-rate mismatch", self.fix_drift),
            ("Correct section-by-section timing shifts", self.fix_sections),
        ):
            self._control(ttk.Checkbutton(
                corrections, text=text, variable=variable,
            )).pack(anchor="w")
        auto_wrap(ttk.Label(
            frame, wraplength=790,
            text="CPU speech-activity matching (ffsubsync): no transcription, translation or model download. "
                 "Preserves subtitle wording. Handles global offset, timing-rate differences and changing "
                 "section delays (up to 60 seconds). Disable section correction for global offsets up to "
                 "ten minutes. Review logged section boundaries; it cannot recover missing dialogue "
                 "or guarantee arbitrary per-line alignment. Image subtitles require OCR and are not supported. "
                 "Other text formats are converted to SRT; format-specific styling may be lost. "
                 "After saving, review the beginning, middle and end in your player.",
        )).grid(row=7, column=0, columnspan=3, sticky="ew", pady=6)
        actions = ttk.Frame(frame)
        actions.grid(row=8, column=0, columnspan=3, sticky="ew", pady=8)
        self._control(ttk.Button(
            actions, text="Synchronize to audio", command=lambda: self._start("sync"),
        )).pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", state="disabled", command=self._cancel)
        self.cancel_button.pack(side="right")
        self.close_button = ttk.Button(actions, text="Close", command=self._close)
        self.close_button.pack(side="right", padx=(0, 8))
        self.progress = ttk.Progressbar(actions, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=12)
        auto_wrap(ttk.Label(frame, textvariable=self.status, wraplength=790)).grid(
            row=9, column=0, columnspan=3, sticky="ew", pady=5,
        )
        logs = ttk.Frame(frame)
        logs.grid(row=10, column=0, columnspan=3, sticky="nsew")
        self.log = tk.Text(logs, wrap="word", height=10, state="disabled")
        scrollbar = ttk.Scrollbar(logs, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _invalidate_tracks(self, *_args: object) -> None:
        self.tracks = []
        self.inspected_inputs = ()
        self.audio.set("")
        self.subtitle.set("")
        self.audio_combo.configure(values=())
        self.subtitle_combo.configure(values=())

    def _choose_video(self) -> None:
        chosen = filedialog.askopenfilename(parent=self, title=self.tr("Choose video or audio"))
        if chosen:
            self.video.set(chosen)
            self.output.set(str(Path(chosen).with_suffix(".audio-synced.srt")))

    def _choose_source(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self, title=self.tr("Choose source SRT"), filetypes=[(self.tr("SubRip subtitles"), "*.srt")],
        )
        if chosen:
            self.source.set(chosen)
            self.source_kind.set("external")
            self.output.set(str(Path(chosen).with_suffix(".audio-synced.srt")))

    def _choose_output(self) -> None:
        current = self.output.get().strip()
        chosen = filedialog.asksaveasfilename(
            parent=self, title=self.tr("Save synchronized SRT"), defaultextension=".srt",
            initialfile=Path(current).name if current else "",
            initialdir=str(Path(current).parent) if current else None,
            filetypes=[(self.tr("SubRip subtitles"), "*.srt")],
        )
        if chosen:
            self.output.set(chosen)

    def _selected_track(self, value: str, kind: str) -> int:
        for track in self.tracks:
            if track.label == value and track.kind == kind:
                return track.index
        raise ValueError(self.tr(
            "Select an audio track after inspecting the file." if kind == "Audio"
            else "Select a subtitle track after inspecting the file."
        ))

    def _start(self, operation: str) -> None:
        if self.busy:
            return
        try:
            if not self.video.get().strip():
                raise ValueError("Choose a video or audio file first.")
            video = Path(self.video.get().strip())
            signature = input_signatures((str(video),))
            source = None
            output = Path(self.output.get().strip())
            audio_track, subtitle_track = -1, None
            fix_drift = self.fix_drift.get()
            fix_sections = self.fix_sections.get()
            if operation == "sync":
                if signature != self.inspected_inputs:
                    raise ValueError("Inspect tracks again after selecting or changing the input file.")
                audio_track = self._selected_track(self.audio.get(), "Audio")
                if self.source_kind.get() == "external":
                    if not self.source.get().strip():
                        raise ValueError("Choose the external source SRT.")
                    source = Path(self.source.get().strip())
                else:
                    subtitle_track = self._selected_track(self.subtitle.get(), "Subtitle")
                audio_sync.validate_paths(video, source, output, self.protected_paths)
        except (OSError, ValueError) as error:
            messagebox.showerror(self.tr("Sync subtitles to audio"), self.tr(str(error)), parent=self)
            return
        overwrite = False
        if operation == "sync" and output.exists():
            overwrite = messagebox.askyesno(
                self.tr("Replace synchronized SRT?"),
                self.tr("Replace this existing file?\n{path}").format(path=output), parent=self,
            )
            if not overwrite:
                return
        self.busy = True
        self.cancel_event.clear()
        for widget, _state in self.settings:
            widget.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.start(12)
        self.status.set(self.tr("Inspecting tracks..." if operation == "inspect" else "Synchronizing to audio on CPU..."))

        def worker() -> None:
            report_log: list[str] = []

            def progress(text: str) -> None:
                if self.generate_report:
                    report_log.append(text)
                self.events.put(("log", text))

            try:
                if operation == "inspect":
                    tracks = audio_sync.inspect_tracks(video, progress, self.cancel_event)
                    if input_signatures((str(video),)) != signature:
                        raise ValueError("The input changed during inspection. Inspect tracks again.")
                    self.events.put(("tracks", tracks))
                else:
                    original = (
                        decode_srt(source.read_bytes())[0]
                        if self.generate_report and source is not None else None
                    )
                    count = audio_sync.synchronize(
                        video, source, output, audio_track, subtitle_track, fix_drift,
                        progress, self.cancel_event, overwrite, self.protected_paths,
                        fix_sections=fix_sections,
                    )
                    report_message = ""
                    if self.generate_report:
                        try:
                            report = write_report(
                                output, method="Audio speech-activity matching (ffsubsync)",
                                original=original, synchronized=decode_srt(output.read_bytes())[0],
                                settings={
                                    "Video / audio": video, "Source SRT": source,
                                    "Audio track": audio_track, "Embedded subtitle track": subtitle_track,
                                    "Correct drift": fix_drift, "Correct sections": fix_sections,
                                    "Processing device": "CPU",
                                },
                                log=report_log,
                                protected_paths=(video, *self.protected_paths, *((source,) if source else ())),
                            )
                            report_message = self.tr("Report saved: {path}").format(path=report)
                        except (OSError, ValueError, UnicodeError) as error:
                            report_message = self.tr(
                                "SRT saved, but the report could not be saved: {error}"
                            ).format(error=error)
                        progress(report_message)
                    self.events.put((
                        "saved", (output, self.tr("Saved {count} cues to {path}").format(count=count, path=output)
                        + ("\n" + self.tr("Review any section boundaries listed in the log.") if fix_sections else "")
                        + ("\n" + report_message if report_message else "")),
                    ))
            except audio_sync.CancelledError as error:
                self.events.put(("cancelled", str(error)))
            except Exception as error:
                self.events.put(("error", str(error)))

        self._pending_signature = signature
        threading.Thread(target=worker, daemon=True).start()
        self._poll_id = self.after(100, self._poll)

    def _poll(self) -> None:
        self._poll_id = None
        for _ in range(150):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log.configure(state="normal")
                self.log.insert("end", str(payload) + "\n")
                if int(self.log.index("end-1c").split(".")[0]) > 2000:
                    self.log.delete("1.0", "500.0")
                self.log.see("end")
                self.log.configure(state="disabled")
                continue
            self.busy = False
            self.progress.stop()
            self.cancel_button.configure(state="disabled")
            for widget, state in self.settings:
                widget.configure(state=state)
            if kind == "tracks" and isinstance(payload, list):
                self.tracks = payload
                self.inspected_inputs = self._pending_signature
                audio = [track.label for track in payload if track.kind == "Audio"]
                subtitles = [
                    track.label for track in payload
                    if track.kind == "Subtitle" and track.codec in audio_sync.TEXT_CODECS
                ]
                self.audio_combo.configure(values=audio)
                self.subtitle_combo.configure(values=subtitles)
                self.audio.set(audio[0])
                self.subtitle.set(subtitles[0] if subtitles else "")
                self.status.set(self.tr(
                    "Select the correct audio and subtitle tracks, then synchronize."
                    if subtitles else "No supported embedded text subtitles found. Choose an external SRT."
                ))
            elif kind == "saved" and isinstance(payload, tuple):
                path, message = payload
                if self.on_saved is not None:
                    self.on_saved(path)
                self.status.set(message)
                messagebox.showinfo(self.tr("Audio synchronization complete"), message, parent=self)
            else:
                self.status.set(self.tr(str(payload)))
                if kind == "error":
                    messagebox.showerror(self.tr("Audio synchronization failed"), self.tr(str(payload)), parent=self)
            return
        if self.busy:
            self._poll_id = self.after(100, self._poll)

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status.set(self.tr("Cancelling audio synchronization..."))
        self.cancel_button.configure(state="disabled")

    def _close(self) -> None:
        if self.busy:
            if messagebox.askyesno(
                self.tr("Work in progress"),
                self.tr("Cancel the current operation? Close this window after cancellation finishes."),
                parent=self,
            ):
                self._cancel()
            return
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.grab_release()
        self.destroy()
