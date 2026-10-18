"""Manual source/reference cue pairing without changing reference timestamps."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from subtitle_alignment import Cue
from sync_subtitles import ms_to_timestamp
from ui_preferences import auto_wrap, make_maximizable


@dataclass(frozen=True)
class SyncSelection:
    source: tuple[Cue, ...]
    reference: tuple[Cue, ...]
    source_number: int
    reference_number: int
    inputs: tuple[tuple[str, int, int], ...]


def input_signatures(paths: tuple[str, ...]) -> tuple[tuple[str, int, int], ...]:
    signatures = []
    for value in paths:
        if not value.strip():
            continue
        path = Path(value).expanduser().resolve()
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


class CuePairDialog:
    def __init__(
        self, parent: tk.Tk, source: tuple[Cue, ...], reference: tuple[Cue, ...],
        accept: Callable[[int, int], None],
    ) -> None:
        self.window = tk.Toplevel(parent)
        self.window.title("Select the initial sync point")
        make_maximizable(self.window)
        self.window.grab_set()
        self.window.geometry("1100x580")
        self.window.minsize(780, 480)
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(1, weight=1)
        auto_wrap(ttk.Label(
            self.window, wraplength=1000,
            text="Select the same dialogue moment in each subtitle, then confirm. "
            "The source cue will start at the reference cue's exact timestamp. "
            "You can choose any cue, not only the first. The reference file is never modified.",
        )).grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=10)
        self.trees: list[ttk.Treeview] = []
        self.previews: list[tk.StringVar] = []
        self.cues = (source, reference)
        self.selected: list[int | None] = [None, None]
        self.searches: list[tk.StringVar] = []
        for column, (title, cues) in enumerate(zip(("Source (to synchronize)", "Reference (correct timing)"), self.cues)):
            frame = ttk.LabelFrame(self.window, text=title, padding=8)
            frame.grid(row=1, column=column, sticky="nsew", padx=8)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(1, weight=1)
            search = tk.StringVar()
            self.searches.append(search)
            entry = ttk.Entry(frame, textvariable=search)
            entry.grid(row=0, column=0, sticky="ew", pady=(0, 6))
            tree = ttk.Treeview(frame, columns=("number", "time", "text"), show="headings", selectmode="browse")
            for name, label, width in (("number", "Cue", 45), ("time", "Starts", 115), ("text", "Dialogue", 300)):
                tree.heading(name, text=label)
                tree.column(name, width=width, stretch=name == "text")
            tree.grid(row=1, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(frame, command=tree.yview)
            scrollbar.grid(row=1, column=1, sticky="ns")
            tree.configure(yscrollcommand=scrollbar.set)
            preview = tk.StringVar(value="Select a cue. Search above by text or cue number.")
            auto_wrap(ttk.Label(frame, textvariable=preview, wraplength=470)).grid(
                row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0)
            )
            self.trees.append(tree)
            self.previews.append(preview)
            search.trace_add("write", lambda *_args, side=column: self._populate(side))
            tree.bind("<<TreeviewSelect>>", lambda _event, side=column: self._select(side))
            self._populate(column)
        self.summary = tk.StringVar(value="Choose one cue on each side.")
        ttk.Label(self.window, textvariable=self.summary).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=12, pady=10
        )
        actions = ttk.Frame(self.window)
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        self.accept = accept
        self.confirm = ttk.Button(actions, text="Use selected pair", state="disabled", command=self._confirm)
        self.confirm.pack(side="right")
        ttk.Button(actions, text="Cancel", command=self.window.destroy).pack(side="right", padx=8)
        interface = getattr(parent, "interface_preferences", None)
        if interface is not None:
            interface.apply_window(self.window)

    def _populate(self, side: int) -> None:
        tree = self.trees[side]
        tree.delete(*tree.get_children())
        query = self.searches[side].get().strip().casefold()
        for number, cue in enumerate(self.cues[side], 1):
            if not query or query in cue.text.casefold() or query == str(number):
                tree.insert("", "end", iid=str(number), values=(number, ms_to_timestamp(cue.start_ms), cue.text))
        self.selected[side] = None
        self.previews[side].set("Select a cue. Search above by text or cue number.")
        if hasattr(self, "confirm"):
            self._update_summary()

    def _select(self, side: int) -> None:
        selection = self.trees[side].selection()
        self.selected[side] = int(selection[0]) if selection else None
        if self.selected[side] is not None:
            cue = self.cues[side][self.selected[side] - 1]
            self.previews[side].set(
                f"{ms_to_timestamp(cue.start_ms)} --> {ms_to_timestamp(cue.end_ms)}\n{cue.text}"
            )
        self._update_summary()

    def _update_summary(self) -> None:
        if None in self.selected:
            self.confirm.configure(state="disabled")
            self.summary.set("Choose one cue on each side.")
            return
        left, right = self.selected
        offset = self.cues[1][right - 1].start_ms - self.cues[0][left - 1].start_ms
        self.summary.set(f"Source cue {left} = reference cue {right}; initial offset: {offset / 1000:+.3f} s")
        self.confirm.configure(state="normal")

    def _confirm(self) -> None:
        left, right = self.selected
        if left is None or right is None:
            messagebox.showwarning("Select both cues", "Select a cue on each side.", parent=self.window)
            return
        self.accept(left, right)
        self.window.destroy()
