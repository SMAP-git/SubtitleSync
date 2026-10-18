"""Desktop-only update notifications and library environment controls."""

from __future__ import annotations

import importlib.metadata
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import library_updates
import managed_updates
from ui_preferences import make_maximizable


class DesktopUpdates:
    def __init__(self, app) -> None:
        self.app = app
        self.root = app.root
        self.enabled = library_updates.desktop_enabled()
        self.manager = library_updates.LibraryManager()
        self.component_manager = managed_updates.ComponentManager()
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.running = False
        self.pending_restart = False
        self.window: tk.Toplevel | None = None
        self.updates: list[dict[str, str]] = []
        self.components: list[dict] = []
        self.report = ""
        self.check_failed = False

    def schedule_check(self) -> None:
        if self.enabled:
            self.root.after(1500, self.check)

    def _start(self, operation: Callable[[], None], modifying: bool = False) -> None:
        if self.running:
            return
        if modifying:
            self.app.setup_busy = True
        self.running = True
        self._refresh()

        def work() -> None:
            try:
                operation()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.events.put(("error", str(error)))
            else:
                self.events.put(("done", ""))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(100, lambda: self._poll(modifying))

    def _poll(self, modifying: bool) -> None:
        if self.app.busy or (self.app.setup_busy and not modifying):
            self.root.after(250, lambda: self._poll(modifying))
            return
        while not self.events.empty():
            kind, detail = self.events.get_nowait()
            if kind == "progress":
                self.report = detail
            elif kind in ("done", "error"):
                self.running = False
                if modifying:
                    self.app.setup_busy = False
                if kind == "error":
                    self.report = detail
                    messagebox.showwarning("Library update failed", detail, parent=self.window or self.root)
                elif modifying:
                    self.pending_restart = True
                    self.report = "Library selection saved. Close the app and reopen it with the desktop launcher."
                    messagebox.showinfo("Restart required", self.report, parent=self.window or self.root)
                elif (self.check_failed or self.updates or any(item.get("candidate") or item.get("error") for item in self.components)) and self.window is None:
                    self.open()
                if not modifying:
                    self.root.after(library_updates.DAY * 1000, self.check)
            self._refresh()

        if self.running:
            self.root.after(100, lambda: self._poll(modifying))

    def check(self, force: bool = False) -> None:
        if not self.enabled or self.running:
            return
        if self.app.busy or self.app.setup_busy:
            if not force:
                self.root.after(1000, self.check)
            else:
                messagebox.showwarning("Work in progress", "Wait for current work to finish.", parent=self.root)
            return

        def work() -> None:
            reports = []
            self.check_failed = False
            try:
                updates = self.manager.check(force=force)
                if updates is not None:
                    self.updates = updates
                    reports.append(
                    "\n".join(f"{item['name']}: {item['version']} -> {item['latest_version']}" for item in updates)
                    if updates else "No newer versions of the managed libraries were found."
                    )
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                reports.append(f"Library check failed: {error}")
                self.check_failed = True
            components = self.component_manager.check(force=force)
            if components is not None:
                self.components = components
                for item in components:
                    reports.append(
                        item["error"] if item.get("error") else
                        f"{item['key']}: {item['version']} -> {item['candidate']['version']}"
                        if item.get("candidate") else f"{item['key']}: no newer compatible version found."
                    )
            if reports:
                self.events.put(("progress", "\n\n".join(reports)))
            else:
                self.events.put(("progress", "Automatic checks were already performed within the last 24 hours. Use Check for updates to check now."))

        self.report = "Checking for library updates..."
        self._start(work)

    def _change(self, restore: bool = False) -> None:
        if self.running or self.app.busy or self.app.setup_busy:
            messagebox.showwarning("Work in progress", "Wait for current work to finish.", parent=self.root)
            return
        detail = (
            "Select the previous library environment for the next launch?"
            if restore else
            "Download the latest stable fastembed, imageio-ffmpeg and CPU ONNX Runtime into a separate "
            "environment? This may use several GB. Previous libraries remain available.\n\n"
            "New versions are not guaranteed compatible. Import and dependency checks will run before "
            "selection. GPU runtimes, models, Python itself and system packages are not updated."
        )
        if not messagebox.askyesno("Library updates", detail, parent=self.window or self.root):
            return
        self._start(
            self.manager.restore if restore else
            lambda: self.manager.install(lambda message: self.events.put(("progress", message))),
            modifying=True,
        )

    def open(self) -> None:
        if not self.enabled:
            return
        if self.window is not None:
            self.window.lift()
            return
        if self.app.busy or self.app.setup_busy:
            messagebox.showwarning("Work in progress", "Wait for current work to finish.", parent=self.root)
            return
        window = self.window = tk.Toplevel(self.root)
        window.title(self.app.interface.tr("Library updates"))
        make_maximizable(window)
        window.geometry("920x560")
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=(
            "Installed desktop libraries, runtimes and models are checked daily. Nothing is installed automatically.\n"
            "Updates and restores take effect after reopening the app with its desktop launcher."
        ), wraplength=700).pack(fill="x")
        versions = []
        for name in library_updates.PACKAGES:
            try:
                versions.append(f"{name} {importlib.metadata.version(name)}")
            except importlib.metadata.PackageNotFoundError:
                versions.append(f"{name}: not installed")
        ttk.Label(frame, text="Running: " + ", ".join(versions), wraplength=700).pack(fill="x", pady=8)
        self.text = tk.Text(frame, height=8, wrap="word")
        self.text.pack(fill="both", expand=True)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(10, 0))
        self.check_button = ttk.Button(buttons, text="Check for updates", command=lambda: self.check(force=True))
        self.install_button = ttk.Button(buttons, text="Install updates", command=self._change)
        self.restore_button = ttk.Button(buttons, text="Restore previous versions", command=lambda: self._change(True))
        for button in (self.check_button, self.install_button, self.restore_button):
            button.pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Close", command=self._close).pack(side="right")
        component_frame = ttk.LabelFrame(frame, text="Installed runtimes and models", padding=8)
        component_frame.pack(fill="x", pady=(10, 0))
        self.component_selection = tk.StringVar()
        self.component_selector = ttk.Combobox(
            component_frame, textvariable=self.component_selection, state="readonly", width=36,
        )
        self.component_selector.pack(side="left", fill="x", expand=True)
        self.component_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh())
        self.component_install = ttk.Button(component_frame, text="Install updates", command=self._change_component)
        self.component_install.pack(side="left", padx=8)
        self.component_restore = ttk.Button(
            component_frame, text="Restore previous versions", command=lambda: self._change_component(True),
        )
        self.component_restore.pack(side="left")
        if not self.components:
            try:
                self.components = self.component_manager.discover()
            except (OSError, ValueError) as error:
                self.report += f"\nCannot read installed components: {error}"
        window.protocol("WM_DELETE_WINDOW", self._close)
        self.app.interface.apply_window(window)
        self._refresh()

    def _refresh(self) -> None:
        if self.window is None:
            return
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.report or "Use Check for updates to check now.")
        self.text.configure(state="disabled")
        self.check_button.configure(state="disabled" if self.running or self.pending_restart else "normal")
        self.install_button.configure(state="normal" if self.updates and not self.running and not self.pending_restart else "disabled")
        try:
            previous = self.manager.state().get("previous")
        except (OSError, ValueError) as error:
            previous = None
            self.text.configure(state="normal")
            self.text.insert("end", f"\nCannot read library selection: {error}")
            self.text.configure(state="disabled")
        self.restore_button.configure(state="normal" if previous is not None and not self.running else "disabled")
        keys = [item["key"] for item in self.components]
        self.component_selector.configure(values=keys)
        if self.component_selection.get() not in keys:
            self.component_selection.set(keys[0] if keys else "")
        selected = next((item for item in self.components if item["key"] == self.component_selection.get()), {})
        available = not self.running and not self.pending_restart
        self.component_selector.configure(state="readonly" if not self.running else "disabled")
        self.component_install.configure(state="normal" if selected.get("candidate") and available else "disabled")
        self.component_restore.configure(state="normal" if selected.get("previous") and available else "disabled")

    def _change_component(self, restore: bool = False) -> None:
        if self.running or self.app.busy or self.app.setup_busy or self.pending_restart:
            messagebox.showwarning("Work in progress", "Restart after a selection change, or wait for current work to finish.", parent=self.root)
            return
        component = next((item for item in self.components if item["key"] == self.component_selection.get()), None)
        if component is None:
            return
        action = "Restore the previous version of" if restore else "Download and validate an update for"
        if not messagebox.askyesno(
            "Library updates",
            f"{action} {component['key']}?\n\n"
            "Current files will be retained. Downloads may use several GB. Restart afterwards. "
            "Checks cannot guarantee compatibility with your hardware; test inference before relying on it.",
            parent=self.window,
        ):
            return
        self._start(
            (lambda: self.component_manager.restore_component(component["key"])) if restore else
            lambda: self.component_manager.install_component(
                component, lambda message: self.events.put(("progress", message)),
            ),
            modifying=True,
        )

    def _close(self) -> None:
        if self.running:
            messagebox.showwarning("Work in progress", "Wait for the library operation to finish.", parent=self.window)
            return
        if self.window is not None:
            self.window.destroy()
            self.window = None
