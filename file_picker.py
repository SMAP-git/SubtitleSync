"""Resizable Linux file dialogs with mounted-share shortcuts; native Windows dialogs."""

from __future__ import annotations

import fnmatch
import os
import re
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ui_preferences import read_settings, settings_path, update_settings
from ui_translations import translate


def mounted_locations(mountinfo: str) -> list[Path]:
    """Include network mounts and user-visible disks, not internal system mounts."""
    locations = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 7 or "-" not in fields:
            continue
        separator = fields.index("-")
        if separator + 1 >= len(fields):
            continue
        filesystem = fields[separator + 1]
        mountpoint = re.sub(
            r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4],
        )
        if (
            filesystem in {"cifs", "smb3", "nfs", "nfs4", "fuse.sshfs", "fuse.gvfsd-fuse"}
            or mountpoint.startswith(("/mnt/", "/media/", "/run/media/"))
        ):
            locations.append(Path(mountpoint))
    return list(dict.fromkeys(locations))


def browse_locations() -> list[Path]:
    locations = [Path.home(), Path("/"), Path("/mnt"), Path("/media")]
    mounts = Path("/proc/self/mountinfo")
    if mounts.is_file():
        locations.extend(mounted_locations(mounts.read_text(encoding="utf-8")))
    # Desktop-mounted SMB/SFTP shares are exposed to ordinary file APIs by GVfs.
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    gvfs = runtime / "gvfs"
    if gvfs.is_dir():
        locations.append(gvfs)
        locations.extend(sorted(path for path in gvfs.iterdir() if path.is_dir()))
    return list(dict.fromkeys(locations))


def sorted_entries(directory: Path, patterns: tuple[str, ...], column: str, reverse: bool):
    entries = []
    with os.scandir(directory) as iterator:
        for entry in iterator:
            directory_entry = entry.is_dir()
            if not directory_entry and not any(
                fnmatch.fnmatch(entry.name.lower(), "*" if pattern == "*.*" else pattern.lower()) for pattern in patterns
            ):
                continue
            stat = entry.stat()
            entries.append((entry.name, directory_entry, stat.st_size, stat.st_mtime))
    index = {"name": 0, "size": 2, "modified": 3}[column]
    key = lambda item: (item[index].casefold() if index == 0 else item[index], item[0].casefold())
    return (
        sorted((item for item in entries if item[1]), key=key, reverse=reverse)
        + sorted((item for item in entries if not item[1]), key=key, reverse=reverse)
    )


class FilePicker(tk.Toplevel):
    def __init__(self, parent, save: bool, **options):
        super().__init__(parent)
        owner = parent
        while owner is not None and not hasattr(owner, "interface_preferences"):
            owner = owner.master
        self.interface = getattr(owner, "interface_preferences", None)
        self.result = ""
        self.save = save
        self.options = options
        self.directory_only = options.pop("directory", False)
        self.title(options.get("title", "Save file" if save else "Open file"))
        # Linux window managers can suppress maximizing transient/dialog windows.
        self.wm_attributes("-type", "normal")
        self.geometry("1000x640")
        self.minsize(620, 400)
        self.resizable(True, True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.directory = Path(options.get("initialdir") or Path.cwd()).expanduser()
        self.location = tk.StringVar(value=str(self.directory))
        self.filename = tk.StringVar(value=options.get("initialfile", ""))
        self.column, self.reverse = "name", False
        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Button(top, text="Up", command=lambda: self.navigate(self.directory.parent)).grid(row=0, column=0)
        location = ttk.Entry(top, textvariable=self.location)
        location.grid(row=0, column=1, sticky="ew", padx=8)
        location.bind("<Return>", lambda _event: self.navigate(Path(self.location.get()).expanduser()))
        ttk.Button(top, text="Home", command=lambda: self.navigate(Path.home())).grid(row=0, column=2)
        ttk.Label(top, text="Locations / mounted shares").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.places = ttk.Combobox(top, state="readonly", postcommand=self.refresh_locations)
        self.places.grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        self.places.bind("<<ComboboxSelected>>", lambda _event: self.navigate(Path(self.places.get())))
        ttk.Button(top, text="Refresh", command=self.refresh_locations).grid(row=1, column=2, pady=(8, 0))
        self.refresh_locations()
        content = ttk.Frame(self, padding=(10, 0))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(content, columns=("name", "size", "modified"), show="headings", selectmode="browse")
        for column, title, width in (("name", "Name", 500), ("size", "Size", 110), ("modified", "Modified", 180)):
            self.tree.heading(column, text=title, command=lambda key=column: self.sort(key))
            self.tree.column(column, width=width, minwidth=80, stretch=column == "name")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(content, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(content, orient="horizontal", command=self.tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        self.tree.bind("<<TreeviewSelect>>", self.select)
        self.tree.bind("<Double-1>", lambda _event: self.open_selected())
        self.tree.bind("<Return>", lambda _event: self.open_selected())
        bottom = ttk.Frame(self, padding=10)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(1, weight=1)
        ttk.Label(bottom, text="Folder mode" if self.directory_only else "File name").grid(row=0, column=0)
        name = ttk.Entry(bottom, textvariable=self.filename)
        name.grid(row=0, column=1, sticky="ew", padx=8)
        name.bind("<Return>", lambda _event: self.accept())
        ttk.Button(bottom, text="Save" if save else "Open", command=self.accept).grid(row=0, column=2)
        types = options.get("filetypes") or (("All files", "*"),)
        self.filters = types
        self.filter = ttk.Combobox(bottom, values=[self.tr(label) for label, _ in types], state="readonly")
        self.filter.current(0)
        self.filter.grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        self.filter.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Button(bottom, text="Cancel", command=self.destroy).grid(row=1, column=2, pady=(8, 0))
        self.bind("<Escape>", lambda _event: self.destroy())
        self.entries = {}
        self.refresh()
        if self.interface:
            self.interface.apply_window(self)
        self.grab_set()
        name.focus_set()

    def tr(self, text):
        return translate(text, self.interface.language if self.interface else "en")

    def sort(self, column):
        self.reverse = not self.reverse if column == self.column else False
        self.column = column
        self.refresh()

    def refresh_locations(self):
        try:
            locations = browse_locations()
        except OSError as error:
            messagebox.showerror(self.tr("Cannot list mounted locations"), str(error), parent=self)
            return
        self.places.configure(values=tuple(str(path) for path in locations))

    def navigate(self, directory):
        previous = self.directory
        self.directory = directory
        if not self.refresh():
            self.directory = previous
        self.location.set(str(self.directory))
        if self.directory_only:
            self.filename.set("")

    def refresh(self):
        pattern = self.filters[self.filter.current()][1]
        patterns = tuple(pattern) if isinstance(pattern, (tuple, list)) else tuple(str(pattern).split())
        try:
            entries = sorted_entries(self.directory, patterns, self.column, self.reverse)
        except OSError as error:
            messagebox.showerror(self.tr("Cannot open folder"), str(error), parent=self)
            return False
        self.tree.delete(*self.tree.get_children())
        self.entries.clear()
        for number, (name, directory, size, modified) in enumerate(entries):
            if self.directory_only and not directory:
                continue
            key = str(number)
            self.entries[key] = (name, directory)
            self.tree.insert("", "end", iid=key, values=(
                name + ("/" if directory else ""), "" if directory else f"{size:,} B",
                datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S"),
            ))
        return True

    def select(self, _event=None):
        selected = self.tree.selection()
        if selected and selected[0] in self.entries:
            name, directory = self.entries[selected[0]]
            if not directory or self.directory_only:
                self.filename.set(name)

    def open_selected(self):
        selected = self.tree.selection()
        if selected:
            name, directory = self.entries[selected[0]]
            if directory:
                self.navigate(self.directory / name)
            else:
                self.filename.set(name)
                self.accept()

    def accept(self):
        if self.directory_only:
            path = self.directory / Path(self.filename.get().strip() or ".").expanduser()
            if not path.is_dir():
                messagebox.showerror(self.tr("Folder not found"), str(path), parent=self)
                return
            self.result = str(path.absolute())
            self.destroy()
            return
        name = self.filename.get().strip()
        if not name:
            return
        path = self.directory / Path(name).expanduser()
        if path.is_dir():
            self.navigate(path)
            return
        if self.save and not path.suffix:
            path = path.with_suffix(self.options.get("defaultextension") or "")
        if not self.save and not path.is_file():
            messagebox.showerror(self.tr("File not found"), str(path), parent=self)
            return
        if self.save and not path.parent.is_dir():
            messagebox.showerror(self.tr("Folder not found"), str(path.parent), parent=self)
            return
        self.result = str(path.absolute())
        self.destroy()


def last_directory() -> str | None:
    """Return the folder of the most recently chosen file, if it still exists."""
    try:
        folder = read_settings(settings_path()).get("last_directory")
    except (OSError, ValueError):
        return None
    return folder if isinstance(folder, str) and Path(folder).is_dir() else None


def remember_directory(path: str) -> None:
    try:
        update_settings(settings_path(), {"last_directory": str(Path(path).absolute().parent)})
    except (OSError, ValueError):
        pass


def _choose(save: bool, **options):
    if not options.get("initialdir"):
        options["initialdir"] = last_directory()
    if not options["initialdir"]:
        options.pop("initialdir")
    if not sys.platform.startswith("linux"):
        native = filedialog.asksaveasfilename if save else filedialog.askopenfilename
        result = native(**options)
    else:
        parent = options.pop("parent", None) or tk._default_root
        previous_grab = parent.grab_current() if parent is not None else None
        dialog = FilePicker(parent, save, **options)
        dialog.wait_window()
        if previous_grab is not None and previous_grab.winfo_exists():
            previous_grab.grab_set()
        result = dialog.result
    if isinstance(result, str) and result:
        remember_directory(result)
        return result
    return ""


def askopenfilename(**options):
    return _choose(False, **options)


def asksaveasfilename(**options):
    return _choose(True, **options)


def askdirectory(**options):
    if not sys.platform.startswith("linux"):
        return filedialog.askdirectory(**options)
    return _choose(False, directory=True, **options)
