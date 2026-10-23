"""Human-readable synchronization reports; never change subtitle content."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from sync_subtitles import TIMESTAMP, timestamp_to_ms
from ui_preferences import APP_VERSION


def report_path(output: Path) -> Path:
    stem = output.stem
    for suffix in (".audio-synced", ".synced"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return output.with_name(stem + ".sync.txt")


def write_report(
    output: Path, *, method: str, original: str | None, synchronized: str,
    settings: Mapping[str, object], log: Sequence[str],
    diagnostics: Mapping[str, object] | None = None,
    anchors: Sequence[Sequence[int | float]] = (),
    protected_paths: Sequence[Path] = (),
) -> Path:
    destination = report_path(output)
    for path in (output, *protected_paths):
        if destination.resolve() == path.resolve() or (
            destination.exists() and path.exists() and destination.samefile(path)
        ):
            raise ValueError(f"Report must not overwrite a selected input or SRT: {destination}")
    if destination.is_symlink():
        raise ValueError(f"Report must not replace a symbolic link: {destination}")
    old = list(TIMESTAMP.finditer(original)) if original is not None else None
    new = list(TIMESTAMP.finditer(synchronized))
    if old is not None and len(old) != len(new):
        raise ValueError("Cannot report timing changes: source and output cue counts differ.")
    lines = [
        f"SubSync {APP_VERSION} synchronization report",
        f"Completed (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Method: {method}", f"Output SRT: {output}", f"Report: {destination}",
        f"Output cues: {len(new)}", "",
        "Timing accuracy is not guaranteed. Review the beginning, middle and end.",
        "Matching coverage is not a timing-accuracy score.", "", "Settings / inputs:",
    ]
    lines.extend(f"{key}: {value}" for key, value in settings.items())
    if diagnostics:
        lines.extend(["", "Matching diagnostics:"])
        lines.extend(f"{key}: {value}" for key, value in diagnostics.items())
    if anchors:
        lines.extend(["", "Timing anchors (1-based cue, offset seconds):"])
        lines.extend(f"{cue}: {offset:+.6f}" for cue, offset in anchors)
    lines.extend(["", "Cue timings (1-based order; shifts in milliseconds):"])
    if old is None:
        lines.append("Original embedded cue timings unavailable here; see engine log for corrections.")
    for index, cue in enumerate(new):
        timing = f"{index + 1}: {cue['start']} --> {cue['end']}"
        if old is not None:
            before = old[index]
            start_shift = timestamp_to_ms(cue["start"]) - timestamp_to_ms(before["start"])
            end_shift = timestamp_to_ms(cue["end"]) - timestamp_to_ms(before["end"])
            timing += (
                f" | original {before['start']} --> {before['end']}"
                f" | start shift {start_shift:+d} ms; end shift {end_shift:+d} ms"
            )
        lines.append(timing)
    lines.extend(["", "Processing log / warnings:", *log, ""])
    with tempfile.TemporaryDirectory(prefix=".sync-report-", dir=output.parent) as folder:
        staged = Path(folder) / destination.name
        staged.write_text("\n".join(lines), encoding="utf-8")
        os.replace(staged, destination)
    return destination
