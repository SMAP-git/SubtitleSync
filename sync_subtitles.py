#!/usr/bin/env python3
"""Retiming SRT subtitles between manually verified audio sync points.

Create an anchor file with entries like:
    [
      {"cue": 1, "offset_seconds": 0.0},
      {"cue": 120, "offset_seconds": 1.25},
      {"cue": 240, "offset_seconds": 3.0}
    ]

Cue numbers are 1-based. An offset is added to both timestamps of that cue;
positive values move a cue later and negative values move it earlier. Offsets
for other cues are linearly interpolated by their original start time.

Run:
    python sync_subtitles.py "input.srt" --anchors "anchors.json"

The default output is "input.synced.srt". The input is never overwritten.
"""

from __future__ import annotations

import argparse
import bisect
import json
import errno
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


TIMESTAMP = re.compile(
    r"(?P<start>\d{2,}:\d{2}:\d{2},\d{3})"
    r"(?P<arrow>\s+-->\s+)"
    r"(?P<end>\d{2,}:\d{2}:\d{2},\d{3})"
)


def publish_new_file(staged: Path, output: Path) -> None:
    """Publish without overwriting, including shares that cannot create hard links."""
    try:
        os.link(staged, output)
        return
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV):
            raise
        logging.info("Hard links unavailable for %s; using exclusive file creation.", output)
    # Exclusive creation still rejects existing files and symlinks on network shares.
    with staged.open("rb") as source:
        destination = output.open("xb")
        try:
            with destination:
                shutil.copyfileobj(source, destination)
        except BaseException:
            output.unlink()
            raise


def decode_srt(data: bytes) -> tuple[str, str, bool]:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8"), "utf-8", True
    try:
        return data.decode("utf-8"), "utf-8", False
    except UnicodeDecodeError:
        return data.decode("cp1252"), "cp1252", False


def timestamp_to_ms(value: str) -> int:
    hours, minutes, remainder = value.split(":")
    seconds, milliseconds = remainder.split(",")
    return (
        ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000
        + int(milliseconds)
    )


def ms_to_timestamp(value: int) -> str:
    value = max(0, value)
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_anchors(path: Path, cue_count: int) -> list[tuple[int, float]]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read anchors file {path}: {error}") from error

    if not isinstance(raw, list) or not raw:
        raise ValueError("The anchors file must contain a non-empty JSON list.")

    anchors: list[tuple[int, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each anchor must be an object with cue and offset_seconds.")
        cue = item.get("cue")
        offset = item.get("offset_seconds")
        if isinstance(cue, bool) or not isinstance(cue, int) or not 1 <= cue <= cue_count:
            raise ValueError(f"Anchor cue must be an integer from 1 to {cue_count}.")
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise ValueError(f"Anchor {cue} must have a numeric offset_seconds value.")
        offset = float(offset)
        if not math.isfinite(offset):
            raise ValueError(f"Anchor {cue} has a non-finite offset.")
        if anchors and cue <= anchors[-1][0]:
            raise ValueError("Anchor cue numbers must be unique and in increasing order.")
        anchors.append((cue, offset))

    return anchors


def interpolate_offset(
    time_ms: int, anchors: list[tuple[int, float]]
) -> float:
    anchor_times = [point[0] for point in anchors]
    next_index = bisect.bisect_right(anchor_times, time_ms)
    if next_index == 0:
        return anchors[0][1]
    if next_index == len(anchors):
        return anchors[-1][1]

    left_time, left_offset = anchors[next_index - 1]
    right_time, right_offset = anchors[next_index]
    if time_ms == left_time:
        return left_offset
    fraction = (time_ms - left_time) / (right_time - left_time)
    return left_offset + fraction * (right_offset - left_offset)


def sync_srt(text: str, anchors: list[tuple[int, float]]) -> tuple[str, int]:
    timestamp_matches = list(TIMESTAMP.finditer(text))
    cue_times = [timestamp_to_ms(match.group("start")) for match in timestamp_matches]
    if not cue_times:
        raise ValueError("No SRT subtitle cues were found.")
    if len(cue_times) < anchors[-1][0]:
        raise ValueError(
            f"The SRT has {len(cue_times)} cues but an anchor refers to cue {anchors[-1][0]}."
        )

    time_anchors = [
        (cue_times[cue - 1], offset)
        for cue, offset in anchors
    ]
    if any(
        right_time <= left_time
        for (left_time, _), (right_time, _) in zip(time_anchors, time_anchors[1:])
    ):
        raise ValueError("Anchor cue timestamps must be strictly increasing.")
    if any(
        right_time + round(right_offset * 1000)
        <= left_time + round(left_offset * 1000)
        for (left_time, left_offset), (right_time, right_offset) in zip(
            time_anchors, time_anchors[1:]
        )
    ):
        raise ValueError("Anchor points must map to strictly increasing video times.")

    lines = text.splitlines(keepends=True)
    cue = 0
    changed = 0
    synced_lines: list[str] = []
    cue_timings: list[tuple[int, int, int, int, int]] = []

    for line_index, line in enumerate(lines):
        match = TIMESTAMP.search(line)
        if not match:
            synced_lines.append(line)
            continue

        cue += 1
        start_ms = timestamp_to_ms(match.group("start"))
        end_ms = timestamp_to_ms(match.group("end"))
        original_start_ms = start_ms
        original_end_ms = end_ms
        offset_ms = round(interpolate_offset(start_ms, time_anchors) * 1000)
        new_start = ms_to_timestamp(start_ms + offset_ms)
        new_end = ms_to_timestamp(end_ms + offset_ms)
        cue_timings.append(
            (
                line_index,
                start_ms + offset_ms,
                end_ms + offset_ms,
                original_start_ms,
                original_end_ms,
            )
        )
        if new_start != match.group("start") or new_end != match.group("end"):
            changed += 1
        synced_lines.append(
            line[: match.start()]
            + new_start
            + match.group("arrow")
            + new_end
            + line[match.end() :]
        )

    if cue == 0:
        raise ValueError("No SRT subtitle cues were found.")

    for cue_index, (current, following) in enumerate(
        zip(cue_timings, cue_timings[1:]), start=1
    ):
        line_index, start_ms, end_ms, original_start_ms, original_end_ms = current
        _, next_start_ms, _, next_original_start_ms, _ = following
        if (
            end_ms > next_start_ms
            and original_end_ms <= next_original_start_ms
        ):
            if next_start_ms <= start_ms:
                raise ValueError(
                    f"Sync anchors collapse subtitle cue {cue_index} to zero or negative duration."
                )
            match = TIMESTAMP.search(synced_lines[line_index])
            if match is None:
                raise ValueError("Internal error while checking adjusted subtitle cues.")
            synced_lines[line_index] = (
                synced_lines[line_index][: match.start("end")]
                + ms_to_timestamp(next_start_ms)
                + synced_lines[line_index][match.end("end") :]
            )

    return "".join(synced_lines), cue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retiming an SRT using piecewise-linear, manually verified sync anchors."
    )
    parser.add_argument("srt", type=Path, help="Input SRT file (left unchanged)")
    parser.add_argument("--anchors", required=True, type=Path, help="JSON file of verified cue offsets")
    parser.add_argument("--output", type=Path, help="Output SRT (default: <input>.synced.srt)")
    args = parser.parse_args()

    try:
        source = args.srt.resolve()
        output = args.output.resolve() if args.output else source.with_name(
            f"{source.stem}.synced{source.suffix}"
        )
        if source == output:
            raise ValueError("Output must be a different file from the input SRT.")

        original_bytes = source.read_bytes()
        text, encoding, had_bom = decode_srt(original_bytes)
        cue_count = len(TIMESTAMP.findall(text))
        anchors = parse_anchors(args.anchors, cue_count)
        synced_text, cue_count = sync_srt(text, anchors)

        encoded = synced_text.encode(encoding)
        if had_bom:
            encoded = b"\xef\xbb\xbf" + encoded
        output.write_bytes(encoded)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Synced {cue_count} cues using {len(anchors)} anchor points.")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
