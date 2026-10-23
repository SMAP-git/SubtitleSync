import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sync_report import report_path, write_report


class SyncReportTests(unittest.TestCase):
    original = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"
    synchronized = "1\n00:00:02,500 --> 00:00:03,500\nHello\n"

    def test_filename_preserves_language_and_removes_generated_suffix(self):
        for name in ("Episode.pt.synced.srt", "Episode.pt.audio-synced.srt", "Episode.pt.srt"):
            self.assertEqual(report_path(Path(name)), Path("Episode.pt.sync.txt"))
        self.assertEqual(report_path(Path("Episode.pt-PT.synced.srt")), Path("Episode.pt-PT.sync.txt"))

    def write(self, output, **kwargs):
        return write_report(
            output, method="Test alignment", original=self.original, synchronized=self.synchronized,
            settings={"Source": "Episode.pt.srt"}, log=["Section boundary: cue 2"], **kwargs,
        )

    def test_utf8_report_contains_actual_timing_changes_diagnostics_anchors_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "Episódio.pt.synced.srt"
            output.write_text(self.synchronized)
            path = self.write(
                output, diagnostics={"matched_cues": 1, "match_percent": 100},
                anchors=[(1, 1.5)],
            )
            text = path.read_text(encoding="utf-8")
            for expected in (
                str(output), "start shift +1500 ms", "end shift +1500 ms",
                "matched_cues: 1", "1: +1.500000", "Section boundary: cue 2",
                "not a timing-accuracy score",
            ):
                self.assertIn(expected, text)
            self.assertEqual(output.read_text(), self.synchronized)
            self.assertFalse(list(Path(directory).glob(".sync-report-*")))
            self.write(output)
            self.assertEqual(path.read_text().count("SubSync"), 1)

    def test_protects_selected_input_and_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode.pt.synced.srt"
            destination = report_path(output)
            destination.write_text("selected input")
            with self.assertRaisesRegex(ValueError, "selected input"):
                self.write(output, protected_paths=[destination])
            alias = Path(directory) / "alias.srt"
            os.link(destination, alias)
            with self.assertRaisesRegex(ValueError, "selected input"):
                self.write(output, protected_paths=[alias])
            self.assertEqual(destination.read_text(), "selected input")

    def test_failure_keeps_previous_report_and_subtitle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode.pt.synced.srt"
            output.write_text(self.synchronized)
            destination = report_path(output)
            destination.write_text("previous report")
            with patch("sync_report.os.replace", side_effect=PermissionError("Read-only report")):
                with self.assertRaises(PermissionError):
                    self.write(output)
            self.assertEqual(destination.read_text(), "previous report")
            self.assertEqual(output.read_text(), self.synchronized)
            self.assertFalse(list(Path(directory).glob(".sync-report-*")))

    def test_embedded_source_reports_available_output_timings_without_inventing_shifts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_report(
                Path(directory) / "episode.en.audio-synced.srt",
                method="ffsubsync", original=None, synchronized=self.synchronized,
                settings={}, log=["offset: 1.5 seconds"],
            )
            text = path.read_text()
            self.assertIn("Original embedded cue timings unavailable", text)
            self.assertIn("00:00:02,500 --> 00:00:03,500", text)
            self.assertNotIn("start shift", text)
            self.assertIn("offset: 1.5 seconds", text)
