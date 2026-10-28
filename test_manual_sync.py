import json
import errno
import os
import queue
import stat
import tempfile
import threading
import time
import tkinter as tk
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import manual_sync
import subtitle_alignment as alignment
import sync_subtitles
from test_subtitle_devices import load_gui


def fixture(cues):
    return "\n\n".join(
        f"{number}\n{sync_subtitles.ms_to_timestamp(cue.start_ms)} --> "
        f"{sync_subtitles.ms_to_timestamp(cue.end_ms)}\n{cue.text}"
        for number, cue in enumerate(cues, 1)
    ) + "\n"


class SyncOutputTests(unittest.TestCase):
    def test_network_share_without_hardlinks_publishes_exclusively(self):
        with patch.object(sync_subtitles.os, "link", side_effect=OSError(errno.EPERM, "Operation not permitted")):
            sync_subtitles.publish_new_file(self.source, self.output)
            self.assertEqual(self.output.read_bytes(), self.source.read_bytes())
            with self.assertRaises(FileExistsError):
                sync_subtitles.publish_new_file(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), self.source.read_bytes())

    def test_network_copy_failure_removes_partial_output(self):
        def fail_copy(source, destination):
            destination.write(b"partial")
            raise OSError("Network disconnected")

        with (
            patch.object(sync_subtitles.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "Not supported")),
            patch.object(sync_subtitles.shutil, "copyfileobj", side_effect=fail_copy),
        ):
            with self.assertRaisesRegex(OSError, "Network disconnected"):
                sync_subtitles.publish_new_file(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.source.exists())

    def test_existing_network_symlink_and_permission_errors_are_not_overwritten(self):
        self.output.symlink_to(self.source)
        with patch.object(sync_subtitles.os, "link", side_effect=OSError(errno.EPERM, "Not supported")):
            with self.assertRaises(FileExistsError):
                sync_subtitles.publish_new_file(self.source, self.output)
        self.assertTrue(self.output.is_symlink())
        self.output.unlink()
        with patch.object(sync_subtitles.os, "link", side_effect=PermissionError(errno.EACCES, "Denied")):
            with self.assertRaises(PermissionError):
                sync_subtitles.publish_new_file(self.source, self.output)
        self.assertFalse(self.output.exists())

    def setUp(self):
        self.gui = load_gui()
        self.addCleanup(self.gui._test_settings_directory.cleanup)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.source = self.folder / "source.srt"
        self.source.write_bytes(b"1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        self.output = self.folder / "output.srt"
        self.app = self.gui.SubtitleSyncApp.__new__(self.gui.SubtitleSyncApp)
        self.app.messages = queue.Queue()
        self.app.sync_cancel = threading.Event()
        self.app.sync_commit_lock = threading.Lock()
        self.app.sync_committed = False

    def synchronize(self):
        with patch.object(self.gui.subtitle_alignment, "analyze_subtitles", return_value=(
            [(1, 0)], {"matched_cues": 1},
        )):
            self.app._sync_worker(None, self.source, self.source, self.output, 1000, 70)
        messages = []
        while not self.app.messages.empty():
            messages.append(self.app.messages.get())
        self.assertFalse(list(self.folder.glob(".subtitle-sync-*")))
        return messages[-1]

    @unittest.skipUnless(os.name == "posix", "POSIX file permissions")
    def test_new_and_replaced_outputs_respect_umask(self):
        for mask, expected in ((0o002, 0o664), (0o022, 0o644), (0o077, 0o600)):
            for existing in (False, True):
                with self.subTest(umask=oct(mask), existing=existing):
                    self.output.unlink(missing_ok=True)
                    if existing:
                        self.output.write_text("original", encoding="utf-8")
                        self.output.chmod(0o600)
                    self.app.sync_committed = False
                    previous = os.umask(mask)
                    try:
                        message = self.synchronize()
                    finally:
                        os.umask(previous)
                    self.assertEqual(message[0], "success")
                    self.assertTrue(self.app.sync_committed)
                    self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), expected)
                    self.assertEqual(self.output.read_bytes(), self.source.read_bytes())

    def test_cancellation_after_staging_preserves_existing_output(self):
        self.output.write_bytes(b"original")
        write_bytes = Path.write_bytes

        def cancel_after_write(path, data):
            written = write_bytes(path, data)
            self.app.sync_cancel.set()
            return written

        with patch.object(Path, "write_bytes", new=cancel_after_write):
            message = self.synchronize()
        self.assertEqual(message[0], "cancelled")
        self.assertFalse(self.app.sync_committed)
        self.assertEqual(self.output.read_bytes(), b"original")

    def test_replace_failure_is_reported_and_preserves_existing_output(self):
        self.output.write_bytes(b"original")
        with patch.object(Path, "replace", side_effect=PermissionError("Cannot replace output")):
            message = self.synchronize()
        self.assertEqual(message, ("error", "Cannot replace output"))
        self.assertFalse(self.app.sync_committed)
        self.assertEqual(self.output.read_bytes(), b"original")


class TimingScaleTests(unittest.TestCase):
    def test_real_episode_timing_noise_does_not_masquerade_as_a_section_cut(self):
        # Timing-only samples from the reported 25-FPS failure; no dialogue/model dependency.
        pairs = [
            (19841, 20588), (65320, 67901), (162360, 169603), (210921, 221922),
            (285801, 298532), (376480, 393594), (424560, 442743), (452921, 472673),
            (503721, 526059), (613201, 640107), (651361, 681414), (699080, 730063),
            (767881, 802169), (819080, 856523), (859600, 898365), (898201, 939006),
            (940240, 985085), (960000, 1006073), (1016880, 1064331), (1053400, 1101301),
            (1076560, 1125292), (1106681, 1156456), (1127001, 1178378), (1154920, 1206973),
            (1188041, 1241741), (1237201, 1292892), (1385360, 1450683), (1415440, 1480212),
            (1439921, 1508207), (1480241, 1548347), (1564960, 1636969), (1614960, 1689355),
            (1669321, 1745878), (1704641, 1783950), (1802441, 1886118), (1831000, 1915982),
            (1961241, 2050816), (2002040, 2093692), (2042040, 2136602), (2154320, 2255487),
            (2182001, 2284350), (2211761, 2315781), (2257481, 2363161), (2344201, 2453652),
            (2406720, 2519184),
        ]
        source = [alignment.Cue(start, start + 2000, "Distinct dialogue") for start, _ in pairs]
        matches = [(i + 1, i + 1, end - start, 0.9) for i, (start, end) in enumerate(pairs)]
        estimate = alignment._estimate_timing_scale(source, matches)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.scale, 1.046917, places=5)
        self.assertLess(estimate.validation_error_ms, 1500)
        residuals = [
            target - (estimate.scale * start + estimate.offset_ms) for start, target in pairs
        ]
        self.assertTrue(any(
            1500 < abs(left) < 4500 and 1500 < abs(right) < 4500 and left * right > 0
            for left, right in zip(residuals, residuals[1:])
        ))
        for factor in (0.5, 1.0, 2.0):
            with self.subTest(direction=factor):
                changed = [replace(
                    cue, start_ms=round(cue.start_ms * factor), end_ms=round(cue.end_ms * factor),
                ) for cue in source]
                changed_matches = [
                    (i + 1, i + 1, target - changed[i].start_ms, 0.9)
                    for i, (_, target) in enumerate(pairs)
                ]
                adjusted = alignment._estimate_timing_scale(changed, changed_matches)
                self.assertIsNotNone(adjusted)
                self.assertAlmostEqual(adjusted.scale * factor, estimate.scale, places=5)

    def source_cues(self, count=24):
        return [
            alignment.Cue(120000 + i * 300000, 122000 + i * 300000, f"Distinct timing dialogue {i}")
            for i in range(count)
        ]

    def matches(self, source, scale, offset=15000):
        return [
            (i + 1, i + 1, round(cue.start_ms * scale + offset) - cue.start_ms, 1.0)
            for i, cue in enumerate(source)
        ]

    def test_estimator_supports_common_rates_in_every_direction(self):
        source = self.source_cues()
        rates = (24000 / 1001, 24, 25, 30000 / 1001, 30, 50, 60000 / 1001, 60)
        for before in rates:
            for after in rates:
                with self.subTest(before=before, after=after):
                    scale = before / after
                    matches = self.matches(source, scale)
                    estimate = alignment._estimate_timing_scale(source, matches)
                    if before == after:
                        self.assertIsNone(estimate)
                    else:
                        self.assertIsNotNone(estimate)
                        self.assertAlmostEqual(estimate.scale, scale, places=6)
                        self.assertLess(estimate.validation_error_ms, 2)

    def test_estimator_ignores_sparse_outliers_but_preserves_manual_anchor(self):
        source = self.source_cues()
        scale = 25 / (24000 / 1001)
        matches = self.matches(source, scale)
        for index in (4, 17):
            number, ref, shift, score = matches[index]
            matches[index] = number, ref, shift + 40000, score
        manual = source[12].start_ms, round(source[12].start_ms * scale + 15000)
        estimate = alignment._estimate_timing_scale(source, matches, manual)
        self.assertIsNotNone(estimate)
        self.assertEqual(len(estimate.matches), 22)
        self.assertAlmostEqual(manual[0] * estimate.scale + estimate.offset_ms, manual[1])
        self.assertLess(estimate.validation_error_ms, 2)
        self.assertIsNone(alignment._estimate_timing_scale(
            source, matches, (manual[0], manual[1] + 20000),
        ))

    def test_estimator_rejects_cuts_nonlinear_drift_and_failed_held_out_predictions(self):
        source = self.source_cues()
        for shifts in (
            [0 if i < 12 else 30000 for i in range(24)],
            [round(cue.start_ms * 0.04) + (0 if i < 22 else 30000)
             for i, cue in enumerate(source)],
            [i * i * 250 for i in range(24)],
            [round(cue.start_ms * 0.04) + (0 if i % 2 == 0 else 8000)
             for i, cue in enumerate(source)],
        ):
            with self.subTest(shifts=shifts):
                matches = [(i + 1, i + 1, shift, 1.) for i, shift in enumerate(shifts)]
                self.assertIsNone(alignment._estimate_timing_scale(source, matches))

    def test_estimator_rejects_insufficient_or_concentrated_evidence(self):
        source = self.source_cues()
        matches = self.matches(source, 1.04)
        self.assertIsNone(alignment._estimate_timing_scale(source, matches[:8]))
        self.assertIsNone(alignment._estimate_timing_scale(source, matches[:10]))

    def test_analysis_recovers_long_drift_in_both_directions_and_keeps_manual_pair_exact(self):
        source = self.source_cues()
        for scale in (25 / (24000 / 1001), (24000 / 1001) / 25, 2.5, 0.4):
            for automatic in (False, True):
                with self.subTest(scale=scale, automatic=automatic):
                    reference = [
                        replace(cue, start_ms=round(cue.start_ms * scale + 15000),
                                end_ms=round(cue.end_ms * scale + 15000),
                                text=f"Reference translation sentence {i}")
                        for i, cue in enumerate(source)
                    ]
                    model = Mock()
                    model.embed.side_effect = lambda texts, **_: [
                        np.eye(len(source))[int(text.rsplit(" ", 1)[-1])] for text in texts
                    ]
                    messages = []
                    with patch.object(alignment, "_load_embedding_model", return_value=model):
                        anchors, diagnostics = alignment.analyze_subtitles(
                            source, reference, 0, source_anchor=12, reference_anchor=12,
                            automatic_initial=automatic, progress=messages.append,
                        )
                    self.assertEqual(diagnostics["timing_scale_applied"], 1)
                    self.assertAlmostEqual(diagnostics["timing_scale"], scale, places=6)
                    self.assertEqual(diagnostics["matched_cues"], len(source) - 1)
                    if not automatic:
                        self.assertIn((12, (reference[11].start_ms - source[11].start_ms) / 1000), anchors)
                    synced, count = sync_subtitles.sync_srt(
                        fixture(source), anchors, timing_scale=diagnostics["timing_scale"],
                    )
                    self.assertEqual(count, len(source))
                    for actual, expected, original in zip(alignment.parse_srt(synced), reference, source):
                        self.assertLessEqual(abs(actual.start_ms - expected.start_ms), 1)
                        self.assertLessEqual(abs(actual.end_ms - expected.end_ms), 2)
                        self.assertEqual(actual.text, original.text)
                    model.embed.assert_called_once()
                    self.assertTrue(any("Timing scale accepted" in message for message in messages))

    def test_scale_retimes_both_boundaries_and_extrapolates_outside_anchors(self):
        source = self.source_cues(5)
        scale = 1.25
        anchors = [(3, (source[2].start_ms * (scale - 1) + 5000) / 1000)]
        synced, _ = sync_subtitles.sync_srt(fixture(source), anchors, timing_scale=scale)
        for cue, original in zip(alignment.parse_srt(synced), source):
            self.assertEqual(cue.start_ms, round(original.start_ms * scale + 5000))
            self.assertEqual(cue.end_ms, round(original.end_ms * scale + 5000))
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                sync_subtitles.sync_srt(fixture(source), anchors, timing_scale=invalid)
        with self.assertRaisesRegex(ValueError, "negative timestamps"):
            sync_subtitles.sync_srt(fixture(source), [(3, -1000.)], timing_scale=scale)

    def test_vulkan_analysis_uses_the_same_scale_detection_and_serializable_diagnostics(self):
        source = self.source_cues(12)
        reference = [replace(
            cue, start_ms=round(cue.start_ms * 1.04), end_ms=round(cue.end_ms * 1.04),
        ) for cue in source]
        model = Mock()
        model.embed.side_effect = lambda texts, **_: [
            np.eye(len(source))[int(text.rsplit(" ", 1)[-1])] for text in texts
        ]
        with patch.object(alignment.vulkan_embeddings, "VulkanEmbedding", return_value=model):
            anchors, diagnostics = alignment.analyze_subtitles(
                source, reference, 0, backend="Vulkan", automatic_initial=True,
            )
        self.assertEqual(diagnostics["timing_scale_applied"], 1)
        self.assertAlmostEqual(diagnostics["timing_scale"], 1.04, places=6)
        self.assertEqual(len(anchors), len(source))
        self.assertEqual(json.loads(json.dumps(diagnostics)), diagnostics)
        model.close.assert_called_once()

    def test_estimator_accepts_small_boundary_jitter_without_snapping_to_an_fps_ratio(self):
        source = self.source_cues()
        scale = 1.037123
        matches = self.matches(source, scale)
        matches = [
            (number, ref, shift + ((index % 3) - 1) * 300, score)
            for index, (number, ref, shift, score) in enumerate(matches)
        ]
        estimate = alignment._estimate_timing_scale(source, matches)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.scale, scale, delta=0.0001)
        self.assertLess(estimate.validation_error_ms, 500)

    def test_repeated_dialogue_is_not_global_scale_evidence(self):
        source = self.source_cues()
        vectors = {cue.text: np.array([1., 0.]) for cue in source}
        self.assertEqual(alignment._global_dialogue_matches(
            source, source, vectors, 70, lambda _: None,
        ), [])

    def test_timing_estimation_checks_cancellation_between_hypothesis_groups(self):
        source = self.source_cues()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            alignment._estimate_timing_scale(
                source, self.matches(source, 1.04),
                progress=Mock(side_effect=RuntimeError("cancelled")),
            )


class ContextMatchingTests(unittest.TestCase):
    def test_automatic_initial_pair_ignores_credits_and_finds_consistent_nonfirst_dialogue(self):
        source = [
            alignment.Cue(1000, 3000, "Downloaded From www.example.org"),
            *[alignment.Cue(20000 * i + 10000, 20000 * i + 12000, f"Distinct source dialogue {i}") for i in range(5)],
        ]
        reference = [
            alignment.Cue(1000, 3000, "Unrelated introduction"),
            *[alignment.Cue(20000 * i + 17000, 20000 * i + 19000, f"Reference dialogue translation {i}") for i in range(5)],
        ]
        vectors = {"Unrelated introduction": np.eye(6)[5]}
        for i in range(5):
            vectors[source[i + 1].text] = vectors[reference[i + 1].text] = np.eye(6)[i]
        left, right = alignment._automatic_initial_pair(source, reference, vectors, 70, lambda _: None)
        self.assertGreater(left, 1)
        self.assertEqual(left, right)
        self.assertEqual(reference[right - 1].start_ms - source[left - 1].start_ms, 7000)

    def test_automatic_initial_pair_rejects_ambiguous_and_insufficient_matches(self):
        cues = [alignment.Cue(i * 20000, i * 20000 + 1000, f"Repeated dialogue wording {i}") for i in range(4)]
        vectors = {cue.text: np.array([1., 0.]) for cue in cues}
        for source, reference in ((cues, cues), (cues[:2], cues)):
            with self.assertRaisesRegex(ValueError, "Select matching cues manually"):
                alignment._automatic_initial_pair(source, reference, vectors, 70, lambda _: None)

    def test_automatic_initial_pair_rejects_competing_offset_groups(self):
        source = [alignment.Cue(i * 20000, i * 20000 + 1000, f"Source dialogue content {i}") for i in range(6)]
        reference = [
            alignment.Cue(cue.start_ms + (5000 if i < 3 else 25000), cue.end_ms + (5000 if i < 3 else 25000),
                          f"Reference dialogue content {i}")
            for i, cue in enumerate(source)
        ]
        vectors = {}
        for i in range(6):
            vectors[source[i].text] = vectors[reference[i].text] = np.eye(6)[i]
        with self.assertRaisesRegex(ValueError, "Select matching cues manually"):
            alignment._automatic_initial_pair(source, reference, vectors, 70, lambda _: None)

    def test_analysis_automatically_selects_anchor_using_existing_embeddings_once(self):
        source = [alignment.Cue(i * 60000 + 10000, i * 60000 + 12000, f"Unique dialogue phrase {i}") for i in range(5)]
        reference = [replace(cue, start_ms=cue.start_ms + 5000, end_ms=cue.end_ms + 5000) for cue in source]
        vectors = {cue.text: np.eye(5)[i] for i, cue in enumerate(source)}
        model = Mock()
        model.embed.side_effect = lambda texts, **kwargs: [vectors[text] for text in texts]
        with patch.object(alignment, "_load_embedding_model", return_value=model):
            anchors, diagnostics = alignment.analyze_subtitles(source, reference, 0, automatic_initial=True)
        self.assertEqual(anchors, [(i, 5.) for i in range(1, 6)])
        self.assertEqual(diagnostics["automatic_initial"], 1)
        self.assertEqual(diagnostics["reference_anchor"], diagnostics["source_anchor"])
        model.embed.assert_called_once()

    def test_out_of_order_credit_does_not_block_dialogue_alignment(self):
        source = [
            alignment.Cue(10000, 11000, "First dialogue"),
            alignment.Cue(70000, 71000, "Second dialogue"),
            alignment.Cue(3000, 5000, "Downloaded From www.example.org"),
        ]
        reference = [
            replace(cue, start_ms=cue.start_ms + 2000, end_ms=cue.end_ms + 2000)
            for cue in source[:2]
        ]
        model = Mock()
        model.embed.side_effect = lambda texts, **kwargs: [
            np.array([1., 0.]) if "First" in text else np.array([0., 1.]) for text in texts
        ]
        messages = []
        with patch.object(alignment, "_load_embedding_model", return_value=model):
            anchors, diagnostics = alignment.analyze_subtitles(
                source, reference, 12000, source_anchor=1, reference_anchor=1, progress=messages.append,
            )
        synced, count = sync_subtitles.sync_srt(fixture(source), anchors)
        self.assertEqual(count, 3)
        self.assertEqual([cue.text for cue in alignment.parse_srt(synced)], [cue.text for cue in source])
        self.assertEqual(anchors, [(1, 2.), (2, 2.)])
        self.assertTrue(any("Out-of-order credit" in message for message in messages))
        self.assertEqual(diagnostics["total_cues"], 3)

    def test_out_of_order_dialogue_is_still_rejected(self):
        cues = [
            alignment.Cue(10000, 11000, "First dialogue"),
            alignment.Cue(3000, 4000, "Misplaced dialogue"),
        ]
        with self.assertRaisesRegex(ValueError, "dialogue cues must be in chronological order"):
            alignment.analyze_subtitles(cues, cues, 10000)

    def test_long_single_whisper_cues_are_usable_without_expanding_multi_cue_windows(self):
        cues = [
            alignment.Cue(2000, 20000, "A complete sentence from a speech recognition segment."),
            alignment.Cue(21000, 23000, "The next sentence."),
        ]
        windows = alignment._build_windows(cues, 70)
        self.assertEqual(
            windows,
            [alignment.DialogueWindow(cue.start_ms, cue.end_ms, cue.text) for cue in cues],
        )

    def test_windows_preserve_non_latin_dialogue_and_split_sentences(self):
        for text in ("你好，世界", "こんにちは", "مرحبا بالعالم", "Привет мир", "안녕하세요"):
            windows = alignment._build_windows([alignment.Cue(1000, 2000, text)], 70)
            self.assertEqual(windows[0].text, text)
        cues = [
            alignment.Cue(1000, 2000, "We should"),
            alignment.Cue(2200, 3200, "leave before dawn."),
            alignment.Cue(12000, 13000, "New scene"),
        ]
        texts = [window.text for window in alignment._build_windows(cues, 70)]
        self.assertIn("We should leave before dawn.", texts)
        self.assertNotIn("We should leave before dawn. New scene", texts)
        context = alignment._window_context(alignment.DialogueWindow(2200, 3200, "leave before dawn."), cues)
        self.assertEqual(context, "We should leave before dawn.")

    def test_context_can_prefer_paraphrase_over_literal_distractor(self):
        source = [
            alignment.Cue(0, 1000, "Manually paired opening"),
            alignment.Cue(20000, 21000, "Source idiomatic expression"),
        ]
        reference = [
            alignment.Cue(0, 1000, "Unrelated wording for selected opening"),
            alignment.Cue(20000, 21000, "Contextually correct paraphrase"),
            alignment.Cue(22000, 23000, "Literal but wrong scene"),
        ]
        left = [alignment.DialogueWindow(c.start_ms, c.end_ms, c.text) for c in source]
        right = [alignment.DialogueWindow(c.start_ms, c.end_ms, c.text) for c in reference]
        e0, e1, e2, e3 = np.eye(4)
        embeddings = {window.text: e0 for window in left + right}
        embeddings[right[1].text] = 0.65 * e0 + np.sqrt(1 - 0.65 ** 2) * e1
        embeddings[right[2].text] = 0.75 * e0 + np.sqrt(1 - 0.75 ** 2) * e2
        embeddings.update({"context:" + window.text: e3 for window in left + right})
        embeddings["context:" + right[2].text] = e2
        model = Mock()
        model.embed.side_effect = lambda texts, **_: [embeddings[text] for text in texts]
        with (
            patch.object(alignment, "_build_windows", side_effect=[left, right]),
            patch.object(alignment, "_window_context", side_effect=lambda window, _: "context:" + window.text),
            patch.object(alignment, "_load_embedding_model", return_value=model),
        ):
            anchors, diagnostics = alignment.analyze_subtitles(
                source, reference, 9999, source_anchor=1, reference_anchor=1
            )
        self.assertEqual(anchors, [(1, 0.0), (2, 0.0)])
        self.assertEqual(diagnostics["anchor_time_ms"], 0)
        embedded = model.embed.call_args.args[0]
        self.assertIn("context:" + left[1].text, embedded)
        self.assertIn("context:" + right[1].text, embedded)

    def test_middle_manual_anchor_is_exact_and_timeline_is_aligned_on_both_sides(self):
        source = [
            alignment.Cue(i * 60000 + 10000, i * 60000 + 12000, f"Distinct dialogue {i}")
            for i in range(5)
        ]
        reference = [replace(cue, start_ms=cue.start_ms + 20000, end_ms=cue.end_ms + 20000) for cue in source]
        model = Mock()
        model.embed.return_value = np.eye(5)
        with patch.object(alignment, "_load_embedding_model", return_value=model):
            anchors, diagnostics = alignment.analyze_subtitles(
                source, reference, 0, source_anchor=3, reference_anchor=3
            )
        self.assertEqual(anchors, [(i, 20.0) for i in range(1, 6)])
        self.assertEqual(diagnostics["anchor_time_ms"], reference[2].start_ms)
        output, count = sync_subtitles.sync_srt(fixture(source), anchors)
        self.assertEqual(count, 5)
        self.assertEqual(alignment.parse_srt(output), reference)

    def test_invalid_manual_indices_fail_before_inference(self):
        cues = [alignment.Cue(0, 1000, "Dialogue")]
        for left, right in ((0, 1), (2, 1), (1, 2), (True, 1), (None, 1)):
            with (
                self.subTest(left=left, right=right),
                patch.object(alignment, "_load_embedding_model") as load,
                self.assertRaises(ValueError),
            ):
                alignment.analyze_subtitles(cues, cues, 0, source_anchor=left, reference_anchor=right)
            load.assert_not_called()

    def test_manual_pair_can_have_different_source_and_reference_cue_numbers(self):
        source = [
            alignment.Cue(10000 + i * 60000, 12000 + i * 60000, f"Distinct dialogue {i}")
            for i in range(5)
        ]
        reference = [
            replace(cue, start_ms=cue.start_ms + 7000, end_ms=cue.end_ms + 7000)
            for cue in source[1:]
        ]
        model = Mock()
        model.embed.side_effect = lambda texts, **_: [
            np.eye(5)[int(text.rsplit(" ", 1)[-1])] for text in texts
        ]
        with patch.object(alignment, "_load_embedding_model", return_value=model):
            anchors, _ = alignment.analyze_subtitles(
                source, reference, 0, source_anchor=3, reference_anchor=2
            )
        self.assertIn((3, 7.0), anchors)
        self.assertTrue(all(offset == 7.0 for _, offset in anchors))
        self.assertGreaterEqual(len(anchors), 3)

    def test_all_embedded_subrip_languages_are_offered(self):
        tracks = [
            {"id": 1, "type": "subtitles", "codec": "SubRip/SRT", "properties": {"language": "jpn"}},
            {"id": 2, "type": "subtitles", "codec": "SubRip/SRT", "properties": {"language_ietf": "es", "track_name": "Spanish"}},
            {"id": 3, "type": "subtitles", "codec": "SubRip/SRT", "properties": {"language": "eng"}},
            {"id": 4, "type": "subtitles", "codec": "HDMV PGS", "properties": {"language": "fra"}},
        ]
        with (
            patch.object(alignment, "_find_tool", side_effect=lambda name: name),
            patch.object(alignment.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps({"tracks": tracks}), stderr=""
            )),
        ):
            found = alignment.list_subtitle_tracks(Path("video.mkv"))
        self.assertEqual([track.track_id for track in found], [1, 2, 3])
        self.assertIn("jpn", found[0].label)
        self.assertIn("Spanish", found[1].label)

    def test_extract_uses_explicit_reference_track_id(self):
        expected = [alignment.Cue(2000, 3000, "Hola")]

        def extract(command, **_):
            track_id, path = command[-1].split(":", 1)
            self.assertEqual(track_id, "7")
            Path(path).write_text(fixture(expected), encoding="utf-8")
            return SimpleNamespace(returncode=0, stderr="")

        with (
            patch.object(alignment, "_find_tool", return_value="mkvextract"),
            patch.object(alignment.subprocess, "run", side_effect=extract),
        ):
            self.assertEqual(alignment.extract_reference_srt(Path("video.mkv"), 7), expected)


@unittest.skipUnless(os.environ.get("DISPLAY") or os.name == "nt", "Needs a display")
class ManualSyncGuiTests(unittest.TestCase):
    def setUp(self):
        self.gui = load_gui()
        self.root = tk.Tk()
        self.app = self.gui.SubtitleSyncApp(self.root)
        self.source = tuple(
            alignment.Cue(i * 60000, i * 60000 + 2000, text)
            for i, text in enumerate(("Bonjour", "Nous partons demain", "A bientot"))
        )
        self.reference = tuple(
            alignment.Cue(i * 60000 + 5000, i * 60000 + 7000, text)
            for i, text in enumerate(("こんにちは", "明日出発します", "また会いましょう"))
        )

    def tearDown(self):
        for identifier in self.root.tk.call("after", "info"):
            self.root.after_cancel(identifier)
        self.root.destroy()

    def test_popup_requires_both_selections_and_preserves_original_cue_numbers_after_search(self):
        accept = Mock()
        dialog = manual_sync.CuePairDialog(self.root, self.source, self.reference, accept)
        self.root.update()
        self.assertEqual(str(dialog.confirm.cget("state")), "disabled")
        dialog.searches[0].set("demain")
        self.root.update()
        self.assertEqual(dialog.trees[0].get_children(), ("2",))
        dialog.trees[0].selection_set("2")
        dialog.trees[1].selection_set("3")
        self.root.update()
        self.assertEqual(str(dialog.confirm.cget("state")), "normal")
        self.assertIn("Source cue 2 = reference cue 3", dialog.summary.get())
        dialog.confirm.invoke()
        accept.assert_called_once_with(2, 3)

    def test_app_accepts_pair_and_invalidates_it_when_an_input_changes(self):
        self.app._show_pair_picker(self.source, self.reference, ())
        self.root.update()
        window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        widgets = list(descendants(window))
        trees = [widget for widget in widgets if widget.winfo_class() == "Treeview"]
        trees[0].selection_set("2")
        trees[1].selection_set("3")
        self.root.update()
        next(widget for widget in widgets if widget.winfo_class() == "TButton"
             and widget.cget("text") == "Use selected pair").invoke()
        self.assertEqual(self.app.sync_selection.source_number, 2)
        self.assertEqual(self.app.sync_selection.reference_number, 3)
        self.assertEqual(self.app.initial_time.get(), self.gui.format_time(self.reference[2].start_ms))
        self.app.reference_path.set("a-different-reference.srt")
        self.assertIsNone(self.app.sync_selection)
        self.assertEqual(self.app.initial_time.get(), "")

    def test_analysis_requires_manual_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            source.write_text(fixture(self.source), encoding="utf-8")
            self.app.source_path.set(str(source))
            self.app.output_path.set(str(Path(folder) / "output.srt"))
            with (
                patch.object(self.gui.messagebox, "showwarning") as warning,
                patch.object(self.gui.threading, "Thread") as worker,
            ):
                self.app._start_sync()
            warning.assert_called_once()
            worker.assert_not_called()

    def test_worker_forwards_manual_pair_to_all_backends_and_keeps_reference_unchanged(self):
        for backend, device, private in (("ONNX", "CPU", False), ("Vulkan", "AMD", False),
                                         ("ONNX", "NVIDIA", True)):
            with self.subTest(backend=backend, device=device), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "source.srt"
                reference = Path(folder) / "reference.srt"
                output = Path(folder) / "output.srt"
                source.write_text(fixture(self.source), encoding="utf-8")
                reference.write_text(fixture(self.reference), encoding="utf-8")
                selection = manual_sync.SyncSelection(
                    self.source, self.reference, 2, 2,
                    manual_sync.input_signatures((str(source), str(reference))),
                )
                with (
                    patch.object(self.gui.gpu_runtime, "runtime_ready", return_value=private),
                    patch.object(alignment, "analyze_subtitles", return_value=([(2, 5.0)], {})) as analyze,
                    patch.object(self.gui.gpu_runtime, "run_worker", return_value={
                        "anchors": [[2, 5.0]], "diagnostics": {},
                    }) as worker,
                ):
                    self.app._sync_worker(None, source, reference, output, 65000, 70, device, 0, backend, selection)
                self.assertTrue(output.is_file())
                self.assertEqual(alignment.read_srt(output)[1].start_ms, self.reference[1].start_ms)
                self.assertEqual(reference.read_text(), fixture(self.reference))
                if private:
                    self.assertEqual(worker.call_args.args[2]["source_anchor"], 2)
                    self.assertEqual(worker.call_args.args[2]["reference_anchor"], 2)
                else:
                    self.assertEqual(analyze.call_args.kwargs["source_anchor"], 2)
                    self.assertEqual(analyze.call_args.kwargs["reference_anchor"], 2)

    def test_file_changes_after_selection_abort_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            reference = Path(folder) / "reference.srt"
            output = Path(folder) / "output.srt"
            source.write_text(fixture(self.source), encoding="utf-8")
            reference.write_text(fixture(self.reference), encoding="utf-8")
            selection = manual_sync.SyncSelection(
                self.source, self.reference, 2, 2,
                manual_sync.input_signatures((str(source), str(reference))),
            )
            reference.write_text("changed externally", encoding="utf-8")
            with patch.object(alignment, "analyze_subtitles") as analyze:
                self.app._sync_worker(None, source, reference, output, 65000, 70, selection=selection)
            analyze.assert_not_called()
            self.assertFalse(output.exists())
            messages = []
            while not self.app.messages.empty():
                messages.append(self.app.messages.get_nowait())
            self.assertTrue(any(kind == "error" and "changed" in text for kind, text in messages))

    def test_loading_external_subtitles_opens_the_pair_picker(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            reference = Path(folder) / "reference.srt"
            source.write_text(fixture(self.source), encoding="utf-8")
            reference.write_text(fixture(self.reference), encoding="utf-8")
            self.app.source_path.set(str(source))
            self.app.reference_path.set(str(reference))
            with patch.object(self.app, "_show_pair_picker") as show:
                self.app._open_initial_sync()
                deadline = time.monotonic() + 3
                while self.app.busy and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.01)
                self.assertFalse(self.app.busy)
                show.assert_called_once()
                self.assertEqual(show.call_args.args[:2], (self.source, self.reference))

    def test_embedded_track_picker_loads_user_selected_language(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            video = Path(folder) / "video.mkv"
            source.write_text(fixture(self.source), encoding="utf-8")
            video.write_bytes(b"mock video")
            self.app.source_path.set(str(source))
            self.app.video_path.set(str(video))
            with (
                patch.object(alignment, "list_subtitle_tracks", return_value=[
                    alignment.SubtitleTrack(3, "3: eng"),
                    alignment.SubtitleTrack(7, "7: jpn"),
                ]),
                patch.object(alignment, "extract_reference_srt", return_value=list(self.reference)) as extract,
                patch.object(self.app, "_show_pair_picker") as show,
            ):
                self.app._open_initial_sync()
                deadline = time.monotonic() + 3
                while self.app.busy and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.01)
                window = next(child for child in self.root.winfo_children() if isinstance(child, tk.Toplevel))
                combo = next(child for child in window.winfo_children() if child.winfo_class() == "TCombobox")
                combo.current(1)
                next(child for child in window.winfo_children() if child.winfo_class() == "TButton").invoke()
                deadline = time.monotonic() + 3
                while self.app.busy and time.monotonic() < deadline:
                    self.root.update()
                    time.sleep(0.01)
                extract.assert_called_once_with(video, 7)
                self.assertEqual(show.call_args.args[:2], (self.source, self.reference))

    def test_reference_cannot_be_used_as_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.srt"
            reference = Path(folder) / "reference.srt"
            source.write_text(fixture(self.source), encoding="utf-8")
            reference.write_text(fixture(self.reference), encoding="utf-8")
            self.app.source_path.set(str(source))
            self.app.reference_path.set(str(reference))
            self.app.output_path.set(str(reference))
            self.app.sync_selection = manual_sync.SyncSelection(
                self.source, self.reference, 2, 2,
                manual_sync.input_signatures((str(source), str(reference))),
            )
            with (
                patch.object(self.gui.messagebox, "showerror") as error,
                patch.object(self.gui.threading, "Thread") as worker,
            ):
                self.app._start_sync()
            error.assert_called_once()
            worker.assert_not_called()
            self.assertEqual(reference.read_text(), fixture(self.reference))


if __name__ == "__main__":
    unittest.main()
