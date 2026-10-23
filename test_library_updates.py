import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import desktop_updates
import library_updates as updates


class LibraryManagerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manager = updates.LibraryManager(self.root)
        enabled = patch.object(updates, "desktop_enabled", return_value=True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def base(self):
        executable = self.manager.python("base")
        executable.parent.mkdir(parents=True)
        executable.touch()
        return executable

    def test_default_and_invalid_selection(self):
        base = self.base()
        self.assertEqual(self.manager.launch_python(), base)
        self.manager._save({"active": "../../escape", "previous": None})
        with self.assertRaisesRegex(ValueError, "Invalid library environment"):
            self.manager.launch_python()

    def test_daily_checks_filter_packages_and_manual_check_bypasses_interval(self):
        response = json.dumps([
            {"name": "fastembed", "version": "0.9.0", "latest_version": "0.10.0"},
            {"name": "pip", "version": "24", "latest_version": "25"},
            {"name": "imageio_ffmpeg", "version": "0.6.0", "latest_version": "0.7.0"},
        ])
        with patch.object(self.manager, "_run", return_value=response) as run:
            self.assertEqual(len(self.manager.check()), 2)
            self.assertIsNone(self.manager.check())
            run.assert_called_once()
            self.assertEqual(len(self.manager.check(force=True)), 2)
            state = self.manager.state()
            state["last_check"] -= updates.DAY + 1
            self.manager._save(state)
            self.assertEqual(len(self.manager.check()), 2)
            self.assertEqual(run.call_count, 3)

    def test_failed_check_is_explicit_and_not_retried_each_startup(self):
        with patch.object(self.manager, "_run", side_effect=RuntimeError("offline")) as run:
            with self.assertRaisesRegex(RuntimeError, "offline"):
                self.manager.check()
            self.assertIsNone(self.manager.check())
            run.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "offline"):
                self.manager.check(force=True)

    def test_malformed_index_response_is_an_error(self):
        for response in ('{}', '[{"name": "fastembed"}]', '[{"name": 5}]'):
            with self.subTest(response=response), patch.object(self.manager, "_run", return_value=response):
                with self.assertRaises(ValueError):
                    self.manager.check(force=True)

    def test_install_is_isolated_validated_and_reversible(self):
        base = self.base()

        def run(command, **kwargs):
            if command[1:3] == ["-m", "venv"]:
                executable = Path(command[3]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                executable.parent.mkdir(parents=True)
                executable.touch()
            return ""

        with (
            patch.object(updates.sys, "executable", str(base)),
            patch.object(updates.sys, "prefix", "/private-app-env"),
            patch.object(updates.sys, "base_prefix", "/base-python"),
            patch.object(self.manager, "_run", side_effect=run) as commands,
        ):
            self.manager.install(Mock())
        state = self.manager.state()
        self.assertEqual(state["previous"], "base")
        self.assertNotEqual(state["active"], "base")
        self.assertTrue(base.exists())
        self.assertEqual(commands.call_count, 4)
        install_command = commands.call_args_list[1].args[0]
        self.assertEqual(install_command[-len(updates.PACKAGES):], list(updates.PACKAGES))
        self.assertIn("--only-binary=:all:", install_command)
        self.assertIn("check", commands.call_args_list[2].args[0])
        self.assertIn("runpy.run_path('subtitle_sync_app.pyw')", commands.call_args_list[3].args[0][-1])
        self.assertEqual(self.manager.launch_python(), self.manager.python(state["active"]))
        self.manager.restore()
        self.assertEqual(self.manager.launch_python(), base)
        self.manager.restore()
        self.assertEqual(self.manager.state()["active"], state["active"])

    def test_install_failure_keeps_selection_and_removes_only_failed_environment(self):
        base = self.base()

        def fail(command, **kwargs):
            if command[1:3] == ["-m", "venv"]:
                Path(command[3]).mkdir()
                return ""
            raise RuntimeError("package resolver failed")

        with (
            patch.object(updates.sys, "executable", str(base)),
            patch.object(updates.sys, "prefix", "/private"),
            patch.object(updates.sys, "base_prefix", "/system"),
            patch.object(self.manager, "_run", side_effect=fail),
            self.assertRaisesRegex(RuntimeError, "resolver failed"),
        ):
            self.manager.install(Mock())
        self.assertEqual(self.manager.state()["active"], "base")
        self.assertEqual(list(self.manager.directory.glob("env-*")), [])
        self.assertTrue(base.exists())

    def test_import_validation_failure_does_not_activate_new_environment(self):
        base = self.base()
        with (
            patch.object(updates.sys, "executable", str(base)),
            patch.object(updates.sys, "prefix", "/private"),
            patch.object(updates.sys, "base_prefix", "/system"),
            patch.object(self.manager, "_run", side_effect=["", "", "", RuntimeError("bad import")]),
            self.assertRaisesRegex(RuntimeError, "bad import"),
        ):
            self.manager.install(Mock())
        self.assertEqual(self.manager.state()["active"], "base")

    def test_stale_or_system_python_cannot_install(self):
        with patch.object(updates.sys, "prefix", updates.sys.base_prefix):
            with self.assertRaisesRegex(RuntimeError, "desktop launcher"):
                self.manager.install(Mock())
        with (
            patch.object(updates.sys, "prefix", "/private"),
            patch.object(updates.sys, "base_prefix", "/system"),
            self.assertRaisesRegex(RuntimeError, "selection has changed"),
        ):
            self.manager.install(Mock())

    def test_missing_environment_has_recovery_instructions(self):
        with self.assertRaisesRegex(ValueError, "--restore-libraries"):
            self.manager.launch_python()
        with self.assertRaisesRegex(ValueError, "No previous"):
            self.manager.restore()

    def test_recovery_works_even_if_selected_environment_is_missing(self):
        base = self.base()
        self.manager._save({
            "active": "env-" + "a" * 32, "previous": "base", "last_check": 0,
        })
        self.manager.restore()
        self.assertEqual(self.manager.launch_python(), base)

    def test_failed_state_publication_does_not_change_active_environment(self):
        base = self.base()
        with (
            patch.object(updates.sys, "executable", str(base)),
            patch.object(updates.sys, "prefix", "/private"),
            patch.object(updates.sys, "base_prefix", "/system"),
            patch.object(self.manager, "_run", return_value=""),
            patch.object(self.manager, "_save", side_effect=PermissionError("state is read-only")),
            self.assertRaisesRegex(PermissionError, "state is read-only"),
        ):
            self.manager.install(Mock())
        self.assertEqual(self.manager.state()["active"], "base")

    def test_concurrent_operations_are_rejected_and_lock_releases(self):
        with self.manager._locked():
            with self.assertRaisesRegex(RuntimeError, "Another library operation"):
                self.manager.restore()
        with self.manager._locked():
            pass

    def test_offline_pip_warning_is_not_success_shaped(self):
        with patch.object(updates.subprocess, "run", return_value=SimpleNamespace(
            returncode=0, stdout="[]", stderr="WARNING: connection failed",
        )):
            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                self.manager._run(["python", "-m", "pip"], strict_stderr=True)

    def test_timeout_and_nonzero_exit_propagate(self):
        with patch.object(updates.subprocess, "run", side_effect=subprocess.TimeoutExpired("pip", 1)):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.manager._run(["python"])
        with patch.object(updates.subprocess, "run", return_value=SimpleNamespace(
            returncode=1, stdout="", stderr="resolver failure",
        )):
            with self.assertRaisesRegex(RuntimeError, "resolver failure"):
                self.manager._run(["python"])

    def test_docker_performs_no_checks_and_rejects_changes(self):
        with patch.object(updates, "desktop_enabled", return_value=False), patch.object(self.manager, "_run") as run:
            self.assertIsNone(self.manager.check(force=True))
            with self.assertRaisesRegex(RuntimeError, "desktop"):
                self.manager.install(Mock())
            with self.assertRaisesRegex(RuntimeError, "desktop"):
                self.manager.restore()
            run.assert_not_called()
        self.assertFalse(self.manager.directory.exists())


class DesktopUpdateTests(unittest.TestCase):
    def setUp(self):
        self.app = SimpleNamespace(
            root=Mock(), busy=False, setup_busy=False, interface=SimpleNamespace(tr=lambda text: text),
        )
        with patch.object(updates, "desktop_enabled", return_value=True):
            self.controller = desktop_updates.DesktopUpdates(self.app)
        self.controller.manager = Mock()

    def test_container_detection_and_explicit_disable(self):
        for marker in ("/.dockerenv", "/run/.containerenv"):
            with patch.object(Path, "exists", lambda path: str(path) == marker):
                self.assertFalse(updates.desktop_enabled())
        with patch.dict(os.environ, {"SUBTITLE_SYNC_DESKTOP_UPDATES": "0"}):
            self.assertFalse(updates.desktop_enabled())

    def test_no_container_schedule_or_controls(self):
        self.controller.enabled = False
        self.controller.schedule_check()
        self.controller.open()
        self.controller.check(force=True)
        self.app.root.after.assert_not_called()
        self.controller.manager.check.assert_not_called()
        self.assertIsNone(self.controller.window)

    def test_background_check_is_scheduled(self):
        self.controller.schedule_check()
        self.app.root.after.assert_called_once_with(1500, self.controller.check)

    def test_available_updates_open_controls_and_failures_are_visible(self):
        self.controller.updates = [{"name": "fastembed"}]
        self.controller.events.put(("done", ""))
        self.controller.running = True
        with patch.object(self.controller, "open") as open_window:
            self.controller._poll(False)
            open_window.assert_called_once()
        self.assertFalse(self.controller.running)
        self.controller.events.put(("error", "offline"))
        with patch.object(desktop_updates.messagebox, "showwarning") as warning:
            self.controller._poll(False)
            warning.assert_called_once()
        self.assertEqual(self.controller.report, "offline")

    def test_install_confirmation_and_restart_required(self):
        with (
            patch.object(desktop_updates.messagebox, "askyesno", return_value=False),
            patch.object(self.controller, "_start") as start,
        ):
            self.controller._change()
            start.assert_not_called()
        with (
            patch.object(desktop_updates.messagebox, "askyesno", return_value=True),
            patch.object(self.controller, "_start") as start,
        ):
            self.controller._change()
            operation = start.call_args.args[0]
            operation()
            self.controller.manager.install.assert_called_once()
            self.assertTrue(start.call_args.kwargs["modifying"])
        self.app.setup_busy = True
        self.controller.events.put(("done", ""))
        with patch.object(desktop_updates.messagebox, "showinfo"):
            self.controller._poll(True)
        self.assertTrue(self.controller.pending_restart)
        self.assertFalse(self.app.setup_busy)

    def test_check_result_defers_during_analysis(self):
        self.app.busy = True
        self.controller.events.put(("done", ""))
        self.controller._poll(False)
        self.assertFalse(self.controller.events.empty())
        self.app.root.after.assert_called_once()

    def test_launcher_and_container_wiring_without_building_archives(self):
        root = updates.ROOT
        linux = (root / "Start_Subtitle_Sync.sh").read_text()
        windows = (root / "start_windows.ps1").read_text()
        for script in (linux, windows):
            self.assertIn("library_updates.py restore", script)
            self.assertIn("library_updates.py launch-python", script)
        self.assertIn('--restore-libraries', (root / "Start_Subtitle_Sync.bat").read_text())
        self.assertIn("SUBTITLE_SYNC_DESKTOP_UPDATES=0", (root / "Dockerfile").read_text())
        subprocess.run(["bash", "-n", str(root / "Start_Subtitle_Sync.sh")], check=True)
        from build_portable import COMMON
        self.assertIn("library_updates.py", COMMON)
        self.assertIn("desktop_updates.py", COMMON)

    @unittest.skipUnless(os.name == "posix", "Linux launcher")
    def test_linux_launcher_selects_managed_python_and_supports_recovery(self):
        with tempfile.TemporaryDirectory(prefix="subtitle updates ") as folder:
            root = Path(folder)
            shutil.copy(updates.ROOT / "Start_Subtitle_Sync.sh", root)
            base = root / ".venv-linux" / "bin" / "python"
            base.parent.mkdir(parents=True)
            base.write_text(
                '#!/bin/sh\n'
                'if [ "$1" = "-c" ]; then exit 0; fi\n'
                'if [ "$1" = "library_updates.py" ] && [ "$2" = "restore" ]; then\n'
                '  printf "restored\\n" > recovery-result; exit 0\n'
                'fi\n'
                'if [ "$1" = "library_updates.py" ] && [ "$2" = "restore-components" ]; then\n'
                '  printf "components restored\\n" > component-recovery-result; exit 0\n'
                'fi\n'
                'if [ "$1" = "library_updates.py" ] && [ "$2" = "launch-python" ]; then\n'
                '  printf "%s/selected-python\\n" "$PWD"; exit 0\n'
                'fi\n'
                'echo "Unexpected command" >&2; exit 1\n',
                encoding="utf-8",
            )
            base.chmod(0o755)
            selected = root / "selected-python"
            selected.write_text(
                '#!/bin/sh\n[ "$1" = "subtitle_sync_app.pyw" ] || exit 1\n'
                'printf "selected\\n" > launch-result\n',
                encoding="utf-8",
            )
            selected.chmod(0o755)
            for arguments in ([], ["--restore-libraries"], ["--restore-components"]):
                subprocess.run(
                    ["bash", str(root / "Start_Subtitle_Sync.sh"), *arguments],
                    check=True, capture_output=True, text=True,
                )
                self.assertEqual((root / "launch-result").read_text(), "selected\n")
            self.assertEqual((root / "recovery-result").read_text(), "restored\n")
            self.assertEqual((root / "component-recovery-result").read_text(), "components restored\n")


if __name__ == "__main__":
    unittest.main()
