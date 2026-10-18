import hashlib
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import build_portable
import ui_preferences


class PortableReleaseTests(unittest.TestCase):
    def test_archives_have_exact_allowlist_and_no_user_files_or_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = build_portable.build(Path(folder))
            windows, linux, checksums = outputs
            with zipfile.ZipFile(windows) as archive:
                names = {Path(name).name for name in archive.namelist()}
                self.assertEqual(names, set(build_portable.COMMON) | {
                    "Start_Subtitle_Sync.bat", "start_windows.ps1", "PORTABLE",
                })
                self.assertIsNone(archive.testzip())
            with tarfile.open(linux) as archive:
                self.assertEqual(
                    {Path(item.name).name for item in archive.getmembers()},
                    set(build_portable.COMMON) | {"Start_Subtitle_Sync.sh", "PORTABLE"},
                )
                launcher = next(item for item in archive.getmembers() if item.name.endswith(".sh"))
                self.assertEqual(launcher.mode, 0o755)
            for path in (windows, linux):
                self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(), checksums.read_text())

    def test_portable_settings_override_does_not_change_normal_default(self):
        with patch.dict("os.environ", {"SUBTITLE_SYNC_SETTINGS_DIR": "/portable/.config"}):
            self.assertEqual(ui_preferences.settings_path(), Path("/portable/.config/settings.json"))


if __name__ == "__main__":
    unittest.main()
