#!/usr/bin/env python3
"""Exercise the downloadable guide script against disposable Git repositories."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'static/downloads/git-worktrees/cw.sh'


class WorktreeManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='worktree guide ')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / 'project repo'
        self.repo.mkdir()
        self.git('init', '-b', 'main')
        for name in ['packages/api/index.js', 'packages/shared/index.js',
                     'packages/web/index.js', '.claude/settings.json', 'README.md']:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('initial\n')
        (self.repo / '.gitignore').write_text('.env\n')
        (self.repo / '.env').write_text('LOCAL_TEST_VALUE=example\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Guide Test', '-c', 'user.email=guide@example.test',
                 'commit', '-m', 'Initial files')

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.repo), *args], check=True,
                              capture_output=True, text=True)

    def manager(self, *args, copy_env=False, check=True):
        env = {**os.environ, 'CW_COPY_ENV': '1' if copy_env else '0'}
        return subprocess.run(['bash', str(SCRIPT), *args], cwd=self.repo / 'packages/api',
                              env=env, check=check, capture_output=True, text=True)

    def path(self, branch):
        return self.repo.parent / 'project repo-worktrees' / branch

    def test_nested_branch_environment_and_dirty_cleanup(self):
        self.manager('setup', 'feature/auth', copy_env=True)
        path = self.path('feature/auth')
        self.assertEqual((path / '.env').read_text(), (self.repo / '.env').read_text())
        self.assertEqual((path / '.claude/settings.json').read_text(), 'initial\n')
        (path / 'README.md').write_text('Uncommitted work\n')
        result = self.manager('cleanup', 'feature/auth', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((path / 'README.md').read_text(), 'Uncommitted work\n')
        self.git('show-ref', '--verify', 'refs/heads/feature/auth')
        subprocess.run(['git', '-C', str(path), 'restore', 'README.md'], check=True)
        self.manager('cleanup', 'feature/auth')
        self.assertFalse(path.exists())

    def test_existing_branch_sparse_checkout_and_failed_creation(self):
        self.git('branch', 'feature/api')
        self.manager('setup', 'feature/api', 'packages/api')
        path = self.path('feature/api')
        self.assertTrue((path / 'packages/api/index.js').is_file())
        self.assertFalse((path / 'packages/web/index.js').exists())
        self.assertTrue((path / 'README.md').is_file())
        self.assertFalse((path / '.env').exists())
        (path / '.env').write_text('KEEP_EXISTING=1\n')
        result = self.manager('setup', 'feature/api', copy_env=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((path / '.env').read_text(), 'KEEP_EXISTING=1\n')
        self.manager('cleanup', 'feature/api')
        self.assertFalse(path.exists())

    def test_rejects_invalid_branch_and_main_checkout_branch(self):
        self.assertNotEqual(self.manager('setup', '../escape', check=False).returncode, 0)
        self.assertNotEqual(self.manager('setup', 'main', check=False).returncode, 0)
        self.assertTrue((self.repo / 'README.md').is_file())
        self.assertIn(str(self.repo), self.manager('list').stdout)


if __name__ == '__main__':
    unittest.main()
