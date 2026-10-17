"""Dataset ingestion regressions, independent of native image feature libraries."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from solarvision.data import PREFIX, REVISION, prepare, read_labels


class DatasetPathTests(unittest.TestCase):
    def archive(self, root, name):
        archive = root / 'fixture.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(PREFIX + 'labels.csv', 'images/a.png 0 mono\n')
            z.writestr(PREFIX + name, b'fixture')
            z.writestr(f'elpv-dataset-{REVISION}/LICENSE', 'Fixture license')
        return archive

    def test_windows_and_posix_archive_escapes_rejected_before_extraction(self):
        for name in ('../escape.txt', 'images/../../escape.txt',
                     r'images\..\..\escape.txt', r'C:\escape.txt',
                     '/escape.txt', 'images/a.png:stream', 'images/.. /escape.txt', 'images/NUL'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                archive = self.archive(root, name)
                with self.assertRaises(ValueError):
                    prepare(root / 'data', archive)
                self.assertFalse((root / 'escape.txt').exists())
                self.assertFalse((root / 'data/labels.csv').exists())

    def test_valid_archive_still_prepares(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = prepare(root / 'data', self.archive(root, 'images/a.png'))
            self.assertEqual(manifest['count'], 1)
            self.assertEqual((root / 'data/images/a.png').read_bytes(), b'fixture')

    def test_label_paths_reject_windows_escapes_and_aliases(self):
        for name in (r'images/a\..\..\escape.png', 'images/a.png:stream',
                     'images/../escape.png', 'images//a.png', 'images'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                (Path(d) / 'labels.csv').write_text(f'{name} 0 mono\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_labels(d)

    def test_existing_symlink_cannot_redirect_archive_or_label(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data, outside = root / 'data', root / 'outside'
            data.mkdir(); outside.mkdir()
            try:
                (data / 'images').symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest('Creating symlinks requires permission on this platform')
            with self.assertRaises(ValueError):
                prepare(data, self.archive(root, 'images/a.png'))
            (data / 'labels.csv').write_text('images/a.png 0 mono\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_labels(data)
            self.assertEqual(list(outside.iterdir()), [])
