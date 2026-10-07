import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('primary_storage', Path(__file__).parents[1]/'scripts/eventwm_primary_storage.py')
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def setup(tmp_path):
    dataset = tmp_path/'dataset'
    (dataset/'roots').mkdir(parents=True)
    primary = storage.configure(dataset, tmp_path/'primary', 'frozen-scientific-commit')
    return dataset, primary


def actual_relative_link_write_and_read(tmp_path):
    dataset, primary = setup(tmp_path)
    assert storage.reserve_root(dataset, primary, 'v3m_test')
    logical = dataset/'roots/v3m_test'
    (logical/'probe.json').write_text('{"filesystem_probe": true}')
    assert json.loads((primary/'roots/v3m_test/probe.json').read_text()) == {'filesystem_probe': True}
    assert logical.resolve() == primary/'roots/v3m_test'
    assert not storage.reserve_root(dataset, primary, 'v3m_test')
    assert (logical/'probe.json').is_file()
    marker = json.loads((primary/'manifest.json').read_text())
    assert marker['protocol_version'] == storage.PROTOCOL
    assert len((primary/'manifest.json').read_bytes()) < 1024


def preserve_existing_root_and_reject_wrong_link(tmp_path):
    dataset, primary = setup(tmp_path)
    existing = dataset/'roots/v3m_existing'
    existing.mkdir()
    (existing/'original').write_text('preserved')
    assert not storage.reserve_root(dataset, primary, 'v3m_existing')
    assert not existing.is_symlink() and (existing/'original').read_text() == 'preserved'
    (dataset/'roots/v3m_wrong').symlink_to(existing, target_is_directory=True)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'Unexpected root link'):
        storage.reserve_root(dataset, primary, 'v3m_wrong')


def reject_foreign_storage_and_untracked_root_data(tmp_path):
    dataset, primary = setup(tmp_path)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'Existing storage metadata differs'):
        storage.configure(dataset, primary, 'different-scientific-commit')
    untracked = primary/'roots/v3m_untracked'
    untracked.mkdir()
    (untracked/'original').write_text('preserved')
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'unidentified root data'):
        storage.reserve_root(dataset, primary, 'v3m_untracked')
    assert (untracked/'original').read_text() == 'preserved'
    with unittest.TestCase().assertRaises(ValueError):
        storage.reserve_root(dataset, primary, 'v3m_../../unsafe')


def reject_unidentified_nonempty_storage(tmp_path):
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    primary = tmp_path/'primary'
    primary.mkdir()
    (primary/'unrelated').write_text('preserved')
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'unidentified nonempty storage'):
        storage.configure(dataset, primary, 'commit')
    assert (primary/'unrelated').read_text() == 'preserved'


class PrimaryStorageTests(unittest.TestCase):
    def run_fixture(self, check):
        with tempfile.TemporaryDirectory(prefix='eventwm-primary-storage-') as directory:
            check(Path(directory))

    def test_relative_link(self):
        self.run_fixture(actual_relative_link_write_and_read)

    def test_existing_roots(self):
        self.run_fixture(preserve_existing_root_and_reject_wrong_link)

    def test_foreign_storage(self):
        self.run_fixture(reject_foreign_storage_and_untracked_root_data)

    def test_unidentified_storage(self):
        self.run_fixture(reject_unidentified_nonempty_storage)


if __name__ == '__main__':
    unittest.main()
