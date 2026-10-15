import json
from pathlib import Path
import tempfile
import unittest

from eventforecastwm.performance_v2.release import digest,relative_file,verify,validate_release_choice


class ReleaseIntegrityTest(unittest.TestCase):
    def test_confirmation_cannot_swap_seeds_or_select_an_unfrozen_alternative(self):
        new={role:dict(sha256='new-'+role) for role in ('primary','replicate')}
        old={role:dict(sha256='old-'+role) for role in ('primary','replicate')}
        selected=dict(models=new,fallback_models=old)
        review=dict(status='confirmation_review_complete',reason='Synthetic policy fixture',decision='promote_frozen_candidate')
        validate_release_choice(selected,review,new)
        with self.assertRaisesRegex(ValueError,'frozen primary/replicate'):
            validate_release_choice(selected,review,dict(primary=new['replicate'],replicate=new['primary']))
        review['decision']='preserve_frozen_incumbent';validate_release_choice(selected,review,old)
        with self.assertRaisesRegex(ValueError,'frozen primary/replicate'):
            validate_release_choice(selected,review,new)
        with self.assertRaisesRegex(ValueError,'frozen before confirmation'):
            validate_release_choice(dict(models=new),review,old)

    def test_changed_bytes_and_unlisted_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for role in ('primary','replicate'):(root/(role+'.pt')).write_bytes(role.encode())
            files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in root.iterdir()}
            manifest=dict(status='verified_release',source_commit='test-fixture',files=files,
                models={role:dict(sha256=files[role+'.pt']['sha256']) for role in ('primary','replicate')})
            (root/'release_manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify(root)['files_verified'],2)
            (root/'primary.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes changed'):verify(root)
            (root/'primary.pt').write_bytes(b'primary')
            (root/'unexpected').write_text('extra')
            with self.assertRaisesRegex(ValueError,'file set'):verify(root)

    def test_paths_and_symlinks_cannot_escape_the_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'release';root.mkdir()
            for name in ('../outside','/tmp/outside',''):
                with self.assertRaises(ValueError):relative_file(root,name)
            outside=Path(directory)/'outside';outside.mkdir()
            (root/'link').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):relative_file(root,'link/weights.pt')
            (root/'release_manifest.json').write_text(json.dumps(dict(status='verified_release')))
            with self.assertRaisesRegex(ValueError,'symlinks'):verify(root)


if __name__=='__main__':unittest.main()
