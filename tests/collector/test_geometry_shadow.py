import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import hashlib, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import eventwm_geometry_v4 as geometry
import eventwm_pipeline_io as io


class ShadowReference(unittest.TestCase):
    def test_reference_uses_shadow_and_train_has_separate_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);nav=base/'navmesh';nav.write_bytes(b'static')
            scene={'scene':{'navmesh':str(nav)},'navmesh_sha256':hashlib.sha256(b'static').hexdigest()}
            original={'start_position':[6,0,0],'info':{}}
            feasible={'start_position':[12,0,0],'info':{}}
            root=dict(root_id='test',seed=1,human_count=2,scene='scene',recipe={'case':'case'},_v3_preparation_covered={'2_0_0':7})
            with patch('eventwm_mass_collection.episode_for',return_value=original) as v3:
                result=io.prepare_episode(base,scene,root,{'4_0_0':99})
                self.assertIsNone(result['error']);self.assertEqual(result['start_region'],'2_0_0')
                self.assertEqual(v3.call_args.args[-1],{'2_0_0':7})
            root['root_id']='train';root['recipe']['episode_sampler']={'version':geometry.VERSION,'component':'geometry_feasible'}
            with patch('eventwm_mass_collection.episode_for',return_value=original),patch('eventwm_geometry_v4.episode',return_value=(feasible,{})) as changed:
                result=io.prepare_episode(base,scene,root,{'4_0_0':99})
                self.assertEqual(result['start_region'],'4_0_0');self.assertEqual(result['v3_shadow_start_region'],'2_0_0')
                self.assertEqual(changed.call_args.args[3],{'4_0_0':99})

    def test_shadow_failure_does_not_fabricate_original_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);nav=base/'navmesh';nav.write_bytes(b'static')
            scene={'scene':{'navmesh':str(nav)},'navmesh_sha256':hashlib.sha256(b'static').hexdigest()}
            root=dict(root_id='failure',seed=1,human_count=0,scene='scene',recipe={'case':'case'},_v3_preparation_covered={})
            with patch('eventwm_mass_collection.episode_for',side_effect=RuntimeError('static_episode_unavailable: no connected 4-10m goal')):
                result=io.prepare_episode(base,scene,root,{})
                self.assertIsNone(result['v3_shadow_start_region']);self.assertTrue(result['error'])
