import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from eventbeliefwm.data import sha256
from eventforecastwm.performance_v2.evaluation_population import population


class TestDiagnosticPopulation(unittest.TestCase):
    def test_post_confirmation_requires_exact_frozen_panel(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def save(name,value):
                path=root/name;path.write_text(json.dumps(value));return str(path)
            checkpoint=root/'model.pt';checkpoint.write_bytes(b'frozen fixture')
            membership=save('membership.json',dict(banks={'natural_reference':['new']}))
            manifest=save('manifest.json',dict(roots=[dict(root_id='new')]))
            decision=save('decision.json',dict(membership_sha256=sha256(membership),manifest_sha256=sha256(manifest),
                models={'primary':dict(sha256=sha256(checkpoint))}))
            identity=save('identity.json',dict(decision_sha256=sha256(decision),membership_sha256=sha256(membership),manifest_sha256=sha256(manifest)))
            receipt=save('result.json',dict(identity_sha256=sha256(identity),models={'primary':{'natural_reference':{}}}))
            cfg=dict(population='confirmation',confirmation_receipt=receipt,decision=decision,membership=membership,
                manifest=manifest,checkpoint=str(checkpoint),bank='natural_reference')
            ids,evidence=population(cfg);self.assertEqual(ids,['new']);self.assertTrue(evidence['confirmation'])
            # Regression: supplementary physics used the old development IDs
            # even after the confirmation panel was completed and frozen.
            from eventforecastwm.performance_v2 import physics_diagnostics
            wrong_protocol=save('old_protocol.json',dict(development=['old'],subsets={'full':['fit']}))
            physics_cfg=dict(cfg,run=str(root/'physics'),protocol=wrong_protocol)
            with patch.object(physics_diagnostics,'CooperativeGuard'), \
                 patch.object(physics_diagnostics.torch,'load',return_value={'fit_root_ids':['fit']}), \
                 patch.object(physics_diagnostics,'load_model',return_value=Mock()), \
                 patch.object(physics_diagnostics,'load_roots',side_effect=RuntimeError('checked population boundary')) as load:
                with self.assertRaisesRegex(RuntimeError,'checked population boundary'):
                    physics_diagnostics.run(physics_cfg)
                self.assertEqual(load.call_args.args[1],['new'])
            checkpoint.write_bytes(b'new unconfirmed model')
            with self.assertRaisesRegex(ValueError,'not in completed'):population(cfg)
            checkpoint.write_bytes(b'frozen fixture');save('membership.json',dict(banks={'natural_reference':['changed']}))
            with self.assertRaisesRegex(ValueError,'identity mismatch'):population(cfg)

    def test_development_is_never_a_fitting_fold(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'protocol.json';path.write_text(json.dumps(dict(development=['a'],subsets={'full':['a']})))
            with self.assertRaisesRegex(ValueError,'leakage'):population(dict(protocol=str(path)))


if __name__=='__main__':unittest.main()
