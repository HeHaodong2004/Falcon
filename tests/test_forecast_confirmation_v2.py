import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eventbeliefwm.data import sha256
from eventforecastwm.performance_v2.confirmation_population import build


class TestConfirmationV2(unittest.TestCase):
    def test_extraction_requires_frozen_population_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);protocol=directory/'protocol';protocol.mkdir()
            membership=protocol/'membership.json';membership.write_text('{}')
            checkpoint=directory/'model.pt';checkpoint.write_bytes(b'frozen fixture')
            selection=directory/'selection.json'
            cfg=dict(run=str(directory/'cache'),protocol_dir=str(protocol),selection=str(selection))
            selected=dict(status='pending',membership_sha256=sha256(membership),
                models={'primary':dict(checkpoint=str(checkpoint),sha256=sha256(checkpoint))})
            with patch('eventforecastwm.confirmation.build') as extract:
                selection.write_text(json.dumps(selected))
                with self.assertRaisesRegex(ValueError,'Freeze development'):build(cfg)
                selected['status']='development_selection_frozen';selected['membership_sha256']='changed'
                selection.write_text(json.dumps(selected))
                with self.assertRaisesRegex(ValueError,'population changed'):build(cfg)
                selected['membership_sha256']=sha256(membership);checkpoint.write_bytes(b'changed')
                selection.write_text(json.dumps(selected))
                with self.assertRaisesRegex(ValueError,'model changed'):build(cfg)
                extract.assert_not_called()
                selected['models']['primary']['sha256']=sha256(checkpoint)
                selection.write_text(json.dumps(selected))
                extract.side_effect=RuntimeError('Reserved raw extraction requires sustained recovery')
                with self.assertRaisesRegex(RuntimeError,'Research stopped: Reserved raw'):build(cfg)
                extract.assert_called_once_with(str(protocol),directory/'cache')


if __name__=='__main__':unittest.main()
