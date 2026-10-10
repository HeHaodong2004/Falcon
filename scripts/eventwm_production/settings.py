"""Workstation locations live in the explicit production configuration."""
import json
import os
from pathlib import Path

RELEASE_REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get('EVENTWM_PRODUCTION_CONFIG',
                                 str(RELEASE_REPO / 'configs/eventwm_300k_production.json'))).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text())
PROJECT = Path(CONFIG['project'])
SCIENCE = Path(CONFIG['scientific_checkout'])
DATA = Path(CONFIG['output'])
LOGS = Path(CONFIG['logs'])
TARGET = CONFIG['retained_target']


def publication_epoch():
    """Measure publication from the baseline of this scientific source epoch."""
    if CONFIG.get('recovery_gate'):
        receipt = json.loads((DATA / 'recovery_epochs' / CONFIG['scientific_commit'] / 'APPLIED.json').read_text())
        return dict(time=receipt['applied_at'],
                    retained_before_resume=receipt['retained_new_after_old_publication'])
    return json.loads((PROJECT / 'output/eventwm_72h_v4_resume_checkpoint.json').read_text())


MEASUREMENT_PREFIX = 'eventwm_static_recovery' if CONFIG.get('recovery_gate') else 'eventwm_72h_v4'
