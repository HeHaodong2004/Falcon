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
