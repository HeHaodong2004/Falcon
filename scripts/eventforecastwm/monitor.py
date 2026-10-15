"""Read-only production recovery monitoring; updates research guard evidence."""
import argparse
import json
from pathlib import Path
import time
from eventforecastwm.guard import Guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--minutes',type=float,default=30.)
    a=p.parse_args();directory=Path(a.directory);started=time.time()
    while time.time()-started<a.minutes*60:
        try:
            guard=Guard(directory)
            print(json.dumps(dict(recovered=True,timestamp=time.time(),baseline=guard.state['baseline_qph'])),flush=True)
            return
        except RuntimeError as error:
            print(json.dumps(dict(recovered=False,timestamp=time.time(),reason=str(error))),flush=True)
        time.sleep(60.)


if __name__=='__main__':main()
