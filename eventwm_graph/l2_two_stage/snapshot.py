"""Reuse the audited online SQLite backup and independent-inode copy contract.

Only output paths are redirected. Source roots and collector settings are read
only. All sealed roots at the new cutoff are inventoried, with the original
scene exclusions, not an outcome cap. Compatible older futures are deduplicated.
"""
import json
import shutil
import sqlite3
from .common import OUT, PREVIOUS, dump, sha


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    from eventwm_graph.l2_final import snapshot, supplement
    snapshot.OUT = OUT
    if not (OUT/'snapshot/manifest.json').exists():snapshot.run()
    source_inventory = PREVIOUS / "campaign_inventory_initial.json"
    shutil.copyfile(source_inventory, OUT / "campaign_inventory_initial.json")
    supplement.OUT = OUT
    if (OUT/'supplement').exists() and not list((OUT/'supplement').iterdir()):
        (OUT/'supplement').rmdir()
    # The completed copy is immutable. Its WAL header otherwise asks SQLite
    # to create sidecars in our sealed directory. Never apply immutable mode
    # to the live source journal.
    original_connect=sqlite3.connect
    def connect(path,*args,**kwargs):
        if str(path)=='file:'+str(OUT/'snapshot/journal.sqlite3')+'?mode=ro':
            path=str(path)+'&immutable=1'
        return original_connect(path,*args,**kwargs)
    sqlite3.connect=connect
    try:supplement.run()
    finally:sqlite3.connect=original_connect
    dump(OUT / "snapshot_provenance.json", {
        "schema": "two_stage_snapshot_provenance_1",
        "snapshot_implementation_sha256": sha(snapshot.__file__),
        "supplement_implementation_sha256": sha(supplement.__file__),
        "inventory_sha256": sha(OUT / "campaign_inventory_initial.json"),
        "controls": [], "protected_payloads_opened": False,
        "inventory_note": "Reuse previous campaign inventory; reviewer verifies no new omitted real campaigns.",
    })


if __name__ == "__main__":
    run()
