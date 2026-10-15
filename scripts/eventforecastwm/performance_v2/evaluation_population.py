"""Explicit development or already-confirmed populations for diagnostics."""
from pathlib import Path
from eventbeliefwm.data import read,sha256


def population(cfg):
    if cfg.get('population','development')=='development':
        protocol=read(cfg['protocol']);ids=protocol['development']
        if set(ids)&set(protocol['subsets']['full']):raise ValueError('Development/fitting root leakage')
        return ids,dict(population='development',protocol_sha256=sha256(cfg['protocol']),confirmation=False)
    if cfg['population']!='confirmation':raise ValueError('Unknown diagnostic population')
    receipt_path=Path(cfg['confirmation_receipt']);receipt=read(receipt_path)
    identity_path=receipt_path.parent/'identity.json';identity=read(identity_path)
    decision=read(cfg['decision']);membership=read(cfg['membership'])
    expected=dict(decision_sha256=sha256(cfg['decision']),membership_sha256=sha256(cfg['membership']),
                  manifest_sha256=sha256(cfg['manifest']))
    if receipt.get('identity_sha256')!=sha256(identity_path) or any(identity.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed confirmation identity mismatch')
    if decision['membership_sha256']!=expected['membership_sha256'] or decision['manifest_sha256']!=expected['manifest_sha256']:
        raise ValueError('Frozen decision population mismatch')
    names=[name for name,spec in decision['models'].items() if spec['sha256']==sha256(cfg['checkpoint'])]
    bank=cfg['bank']
    if not names or not any(bank in receipt['models'].get(name,{}) for name in names):
        raise ValueError('Diagnostic checkpoint/bank was not in completed frozen confirmation')
    return membership['banks'][bank],dict(population='confirmation',bank=bank,confirmation=True,
        **expected,confirmation_receipt_sha256=sha256(receipt_path),source_sha256=sha256(__file__))
