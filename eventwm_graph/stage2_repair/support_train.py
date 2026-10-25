"""Matched auxiliary-supervision experiment; teacher arrays are loss-only.

Both variants have exactly the same deployed architecture and legal inputs.
Only the training objective differs. No teacher head or private target file
is needed by SpatialRangePredictor.
"""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .diagnose import OUT, read, save, sha
from .fresh_data import FreshGroups, keep
from .range_spatial import SpatialEvidenceNet


class SupportGroups(FreshGroups):
    def __init__(self, manifest, targets):
        super().__init__(manifest)
        t = read(targets)
        assert t['source_manifest_sha256'] == sha(manifest)
        byroot = {r['root_id']: r for r in t['records']}
        cursor = 0
        for row in self.manifest['records']:
            teacher = byroot[row['root_id']]
            assert teacher['source_map_sha256'] == row['map_sha256']
            assert teacher['sha256'] == sha(teacher['path'])
            with np.load(teacher['path']) as z:
                labels = z['support_targets']
            for group in row['groups']:
                self.groups[cursor]['support_targets'] = labels[group['map_index']]
                cursor += 1
        assert cursor == len(self.groups)

    def __iter__(self):
        # Parent iteration deliberately exposes only legal arrays and ordinary
        # supervised increment labels. Attach private targets after that draw.
        lookup = {id(g['maps']): g['support_targets'] for g in self.groups}
        for item in super().__iter__():
            item['support_targets'] = lookup[id(item['maps'])]
            yield item


def run(manifest, targets, warm, epochs=4, device='cuda:0', name='support256'):
    torch.set_num_threads(1)
    torch.manual_seed(7811)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = SupportGroups(Path(manifest), Path(targets))
    m = data.manifest
    loader = DataLoader(data, batch_size=8, num_workers=4, collate_fn=keep,
                        persistent_workers=True, multiprocessing_context='fork', prefetch_factor=2)
    initial = SpatialEvidenceNet()
    initial.load_state_dict(torch.load(warm, map_location='cpu')['model'])
    variants = {}
    for variant, coefficient in [('control', 0.), ('teacher', .5)]:
        model = copy.deepcopy(initial).to(device)
        head = nn.Conv2d(24, 2, 1).to(device)
        parameters = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=.0002, weight_decay=.0001)
        variants[variant] = (model, head, parameters, optimizer, torch.cuda.amp.GradScaler(), coefficient)
        destination = OUT/'training'/(name+'_'+variant)
        destination.mkdir(exist_ok=False)
        cfg = dict(schema='legal_range_spatial_1', name=name+'_'+variant,
                   base_checkpoint=read(OUT/'training/range_fresh256/config.json')['base_checkpoint'],
                   warm_checkpoint=str(warm), warm_sha256=sha(warm), epochs=epochs,
                   roots=m['roots'], scenes=m['scenes'], learning_rate=.0002,
                   manifest_sha256=sha(manifest), teacher_target_manifest_sha256=sha(targets),
                   auxiliary_coefficient=coefficient,
                   inputs='Unchanged legal G/path/past-current ideal ranges/tracks. No private teacher array is an input.',
                   supervision='Unchanged fresh original-cell increment BCE/Dice/class-UNION; optional dense projected-navmesh support and signed-distance auxiliary loss.',
                   deployed_architecture='Unchanged SpatialEvidenceNet; two-channel teacher head discarded after training.',
                   sources={f:sha(Path(__file__).with_name(f)) for f in ['support_train.py','support_teacher.py','range_spatial.py','fresh_data.py']})
        save(destination/'config.json', cfg)
    del initial
    tick = time.perf_counter()
    learning = {v: [] for v in variants}
    def tensor(x, dtype=None):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=dtype)).to(device)
    for epoch in range(1, epochs+1):
        data.epoch.value = epoch
        total = {v: np.zeros(4) for v in variants}
        batches = 0
        for items in loader:
            x=tensor(np.concatenate([g['x'] for g in items]),np.float32)
            extra=tensor(np.concatenate([g['extra'] for g in items]),np.float32)
            y=tensor(np.concatenate([g['y'] for g in items]),np.int64)
            weight=tensor(np.concatenate([g['weight'] for g in items]),np.float32)
            indices=tensor(np.concatenate([g['indices'] for g in items]),np.int64)
            ids=tensor(np.repeat(np.arange(len(items)),[len(g['y']) for g in items]),np.int64)
            maps=tensor(np.stack([g['maps'] for g in items]),np.float32)
            certificate=tensor(np.stack([g['aux'] for g in items]),np.float32)
            certificate_mask=tensor(np.stack([g['mask'] for g in items]),np.float32)
            teacher=tensor(np.stack([g['support_targets'] for g in items]),np.float32)
            aw=tensor([g['aux_weight'] for g in items],np.float32)
            domain=(maps[:,12].square()+maps[:,13].square()<=1.44).float()
            domain=domain*(1+3*(1-maps[:,5]).clamp(0,1))
            for variant,(model,head,parameters,optimizer,scaler,coefficient) in variants.items():
                with torch.cuda.amp.autocast():
                    encoded=model.encode(maps)
                    rl,bl,_=model.raw(x,extra,indices,ids,encoded)
                    auxlogit=model.aux(encoded[0])[:,0]
                    teacher_logits=head(encoded[0])
                revelation=(y>0).float()
                ce=F.binary_cross_entropy_with_logits(rl.float(),revelation,pos_weight=rl.new_tensor(8.),reduction='none')
                ce=ce+8*revelation*F.binary_cross_entropy_with_logits(bl.float(),(y==2).float(),reduction='none')
                main=(ce*weight).sum()*len(data.groups)/(m['roots']*len(items))
                pr=(rl.float()-np.log(8.)).sigmoid(); pb=bl.float().sigmoid(); dice=0.
                for cls,prob in [(1,pr*(1-pb)),(2,pr*pb)]:
                    truth=(y==cls).float()
                    dice+=1-(2*(weight*prob*truth).sum()+1.e-7)/((weight*(prob+truth)).sum()+1.e-7)
                ac=F.binary_cross_entropy_with_logits(auxlogit.float(),certificate,reduction='none')
                aux=(((ac*certificate_mask).sum((1,2))/certificate_mask.sum((1,2)).clamp_min(1))*aw).mean()*len(data.groups)/m['roots']
                tc=F.binary_cross_entropy_with_logits(teacher_logits[:,0].float(),teacher[:,0],reduction='none')
                tc=tc+(teacher_logits[:,1].float().tanh()-teacher[:,1]).square()
                support=(((tc*domain).sum((1,2))/domain.sum((1,2)).clamp_min(1))*aw).mean()*len(data.groups)/m['roots']
                loss=main+.1*dice+.25*aux+coefficient*support
                if not torch.isfinite(loss):raise ValueError('nonfinite auxiliary experiment')
                optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer)
                gradient=torch.nn.utils.clip_grad_norm_(parameters,5.)
                scaler.step(optimizer);scaler.update()
                total[variant]+=[float(main),float(dice),float(aux),float(support)]
            batches+=1
        for variant,(model,head,parameters,optimizer,scaler,coefficient) in variants.items():
            destination=OUT/'training'/(name+'_'+variant)
            record=dict(epoch=epoch,losses=(total[variant]/batches).tolist(),seconds=time.perf_counter()-tick)
            learning[variant].append(record);save(destination/'learning.json',learning[variant]);print(variant,record,flush=True)
            cfg=read(destination/'config.json')
            torch.save(dict(schema=cfg['schema'],config=cfg,epoch=epoch,threshold=.2,model=model.state_dict(),training_only_teacher_head=head.state_dict()),destination/('epoch%02d.pt'%epoch))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--targets',required=True);p.add_argument('--warm',required=True);p.add_argument('--epochs',type=int,default=4);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='support256');run(**vars(p.parse_args()))
