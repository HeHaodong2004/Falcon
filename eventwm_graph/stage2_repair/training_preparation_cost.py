"""Matched real TRAIN batch preparation cost; not deployment latency."""
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_online_data import inventory,OnlineRangeGroups
from .range_online_train import make_batch
from .range_online_features import LegalFeatureBuilder
from .range_spatial import SpatialEvidenceNet
from .quiet_owned_jobs import quiet_jobs


def run():
    device='cuda:1';torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;model=SpatialEvidenceNet().to(device);model.load_state_dict(torch.load(OUT/'training/range_online4096/selected_epoch02_argmax.pt',map_location='cpu')['model']);builders={n:LegalFeatureBuilder(model,device,device_canvas=n=='gpu',device_edt=n=='gpu')for n in ['cpu','gpu']};iterator=iter(OnlineRangeGroups(inventory(4)));items=[next(iterator)for _ in range(16)];batches=[items[i:i+4]for i in range(0,16,4)];records=[]
    for name,builder in builders.items():make_batch(batches[0],builder,device)
    for rep in range(3):
        for i,items in enumerate(batches):
            reference=None
            for name in ['cpu','gpu']if rep%2 else['gpu','cpu']:
                torch.cuda.synchronize(device);tick=time.perf_counter();batch,ledger=make_batch(items,builders[name],device);torch.cuda.synchronize(device);ms=(time.perf_counter()-tick)*1000;arrays={k:v.cpu().numpy()for k,v in batch.items()}
                if reference is not None:
                    for key in arrays:np.testing.assert_array_equal(arrays[key],reference[key],err_msg=key)
                reference=arrays;records.append(dict(implementation=name,batch=i,repetition=rep,ms=ms,ledger=ledger));print(name,i,round(ms,2),flush=True)
    summary={name:{'p50_ms':float(np.percentile([r['ms']for r in records if r['implementation']==name],50)),'p95_ms':float(np.percentile([r['ms']for r in records if r['implementation']==name],95))}for name in builders};save(OUT/'training_preparation_cost.json',dict(role='Matched16realTRAINgroups,4batchesx3reps. All features/labels/weights/canvasesexact. Feature preparation only; not full optimizer throughput or deployment latency.',summary=summary,calls=records));print('SUMMARY',summary,flush=True)


if __name__=='__main__':
    with quiet_jobs('training_preparation_cost',extra_training_names=['online_cardinal256_control']):run()
