"""Inspect unused START-view render codes on32realTRAINwindows.

The current partial measured mesh is not a full visibility oracle. Render
codes are legal derived inputs, and targets are attached only after feature
construction. This diagnostic neither changes labels nor selects a model.
"""
from collections import defaultdict
import numpy as np
import torch
from .diagnose import OUT,save
from .range_spatial import SpatialEvidenceNet
from .range_online_data import inventory,OnlineRangeGroups
from .range_online_features import LegalFeatureBuilder
from .full_evaluate import labels


def run():
    device='cuda:1';torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;model=SpatialEvidenceNet().to(device).eval();model.load_state_dict(torch.load(OUT/'training/range_online4096/selected_epoch02_argmax.pt',map_location='cpu')['model']);builder=LegalFeatureBuilder(model,device,device_canvas=True,device_edt=True);data=OnlineRangeGroups(inventory(4));counts=np.zeros((3,3,3,3),np.int64);records=[];initial=[];original=builder.range_kernel.module.launch
    def capture(name,total,values,threads=128):
        result=original(name,total,values,threads)
        if name=='pack_evidence':
            frame,area=values[:2];cells,queries=values[4:6];init=values[9];code=frame.reshape(-1,area.value)[init[queries.long()].long(),cells.long()];initial.append(torch.where(code>0,1,torch.where(code<0,2,0)).to(torch.uint8).cpu().numpy())
        return result
    builder.range_kernel.module.launch=capture
    with torch.inference_mode():
        for item in data:
            windows=item['windows'][:32-len(records)];paths=[e['path']for e in windows];prepared,contexts=builder.prepare(keys=item['keys'],states=item['states'],heights=item['heights'],geometry=item['geometry'],ego=item['ego'],now=item['now'],paths=paths);keys=[c.candidates('both')for c in contexts];initial.clear();xs,extras,canvas,low,_=builder.build(prepared=prepared,contexts=contexts,keysets=keys,known_keys=item['keys'],known_states=item['states'],known_heights=item['heights'],ego=item['ego'],packet=item['packet'],tracks=item['tracks']);assert len(initial)==1;offset=0
            with torch.cuda.amp.autocast():encoded=model.encode(canvas[None])
            for e,k,x,extra in zip(windows,keys,xs,extras):
                target={tuple(kk):(int(s),0.)for kk,s in zip(e['target_keys'],e['target_states'])};y,_,missing=labels(k,target);values=[];indices=torch.from_numpy((k-low).astype(np.int64)).to(device)
                with torch.cuda.amp.autocast():
                    for begin in range(0,len(k),65536):
                        end=min(begin+65536,len(k));p,_=model.probabilities(x[begin:end],extra[begin:end],indices[begin:end],torch.zeros(end-begin,device=device,dtype=torch.long),encoded);values.append(p.float().cpu().numpy())
                pred=np.concatenate(values).argmax(1);init=initial[0][offset:offset+len(k)];offset+=len(k);ex=extra.cpu().numpy();future=np.where(ex[:,0]>.5,1,np.where(ex[:,1]>.5,2,0));cm=np.bincount(((y.astype(np.int64)*3+pred)*3+init)*3+future,minlength=81).reshape(3,3,3,3);counts+=cm;records.append(dict(root_id=item['root_id'],start_node=item['start_node'],end_node=e['end_node'],rows=len(k),true_counts=np.bincount(y,minlength=3).tolist(),out_of_domain_positives=len(missing)))
            if len(records)>=32:break
    summary={}
    for init in [0,1,2]:
        cm=counts[:,:,init,1];summary[str(init)]=dict(future_render_FREE_rows=int(cm.sum()),true_FREE=int(cm[1].sum()),true_BLOCKED=int(cm[2].sum()),true_UNKNOWN=int(cm[0].sum()),predicted_FREE=int(cm[:,1].sum()),false_FREE_on_UNKNOWN=int(cm[0,1]),wrong_FREE_on_BLOCKED=int(cm[2,1]))
    result=dict(role='32realTRAINwindows descriptive lead, not model selection or a measured accuracy gain. All original candidates, no relabeling. Initial code is a PARTIAL-MESH render, not certified visibility.',axes=['target','prediction','initial_partial_render','future_partial_render'],codes={'0':'UNKNOWN','1':'FREE','2':'BLOCKED'},counts=counts.tolist(),future_render_FREE_conditioned_on_initial=summary,records=records);save(OUT/'initial_visibility_diagnostic32.json',result);print(summary,flush=True)


if __name__=='__main__':run()
