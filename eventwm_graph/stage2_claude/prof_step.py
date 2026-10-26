"""Diagnostic: time split between legal feature construction and the training step."""
import time, numpy as np, torch
from eventwm_graph.stage2_claude import train as T
from eventwm_graph.stage2_claude.data import split_roots, load_root
from eventwm_graph.stage2_claude.model import build_model
dev='cuda:0'
saved=torch.load('output/claude_stage2/training/past_v7/final.pt',map_location='cpu'); c=saved['config']
T.GEOMETRY.update(splat=False, sub=2, render=192, occluders=True, near=0.)
from eventwm_graph.stage2_claude.features import channel_index
cidx=channel_index(c)
m=build_model(len(c['map_channels'])+len(c['query_channels']),c).to(dev).to(memory_format=torch.channels_last); m.load_state_dict(saved['model'])
opt=torch.optim.AdamW(m.parameters(),lr=1e-5)
tr,_=split_roots(); rng=np.random.default_rng(0)
rows=[tr[i] for i in rng.choice(len(tr),30,replace=False)]
t_load=t_feat=t_train=0; nwin=0; pool=[]; steps=0
scaler=torch.cuda.amp.GradScaler()
for row in rows:
    t0=time.time(); groups=load_root(row,tracks=True); t_load+=time.time()-t0
    for g in groups:
        g['scene_factor']=1.0
        torch.cuda.synchronize(); t0=time.time()
        ctx=T.build_context(g,dev); s=T.window_samples(g,ctx,288,rng,True,None); torch.cuda.synchronize(); t_feat+=time.time()-t0
        nwin+=len(s); pool.extend(s)
        while len(pool)>=16:
            items=pool[:16]; pool=pool[16:]
            torch.cuda.synchronize(); t0=time.time()
            x,y,A=T.collate_frames(items,dev)
            if cidx is not None: x=x[:,cidx].contiguous(memory_format=torch.channels_last)
            w=torch.tensor([it[2] for it in items],device=dev)
            with torch.cuda.amp.autocast(): lg=m(x)
            loss=T.loss_fn(lg,y,w,3.,2.,2000.)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); torch.cuda.synchronize(); t_train+=time.time()-t0; steps+=1
print('windows',nwin,'steps',steps,'load %.1fs feat %.1fs train %.1fs'%(t_load,t_feat,t_train),'feat/win %.1f ms train/win %.1f ms'%(1e3*t_feat/nwin,1e3*t_train/max(steps*16,1)))
