"""Select a full-fit epoch using only the fixed pilot at operating point .20."""
from pathlib import Path
from .diagnose import OUT,read,save,sha


def main():
    dest=OUT/'training/point_residual_full';complete=read(dest/'complete.json');assert complete['epochs']==6
    rows=[]
    for epoch in range(1,7):
        p=OUT/'pilot'/('point_residual_full_epoch%02d.json'%epoch);r=read(p);metric=r['rules']['threshold_0.20']
        cp=dest/('epoch%02d.pt'%epoch);assert sha(cp)==r['checkpoint_sha256']
        rows.append(dict(epoch=epoch,checkpoint=str(cp),checkpoint_sha256=r['checkpoint_sha256'],label_aware_iou=metric['label_aware_delta_iou'],revelation_iou=metric['revelation_iou']))
    selected=max(rows,key=lambda r:r['label_aware_iou']);path=OUT/'selection.json'
    if path.exists():raise FileExistsError('frozen selection already exists')
    save(path,dict(**selected,threshold=.2,method='Highest original fixed16-scene pilot reconstruction at previously fixed .20 operating point among exactly6full epochs; full108-scene evaluation has not been read.',all_epochs=rows,
        context_pilot='Rejected: .175087/.168183 best pilot IoUs despite improved same-example fitting; no context full fit.',full_development_opened_by_this_run=False))
    print('FROZEN',selected,flush=True)


if __name__=='__main__':main()
