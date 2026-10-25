from .diagnose import OUT,read,save,sha
from .range_spatial_eval import evaluate


def run():
    records=[]
    for epoch in [20,40,60,80]:
        checkpoint=OUT/'training/range_spatial64'/('epoch%02d.pt'%epoch);path=OUT/'pilot'/('range_spatial_dev_epoch%02d.json'%epoch)
        if not path.exists():evaluate(checkpoint,device='cuda:0')
        result=read(path);rule=result['best_rule'];records.append(dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),rule=rule,metrics=result['rules'][rule]))
    best=max(records,key=lambda r:r['metrics']['label_aware_delta_iou']);save(OUT/'range_spatial_selection.json',dict(role='Original fixed development pilot only, not full development or confirmation.',best=best,candidates=records));print('SELECTION',best,flush=True)
    evaluate(OUT/'training/range_spatial64/epoch80.pt',device='cuda:0',full_training=True)


if __name__=='__main__':run()
