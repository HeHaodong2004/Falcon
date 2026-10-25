"""Fixed pilot selection; explicitly no confirmation/model training here."""
from .diagnose import OUT,read,save,sha
from .range_evidence_eval import evaluate


def run():
    reports=[]
    for epoch in [10,20,30,40]:
        cp=OUT/'training/range_evidence64'/('epoch%02d.pt'%epoch)
        evaluate(cp,device='cuda:0')
        report=read(OUT/'pilot'/('range_evidence_dev_epoch%02d.json'%epoch));rule=report['best_rule'];reports.append(dict(epoch=epoch,checkpoint=str(cp),checkpoint_sha256=sha(cp),rule=rule,metrics=report['rules'][rule]))
    best=max(reports,key=lambda r:r['metrics']['label_aware_delta_iou']);save(OUT/'range_selection.json',dict(role='Selected on original fixed development pilot only; this pilot is inspected, not untouched confirmation. Legal current/past range sensor extension. No acceptance without original full-development accuracy and complete latency.',best=best,candidates=reports,targets=dict(label_aware_increment_iou='>0.50',complete_K6_P95_ms='<50')));print('SELECTION',best,flush=True)


if __name__=='__main__':run()
