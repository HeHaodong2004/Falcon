"""Write the diagnostic contract BEFORE any inference or threshold selection."""
import json
from .common import ROOT,OUT,EVALUATION,CHECKPOINT_SHA,THRESHOLDS,EDGES,digest,save,stable
from eventwm_graph.stage2.data import source_rows

def main():
    rows,manifest=source_rows('dev')
    scenes=sorted({r['scene_id'] for r in rows},key=stable)
    selection=scenes[:70]; validation=scenes[70:]
    protocol=dict(schema='stage2_diagnostic_20261025_v1',seed=2525,
        checkpoint_sha256=CHECKPOINT_SHA,original_evaluation_protocol=json.loads((EVALUATION/'protocol.json').read_text()),
        source_manifest_sha256=digest(manifest),roots=len(rows),scenes=len(scenes),expected_segments=47128,
        source_scope='existing development only; protected payloads remain sealed and unavailable through source_rows',
        threshold_selection_scenes=selection,diagnostic_validation_scenes=validation,
        split_caveat='38 scenes held out from NEW threshold/calibration selection, but previously scored in original development; NOT untouched confirmation',
        domain='unchanged initiallyUNKNOWN union11m savedpose discs; assert zero outofdomainGT and exact original argmaxconfusions',
        weighting='window mean withinRoot, Root mean withinScene, Scene mean; rawcounts for confusion-rates, perwindow-cell-normalized proper scores/histograms',
        thresholds=THRESHOLDS.tolist(),decision='P(F)+P(B)>=theta; classFREE ifP(F)>=P(B),elseBLOCKED',
        selected_rules=['max development GraphDeltaIoU','max development binaryrevelationF1','same two selected on70scenes then frozen on38'],
        false_free='report UNKNOWN->FREE separately from actualBLOCKED->FREE; UNKNOWN is not proven physical occupancy',
        proper_scores='multiclass sumBrier, clipped1e-9 natural logloss; binary revealBrier/logloss; unchanged across hard decisionrules',
        rank_metrics=dict(type='hierarchically cell-normalized weighted global histogram AUROC/AP',bins=len(EDGES)-1,
            edges_sha256=__import__('hashlib').sha256(EDGES.astype('<f8').tobytes()).hexdigest(),
            caveat='binned estimates, explicit AUROC and AP binning bounds; AP noninterpolated; report prevalence'),
        quantiles=[0,.01,.05,.25,.5,.75,.95,.99,1],
        probability_cache='float32 all3class probabilities and uint8groundtruth; no lossy quantization; perRoot SHA+segment/domain identities',
        confidence='2000 paired whole-scene bootstrap draws seed2525; full-dev selectedthreshold CIs exploratory/conditional on selection; validation frozen thresholds separate',
        trajectory=dict(selection='existing frozen windows; exact common start and complete timestamp schedule; one diverse pose-selected pair perRoot; no outcome-based selection',
            domain='CAunionCB fixed for all own/cross and neutral controls; report CAintersectionCB network-only support separately',
            controls='normal production predictions paddedUNKNOWN vs same fixedkeys/commoncrop neural prediction; commoncrop effect explicitly quantified',
            neutral='repeat starting pose/yaw on unchanged timestamps: motion-ablation/OOD sensitivity only, NO ownGT/accuracy claim',
            pairing='both recorded paths retain own legalGT; own-v-cross score gaps are specificity diagnostics; report symmetric proper-score andIoU gaps',
            visual_selection='fixedhash examples plus explicitly labeled strongest/weakest specificity examples, not representative claims'),
        geometry=dict(inputs='current legalgraph,supplied savedposes,recorded sensorparams only',
            output='2D potential-revelation proxy only; C-space BLOCKED not opticalgeometry; no fabricated FREE/BLOCKED labels',
            baseline='knownFREE prefix to firstUNKNOWN within baseyaw FOV/range; strict stopUNKNOWN/BLOCKED; also lessrestrictive FOV/range opportunity surrogate, NOT a realvisibility upperbound',
            caveat='future articulatedcamera state and hidden support unavailable; no exact optical visibility claim'),
        decision_relevance=dict(counts='sumprobabilities perunique candidatecell within eachsegment, not sums across overlappingsegments',
            simple_baselines='zero; duration-cohort constantcounts and candidatearea-timesprevalence fitted on70selectionscenes, frozen38validation',
            ranking='same-start same-schedule selectedRootpairs, non-tied actualcounts only; report ties and coverage; no ranking model optimization',
            spatial='coarse spatial mass errors and paired near-path mass shift; dependency-preserving sceneCIs'),
        runtime='evaluate conversion overhead on cached matchedinputs; normalGC; no production modifications, no newarchitecture/training',
        prohibitions=['no retraining','no checkpoint edit','no production edits','no background process changes','no protected payload','no simulator/hiddengeometry'],
        code={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'scripts/stage2_diagnostic').glob('*.py'))})
    path=OUT/'protocol.json'
    if path.exists():raise FileExistsError('frozen protocol exists')
    save(path,protocol);print(path);print('protocol_sha256',digest(path))

if __name__=='__main__':main()
