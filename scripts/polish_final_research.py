"""Summarize completed research evidence without changing scientific artifacts."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
DOC=BASE/'docs/l2/final_stage1_stage2_20261024'
OUT=BASE/'output/research_transition_20261024'


def load(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    a=DOC/'A_FINAL_STAGE1_RESEARCH_RELEASE.md';text=a.read_text()
    text=text.replace('Population calibration is not established as an accepted release requirement.',
        'Population-count calibration still fails the original acceptance requirements. The checkpoint remains a research artifact.')
    persistence=load(OUT/'stage1_research_release/persistence.json')
    marker='Historical physical-feasibility and persistence rerun:'
    if 'Across 1,765 anchors' not in text:
        text=text.replace(marker,
            'Across 1,765 anchors /608 Roots /94 scenes, 141,200 sampled worlds had zero recorded reachability, native-motion or historical-lineage violations, with zero data/model failures. These finite checks retain the original uncertain-geometry limitations. '+marker)
    runtime=load(OUT/'stage1_research_release/runtime.json');startup=runtime['first_startup']
    if 'First empty-JIT raw-history forecast' not in text:
        message=(f"First empty-JIT raw-history forecast: **{startup['first_raw_history_predict_ms']/1000:.3f}s**, after {startup['imports_seconds']:.3f}s of imports; checkpoint loading {startup['model_load_ms']:.3f}ms. "
            f"With the compiled cache, raw-history initialization P95 was {runtime['raw_initialization']['representative']['p95']:.3f}ms representative /{runtime['raw_initialization']['dense']['p95']:.3f}ms dense. "
            f"Peak process RSS across the benchmark groups was {max(runtime['rusage_maxrss_kib'].values())*1024:,} bytes. These startup costs are separate from initialized-call latency.\n\n")
        text=text.replace('Reproduction and inference:',message+'Reproduction and inference:')
    lines=[]
    for line in text.splitlines():
        cells=line.split('|')
        if len(cells)==6 and cells[1].strip() in runtime['metrics']:
            label=cells[1].strip();value=runtime['metrics'][label]
            line=f"| {label} | {value['p50']:.3f} | {value['p95']:.3f} | {value['p99']:.3f} |"
        lines.append(line)
    a.write_text('\n'.join(lines)+'\n')

    b=DOC/'B_STAGE2_RESEARCH.md';lines=b.read_text().splitlines();result=[]
    metrics=load(OUT/'stage2_research_release/evaluation.json');performance=load(OUT/'stage2_research_release/performance.json')
    for line in lines:
        if line.startswith('Substantial-information trajectories '):
            cohort=metrics['substantial_information']
            result.extend([f"The substantial-information cohort (≥100 true new cells) contains {metrics['counts']['substantial_queries']:,} Queries /{cohort['roots']:,} Roots /{cohort['scenes']} scenes. Its pooled results are:",
                '', '| Class | Precision | Recall | F1 |','|---|---:|---:|---:|'])
            for key in ('new_free','new_blocked'):
                v=cohort[key];result.append(f"| {key} | {v['precision']:.6f} | {v['recall']:.6f} | {v['f1']:.6f} |")
        elif line.startswith('Per-class reliability bins/ECE:'):
            suffix=line[line.index('Complete pooled'):]
            result.extend(['Per-class Root-weighted calibration error:', '', '| Class | ECE |','|---|---:|'])
            for key,value in metrics['calibration'].items():result.append(f"| {key} | {value['ece']:.8f} |")
            result.extend(['','The large UNKNOWN region influences aggregate calibration scores. Full probability-bin masses and frequencies are retained for inspection. '+suffix])
        elif line.startswith('Batch throughput and latency:'):
            result.extend(['| Batch | P50 ms | P95 ms | P99 ms | Trajectories/s |','|---|---:|---:|---:|---:|'])
            for key,value in performance['batch'].items():
                result.append(f"| {key} | {value['p50_ms']:.3f} | {value['p95_ms']:.3f} | {value['p99_ms']:.3f} | {value['trajectories_per_second']:.3f} |")
        elif line.startswith('| representative |') or line.startswith('| dense |'):
            key=line.split('|')[1].strip();value=performance['workloads'][key]
            fmt=lambda v:' / '.join(f"{v[k]:.3f}" for k in ('p50_ms','p95_ms','p99_ms'))
            result.append(f"| {key} | {fmt(value['predict'])} | {fmt(value['predict_apply_wire'])} |")
        else:result.append(line)
    text='\n'.join(result)+'\n'
    if 'The fixed-argmax output has low recall' not in text:
        text=text.replace('Scene-balanced label-aware GraphDelta IoU:',
            '**The fixed-argmax output has low recall: 15.1% for FREE and 5.6% for BLOCKED.** This first supervised baseline leaves many true revelations as UNKNOWN. Improving Stage2 recall remains a primary research objective. Deterministic update correctness measures application of a supplied delta; prediction accuracy is measured separately above.\n\nScene-balanced label-aware GraphDelta IoU:')
    if 'Batch sizes use different prefixes' not in text:
        text=text.replace('The original complete-call P95 failures',
            'Dense complete-call P99 remains **216.2ms**. Batch sizes use different prefixes of the input pool, so these values do not establish a matched batch speedup. Peak benchmark RSS includes retained workload examples, libraries and scratch arrays. Optimized cold predict/apply/wire observations conservatively include import-witness hashing overhead.\n\nThe original complete-call P95 failures')
    b.write_text(text)
    status=BASE/'RESEARCH_STATUS.md'
    status.write_text('# Current research priority\n\n**Stage2: Trajectory-Conditioned GraphDelta Predictor** is the primary development objective.\n\n'
        'Stage1 development is closed. Its one final full-data fit/evaluation is complete and preserved as a research checkpoint. Population calibration still fails the original acceptance requirements; no further Stage1 calibration, architecture work or collection is planned.\n\n'
        'Continuous Stage01 collection is disabled and MANUALLY_STOPPED with zero workers. Complete immutable data and all source identities/logs are preserved.\n\n'
        'The initial Stage2 predictor, full-data checkpoint, evaluation, optimized API and benchmarks are complete. Low recall and integrated Layer2 latency remain research limitations. Existing L2.5, L3 and L4 implementations remain intact; no L3 RL training was started.\n\n'
        '[Final Stage1 research release](docs/l2/final_stage1_stage2_20261024/A_FINAL_STAGE1_RESEARCH_RELEASE.md) · '
        '[Stage2 research development](docs/l2/final_stage1_stage2_20261024/B_STAGE2_RESEARCH.md) · '
        '[Reproduction and inference](docs/l2/final_stage1_stage2_20261024/REPRODUCE.md)\n')
    registry_path=DOC/'ARTIFACTS.json';registry=load(registry_path)
    for p in [a,b,status,Path(__file__),BASE/'scripts/package_research_transition.py']:
        registry['files'][str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    registry_path.write_text(json.dumps(registry,indent=2,allow_nan=False)+'\n')
    complete=OUT/'COMPLETE.json';value=load(complete);value['artifacts_sha256']=sha(registry_path)
    complete.write_text(json.dumps(value,indent=2)+'\n')
    print('Research summaries and artifact hashes refreshed; scientific artifacts unchanged.')


if __name__=='__main__':main()
