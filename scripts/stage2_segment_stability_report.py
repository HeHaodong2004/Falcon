"""Report the fixed six-hour service test without redefining primary gates."""
import json
from pathlib import Path
import numpy as np
from eventwm_graph.stage2_segment.common import BASE,OUT,dump,sha,snapshot


def main():
    source=OUT/'stability_release_dede_v1';complete=source/'COMPLETE.json'
    if not complete.exists():raise RuntimeError('Both fixed-duration scenarios must finish first')
    if list(source.rglob('FAILED.json')):raise RuntimeError('Failure artifacts require explicit review')
    dest=OUT/'stability_report_v1';dest.mkdir(exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());summaries=[];minute_rows={};receipts=[]
    for path in sorted(source.rglob('*')):
        if path.is_file():receipts.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    for workers in (4,8):
        folder=source/f'workers_{workers}';summary=json.loads((folder/'summary.json').read_text())
        rows=[json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines()]
        assert summary['status']=='COMPLETED_INTEGRITY_CHECKS_PASS' and summary['configured_seconds']==10800
        assert sum(v['calls'] for r in rows for v in r['pooled'].values())==summary['actual_k4_service_calls']
        assert all(w['gc_enabled'] for r in rows for w in r['workers'])
        assert sum(w['counters'].get('exact_proofs',0) for r in rows for w in r['workers'])==summary['exact_reference_probes']
        assert all(r['proof_roots_seen']==24 and r['final_input_identity_pass'] for r in summary['done'])
        memory={field:dict(slope_mib_per_hour_min=min(v[field]['slope_mib_per_hour'] for v in summary['memory_trends'].values()),
                          slope_mib_per_hour_max=max(v[field]['slope_mib_per_hour'] for v in summary['memory_trends'].values()),
                          largest_first_to_last_median_increase_mib=max((v[field]['last_five_median_bytes']-v[field]['first_five_median_bytes'])/2**20 for v in summary['memory_trends'].values()))
                for field in ('private','pss','rss','cuda_allocated','cuda_reserved')}
        latency={}
        for mode in ('retained_cache','cache_thrash'):
            p95=[r['pooled'][mode]['p95_ms'] for r in rows if r['pooled'][mode]['calls']]
            worst=[w['modes'][mode]['maximum_ms'] for r in rows for w in r['workers'] if w['modes'][mode]['calls']]
            latency[mode]=dict(median_minute_p95_ms=float(np.median(p95)),maximum_minute_p95_ms=max(p95),
                largest_recorded_call_ms=max(worst),per_minute_quantile_scopes=sorted({r['pooled'][mode]['quantile_scope'] for r in rows}),
                scope='Quantiles within synchronized minute windows; no invented full-run percentile by averaging window quantiles')
        item=dict(workers=workers,configured_seconds=10800,calls=summary['actual_k4_service_calls'],graphs=summary['actual_service_segments'],
            proofs=summary['exact_reference_probes'],calls_per_second=summary['calls_per_second_including_proofs_and_telemetry'],
            recorded_window_cpu_seconds=sum(w['process_cpu_seconds'] for r in rows for w in r['workers']),
            mean_aggregate_cpu_cores=sum(w['process_cpu_seconds'] for r in rows for w in r['workers'])/summary['coordinator_seconds'],
            maximum_sampled_worker_rss_gib=max(w['memory']['Rss'] for r in rows for w in r['workers'])/2**30,
            maximum_sampled_summed_worker_pss_gib=max(sum(w['memory']['Pss'] for w in r['workers']) for r in rows)/2**30,
            maximum_sampled_whole_device_gib=max(w['memory']['whole_device_used'] for r in rows for w in r['workers'])/2**30,
            memory_scope='One-minute resident snapshots, not continuous host/device peaks; Torch allocator peak counters remain in raw telemetry',
            memory_trends=memory,latency=latency,integrity='All predetermined exact probes and final input checks passed')
        summaries.append(item);minute_rows[workers]=rows
    host_events=[json.loads(p.read_text()) for p in sorted((source/'host_events').glob('*.json'))]
    report=dict(protocol_sha256=sha(source/'protocol.json'),complete_sha256=sha(complete),source=snapshot(),host_events=host_events,
        total_calls=sum(s['calls'] for s in summaries),total_graphs=sum(s['graphs'] for s in summaries),total_proofs=sum(s['proofs'] for s in summaries),
        scenarios=summaries,artifacts=receipts,primary_acceptance_redefined=False,full_rl_measured=False)
    dump(dest/'results.json',report)
    lines=['# Six-hour repeated-inference service test','',
        'The unchanged released predictor completed two fixed three-hour scenarios, first four workers and then eight. '
        'All predetermined exact reference probes and final input-identity checks passed. Normal GC remained enabled with unchanged thresholds. '
        'This is Stage 2 component stability evidence; no simulator or RL decisions were collected.','',
        f"The run constructed **{report['total_graphs']:,}** complete future graphs in **{report['total_calls']:,}** K4 service calls, "
        f"with **{report['total_proofs']:,}** separately counted exact reference probes. Startup and proof graphs are additional to that service count.",'',
        '| Workers | Service duration | Complete K4 calls | Complete future graphs | Exact probes | K4 calls/s including probe/telemetry overhead | Mean recorded CPU cores |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['workers']} | 3 h | {s['calls']:,} | {s['graphs']:,} | {s['proofs']:,} | {s['calls_per_second']:.2f} | {s['mean_aggregate_cpu_cores']:.2f} |")
    lines+=['','The fixed cohort contains 24 actual saved Root graphs and their first four distinct frozen paths. '
        'Each minute alternates 40 seconds over four retained graphs and 20 seconds cycling all 24 graphs with a four-graph LRU cache. '
        'Periodic probes compare candidate cells, probabilities, heights, hard labels, sparse updates, provenance and complete future snapshots '
        'against a separate unshared same-execution reference. Every worker covered all 24 reference Roots. '
        'Hashes bind the accepted runtime, trained checkpoint, inputs and harness before and after each scenario.','',
        '## Memory and tails','',
        '| Workers | Maximum sampled worker RSS | Maximum sampled summed worker PSS | Maximum sampled whole-device used | Private-memory slope range after first 10 min |',
        '|---:|---:|---:|---:|---:|']
    for s in summaries:
        trend=s['memory_trends']['private'];lines.append(f"| {s['workers']} | {s['maximum_sampled_worker_rss_gib']:.3f} GiB | {s['maximum_sampled_summed_worker_pss_gib']:.3f} GiB | {s['maximum_sampled_whole_device_gib']:.3f} GiB | {trend['slope_mib_per_hour_min']:.3f} to {trend['slope_mib_per_hour_max']:.3f} MiB/h |")
    lines+=['','These are one-minute resident snapshots, not continuous host/device peak measurements. Worker PSS excludes the coordinator. '
        'Whole-device use includes CUDA contexts and any unrelated device allocations; it is not tensor-only memory. '
        'The first ten minutes are excluded from descriptive OLS memory slopes. First/last five-window medians and all host/device trends remain in the JSON report. '
        'Finite fixed-cohort trends do not prove absence of a leak on every future graph distribution.','',
        '| Workers | Mode | Median minute P95 | Worst minute P95 | Largest recorded ordinary call |',
        '|---:|---|---:|---:|---:|']
    for s in summaries:
        for mode,v in s['latency'].items():lines.append(f"| {s['workers']} | {mode} | {v['median_minute_p95_ms']:.2f} ms | {v['maximum_minute_p95_ms']:.2f} ms | {v['largest_recorded_call_ms']:.2f} ms |")
    lines+=['','These are synchronized minute-level diagnostics. They are not fabricated full-run percentiles obtained by averaging quantiles. '
        'The separate original integrated benchmark still has approximately 280 ms P99 and a 303 ms maximum. '
        'This smaller fixed service cohort does not establish that those original tails are solved. '
        'No primary workload, probability score, model-selection rule or acceptance threshold changed.','',
        '## Reproduction and scope','',
        'The committed `reviews/stage2_segment/reviewer_b/SUSTAINED_SERVICE_HARNESS.md` documents the exact command, limits, scheduling and timing boundaries. '
        'Raw minute telemetry, startup records, exact golden digests, completion summaries and source identities remain under '
        '`output/stage2_segment_20261024/stability_release_dede_v1`; the report artifact receipt contains every file SHA256. '
        'These repeated in-memory predictions are not independent trajectories, new simulation experience or complete RL decisions. '
        'End-to-end RL throughput and 48/72-hour convergence remain unverified.','']
    if host_events:
        lines+=['## Incidental host activity','',
            'A read-only review process mistakenly began parsing the 2.0 GiB fitting metadata manifest at approximately 04:38 UTC '
            'and was terminated by 04:39:02 UTC. One process sample observed about one CPU core and 3.91 GiB RSS; '
            'exact duration and continuous peaks were not retained. No model inference, GPU work or source/data mutation occurred. '
            'All affected service minutes remain included. The six-hour host therefore was not strictly exclusive throughout; '
            'the separate primary acceptance benchmarks had already completed. Full event details and their hashes are retained with the telemetry.','']
    target=BASE/'docs/l2/stage2_segment_20261024/STABILITY.md';target.write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,workers in zip(axes,(4,8)):
        rows=minute_rows[workers]
        for number in range(workers):
            ax.plot([r['workers'][number]['elapsed_seconds']/3600 for r in rows],
                [r['workers'][number]['memory']['private_bytes']/2**30 for r in rows],lw=.8,label=f'Worker {number+1}')
        ax.set_title(f'{workers} workers');ax.set_xlabel('Service elapsed time (h)');ax.set_ylim(bottom=0);ax.legend(fontsize=7,ncol=2)
    axes[0].set_ylabel('Worker private resident memory (GiB)');fig.suptitle('Fixed real Stage 2 requests; normal GC; all future graphs constructed')
    fig.tight_layout();figures=BASE/'docs/l2/stage2_segment_20261024/figures'
    fig.savefig(figures/'sustained_memory.png',dpi=160);fig.savefig(figures/'sustained_memory.pdf');plt.close(fig)
    dump(dest/'report_files.json',{str(p):sha(p) for p in (target,figures/'sustained_memory.png',figures/'sustained_memory.pdf')})
    print(json.dumps(dict(calls=report['total_calls'],graphs=report['total_graphs'],proofs=report['total_proofs'],report=str(target))))


if __name__=='__main__':main()
