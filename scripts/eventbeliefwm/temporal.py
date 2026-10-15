"""Native temporal event diagnostics; target identities never enter inference."""
import torch


@torch.no_grad()
def event_summary(forecast, batch, length=None):
    """One result per root, retaining canonical query joins and censor exposure.

    NLL includes complete supported safe intervals plus the observed contact bin. Timing
    uses the exact native first-contact time and the forecast's conditional event
    distribution through its supported query horizon. It is descriptive when
    total event probability is small; zero-mass misses are counted separately.
    """
    cumulative = forecast.cumulative_collision
    length = cumulative.shape[-1] if length is None else length
    cumulative = cumulative[..., :length]
    mass = forecast.collision_interval[..., :length]
    previous = torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], -1)
    hazard = (mass/(1-previous).clamp_min(1e-7)).clamp(1e-7, 1-1e-7)
    labels = batch['targets']['collision_interval'][..., :length].bool()
    query_support = forecast.validity['query_time'][..., :length]
    valid = batch['targets']['collision_valid'][..., :length] & query_support
    loss = -torch.where(labels, hazard.log(), torch.log1p(-hazard))
    times = batch['targets']['times'][..., :length]
    if times.ndim == 1:
        times = times[None].expand(len(cumulative), -1)
    result = []
    for i, metadata in enumerate(batch['metadata']):
        physical = metadata.get('physical_humans_target_only')
        fully_tracked = physical is not None and metadata['association']['associated'] == physical
        queries = []
        for k, row in enumerate(metadata['rows']):
            exposed = valid[i, k]
            selected = labels[i, k] & exposed
            native_event = bool(row.get('human_collision_time_valid', False))
            native_time = float(row.get('human_collision_time_s', 0.))
            usable_event = native_event and bool(selected.any())
            total_mass = float((mass[i, k]*query_support[i, k]).sum())
            predicted_time = None
            if usable_event and total_mass > 1e-6:
                previous_times = torch.cat([times.new_zeros(1), times[i, :-1]])
                midpoints = (times[i]+previous_times)/2
                predicted_time = float((mass[i, k]*query_support[i, k]*midpoints).sum()/total_mass)
            queries.append(dict(branch_id=row['branch_id'], query_id=row['query_id'],
                exposed_intervals=int(exposed.sum()), event_interval_observed=bool(selected.any()),
                temporal_nll=float(loss[i, k][exposed].sum()) if exposed.any() else None,
                native_event_time_s=native_time if native_event else None,
                native_event_timing_eligible=usable_event,
                predicted_event_mass=total_mass,
                conditional_expected_event_time_s=predicted_time,
                conditional_event_time_error_s=abs(predicted_time-native_time) if predicted_time is not None else None))
        nll = [q['temporal_nll'] for q in queries if q['temporal_nll'] is not None]
        timing = [q['conditional_event_time_error_s'] for q in queries if q['conditional_event_time_error_s'] is not None]
        average = lambda values: sum(values)/len(values) if values else None
        result.append(dict(root_id=metadata['root_id'], fully_tracked=fully_tracked,
            queries_exposed=len(nll), exposed_intervals=sum(q['exposed_intervals'] for q in queries),
            observed_event_queries=sum(q['event_interval_observed'] for q in queries),
            native_events_outside_scored_intervals=sum(q['native_event_time_s'] is not None and
                not q['native_event_timing_eligible'] for q in queries),
            zero_mass_event_misses=sum(q['native_event_timing_eligible'] and q['predicted_event_mass'] <= 1e-6 for q in queries),
            temporal_nll=average(nll), fully_tracked_temporal_nll=average(nll) if fully_tracked else None,
            conditional_event_time_MAE_s=average(timing), timing_queries=len(timing),
            fully_tracked_conditional_event_time_MAE_s=average(timing) if fully_tracked else None,
            scope='complete safe intervals plus observed contact bin, within planned canonical support; timing uses0.25s interval midpoints and is conditional on predicted collision',
            queries=queries))
    return result
