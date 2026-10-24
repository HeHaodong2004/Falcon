"""Post-selection privileged diagnostics; never used as an acceptance predicate."""
import numpy as np
from collector.natural_sampling import natural_selection


def root_diagnostics(recipe,state,history,prefix_length):
    profile=recipe.get('sampling',{}).get('profile','explicit')
    tracks=history[-1]['human_tracks']
    ds=[d['distance'] for d in state.get('social_metric',{}).get('distances',[])]
    euclidean=[float(np.linalg.norm(np.asarray(state['robot']['position'])-h['position'])) for h in state['humans']]
    result=dict(sampling_profile=profile,sampling_profile_family='natural' if profile=='natural' else 'targeted' if profile=='wm_train' else 'coverage',
        human_count=len(state['humans']),active_humans=len(state['humans']),root_time=state['timestamp'],prefix_length=prefix_length,
        visible_tracks=sum(t['visible_now'] for t in tracks),lost_tracks=sum(not t['visible_now'] for t in tracks),
        any_human_observed_in_history=any(t['visible_now'] for row in history for t in row['human_tracks']),
        nearest_social_distance_at_root=min(ds) if ds else None,
        nearest_euclidean_distance_at_root=min(euclidean) if euclidean else None,
        already_in_social_violation_at_root=any(d<1. for d in ds),
        audit_only=True,used_for_selection=False)
    if profile=='natural':result.update(natural_selection(recipe['seed'],recipe['sampling']))
    if profile=='wm_train':
        from collector.wm_sampling import legal_root_state, SAMPLER_VERSION
        result.update(sampler_version=SAMPLER_VERSION,
            legal_selection_state=legal_root_state(history,recipe['sampling']['human_count'],recipe['sampling']),
            requested_stratum=recipe['sampling'])
    return result
