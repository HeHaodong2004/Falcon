"""Versioned human-event query/label protocol; no simulator imports."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import numpy as np
from collector.core import Query, build_labels, compile_motion, digest
from collector.static_feasibility import CHECKER_VERSION, StaticConfig, is_static_feasible

PROTOCOL = 'eventwm-human-event-v2'


def guard_output(path, check_protocol=True):
    repo = Path(__file__).resolve().parents[2]
    legacy = (repo/'datasets/raw/eventwm_prod_v1').resolve()
    dest = Path(path).resolve()
    if dest == legacy or legacy in dest.parents or dest in legacy.parents:
        raise ValueError('V1 is read-only legacy: v2 output cannot overlap v1')
    manifest = dest/'manifest.json'
    if check_protocol and manifest.exists():
        import json
        if json.loads(manifest.read_text()).get('protocol_version') != PROTOCOL:
            raise ValueError('Refusing to mix v2 with another dataset protocol')
    return dest


@dataclass(frozen=True)
class QueryConfig:
    movement_queries: int = 8
    max_attempts: int = 128
    radii: tuple = (1.5, 1.0, .65)
    endpoint_separation: float = .30
    include_wait: bool = True

    def __post_init__(self):
        if self.movement_queries < 1 or self.max_attempts < self.movement_queries:
            raise ValueError('Invalid candidate budget')
        if not self.radii or min(self.radii) <= 0 or self.endpoint_separation <= 0:
            raise ValueError('Invalid radius/diversity configuration')


class InsufficientStaticQueries(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('insufficient_static_feasible_queries')


def generate_queries_v2(root, planner, geometry, config=QueryConfig(), static_config=StaticConfig()):
    cfg = dict(directions=config.movement_queries,target_radius=config.radii[0],speed=.7,yaw_speed=1.5,
               max_projection=.25,max_path_length=4.,human_blind=True)
    accepted, endpoints, rejected, records = [], [], [], []
    attempts = 0
    for attempt in range(config.max_attempts):
        if len(endpoints) == config.movement_queries: break
        attempts += 1
        # Original directions first, then deterministic interleaved headings/radii.
        ring = attempt // config.movement_queries
        angle = 2*math.pi*((attempt % config.movement_queries)/config.movement_queries + ring*.3819660112501051)
        radius = config.radii[ring % len(config.radii)]
        local = [radius*math.cos(angle), radius*math.sin(angle)]
        target = np.asarray(root.position)+[radius*math.cos(root.yaw+angle),0,-radius*math.sin(root.yaw+angle)]
        path = planner.path(root.position,target,cfg)
        detail = dict(attempt=attempt,target=target.tolist(),local_target=local)
        reason = None
        if path is None: reason = 'static_path_unavailable'
        elif any(np.linalg.norm((np.array(path[-1])-e)[[0,2]]) < config.endpoint_separation for e in endpoints):
            reason = 'near_duplicate_endpoint'
        if reason:
            rejected.append(dict(detail,rejection_reason=reason)); continue
        variants = []
        for wait, scale in ((0.,1.),(.5,1.),(1.,1.),(0.,.5)):
            q = compile_motion(root,path,local,len(endpoints),wait,scale,cfg)
            precheck = is_static_feasible(root,q,planner,geometry,static_config)
            if not precheck['static_feasible']:
                reason = precheck['rejection_reason']; detail['precheck'] = precheck; break
            body = q.data
            body.update(schema_version=2,protocol_version=PROTOCOL,static_precheck=precheck,
                        spatial_path_id=digest(path),query_generation_v2=asdict(config))
            body.pop('query_id');body['query_id']='q_'+digest(body)[:12]
            variants.append(Query.make(body))
        if reason:
            rejected.append(dict(detail,rejection_reason=reason)); continue
        endpoints.append(np.asarray(path[-1])); accepted.extend(variants)
        records.append(dict(detail,path=path,query_ids=[q.data['query_id'] for q in variants]))
    diagnostics = dict(protocol_version=PROTOCOL,checker_version=CHECKER_VERSION,config=asdict(config),
        static_config=asdict(static_config),geometry=asdict(geometry),attempts=attempts,
        requested_movement_queries=config.movement_queries,accepted_movement_queries=len(endpoints),
        candidate_rejections=rejected,accepted_candidates=records,wait_count=0)
    if len(endpoints) != config.movement_queries:
        raise InsufficientStaticQueries(diagnostics)
    if config.include_wait:
        horizon=max(q.data['planned_duration'] for q in accepted)
        q=compile_motion(root,[list(root.position)],[0.,0.],'WAIT',horizon,0.,cfg)
        precheck=is_static_feasible(root,q,planner,geometry,static_config)
        if not precheck['static_feasible']:
            diagnostics['wait_rejection']=precheck;raise InsufficientStaticQueries(diagnostics)
        body=q.data;body.update(schema_version=2,protocol_version=PROTOCOL,static_precheck=precheck,
                              spatial_path_id=digest(body['spatial_path']),query_generation_v2=asdict(config))
        body.pop('query_id');body['query_id']='q_'+digest(body)[:12];accepted.append(Query.make(body))
        diagnostics['wait_count']=1
    return accepted,diagnostics


def human_event_labels(branch, query, human_count=None):
    label=build_labels(branch)
    static=label['static_contact'];human=label['native_robot_human_contact'] or label['research_disc_contact']
    q=query.data if hasattr(query,'data') else query
    controls=branch['controls'];planned=q['executable_controls']
    horizon=(len(controls)==len(planned) and all(c['index']==i and c['issued_command']==planned[i]
             for i,c in enumerate(controls)) and abs(label['observed_duration']-q['planned_duration'])<q['clock_quantization_tolerance'])
    reason=None
    if static: reason='static_feasibility_violation'
    elif not q.get('static_precheck',{}).get('static_feasible'): reason='missing_static_precheck'
    elif human_count==0 and human: reason='zero_human_label_violation'
    elif branch['termination_reason']=='collection_error': reason='collection_error'
    elif human: pass
    elif not (branch['completed'] and horizon and branch['termination_reason']=='query_completed'):
        reason='incomplete_or_invalid_nonhuman_outcome'
    label.update(protocol_version=PROTOCOL,human_event_valid=reason is None,
                 human_event_label=(int(human) if reason is None else None),invalid_reason=reason,
                 static_feasibility_violation=static,full_horizon_observed=bool(horizon),
                 static_checker_version=q.get('static_precheck',{}).get('checker_version',CHECKER_VERSION),
                 legacy_fields_note='full_query_safe/completed/censored retained as v1 diagnostics; not v2 validity')
    return label
