"""Exhaustive matching counterexamples and independent real legal-set checks."""
import copy
import gzip
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from eventwm_graph.l2_two_stage import association as a


def witness(groups,times=None):
    times=np.arange(len(groups),dtype=np.float64) if times is None else np.asarray(times,np.float64)
    groups=[np.asarray(g,np.float64).reshape(-1,3) for g in groups]
    return SimpleNamespace(timestamps=times,positive_offsets=np.r_[0,np.cumsum([len(g) for g in groups])],
                           positive_xyz=np.concatenate(groups),observation_valid=np.ones(len(groups),bool),
                           human_sensor_available=np.ones(len(groups),bool),horizontal_motion_bound_m_s=3.)


def csr(past):
    offsets=np.r_[0,np.cumsum(past.sum(1))].astype(np.int64)
    indices=np.concatenate([np.flatnonzero(row) for row in past]).astype(np.int64) if len(past) else np.empty(0,np.int64)
    return offsets,indices


def all_matchings(past,terminal):
    matrix=np.concatenate([past,terminal],axis=1);n=len(matrix)
    return {p for p in itertools.permutations(range(matrix.shape[1]),n) if all(matrix[i,j] for i,j in enumerate(p))}


class AssociationTests(unittest.TestCase):
    def test_lazy_endpoint_predicate_is_pointwise_identical_to_eager(self):
        generator=np.random.default_rng(932)
        checked=0
        for frames in (1,2,5,12):
            for count in (0,1,3):
                groups=[np.c_[generator.normal(size=(count,1)),np.zeros((count,1)),
                              generator.normal(size=(count,1))] for _ in range(frames)]
                graph=a.AssociationGraph.from_history(witness(groups,np.arange(frames)*.2))
                for people in (0,1,4,8):
                    points=generator.normal(size=(64,people,2))
                    mask=generator.random((64,people))>.2
                    if count and people>=count:
                        points[:32,:count]=groups[-1][:,[0,2]]
                        mask[:32,:count]=True
                    terminal,_=a._terminal_edges(graph.times,graph.xy,graph.timestamp,points,mask,False)
                    expected=a._feasible(graph.indptr,graph.indices,
                                         graph.maximum_history_matching,terminal)
                    np.testing.assert_array_equal(graph.feasible(points,mask),expected)
                    checked+=len(points)
        self.assertEqual(checked,3072)

    def test_bitset_boundary_order_and_all_uint64_bit_positions(self):
        for bit in range(64):
            self.assertEqual(a._trailing_bit(np.uint64(1)<<np.uint64(bit)),bit)
        generator=np.random.default_rng(91)
        for words in (1,2,4):
            for _ in range(30):
                allowed=generator.random(64*words)<.3;excluded=generator.random(64*words)<.5
                bits=np.zeros(words,np.uint64);mask=np.zeros(words,np.uint64)
                for column in range(64*words):
                    if allowed[column]:bits[column//64]|=np.uint64(1)<<np.uint64(column%64)
                    if excluded[column]:mask[column//64]|=np.uint64(1)<<np.uint64(column%64)
                first=int(generator.integers(64*words));order=[(first+i)%(64*words) for i in range(64*words)]
                candidates=[i for i in order if allowed[i] and not excluded[i]]
                self.assertEqual(a._first_unexcluded(bits,mask,first),candidates[0] if candidates else -1)

    def test_normalized_first_edge_probabilities_are_declared_mixture(self):
        # Two allowed edges, g=(1,3), eta=.2 -> exact categorical p=(.3,.7).
        indptr=np.array([0,2],np.int64);indices=np.array([0,1],np.int64)
        uniforms=np.array([[.1],[.299],[.301],[.999]])
        terminal=np.zeros((4,1,0),bool);tg=np.zeros((4,1,0))
        first,column,logp=a._first_choices(indptr,indices,np.array([1.,3.]),terminal,tg,uniforms,.2)
        np.testing.assert_array_equal(first[:,0],[0,0,1,1])
        np.testing.assert_array_equal(column[:,0],[0,0,1,1])
        np.testing.assert_allclose(logp,np.log([.3,.3,.7,.7]),rtol=0,atol=1e-15)

    def test_exhaustive_tiny_feasibility_matches_all_injective_matchings(self):
        n=3;past_candidates=[(0,1),(0,2),(1,2)]
        cases=0
        for edges in itertools.product((False,True),repeat=len(past_candidates)):
            past=np.zeros((n,n),bool)
            for ij,value in zip(past_candidates,edges):past[ij]=value
            indptr,indices=csr(past);maximum=a._extend_maximum(indptr,indices,np.empty(0,np.int64))
            independent=maximum_bipartite_matching(csr_matrix(past),perm_type='column')
            self.assertEqual(sum(maximum>=0),sum(independent>=0))
            for m in range(3):
                terminals=np.asarray(list(itertools.product((False,True),repeat=n*m)),bool).reshape(-1,n,m) if m else np.empty((1,n,0),bool)
                result=a._feasible(indptr,indices,maximum,terminals)
                valid_lineage,_,_,_=a._lineages(indptr,indices,terminals,
                    np.zeros((len(terminals),n),np.int64),
                    np.repeat(np.arange(n-1,-1,-1,dtype=np.int64)[None],len(terminals),axis=0))
                np.testing.assert_array_equal(valid_lineage,result)
                for terminal,valid in zip(terminals,result):
                    self.assertEqual(bool(valid),bool(all_matchings(past,terminal)))
                    cases+=1
        self.assertEqual(cases,584)

    def test_exhaustive_randomization_support_is_every_feasible_tiny_matching(self):
        # Enumerate auxiliary choices rather than claiming rare Monte Carlo draws prove support.
        n=2;cases=0
        for edge in (False,True):
            past=np.array([[False,edge],[False,False]])
            indptr,indices=csr(past)
            for bits in itertools.product((False,True),repeat=4):
                terminal=np.array(bits,bool).reshape(n,2);expected=all_matchings(past,terminal)
                if not expected:continue
                choices=[]
                for row in range(n):
                    degree=indptr[row+1]-indptr[row]
                    choices.append(list(range(degree))+[degree+j for j in np.flatnonzero(terminal[row])])
                first=[];orders=[]
                for option in itertools.product(*choices):
                    # Exact current production row order, not a union over orders.
                    first.append(option);orders.append(tuple(range(n-1,-1,-1)))
                terminal_batch=np.repeat(terminal[None],len(first),axis=0)
                valid,matching,_,_=a._lineages(indptr,indices,terminal_batch,np.array(first,np.int64),np.array(orders,np.int64))
                self.assertTrue(valid.all())
                self.assertEqual({tuple(row) for row in matching},expected)
                cases+=1
        self.assertGreater(cases,10)

    def test_layered_matching_random_dags_agree_with_independent_scipy(self):
        generator=np.random.default_rng(731);cases=0
        for n in (5,10,20):
            for _ in range(20):
                past=np.triu(generator.random((n,n))<.2,1);indptr,indices=csr(past)
                m=int(generator.integers(1,6));terminal=generator.random((4,n,m))<.3
                first=np.zeros((4,n),np.int64)
                for k in range(4):
                    for u in range(n):
                        degree=indptr[u+1]-indptr[u]
                        allowed=list(range(degree))+[degree+j for j in np.flatnonzero(terminal[k,u])]
                        if allowed:first[k,u]=generator.choice(allowed)
                order=np.array([generator.permutation(n) for _ in range(4)])
                valid,matching,event_person,seen=a._lineages(indptr,indices,terminal,first,order)
                for k in range(4):
                    matrix=np.concatenate([past,terminal[k]],axis=1)
                    independent=maximum_bipartite_matching(csr_matrix(matrix),perm_type='column')
                    self.assertEqual(bool(valid[k]),bool(np.all(independent>=0)))
                    if valid[k]:
                        self.assertEqual(len(set(matching[k])),n)
                        self.assertTrue(all(matrix[u,v] for u,v in enumerate(matching[k])))
                        self.assertTrue((event_person[k]>=0).all())
                    cases+=1
        self.assertEqual(cases,240)

    def test_exact_current_conditioning_and_colocated_multiplicity(self):
        raw=1000.1
        graph=a.AssociationGraph.from_history(witness([[[raw,0.,0.],[raw,0.,0.]]],[.5]))
        self.assertEqual(graph.minimum_current_count,2)
        one=np.array([[[raw,0.],[raw,0.]]],np.float64)
        np.testing.assert_array_equal(graph.feasible(one,[[True,False]]),[False])
        np.testing.assert_array_equal(graph.feasible(one,[[True,True]]),[True])
        rounded=one.astype(np.float32)
        self.assertFalse(graph.feasible(rounded).any())
        moved=one.copy();moved[0,0,0]=np.nextafter(raw,np.inf)
        self.assertFalse(graph.feasible(moved).any())
        with self.assertRaisesRegex(ValueError,'infeasible'):
            graph.lineage(one,[[True,False]],rng=np.random.default_rng(12))

    def test_same_time_and_strict_speed_boundary_counterexamples(self):
        self.assertTrue(a._allowed(0.,0.,0.))
        self.assertFalse(a._allowed(0.,1e-200,0.))
        self.assertFalse(a._allowed(-.1,0.,0.))
        self.assertTrue(a._allowed(1.,3.,0.))
        self.assertTrue(a._allowed(1.,np.nextafter(3.,0.),0.))
        self.assertFalse(a._allowed(1.,np.nextafter(3.,np.inf),0.))
        graph=a.AssociationGraph.from_history(witness([[[0.,0.,0.]],[]],[0.,1.]))
        result=graph.feasible(np.array([[[3.,0.]],[[np.nextafter(3.,np.inf),0.]]]))
        np.testing.assert_array_equal(result,[True,False])

    def test_empty_history_positives_and_absent_endpoints(self):
        graph=a.AssociationGraph.from_history(witness([[],[]],[0.,.048]))
        endpoints=np.zeros((3,2,2));present=np.array([[False,False],[True,False],[True,True]])
        self.assertEqual(graph.minimum_current_count,0)
        np.testing.assert_array_equal(graph.feasible(endpoints,present),True)
        lineage=graph.lineage(endpoints,present,rng=np.random.default_rng(0))
        self.assertEqual(lineage.event_person.shape,(3,0))
        self.assertFalse(lineage.physical_seen.any())
        np.testing.assert_array_equal(lineage.auxiliary_log_probability,0.)

    def test_incremental_cache_matches_rebuild_and_rejects_prefix_changes(self):
        groups=[[[0.,0.,0.],[2.,0.,0.]],[],[[.2,0.,0.],[2.2,0.,0.]],[[.4,0.,0.]]]
        first=witness(groups[:2],[0.,.048]);full=witness(groups,[0.,.048,.104,.152])
        old=a.AssociationGraph.from_history(first);new=old.append(full);rebuilt=a.AssociationGraph.from_history(full)
        for key in ('frame_timestamps','positive_offsets','positive_xyz','times','xy','indptr','indices','geometry'):
            np.testing.assert_array_equal(getattr(new,key),getattr(rebuilt,key))
        self.assertEqual(new.minimum_current_count,rebuilt.minimum_current_count)
        self.assertEqual(old.detections,2)
        changed=copy.deepcopy(full);changed.positive_xyz[0,0]=999.
        with self.assertRaisesRegex(ValueError,'changes or truncates'):
            old.append(changed)
        with self.assertRaisesRegex(ValueError,'changes or truncates'):
            new.append(first)

    def test_first_append_and_readonly_endpoints_need_no_new_jit_types(self):
        first=witness([[[0.,0.,0.]],[]],[0.,.048]);full=witness([[[0.,0.,0.]],[],[[.1,0.,0.]]],[0.,.048,.104])
        graph=a.AssociationGraph.from_history(first)
        append_types=set(a._append_edges.signatures);maximum_types=set(a._extend_maximum.signatures)
        new=graph.append(full)
        self.assertEqual(set(a._append_edges.signatures),append_types)
        self.assertEqual(set(a._extend_maximum.signatures),maximum_types)
        points=np.array([[[.1,0.]]]);new.feasible(points)
        terminal_types=set(a._terminal_edges.signatures)
        points.setflags(write=False);new.feasible(points)
        self.assertEqual(set(a._terminal_edges.signatures),terminal_types)

    def test_graph_and_witness_arrays_are_deeply_immutable(self):
        record=witness([[[0.,0.,0.]],[]],[0.,.048]);graph=a.AssociationGraph.from_history(record)
        record.positive_xyz[0,0]=100.
        self.assertEqual(graph.xy[0,0],0.)
        for key in ('frame_timestamps','positive_offsets','positive_xyz','observation_valid','human_sensor_available','times','xy','indptr','indices','geometry','past_bits','maximum_history_matching'):
            with self.assertRaises(ValueError):getattr(graph,key).setflags(write=True)
        lineage=graph.lineage(np.zeros((2,2,2)),rng=np.random.default_rng(1))
        for key in ('matching','event_person','physical_seen','row_order','first_edge','auxiliary_log_probability'):
            with self.assertRaises(ValueError):getattr(lineage,key).setflags(write=True)
        with self.assertRaisesRegex(ValueError,'uncertified'):
            a.AssociationGraph().validate_certificate()

    def test_lineage_is_reproducible_per_world_ancestry_without_source_ids(self):
        graph=a.AssociationGraph.from_history(witness([[[0.,0.,0.]],[]],[0.,1.]))
        endpoints=np.repeat(np.array([[[.2,0.],[-.2,0.],[10.,0.]]]),100,axis=0)
        left=graph.lineage(endpoints,rng=np.random.default_rng(19));right=graph.lineage(endpoints,rng=np.random.default_rng(19))
        np.testing.assert_array_equal(left.matching,right.matching)
        np.testing.assert_array_equal(left.physical_seen.sum(1),1)
        np.testing.assert_array_equal(left.row_order,np.zeros((100,1),np.int64))
        self.assertFalse(left.physical_seen[:,2].any())
        self.assertEqual(set(left.event_person[:,0]),{0,1})
        self.assertTrue(np.isfinite(left.auxiliary_log_probability).all())
        self.assertTrue((left.auxiliary_log_probability<=0.).all())
        with self.assertRaisesRegex(ValueError,'positive uniform'):
            graph.lineage(endpoints,uniform_mass=0.)

    def test_invalid_clocks_validity_and_endpoint_shapes_fail_closed(self):
        for record in (witness([[],[]],[1.,1.]),witness([[],[]],[1.,.5])):
            with self.assertRaisesRegex(ValueError,'invalid causal'):
                a.AssociationGraph.from_history(record)
        record=witness([[[0.,0.,0.]]]);record.observation_valid[0]=False
        with self.assertRaisesRegex(ValueError,'invalid/unavailable'):
            a.AssociationGraph.from_history(record)
        graph=a.AssociationGraph.from_history(witness([[]]))
        with self.assertRaises(ValueError):graph.feasible(np.zeros((2,3)))
        with self.assertRaises(ValueError):graph.feasible(np.full((1,1,2),np.nan))
        with self.assertRaises(ValueError):graph.feasible(np.zeros((1,1,2)),np.ones((2,1),bool))

    def test_real_legal_roots_independent_scipy_and_lineage_path_audit(self):
        # Fixed original dense IDs, no forecast scores or privileged payloads used.
        base=Path(__file__).resolve().parents[2]
        directory=base/'docs/l2/two_stage_20261023/reviews/runtime_200_30_pilot_k_bounded_unique_20261023'
        protocol=json.loads((directory/'protocol.json').read_text());inventory=json.loads((directory/'inventory.json').read_text())['metadata']
        selected={r['root_id']:r for r in inventory}
        for rid in protocol['dense_roots'][:3]:
            meta=selected[rid]
            with gzip.open(Path(meta['folder'])/'legal_root.json.gz','rt') as stream:legal=json.load(stream)
            frames=legal['history'];groups=[]
            for ob in frames:
                groups.append([tr['position'] for tr in ob['tracks'] if ob.get('observation_valid',True)
                    and ob.get('human_sensor_available',True) and tr.get('visible_now') and tr.get('valid',True)
                    and abs(tr['last_seen_time']-ob['timestamp'])<1e-7])
            record=witness(groups,[ob['timestamp'] for ob in frames]);graph=a.AssociationGraph.from_history(record)
            # Each actual detection has a distinct possible current endpoint at its
            # old position. This broad fixture is not a sampled population model.
            endpoints=np.repeat(graph.xy[None],4,axis=0);present=np.ones(endpoints.shape[:2],bool)
            terminal,_=a._terminal_edges(graph.times,graph.xy,graph.timestamp,endpoints,present,False)
            past=np.zeros((graph.detections,graph.detections),bool)
            for u in range(graph.detections):past[u,graph.indices[graph.indptr[u]:graph.indptr[u+1]]]=True
            independent=maximum_bipartite_matching(csr_matrix(past),perm_type='column')
            self.assertEqual(graph.minimum_current_count,int(sum(independent<0)))
            for k in range(4):
                full=maximum_bipartite_matching(csr_matrix(np.concatenate([past,terminal[k]],axis=1)),perm_type='column')
                self.assertEqual(bool(graph.feasible(endpoints)[k]),bool(np.all(full>=0)))
            lineage=graph.lineage(endpoints,present,rng=np.random.default_rng(77))
            for k in range(4):
                self.assertEqual(len(set(lineage.matching[k])),graph.detections)
                for u,column in enumerate(lineage.matching[k]):
                    if column<graph.detections:
                        self.assertTrue(past[u,column]);self.assertGreater(graph.times[column],graph.times[u])
                    else:self.assertTrue(terminal[k,u,column-graph.detections])
                    person=lineage.event_person[k,u]
                    self.assertTrue(lineage.physical_seen[k,person])
                    current=u
                    for _ in range(graph.detections+1):
                        successor=lineage.matching[k,current]
                        if successor>=graph.detections:
                            self.assertEqual(successor-graph.detections,person);break
                        current=successor
                    else:self.fail('historical lineage cycle')


if __name__=='__main__':unittest.main()
