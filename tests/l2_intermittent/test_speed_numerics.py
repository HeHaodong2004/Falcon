"""Independent scalar-law and NumPy RNG witnesses for runtime optimizations."""
import unittest
import numpy as np
from eventwm_graph.l2_partial.state import ObservedGraph
from eventwm_graph.l2_intermittent.dynamics import (
    DEFAULT,Component,evolve,evolve_many,keyed_rng,categorical)
from eventwm_graph.l2_intermittent.neural import observed_features
from eventwm_graph.l2_intermittent.random_kernel import seed_words,KeyedWorkspace


class SpeedNumericsTests(unittest.TestCase):
    def test_batched_propagation_matches_scalar_all_fields(self):
        rng=np.random.default_rng(923)
        components=[];start=.123456789
        for j,n in enumerate((0,1,3,6,2)):
            P=rng.normal(size=(n,4,4));P=P@P.swapaxes(-1,-2)
            c=Component(.2,rng.normal(size=(n,4)),P,np.arange(n),np.full(n,start),
                np.full(n,start),rng.normal(size=(n,2)),rng.random(n)<.5,
                rng.normal(size=(n,2)),rng.uniform(.001,.3,n),start+rng.uniform(0.,.4,n),
                np.zeros(n,dtype=np.int64),j+221,np.array([.5,.5]),
                hidden_mean=rng.normal(size=4),hidden_covariance=np.diag([100.,100.,.36,.36]))
            if j%2:c=c.fork(mean=c.mean.astype(np.float32),covariance=c.covariance.astype(np.float32),
                diffusion=c.diffusion.astype(np.float32),intent=c.intent.astype(np.float32))
            components.append(c)
        for dt in (0.,1e-14,5e-13,2e-12,.000013,.048,.5,1.003,3.):
            fast=evolve_many(components,start,start+dt,DEFAULT)
            reference=[evolve(c,start,start+dt,DEFAULT) for c in components]
            for a,b in zip(fast,reference):
                for key,value in vars(a).items():
                    if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(b,key),err_msg=key)
                    else:self.assertEqual(value,getattr(b,key))
        # Nonstandard event storage preserves the original scalar arithmetic.
        fallback=[c.fork(counters=c.counters.astype(np.int32)) for c in components]
        for a,b in zip(evolve_many(fallback,start,start+.5,DEFAULT),[evolve(c,start,start+.5,DEFAULT) for c in fallback]):
            np.testing.assert_array_equal(a.mean,b.mean)
            np.testing.assert_array_equal(a.covariance,b.covariance)
            self.assertEqual(a.counters.dtype,b.counters.dtype)

    def test_keyed_generator_and_choice_stream_matches_numpy(self):
        import hashlib
        for i in range(1000):
            raw=hashlib.blake2b(('%s:%s:%s'%(i,3,19)).encode(),digest_size=16).digest()
            a=keyed_rng(i,3,19);b=np.random.default_rng(int.from_bytes(raw,'little'))
            self.assertEqual(a.exponential(.4),b.exponential(.4))
            p=np.random.default_rng(i).random(3).astype(np.float32 if i%2 else float)
            if i%7==0:p[0]=0.
            p/=p.sum()
            self.assertEqual(categorical(a,p),int(b.choice(3,p=p)))
            self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        for p in ([0.,0.,0.],[-.1,.5,.6],[np.nan,.5,.5],[.1,.1,.1]):
            with self.assertRaises(ValueError):categorical(np.random.default_rng(17),p)

    def test_observed_cell_stencil_reuse_is_exact(self):
        graph=ObservedGraph('cell-reuse-witness',0,.2,0.,
            {(x,z):(1+(x%2),0.) for x in range(-10,10) for z in range(-10,10)})
        graph=graph.update([dict(cell=[0,0],state=2,height=0.,source='observed')])
        r=np.random.default_rng(32);positions=np.round(r.normal(size=(32,6,2)),1)
        np.testing.assert_array_equal(observed_features(graph,positions),graph.local_features(positions))

    def test_seed_sequence_specialization_and_workspace_matches_reference(self):
        import hashlib
        boundary=[0,1,2**32-1,2**32,2**64-1,2**64,2**96-1,2**96,2**128-1]
        for entropy in boundary:
            words=np.frombuffer(entropy.to_bytes(16,'little'),dtype='<u4')
            np.testing.assert_array_equal(seed_words(words),np.random.SeedSequence(entropy).generate_state(4,np.uint64))
        workspace=KeyedWorkspace()
        for i in range(1000):
            raw=hashlib.blake2b(('%s:%s:%s'%(i,3,19)).encode(),digest_size=16).digest()
            a=workspace.generator(i,3,19);b=np.random.default_rng(int.from_bytes(raw,'little'))
            self.assertEqual(a.bit_generator.state,b.bit_generator.state)
            self.assertEqual(a.exponential(.4),b.exponential(.4));self.assertEqual(a.random(),b.random())
            self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        workspace.fast=False
        self.assertEqual(workspace.generator(17,3,19).bit_generator.state,keyed_rng(17,3,19).bit_generator.state)


if __name__=='__main__':unittest.main()
