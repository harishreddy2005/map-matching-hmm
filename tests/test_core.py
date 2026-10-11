import unittest
import numpy as np
from src.hmm import decode

class ViterbiTests(unittest.TestCase):
    def inputs(self,n=20):
        rng=np.random.default_rng(4)
        c=[np.array([0,1]) for _ in range(n)]
        e=[rng.uniform(.01,1,(2,3)) for _ in range(n)]
        def tr(t,visited):
            return np.array([[[1,1],[1,.01]],[[1,.01],[1,1]]])
        return c,e,tr
    def test_committed_outputs_are_causal(self):
        c,e,tr=self.inputs()
        full=decode(c,e,tr,T=5,N=2)
        for stop in range(3,20):
            prefix=decode(c[:stop],e[:stop],tr,T=5,N=2)
            np.testing.assert_array_equal(prefix[:-2],full[:stop-2])
    def test_zero_lag_is_causal(self):
        c,e,tr=self.inputs()
        full=decode(c,e,tr,T=5,N=0)
        for stop in range(1,20):
            np.testing.assert_array_equal(decode(c[:stop],e[:stop],tr,T=5,N=0),full[:stop])
    def test_delayed_decision_uses_future_evidence(self):
        c=[np.array([0,1])]*2
        e=[np.array([[.6,0,0],[.4,0,0]]),np.array([[.01,0,0],[1,0,0]])]
        def tr(t,v):return np.array([[[1,1],[.001,.001]],[[.001,.001],[1,1]]])
        self.assertEqual(decode(c,e,tr,2,0,[1,0,0],[1,0])[0],0)
        self.assertEqual(decode(c,e,tr,2,1,[1,0,0],[1,0])[0],1)
    def test_validation_and_flush(self):
        c,e,tr=self.inputs(2)
        self.assertEqual(len(decode(c,e,tr,5,4)),2)
        with self.assertRaises(ValueError):decode(c,e,tr,5,5)

if __name__=='__main__':unittest.main()
