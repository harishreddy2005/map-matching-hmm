import unittest
from types import SimpleNamespace
import numpy as np
from shapely.geometry import LineString
from src.features import emissions, transitions

class FeatureTests(unittest.TestCase):
    def data(self):
        return SimpleNamespace(xy=np.array([[0.,0.],[20.,0.]]),speed=np.array([0.,20.]),
                               heading=np.array([[0.,0.],[1.,0.]]),
                               lines=[LineString([(0,0),(100,0)]),LineString([(100,0),(0,0)])],
                               limits=np.array([10.,30.]),midpoints=np.array([[50.,0.],[50.,0.]]))
    def test_directed_heading_and_high_speed_gate(self):
        d=self.data();f=emissions(d,1,np.array([0,1]),np.zeros(2))
        np.testing.assert_allclose(f[:,1],[1,-1])
        np.testing.assert_allclose(f[:,2],[0,1])
    def test_stopped_heading_is_neutral(self):
        d=self.data();f=emissions(d,0,np.array([0,1]),np.array([0.,10.]))
        np.testing.assert_allclose(f[:,1],0)
        self.assertAlmostEqual(f[1,0]/f[0,0],np.exp(-.5))
    def test_visited_penalty_exempts_staying(self):
        d=self.data();f=transitions(d,1,np.array([0,1]),np.array([0,1]),{0,1})
        np.testing.assert_allclose(f[:,:,0],np.eye(2))
        np.testing.assert_allclose(f[:,:,1],1/21)

if __name__=='__main__':unittest.main()
