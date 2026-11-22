import unittest
import numpy as np
from study.core import Snapshot,Lane,LEFT,IDLE,RIGHT,FASTER,SLOWER,episode_feature
from study.controllers import (METHODS, ttc_risk, rss_risk, cbf_risk,
                               rss_safe_front_gap, choose_method)
from study.memory import EpisodicMemory

class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.safe=Snapshot(speed=25,lane_id=1,lanes_count=3,
                           lanes=(Lane(),Lane(),Lane()),nearby_count=0)
        self.close=Snapshot(speed=28,lane_id=1,lanes_count=3,
                            lanes=(Lane(),Lane(front_gap=18,front_closing=9,front_speed=19),Lane()),
                            nearby_count=1)
    def test_six_methods(self):
        self.assertEqual(len(METHODS),6)
        for meth in METHODS:
            d=choose_method(meth,self.safe,set(range(5)))
            self.assertEqual(d.intervention,int(d.proposed!=d.executed))
    def test_ttc_detects_imminent_closing(self):
        self.assertTrue(ttc_risk(self.close,IDLE))
        self.assertFalse(ttc_risk(self.safe,IDLE))
        self.assertFalse(ttc_risk(self.close,SLOWER))
    def test_rss_has_speed_aware_safe_gap(self):
        self.assertGreater(rss_safe_front_gap(30,15),rss_safe_front_gap(20,15))
        self.assertTrue(rss_risk(self.close,IDLE))
        self.assertFalse(rss_risk(self.safe,IDLE))
    def test_cbf_discrete_filter(self):
        self.assertTrue(cbf_risk(self.close,FASTER))
        self.assertFalse(cbf_risk(self.safe,IDLE))
    def test_no_override_when_base_is_braking(self):
        s=Snapshot(speed=29,lane_id=1,lanes_count=3,
                   lanes=(Lane(front_gap=0,rear_gap=0),
                          Lane(front_gap=2,front_closing=15,front_speed=14),
                          Lane(front_gap=0,rear_gap=0)),nearby_count=3)
        d=choose_method('ttc_shield',s,set(range(5)))
        self.assertEqual(d.executed,SLOWER)
        self.assertEqual(d.intervention,0)
    def test_frozen_memory_shape_and_lookup(self):
        feat=episode_feature(self.close,IDLE)
        rec=[dict(feature=feat,hazard_label=1),dict(feature=feat+.02,hazard_label=1)]
        memory=EpisodicMemory.from_records(rec)
        self.assertEqual(memory.features.shape[1],21)
        sc,dist,match=memory.lookup(self.close,IDLE)
        self.assertTrue(match)
        self.assertGreater(sc,0.5)
    def test_method_baseline_never_intervenes(self):
        d=choose_method('baseline',self.close,set(range(5)))
        self.assertEqual(d.intervention,0)

if __name__=='__main__':unittest.main()
