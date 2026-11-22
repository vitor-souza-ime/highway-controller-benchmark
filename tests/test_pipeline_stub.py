"""STUB INTEGRATION TESTS ONLY! No mock CSV in distributed experimental results."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np
from study.runner import Settings,run_study
from study.report import create_report,wilson_ci,paired_cluster_ci
from study.quality import check_results,save_quality
from study.scenarios import SCENARIOS
from study.controllers import METHODS

class Vehicle:
    LENGTH=5.0
    heading=0.0
    crashed=False
    def __init__(self,x,lane,speed):
        self.position=[float(x),4.*lane]
        self.lane_index=('a','b',lane)
        self.target_lane_index=self.lane_index
        self.speed=float(speed)
        self.crashed=False

class ActionType:
    def get_available_actions(self):return [0,1,2,3,4]

class EnvStub:
    """Lightweight API double, not highway-env or a traffic physics model."""
    def __init__(self,scenario,render):
        self.unwrapped=self
        self.config={'lanes_count':3}
        self.action_type=ActionType()
        self.s=scenario
        self.t=0
    def reset(self,seed):
        self.t=0
        self.vehicle=Vehicle(0,1,26)
        gap={'low':50,'medium':28,'high':17}[self.s.density_label]+(seed%4)
        self.other=Vehicle(gap,1,17 if self.s.behavior_label=='IDM' else 22)
        self.road=type('Road',(),{'vehicles':[self.vehicle,self.other]})()
        return np.zeros((8,5)),{}
    def step(self,action):
        if action==4:self.vehicle.speed=max(6,self.vehicle.speed-2)
        if action==3:self.vehicle.speed=min(31,self.vehicle.speed+2)
        self.vehicle.position[0]+=self.vehicle.speed
        self.other.position[0]+=self.other.speed
        self.t+=1
        if self.other.position[0]-self.vehicle.position[0]<5:
            self.vehicle.crashed=True
        return np.zeros((8,5)),.2,bool(self.vehicle.crashed),self.t>=7,{}
    def render(self):
        # Assertions test rendering pipeline shape only, NOT simulator graphic quality.
        return np.full((180,640,3),220,dtype=np.uint8)
    def close(self):pass

class PipelineTests(unittest.TestCase):
    def test_stub_pipeline_creates_every_file(self):
        with TemporaryDirectory() as td:
            out=Path(td)/'pilot'
            with patch('study.runner.make_env',side_effect=lambda s,render: EnvStub(s,render)):
                run_study(Settings('smoke',out,1,1,3101,True))
            create_report(out)
            result=check_results(out)
            save_quality(out,result)
            self.assertNotEqual(result['status'],'FAIL',result['errors'])
            self.assertEqual(result['metrics']['evaluation_episodes'],6*6)
            self.assertTrue((out/'figures'/'highway_env_high_idm.png').is_file())
            self.assertTrue((out/'figures'/'highway_env_high_aggr.png').is_file())
            self.assertEqual(len(list((out/'figures').glob('fig_*.pdf'))),12)
            self.assertEqual(len(list((out/'figures').glob('panel_*.pdf'))),2)
            self.assertEqual(len(list((out/'tables').glob('table_*.tex'))),9)
            self.assertTrue((out/'QUALITY_CHECK.txt').exists())
            self.assertEqual(len(list((out/'figures').glob('fig_*.svg'))),12)
            self.assertTrue((out/'paper_assets.tex').exists())
    def test_wilson(self):
        lo,hi=wilson_ci(3,10)
        self.assertLess(lo,.3)
        self.assertGreater(hi,.3)
    def test_scenarios(self):
        self.assertEqual(len(SCENARIOS),6)
        self.assertEqual({s.behavior_label for s in SCENARIOS},{'IDM','aggressive'})
        self.assertEqual(len(METHODS),6)

if __name__=='__main__':unittest.main()
