"""Predeclared 3 density levels x 2 traffic driving styles on highway-v0.
The deliberately fixed number AND density avoid the v2 pilot's degenerate
traffic counts. Actual traffic difficulty must still be assessed empirically.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Scenario:
    name: str
    density_label: str
    behavior_label: str
    vehicles_count: int
    vehicles_density: float
    behavior_class: str

SCENARIOS = (
    Scenario('low_idm','low','IDM',16,0.65,'highway_env.vehicle.behavior.IDMVehicle'),
    Scenario('mid_idm','medium','IDM',25,1.25,'highway_env.vehicle.behavior.IDMVehicle'),
    Scenario('high_idm','high','IDM',38,2.10,'highway_env.vehicle.behavior.IDMVehicle'),
    Scenario('low_aggr','low','aggressive',16,0.65,'highway_env.vehicle.behavior.AggressiveVehicle'),
    Scenario('mid_aggr','medium','aggressive',25,1.25,'highway_env.vehicle.behavior.AggressiveVehicle'),
    Scenario('high_aggr','high','aggressive',38,2.10,'highway_env.vehicle.behavior.AggressiveVehicle'),
)
SCENARIO_MAP = {s.name:s for s in SCENARIOS}

def config(s: Scenario) -> dict:
    return {
        'lanes_count':3, 'vehicles_count':s.vehicles_count,
        'vehicles_density':s.vehicles_density,
        'other_vehicles_type':s.behavior_class,
        'duration':30, 'policy_frequency':1, 'simulation_frequency':10,
        'observation':{'type':'Kinematics','vehicles_count':8},
        'action':{'type':'DiscreteMetaAction'},
        'screen_width':1100, 'screen_height':380,
        'scaling':4.3, 'offscreen_rendering':True,
        'real_time_rendering':False,
    }
