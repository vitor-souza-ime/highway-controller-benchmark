"""Six *heuristic* CPU controllers; no formal safety guarantee.

The benchmarks share an identical proposal policy and SLOWER fallback.
RSS is RSS-inspired longitudinal spacing (not the full RSS specification).
CBF is a one-step discrete barrier surrogate, not a certified CBF.
TTC uses closing speed / closing time, with separate merging-lane checks.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from .core import (Snapshot, Lane, LEFT, IDLE, RIGHT, FASTER, SLOWER,
                   base_action, safe_fallback)
from .memory import EpisodicMemory

METHODS = ('baseline', 'ttc_shield', 'rss_shield', 'cbf_discrete', 'memory_only', 'hybrid')
DISPLAY = {'baseline':'Baseline', 'ttc_shield':'TTC', 'rss_shield':'RSS*',
           'cbf_discrete':'CBF*', 'memory_only':'Memory', 'hybrid':'Hybrid'}
# Fixed a priori; no tuning on evaluation episodes.
TTC_FRONT_S = 3.0
TTC_LANE_REAR_S = 2.2
RSS_REACTION_S = 0.8
RSS_ACCEL_MAX = 1.5
RSS_BRAKE_EGO = 4.0
RSS_BRAKE_FRONT = 4.0
RSS_MIN_GAP_M = 4.0
CBF_HEADWAY_S = 1.0
CBF_MIN_GAP_M = 4.0
CBF_KAPPA = 0.50
MEMORY_THRESHOLD = 0.58

@dataclass(frozen=True)
class Decision:
    proposed: int
    executed: int
    intervention: int
    reason: str
    ttc_trigger: bool = False
    rss_trigger: bool = False
    cbf_trigger: bool = False
    memory_trigger: bool = False
    memory_score: float = 0.0
    memory_distance: float = math.inf
    memory_match: bool = False
    memory_additive: int = 0


def ttc_risk(state: Snapshot, action: int) -> bool:
    if action == SLOWER:
        return False
    lane = state.lane(state.target_of(action))
    if lane is None:
        return True
    horizon = TTC_FRONT_S + (0.5 if action == FASTER else 0.0)
    unsafe = lane.front_ttc < horizon
    if action in (LEFT, RIGHT):
        unsafe |= lane.rear_ttc < TTC_LANE_REAR_S
        # Continuing a lane change must not ignore imminent risk in current lane.
        unsafe |= state.lanes[state.lane_id].front_ttc < TTC_FRONT_S
    if state.in_lane_change and state.target_lane_id is not None:
        target = state.lane(state.target_lane_id)
        if target is not None:
            unsafe |= target.front_ttc < TTC_FRONT_S
            unsafe |= target.rear_ttc < TTC_LANE_REAR_S
    return bool(unsafe)


def rss_safe_front_gap(v: float, lead_v: float) -> float:
    """One-dimensional RSS-inspired safe distance [m]."""
    tau, a = RSS_REACTION_S, RSS_ACCEL_MAX
    d = (v * tau + 0.5*a*tau*tau
         + (v+a*tau)**2/(2*RSS_BRAKE_EGO)
         - lead_v**2/(2*RSS_BRAKE_FRONT))
    return RSS_MIN_GAP_M + max(0., d)


def rss_risk(state: Snapshot, action: int) -> bool:
    if action == SLOWER:
        return False
    lane = state.lane(state.target_of(action))
    if lane is None:
        return True
    v = state.speed + (2.0 if action == FASTER else 0.)
    risky = lane.front_gap < rss_safe_front_gap(v, lane.front_speed)
    if action in (LEFT, RIGHT):
        # Conservative surrogate for lateral merge occupancy; not full lateral RSS.
        rear_safe = RSS_MIN_GAP_M + max(7.0, lane.rear_speed*0.55)
        risky |= lane.rear_gap < rear_safe
        risky |= state.lanes[state.lane_id].front_gap < RSS_MIN_GAP_M
    if state.in_lane_change and state.target_lane_id is not None:
        target = state.lane(state.target_lane_id)
        if target:
            risky |= target.front_gap < rss_safe_front_gap(v, target.front_speed)
            risky |= target.rear_gap < RSS_MIN_GAP_M + max(7, target.rear_speed*0.55)
    return bool(risky)


def _cbf_front(state: Snapshot, lane: Lane, action: int) -> bool:
    """Discrete-time barrier inequality: h_next >= (1-kappa)*h_now."""
    v = state.speed
    accel = 2.0 if action == FASTER else (-3.5 if action == SLOWER else 0.0)
    v_next = max(0., v+accel)
    h = lane.front_gap - (CBF_MIN_GAP_M + CBF_HEADWAY_S*v)
    next_gap = lane.front_gap + lane.front_speed - v_next
    h_next = next_gap - (CBF_MIN_GAP_M + CBF_HEADWAY_S*v_next)
    return bool(h_next < (1-CBF_KAPPA)*h)


def cbf_risk(state: Snapshot, action: int) -> bool:
    if action == SLOWER:
        return False
    lane = state.lane(state.target_of(action))
    if lane is None:
        return True
    risky = _cbf_front(state, lane, action)
    if action in (LEFT, RIGHT):
        # Rear-only approximation for discrete lateral actions.
        risky |= lane.rear_gap < max(8., 0.65*lane.rear_speed)
        risky |= _cbf_front(state, state.lanes[state.lane_id], action)
    if state.in_lane_change and state.target_lane_id is not None:
        target = state.lane(state.target_lane_id)
        if target:
            risky |= _cbf_front(state, target, action)
            risky |= target.rear_gap < max(8., 0.65*target.rear_speed)
    return bool(risky)


def choose_method(method: str, state: Snapshot, available: set[int],
                  memory: EpisodicMemory | None = None) -> Decision:
    if method not in METHODS:
        raise ValueError('Unknown controller: '+method)
    proposed = base_action(state, available)
    if proposed not in available:
        raise RuntimeError(f'Base action unavailable: {proposed}, {available}')
    if method == 'baseline':
        return Decision(proposed, proposed, 0, 'baseline')
    ttc = method in ('ttc_shield','hybrid') and ttc_risk(state, proposed)
    rss = method == 'rss_shield' and rss_risk(state, proposed)
    cbf = method == 'cbf_discrete' and cbf_risk(state, proposed)
    score, distance, found = (memory.lookup(state, proposed) if memory is not None
                              and method in ('memory_only','hybrid')
                              else (0., math.inf, False))
    mem = method in ('memory_only','hybrid') and bool(found) and score >= MEMORY_THRESHOLD
    triggered = ttc or rss or cbf or mem
    fallback = safe_fallback(state, available)
    override = bool(triggered and proposed != fallback and fallback in available)
    reason = ('none' if not triggered else
              'not_actionable' if not override else
              'ttc_memory' if ttc and mem else
              'ttc' if ttc else 'rss' if rss else 'cbf' if cbf else
              'memory_additive' if method == 'hybrid' else 'memory')
    return Decision(proposed, fallback if override else proposed, int(override), reason,
                    bool(ttc), bool(rss), bool(cbf), bool(mem),
                    float(score), float(distance), bool(found),
                    int(override and method == 'hybrid' and mem and not ttc))
