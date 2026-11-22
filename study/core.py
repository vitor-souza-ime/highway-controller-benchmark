"""Controller and safety features for straight highway-v0 roads.

All safety indicators are inexpensive, deterministic approximations. They
are not formal collision guarantees and they are not suitable for deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

LEFT, IDLE, RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4
ACTION_NAMES = {LEFT: "LANE_LEFT", IDLE: "IDLE", RIGHT: "LANE_RIGHT", FASTER: "FASTER", SLOWER: "SLOWER"}
FAR = 120.0


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


@dataclass(frozen=True)
class Lane:
    front_gap: float = FAR
    rear_gap: float = FAR
    front_closing: float = 0.0  # ego longitudinal velocity minus front velocity
    rear_closing: float = 0.0  # rear velocity minus ego longitudinal velocity
    front_speed: float = 25.0  # observed leading vehicle speed (m/s)
    rear_speed: float = 25.0   # observed following vehicle speed (m/s)

    @property
    def front_ttc(self) -> float:
        return max(0.0, self.front_gap) / self.front_closing if self.front_closing > 1e-6 else math.inf

    @property
    def rear_ttc(self) -> float:
        return max(0.0, self.rear_gap) / self.rear_closing if self.rear_closing > 1e-6 else math.inf


@dataclass(frozen=True)
class Snapshot:
    speed: float
    lane_id: int
    lanes_count: int
    lanes: tuple[Lane, ...]
    nearby_count: int
    in_lane_change: bool = False
    target_lane_id: int | None = None

    def lane(self, lane_id: int) -> Lane | None:
        return self.lanes[lane_id] if 0 <= lane_id < len(self.lanes) else None

    def target_of(self, action: int) -> int:
        return self.lane_id + (1 if action == RIGHT else -1 if action == LEFT else 0)


def _lane_id(vehicle: Any) -> int | None:
    lane_index = getattr(vehicle, 'lane_index', None)
    if not isinstance(lane_index, (tuple, list)) or len(lane_index) < 3:
        return None
    try:
        return int(lane_index[-1])
    except (TypeError, ValueError):
        return None


def _longitudinal_velocity(vehicle: Any) -> float:
    return float(vehicle.speed) * math.cos(float(getattr(vehicle, 'heading', 0.0)))


def snapshot_from_env(env: Any) -> Snapshot:
    """Read ground-truth simulator state, not a simulated sensor observation.

    Only the straight, same-direction lanes of highway-v0 are supported.
    """
    obj = env.unwrapped
    ego = obj.vehicle
    lanes_count = int(obj.config['lanes_count'])
    lane_id = _lane_id(ego)
    if lane_id is None:
        raise RuntimeError('Ego lane_index unavailable; this experiment requires highway-v0.')
    tgt_idx = getattr(ego, 'target_lane_index', None)
    tgt_id = int(tgt_idx[-1]) if isinstance(tgt_idx, (tuple, list)) and len(tgt_idx) >= 3 else lane_id
    front = [FAR] * lanes_count
    rear = [FAR] * lanes_count
    front_closing = [0.0] * lanes_count
    rear_closing = [0.0] * lanes_count
    ex = float(ego.position[0])
    ev = _longitudinal_velocity(ego)
    front_speed = [max(0.0, ev)] * lanes_count
    rear_speed = [max(0.0, ev)] * lanes_count
    ego_len = float(getattr(ego, 'LENGTH', 5.0))
    nearby = 0
    for other in obj.road.vehicles:
        if other is ego:
            continue
        other_id = _lane_id(other)
        if other_id is None or not 0 <= other_id < lanes_count:
            continue
        dx = float(other.position[0]) - ex
        if abs(dx) < 60.0:
            nearby += 1
        bumper_gap = max(0.0, abs(dx) - 0.5 * (ego_len + float(getattr(other, 'LENGTH', 5.0))))
        ov = _longitudinal_velocity(other)
        if dx >= 0 and bumper_gap < front[other_id]:
            front[other_id] = bumper_gap
            front_closing[other_id] = max(0.0, ev - ov)
            front_speed[other_id] = max(0.0, ov)
        elif dx < 0 and bumper_gap < rear[other_id]:
            rear[other_id] = bumper_gap
            rear_closing[other_id] = max(0.0, ov - ev)
            rear_speed[other_id] = max(0.0, ov)
    return Snapshot(
        speed=max(0.0, ev), lane_id=lane_id, lanes_count=lanes_count,
        lanes=tuple(Lane(front[i], rear[i], front_closing[i], rear_closing[i], front_speed[i], rear_speed[i]) for i in range(lanes_count)),
        nearby_count=nearby, in_lane_change=(tgt_id != lane_id), target_lane_id=tgt_id,
    )


def risk_score(state: Snapshot, action: int) -> float:
    """Uncalibrated [0, 1] heuristic risk proxy for a proposed maneuver."""
    target = state.target_of(action)
    lane = state.lane(target)
    if lane is None:
        return 1.0
    if action == SLOWER:
        return 0.0
    v = max(1.0, state.speed)
    front_headway = lane.front_gap / v
    risk = max(
        clip01((1.8 - front_headway) / 1.8),
        clip01((3.5 - lane.front_ttc) / 3.5),
    )
    if action in (LEFT, RIGHT):
        # Rear headway is computed with the closing vehicle's relative speed.
        rear_t = lane.rear_gap / max(1.0, state.speed + lane.rear_closing)
        risk = max(risk,
            clip01((1.2 - rear_t) / 1.2),
            clip01((3.0 - lane.rear_ttc) / 3.0),
        )
    return float(risk)


def base_action(state: Snapshot, available: set[int]) -> int:
    """Fixed efficiency-oriented, somewhat opportunistic rule policy."""
    if state.in_lane_change:
        return IDLE
    own = state.lanes[state.lane_id]
    v = state.speed
    if own.front_gap < 43.0 and v > 14.0:
        candidates = []
        for act in (LEFT, RIGHT):
            if act not in available:
                continue
            adj = state.lane(state.target_of(act))
            if adj and adj.front_gap > max(9.0, 0.58 * v) and adj.rear_gap > max(7.0, 0.36 * v):
                candidates.append((adj.front_gap + 0.1 * adj.rear_gap, -act, act))
        if candidates:
            return max(candidates)[2]
    if own.front_gap < max(11.0, 0.68 * v) or own.front_ttc < 2.1:
        return SLOWER if SLOWER in available else IDLE
    if v < 29.0 and own.front_gap > 28.0 and FASTER in available:
        return FASTER
    return IDLE


def safe_fallback(state: Snapshot, available: set[int]) -> int:
    """Conservative replacement, not a collision avoidance guarantee."""
    return SLOWER if SLOWER in available else IDLE


def hazard_ttc(state: Snapshot) -> float:
    """Minimum front/rear closing TTC on occupied lane(s), capped at 15 seconds."""
    lanes = [state.lanes[state.lane_id]]
    if state.in_lane_change and state.target_lane_id is not None:
        adj = state.lane(state.target_lane_id)
        if adj is not None:
            lanes.append(adj)
    values = [min(x.front_ttc, x.rear_ttc) for x in lanes]
    return float(min(15.0, *values))


def minimum_gap(state: Snapshot) -> float:
    """Minimum bumper gap on ego's occupied/target lane(s), capped at 120m."""
    lanes = [state.lanes[state.lane_id]]
    if state.in_lane_change and state.target_lane_id is not None:
        adj = state.lane(state.target_lane_id)
        if adj is not None:
            lanes.append(adj)
    return float(min(min(x.front_gap, x.rear_gap) for x in lanes))


def episode_feature(state: Snapshot, action: int) -> np.ndarray:
    """Fixed-scale feature vector for 7-nearest-neighbor episodic retrieval.

    The vector does not include an outcome. Outcome labels are learned on
    calibration episodes only. One-hot encoding prevents action-code distances.
    """
    own = state.lanes[state.lane_id]
    other = state.lane(state.target_of(action))
    if other is None:
        other = Lane(0.0, 0.0, 20.0, 20.0)
    def packed(lane: Lane) -> list[float]:
        return [min(lane.front_gap, 120) / 120, min(lane.rear_gap, 120) / 120,
                min(lane.front_closing, 30) / 30, min(lane.rear_closing, 30) / 30,
                min(lane.front_ttc, 15) / 15, min(lane.rear_ttc, 15) / 15]
    return np.asarray(
        [min(state.speed, 40) / 40, state.lane_id / max(1, state.lanes_count-1),
         min(state.nearby_count, 20) / 20, float(state.in_lane_change)]
        + packed(own) + packed(other) + [float(action == i) for i in range(5)],
        dtype=np.float32,
    )
