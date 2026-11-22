"""CPU-only matched-seed benchmark on highway-v0 with oracle simulator state.

No online memory update; separate offline baseline calibration and test seeds.
Experimental safety metrics are proxies. Rendered screenshots are true env output.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
import json
import platform
import sys
import importlib.metadata as metadata
import numpy as np
import pandas as pd

from .core import snapshot_from_env, hazard_ttc, minimum_gap, episode_feature
from .controllers import METHODS, choose_method
from .memory import EpisodicMemory, save_training_records
from .scenarios import SCENARIOS, Scenario, config

PROTOCOL_VERSION = '3.0.0'
DURATION = 30
NEAR_TTC_S = 2.5
NEAR_GAP_M = 6.0

@dataclass(frozen=True)
class Settings:
    mode: str
    out: Path
    n_calibration: int
    n_test: int
    base_seed: int
    screenshots: bool = True


def make_env(scenario: Scenario, render: bool):
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Missing highway-env. Run setup_windows.bat first.') from exc
    return gym.make('highway-v0', config=config(scenario),
                    render_mode='rgb_array' if render else None)


def save_frame(env, path: Path) -> None:
    from PIL import Image
    frame = env.render()
    if frame is None:
        raise RuntimeError('Rendering returned None; check pygame-ce and render_mode.')
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f'Invalid frame shape {arr.shape}')
    if arr.shape[0] < 100 or arr.shape[1] < 200:
        raise ValueError(f'Unexpectedly small frame {arr.shape}')
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).convert('RGB').save(path)


def run_episode(scenario: Scenario, seed: int, method: str,
                memory: EpisodicMemory | None = None,
                calibrate: bool = False, screenshot: Path | None = None):
    if calibrate and method != 'baseline':
        raise ValueError('Baseline-only calibration')
    env = make_env(scenario, screenshot is not None)
    try:
        env.reset(seed=int(seed))
        start_state = snapshot_from_env(env)
        initial_nearby = int(start_state.nearby_count)
        initial_front_gap = float(start_state.lanes[start_state.lane_id].front_gap)
        framesaved = False
        if screenshot is not None:
            # Save a real initial frame in case crash occurs before the 5th step.
            save_frame(env, screenshot)
            framesaved = True
        steps, samples = [], []
        total_reward, total_distance = 0., 0.
        previous_x = float(env.unwrapped.vehicle.position[0])
        completed = False
        for step in range(DURATION+3):
            start_ns = perf_counter_ns()
            state = snapshot_from_env(env)
            actions = set(int(a) for a in env.unwrapped.action_type.get_available_actions())
            decision = choose_method(method, state, actions, memory)
            decision_ms = (perf_counter_ns()-start_ns)/1e6
            feature = episode_feature(state, decision.proposed) if calibrate else None
            _, reward, terminated, truncated, _info = env.step(decision.executed)
            after = snapshot_from_env(env)
            collision = int(bool(env.unwrapped.vehicle.crashed))
            ttc, gap = hazard_ttc(after), minimum_gap(after)
            hit_ttc, hit_gap = int(ttc < NEAR_TTC_S), int(gap < NEAR_GAP_M)
            hazard = int(bool(hit_ttc or hit_gap or collision))
            x = float(env.unwrapped.vehicle.position[0])
            total_distance += max(0., x-previous_x)
            previous_x = x
            total_reward += float(reward)
            steps.append({
                'scenario':scenario.name, 'density':scenario.density_label,
                'behavior':scenario.behavior_label, 'seed':seed,
                'method':method, 'step':step,
                'proposed_action':decision.proposed, 'executed_action':decision.executed,
                'intervention':decision.intervention, 'intervention_reason':decision.reason,
                'ttc_trigger':int(decision.ttc_trigger), 'rss_trigger':int(decision.rss_trigger),
                'cbf_trigger':int(decision.cbf_trigger), 'memory_trigger':int(decision.memory_trigger),
                'memory_additive':decision.memory_additive,
                'memory_score':decision.memory_score, 'memory_match':int(decision.memory_match),
                'memory_distance':(decision.memory_distance if np.isfinite(decision.memory_distance) else -1),
                'post_ttc_s':ttc, 'post_gap_m':gap,
                'near_miss_step':hazard,'ttc_low_step':hit_ttc,'gap_low_step':hit_gap,
                'speed_mps':after.speed,'collision_step':collision,
                'decision_latency_ms':decision_ms,
            })
            if calibrate:
                samples.append({
                    'scenario':scenario.name,'seed':seed,'step':step,
                    'action':decision.proposed,'hazard_label':hazard,
                    'post_min_ttc_s':ttc,'post_min_gap_m':gap, 'feature':feature,
                })
            if screenshot is not None and step == 4:
                save_frame(env, screenshot)
                framesaved = True
            completed = bool(terminated or truncated or collision)
            if completed:
                break
        if not completed:
            raise RuntimeError(f'Episode did not finish within {DURATION+3} steps')
        df = pd.DataFrame(steps)
        result = {
            'scenario':scenario.name, 'density':scenario.density_label, 'behavior':scenario.behavior_label,
            'vehicles_count':scenario.vehicles_count, 'vehicles_density':scenario.vehicles_density,
            'seed':seed,'method':method,
            'steps':len(steps), 'collision':int(bool(env.unwrapped.vehicle.crashed)),
            'success':int(not bool(env.unwrapped.vehicle.crashed)),
            'initial_nearby_count':initial_nearby, 'initial_front_gap_m':initial_front_gap,
            'cumulative_reward':float(total_reward),
            'distance_m':float(total_distance),
            'distance_per_sim_s':float(total_distance/max(len(steps),1)),
            'mean_speed_mps':float(df.speed_mps.mean()),
            'near_miss':int(bool(df.near_miss_step.any())),
            'near_miss_step_rate':float(df.near_miss_step.mean()),
            'ttc_low_step_rate':float(df.ttc_low_step.mean()),
            'gap_low_step_rate':float(df.gap_low_step.mean()),
            'min_ttc_s_cap15':float(df.post_ttc_s.min()),
            'min_gap_m_cap120':float(df.post_gap_m.min()),
            'interventions':int(df.intervention.sum()),
            'intervention_rate':float(df.intervention.mean()),
            'memory_additive_interventions':int(df.memory_additive.sum()),
            'mean_decision_latency_ms':float(df.decision_latency_ms.mean()),
            'p95_decision_latency_ms':float(df.decision_latency_ms.quantile(.95)),
            'render_saved':int(framesaved) if screenshot is not None else -1,
        }
        return result, steps, samples
    finally:
        env.close()


def run_study(cfg: Settings) -> dict:
    out = cfg.out
    if any((out/n).exists() for n in ('run_manifest.json','episodes.csv','decisions.csv','frozen_memory.npz')):
        raise FileExistsError(f'{out} already contains a run. Use --out NEW_FOLDER to prevent mixing trials.')
    (out/'figures').mkdir(parents=True,exist_ok=True)
    (out/'tables').mkdir(parents=True,exist_ok=True)
    rows = []
    print(f'[CALIBRATION] {len(SCENARIOS)} x {cfg.n_calibration} independent baseline episodes',flush=True)
    for scenario in SCENARIOS:
        for i in range(cfg.n_calibration):
            seed = cfg.base_seed + 1000 + i
            _, _, rec = run_episode(scenario, seed, 'baseline', calibrate=True)
            rows.extend(rec)
        print(f'  {scenario.name}: OK',flush=True)
    save_training_records(rows, out)
    memory = EpisodicMemory.from_records(rows, max_per_class=900, seed=cfg.base_seed)
    memory.save(out)
    positives = int(memory.labels.sum())
    if positives == 0:
        print('[WARNING] No positive calibration hazards: Memory cannot intervene. QA will flag.',flush=True)
    episodes, decisions = [], []
    total = len(SCENARIOS)*cfg.n_test*len(METHODS)
    print(f'[EVALUATION] {total} episodes | 6 controllers | six scenarios',flush=True)
    for scenario in SCENARIOS:
        for i in range(cfg.n_test):
            seed = cfg.base_seed + 10000 + i
            for method in METHODS:
                capture = (cfg.screenshots and method == 'baseline' and i == 0
                           and scenario.name in ('high_idm','high_aggr'))
                imagefile = (out/'figures'/f'highway_env_{scenario.name}.png') if capture else None
                row, step_rows, _ = run_episode(scenario, seed, method, memory,
                                                screenshot=imagefile)
                episodes.append(row)
                decisions.extend(step_rows)
        print(f'  {scenario.name}: {cfg.n_test*len(METHODS)} episodes completed',flush=True)
    pd.DataFrame(episodes).to_csv(out/'episodes.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'decisions.csv',index=False)
    installed = {}
    for lib in ('highway-env','gymnasium','numpy','pandas','matplotlib','pygame-ce','Pillow','scipy'):
        try: installed[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError: installed[lib] = None
    settings = {
        'mode':cfg.mode,'protocol_version':PROTOCOL_VERSION,
        'base_seed':cfg.base_seed, 'calibration_episodes_per_scenario':cfg.n_calibration,
        'test_episodes_per_scenario':cfg.n_test,
        'total_evaluation_episodes':total,
        'methods':list(METHODS),
        'scenarios':[s.__dict__ for s in SCENARIOS],
        'duration_s':DURATION, 'policy_frequency_hz':1, 'simulation_frequency_hz':10,
        'calibration_seed_range':[cfg.base_seed+1000,cfg.base_seed+1000+cfg.n_calibration-1],
        'test_seed_range':[cfg.base_seed+10000,cfg.base_seed+10000+cfg.n_test-1],
        'memory_examples':len(memory.labels),'memory_positive_samples':positives,
        'memory_frozen':True,
        'ttc_threshold_s':3.0,'rss_reaction_s':.8,'cbf_horizon_s':1.0,
        'near_ttc_threshold_s':NEAR_TTC_S,'near_gap_threshold_m':NEAR_GAP_M,
    }
    manifest = {
        'status':'completed','settings':settings,
        'platform':platform.platform(),'python':sys.version,'dependencies':installed,
        'limitations':[
            'Only straight highway-v0; not valid on merge/intersection topology.',
            'All policies use privileged state read from simulator road objects, not noisy onboard sensors.',
            'RSS-inspired longitudinal safe distance is not a complete RSS implementation.',
            'CBF-inspired discrete one-step barrier is not a formal CBF with invariance proof.',
            'Memory training labels are 1-step hazards after baseline actions, not counterfactual outcomes.',
            'Same reset seed matches initial conditions across methods; trajectories diverge under interventions.',
            'Decision time excludes simulation step, rendering, file I/O and memory training.',
            'Different vehicle behavior classes are scenario modifications, not new maps.',
            'Collision safety outside this simulator is not established.',
        ],
    }
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest
