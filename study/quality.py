"""Reject malformed experiments and flag degenerate traffic situations.

'PASS' means consistent files, not superiority of a controller. Negative
results are allowed in a valid benchmark and never automatically fail QA.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
from .controllers import METHODS
from .scenarios import SCENARIOS


def check_results(folder:Path)->dict:
    errors=[];warnings=[];checks=[];metrics={}
    required=['episodes.csv','decisions.csv','calibration_transitions.csv',
              'frozen_memory.npz','run_manifest.json']
    missing=[p for p in required if not (folder/p).is_file()]
    if missing:
        return dict(status='FAIL',errors=[f'Missing required files: {missing}'],warnings=[],checks=[],metrics={})
    try:
        ep=pd.read_csv(folder/'episodes.csv')
        d=pd.read_csv(folder/'decisions.csv')
        cal=pd.read_csv(folder/'calibration_transitions.csv')
        manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
        with np.load(folder/'frozen_memory.npz') as mem:
            mlen=len(mem['labels']); mshape=mem['features'].shape
    except Exception as ex:
        return dict(status='FAIL',errors=[f'Cannot read inputs: {ex}'],warnings=[],checks=[],metrics={})
    settings=manifest.get('settings',{})
    ntest=int(settings.get('test_episodes_per_scenario',0))
    ncal=int(settings.get('calibration_episodes_per_scenario',0))
    expected=len(SCENARIOS)*len(METHODS)*ntest
    metrics.update(evaluation_episodes=len(ep), decisions=len(d), calibration_steps=len(cal),
                   memory_examples=mlen, expected_episodes=expected)
    required_ep=['scenario','seed','method','collision','steps','initial_nearby_count',
                 'initial_front_gap_m','near_miss_step_rate','memory_additive_interventions']
    required_d=['scenario','seed','method','step','proposed_action','executed_action',
                'intervention','intervention_reason','memory_additive','decision_latency_ms']
    for name, frame, cols in [('episodes',ep,required_ep),('decisions',d,required_d)]:
        absent=[col for col in cols if col not in frame.columns]
        if absent: errors.append(f'{name}: missing columns {absent}')
    if errors: return dict(status='FAIL',errors=errors,warnings=warnings,checks=checks,metrics=metrics)
    if len(ep)!=expected:
        errors.append(f'Expected {expected} episodes, got {len(ep)}')
    if not set(ep.method)==set(METHODS):
        errors.append('Incomplete controller set')
    if not set(ep.scenario)=={s.name for s in SCENARIOS}:
        errors.append('Incomplete scenario set')
    if ep.duplicated(['scenario','seed','method']).any():
        errors.append('Duplicate episode rows')
    if d.duplicated(['scenario','seed','method','step']).any():
        errors.append('Duplicate action rows')
    for scenario in SCENARIOS:
        bases=[]
        for method in METHODS:
            seeds=set(ep[(ep.scenario==scenario.name)&(ep.method==method)].seed)
            if len(seeds)!=ntest:
                errors.append(f'{scenario.name}/{method} missing test seeds')
            bases.append(seeds)
        if len({frozenset(x) for x in bases})!=1:
            errors.append(f'{scenario.name}: unmatched paired seed sets')
    es=ep.set_index(['scenario','seed','method']).steps.sort_index()
    ds=d.groupby(['scenario','seed','method']).size().sort_index()
    if not es.equals(ds):
        errors.append('Step log count != episode steps')
    if not (d.intervention.astype(int)==(d.executed_action!=d.proposed_action).astype(int)).all():
        errors.append('Intervention attribution inconsistent with action changes')
    if (d[(d.method=='baseline')].intervention!=0).any():
        errors.append('Baseline interventions should always be zero')
    if (d.decision_latency_ms < 0).any() or not np.isfinite(d.decision_latency_ms).all():
        errors.append('Negative/nonfinite decision times')
    if not (d.executed_action.between(0,4)).all():
        errors.append('Invalid discrete actions')
    if not set(ep.seed).isdisjoint(set(cal.seed)):
        errors.append('Calibration/test seed overlap!')
    if mshape!=(mlen,21):
        errors.append(f'Corrupted episodic memory array {mshape}')
    if len(cal)<ncal*len(SCENARIOS):
        errors.append('Too few memory calibration transitions')
    if not errors:
        checks.append('Complete paired coverage, independent training seeds, valid actions and step logs')
    bad_add=d[(d.memory_additive==1) & ((d.method!='hybrid') |
             (d.intervention!=1) | (d.intervention_reason!='memory_additive'))]
    if len(bad_add): errors.append('Memory additive attribution inconsistent')
    epi_add=ep.set_index(['scenario','seed','method']).memory_additive_interventions.sort_index()
    action_add=d.groupby(['scenario','seed','method']).memory_additive.sum().sort_index()
    if not epi_add.equals(action_add): errors.append('Episode and decision memory-additive counts differ')
    # Headline scenario integrity: changing only vehicle count was degenerate in v2.
    base=ep[ep.method=='baseline']
    density_diag={}
    for behavior in ('IDM','aggressive'):
        sub=base[base.behavior==behavior]
        means={k:float(sub[sub.density==k].initial_nearby_count.mean())
               for k in ('low','medium','high')}
        density_diag[behavior]=means
        if all(np.isfinite(list(means.values()))) and max(means.values())-min(means.values())<.25:
            warnings.append(f'Initial local traffic does not distinguish density levels for {behavior}: {means}')
    metrics['initial_neighbor_means']=density_diag
    # Check whether IDM/aggressive alter observed trajectories, not just their labels.
    for level in ('low','medium','high'):
        pair=base[base.density==level].pivot(index='seed',columns='behavior',values='mean_speed_mps')
        if 'IDM' in pair and 'aggressive' in pair and np.allclose(pair.IDM,pair.aggressive,rtol=0,atol=1e-9):
            warnings.append(f'{level}: baseline mean velocities identical for both traffic behaviors')
    # Degenerate-trajectory detector catches the v2 problem where increasing
    # the traffic count did not alter the actual ego observations/trajectories.
    trajectory_diagnostics={}
    for behavior in ('IDM','aggressive'):
        identical=0; paired=0
        for seed in sorted(base[base.behavior==behavior].seed.unique()):
            a=d[(d.behavior==behavior)&(d.scenario==('mid_idm' if behavior=='IDM' else 'mid_aggr'))
                 &(d.seed==seed)&(d.method=='baseline')].sort_values('step')
            b=d[(d.behavior==behavior)&(d.scenario==('high_idm' if behavior=='IDM' else 'high_aggr'))
                 &(d.seed==seed)&(d.method=='baseline')].sort_values('step')
            if a.empty or b.empty: continue
            paired+=1
            keys=['executed_action','speed_mps','post_gap_m','post_ttc_s']
            if len(a)==len(b) and np.allclose(a[keys].to_numpy(float),b[keys].to_numpy(float),rtol=0,atol=1e-7):
                identical+=1
        trajectory_diagnostics[behavior]={'identical_mid_high':identical,'paired_seeds':paired}
        if paired>=3 and identical/paired>=.9:
            warnings.append(f'{behavior}: {identical}/{paired} medium/high baseline trajectories identical')
    metrics['medium_high_trajectory_identity']=trajectory_diagnostics
    repeated_trajectories=any(v['paired_seeds']>=3 and v['identical_mid_high']/v['paired_seeds']>=.9
                             for v in trajectory_diagnostics.values())
    # Compare controllers only as diagnostic (equal results are not scientifically forbidden).
    per=d.groupby(['scenario','seed','method'],sort=False).executed_action.apply(tuple)
    delta=0
    for s in SCENARIOS:
        seeds=ep[ep.scenario==s.name].seed.unique()
        for seed in seeds:
            try:
                if per.loc[(s.name,seed,'ttc_shield')]!=per.loc[(s.name,seed,'hybrid')]:
                    delta+=1
            except KeyError: pass
    metrics['distinct_ttc_vs_hybrid_episodes']=int(delta)
    memadd=int(d[d.method=='hybrid'].memory_additive.sum())
    metrics['hybrid_memory_additive_actions']=memadd
    if delta==0:
        warnings.append('Hybrid and TTC have identical executed actions in all matched episodes')
    if memadd==0:
        warnings.append('Hybrid memory produces no additional actions; valid negative benchmark result')
    if mlen==0 or (mlen>0 and int(cal.hazard_label.sum())==0):
        warnings.append('Memory calibration contains no hazards; interpret memory methods cautiously')
    for item in ('high_idm','high_aggr'):
        shot=folder/'figures'/f'highway_env_{item}.png'
        if not shot.exists():
            warnings.append(f'Missing rendered screenshot: {shot.name}')
        else:
            try:
                with Image.open(shot) as pic:
                    pic.verify()
                with Image.open(shot) as pic:
                    if pic.size[0]<200 or pic.size[1]<100:
                        errors.append(f'Invalid screenshot dimensions: {pic.size}')
                    else:
                        checks.append(f'Validated real simulator screenshot: {item} {pic.size}')
            except Exception as exc:
                errors.append(f'Invalid screenshot {item}: {exc}')
    fig_count=len(list((folder/'figures').glob('fig_*.pdf')))
    table_count=len(list((folder/'tables').glob('table_*.tex')))
    metrics['figures_pdf']=fig_count
    metrics['tables_tex']=table_count
    if fig_count<6 or table_count<3:
        errors.append(f'Insufficient paper assets: {fig_count} figures, {table_count} tables')
    else:
        checks.append('At least 6 plot PDFs and 3 LaTeX tables generated')
    # REVIEW only if BOTH traffic types fail to change the initial local traffic.
    degenerate_traffic=all(max(vals.values())-min(vals.values())<.25
                           for vals in density_diag.values())
    if degenerate_traffic:
        warnings.append('ALL density levels appear indistinguishable at initialization; review settings before full')
    if repeated_trajectories:
        warnings.append('Medium/high scenarios are trajectory-degenerate: do not run FULL yet')
    metrics['collision_rate_by_method']={m:float(ep[ep.method==m].collision.mean()) for m in METHODS}
    checks.append('QA status verifies integrity only; it does not certify method superiority')
    status=('FAIL' if errors else 'REVIEW' if (degenerate_traffic or repeated_trajectories) else 'PASS')
    return dict(status=status,errors=errors,warnings=warnings,checks=checks,metrics=metrics,
                interpretation='PASS = structurally consistent and density nondegenerate. It does not imply performance gains.')


def save_quality(out:Path, result:dict):
    (out/'QUALITY_CHECK.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    report=[f"QUALITY STATUS: {result['status']}",
            'PASS = complete benchmark data, not superiority or statistical significance.',
            '\nCHECKS:']
    report.extend('  OK: '+x for x in result['checks'])
    report.append('\nWARNINGS:')
    report.extend('  - '+x for x in result['warnings'])
    if not result['warnings']: report.append('  none')
    report.append('\nERRORS:')
    report.extend('  - '+x for x in result['errors'])
    if not result['errors']: report.append('  none')
    report.append('\nMETRICS:')
    report.extend('  '+str(k)+': '+str(v) for k,v in result['metrics'].items())
    (out/'QUALITY_CHECK.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')


def enforce_pilot_gate(path:Path):
    result=check_results(path)
    if path.is_dir(): save_quality(path,result)
    if result['status']!='PASS':
        raise SystemExit(f'PILOT QA = {result["status"]}. Full blocked; inspect {path/"QUALITY_CHECK.txt"}. '
                         'Use --skip-pilot-gate only after manual scientific review.')
    print('[PASS] Pilot integrity / scenario-differentiation gate. Full authorized.',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('results/pilot'))
    p.add_argument('--strict',action='store_true')
    args=p.parse_args()
    result=check_results(args.out)
    if args.out.is_dir(): save_quality(args.out,result)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.strict and result['status']!='PASS':
        raise SystemExit(2)

if __name__=='__main__':main()
