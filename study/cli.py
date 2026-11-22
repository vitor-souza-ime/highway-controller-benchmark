"""Command line entry point for the CPU-only benchmark."""
from __future__ import annotations
import argparse
from pathlib import Path
from .runner import Settings, run_study
from .report import create_report
from .quality import check_results, save_quality, enforce_pilot_gate

# Presets have non-overlapping calibration and evaluation seed ranges.
PRESETS = {
    'smoke':dict(calibration=1, test=1, seed=3101),     # 36 evaluation episodes
    'pilot':dict(calibration=5, test=5, seed=84001),   # 180 evaluation episodes
    'full':dict(calibration=8, test=20, seed=190001),  # 720 evaluation episodes
}

def main():
    parser = argparse.ArgumentParser(description='HighwayEnv vehicle safety benchmark v3 (CPU)')
    parser.add_argument('--mode',choices=PRESETS,default='smoke')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--calibration-episodes',type=int)
    parser.add_argument('--test-episodes',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--no-screenshots',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--skip-pilot-gate',action='store_true',help='Explicit override; use only after reviewing pilot.')
    args=parser.parse_args()
    out=args.out or Path('results')/args.mode
    if args.report_only:
        create_report(out)
        save_quality(out,check_results(out))
        return
    profile=PRESETS[args.mode]
    cal=args.calibration_episodes if args.calibration_episodes is not None else profile['calibration']
    num=args.test_episodes if args.test_episodes is not None else profile['test']
    seed=args.seed if args.seed is not None else profile['seed']
    if cal <1 or num<1:
        parser.error('calibration and test episode counts must be positive')
    if args.mode=='full' and not args.skip_pilot_gate:
        enforce_pilot_gate(Path('results/pilot'))
    run_study(Settings(args.mode,out,cal,num,seed,not args.no_screenshots))
    create_report(out)
    result=check_results(out)
    save_quality(out,result)
    print(f'\nDONE: {out.resolve()}',flush=True)
    print(f"QUALITY: {result['status']} | warnings={len(result['warnings'])} | errors={len(result['errors'])}",flush=True)
    print('Send results/pilot (ZIP/RAR) after pilot before running FULL.',flush=True)
    if result['status']=='FAIL':
        raise SystemExit('Quality validation FAILED. See QUALITY_CHECK.txt')

if __name__=='__main__':
    main()
