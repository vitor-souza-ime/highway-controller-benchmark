"""Export real simulation results into editable IEEE-ready SVG/PDF/PNG plots,
CSV/Markdown/LaTeX tables, paired descriptive confidence intervals, and text.
Never synthesize experiment results. No claim of statistical significance.
"""
from __future__ import annotations
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .controllers import METHODS, DISPLAY
from .scenarios import SCENARIOS

PALETTE = ['#27415B','#527A8A','#A47B52','#5C906B','#937DB0','#C17F8F']
LABELS = [DISPLAY[m] for m in METHODS]

def wilson_ci(x: int, n: int, z: float=1.95996398454):
    if n == 0:
        return np.nan, np.nan
    p=x/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    radius=z/den*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return max(0,mid-radius),min(1,mid+radius)


def paired_cluster_ci(df, method, reference='ttc_shield', reps=1500, seed=1784):
    """Paired difference in collision risk (percentage points) with seed bootstrap.

    A seed is resampled as a cluster across all 6 traffic settings. This is
    descriptive, not a significance declaration; pilot estimates are uncertain.
    """
    if method == reference:
        return 0., 0., 0.
    wide=df.pivot(index=['scenario','seed'],columns='method',values='collision')
    if method not in wide or reference not in wide:
        return float('nan'),float('nan'),float('nan')
    delta=(wide[method]-wide[reference]).dropna().reset_index(name='delta')
    if delta.empty:
        return float('nan'),float('nan'),float('nan')
    avg=float(delta.delta.mean())
    per_seed = {k:v.delta.to_numpy() for k,v in delta.groupby('seed')}
    rng=np.random.default_rng(seed)
    keys=list(per_seed)
    samples=np.empty(reps)
    for k in range(reps):
        drawn=rng.choice(keys,len(keys),replace=True)
        samples[k]=np.concatenate([per_seed[x] for x in drawn]).mean()
    lo,hi=np.quantile(samples,[.025,.975])
    return 100*avg,100*float(lo),100*float(hi)


def _style(ax,title,ylabel='',xlabel=''):
    ax.set_title(title,fontsize=10,pad=9)
    ax.set_ylabel(ylabel,fontsize=9)
    ax.set_xlabel(xlabel,fontsize=9)
    ax.tick_params(axis='both',labelsize=8)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)


def _export(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def _columns(ax, values, title, ylabel, percent=False, ylim=None):
    x=np.arange(len(METHODS))
    ax.bar(x,values,color=PALETTE,edgecolor='#333333',linewidth=.5,width=.72)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS,rotation=20,ha='right')
    _style(ax,title,ylabel)
    if ylim: ax.set_ylim(*ylim)
    if percent: ax.set_ylim(0,max(10,100 if max(values,default=0)>60 else max(values,default=0)*1.25))


def _table_files(out:Path, name:str, df:pd.DataFrame):
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/(name+'.csv'),index=False)
    (out/(name+'.md')).write_text(df.to_markdown(index=False)+'\n',encoding='utf-8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tex = df.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.3f}')
    (out/(name+'.tex')).write_text(tex,encoding='utf-8')


def create_report(out:Path):
    ep_file=out/'episodes.csv'
    d_file=out/'decisions.csv'
    if not ep_file.exists() or not d_file.exists():
        raise FileNotFoundError(f'Raw simulation CSV missing in {out}. Run experiment first.')
    ep=pd.read_csv(ep_file)
    step=pd.read_csv(d_file)
    figs=out/'figures'; tables=out/'tables'
    figs.mkdir(exist_ok=True,parents=True); tables.mkdir(exist_ok=True,parents=True)
    if not set(METHODS).issubset(set(ep.method)):
        raise ValueError('Missing evaluation controllers in episodes.csv')
    grouped=ep.groupby('method',sort=False)
    rows=[]
    for method in METHODS:
        df=grouped.get_group(method)
        n=len(df); ncrash=int(df.collision.sum())
        low,high=wilson_ci(ncrash,n)
        dsub=step[step.method==method]
        rows.append({
            'method':DISPLAY[method], 'n':n,'collisions':ncrash,
            'collision_pct':round(100*ncrash/n,2),
            'CI95_low_pct':round(100*low,2),'CI95_high_pct':round(100*high,2),
            'near_miss_exposure_pct':round(100*df.near_miss_step_rate.mean(),2),
            'mean_speed_mps':round(df.mean_speed_mps.mean(),2),
            'distance_per_sim_s':round(df.distance_per_sim_s.mean(),2),
            'mean_reward':round(df.cumulative_reward.mean(),2),
            'intervention_pct':round(100*df.intervention_rate.mean(),2),
            'latency_p50_ms':round(float(dsub.decision_latency_ms.quantile(.5)),4),
            'latency_p95_ms':round(float(dsub.decision_latency_ms.quantile(.95)),4),
        })
    summary=pd.DataFrame(rows)
    protocol=pd.DataFrame([{
        'scenario':s.name,'traffic':s.density_label,
        'behavior':s.behavior_label,'vehicles':s.vehicles_count,
        'density_parameter':s.vehicles_density,
        'replicates_per_method':int(ep.loc[(ep.scenario==s.name)&(ep.method==METHODS[0])].shape[0]),
    } for s in SCENARIOS])
    bycondition=(ep.groupby(['scenario','density','behavior','method'],as_index=False)
                 .agg(n=('collision','size'),collision_pct=('collision',lambda x:100*x.mean()),
                      exposure_pct=('near_miss_step_rate',lambda x:100*x.mean()),
                      speed_mps=('mean_speed_mps','mean')))
    bycondition['collision_pct']=bycondition.collision_pct.round(2)
    bycondition['exposure_pct']=bycondition.exposure_pct.round(2)
    bycondition['speed_mps']=bycondition.speed_mps.round(2)
    paired=[]
    for method in METHODS:
        delta,lo,hi=paired_cluster_ci(ep,method)
        paired.append({'method':DISPLAY[method], 'reference':'TTC',
                       'collision_diff_pp':round(delta,2),'CI95_low_pp':round(lo,2),
                       'CI95_high_pp':round(hi,2)})
    attribution=(step.groupby(['method','intervention_reason'],as_index=False)
                 .agg(steps=('step','size'),interventions=('intervention','sum')))
    attribution['method']=attribution.method.map(DISPLAY)
    latency=summary[['method','latency_p50_ms','latency_p95_ms','intervention_pct']].copy()
    init=(ep[ep.method=='baseline']
          .groupby(['density','behavior'],as_index=False)
          .agg(initial_nearby_mean=('initial_nearby_count','mean'),
               initial_front_gap_mean_m=('initial_front_gap_m','mean'),
               collision_pct=('collision',lambda x:100*x.mean())))
    compact=summary[['method','n','collision_pct','near_miss_exposure_pct','mean_speed_mps','latency_p95_ms']].copy()
    compact.columns=['Method','N','Crash (%)','Exposure (%)','Speed (m/s)','P95 (ms)']
    compact_ci=pd.DataFrame(paired).copy()
    compact_ci.columns=['Method','Reference','Delta crash (pp)','CI low (pp)','CI high (pp)']
    for name,tab in [
        ('table_01_protocol',protocol),('table_02_overall',summary),
        ('table_03_scenario_results',bycondition),('table_04_paired_vs_ttc',pd.DataFrame(paired)),
        ('table_05_latency',latency),('table_06_intervention_attribution',attribution),
        ('table_07_initial_conditions',init),
        ('table_08_ieee_compact',compact),('table_09_ieee_pairwise_ci',compact_ci),
    ]:
        _table_files(tables,name,tab)

    # Fig 1: collision rates with Wilson CI (highly uncertain in pilot)
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    x=np.arange(len(METHODS)); y=summary.collision_pct.to_numpy()
    low=y-summary.CI95_low_pct.to_numpy(); up=summary.CI95_high_pct.to_numpy()-y
    ax.bar(x,y,color=PALETTE,edgecolor='#444',linewidth=.5)
    ax.errorbar(x,y,yerr=np.vstack([np.maximum(0,low),np.maximum(0,up)]),
                fmt='none',ecolor='#222',capsize=3,linewidth=1)
    ax.set_xticks(x);ax.set_xticklabels(LABELS)
    _style(ax,'Ego collision rate (Wilson 95% CI)','Episodes with collision (%)')
    ax.set_ylim(0,min(100,max(30,float((y+up).max())*1.1)))
    _export(fig,figs,'fig_01_collision_wilson')
    # Fig 2: exposure to near-miss proxies
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    _columns(ax,summary.near_miss_exposure_pct.to_numpy(),
             'Mean per-episode near-miss proxy exposure','Share of time steps (%)')
    _export(fig,figs,'fig_02_risk_exposure')
    # Fig 3: mobility
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    _columns(ax,summary.mean_speed_mps.to_numpy(),
             'Mean ego velocity','Velocity (m/s)')
    _export(fig,figs,'fig_03_velocity')
    # Fig 4: per decision CPU latency
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    x=np.arange(len(METHODS));w=.36
    ax.bar(x-w/2,summary.latency_p50_ms.to_numpy(),w,label='P50',color='#6E8795')
    ax.bar(x+w/2,summary.latency_p95_ms.to_numpy(),w,label='P95',color='#30495E')
    ax.set_xticks(x);ax.set_xticklabels(LABELS)
    ax.legend(frameon=False,ncol=2,fontsize=8)
    _style(ax,'Controller + state extraction CPU latency','Time per decision (ms)')
    _export(fig,figs,'fig_04_cpu_latency')
    # Fig 5: interventions
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    _columns(ax,summary.intervention_pct.to_numpy(),
             'Safety overrides per controller','Overridden proposals (%)')
    _export(fig,figs,'fig_05_intervention_rate')
    # Fig 6: effect of traffic density (collision across both styles)
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    dlevels=['low','medium','high']; xs=np.arange(3)
    for i,method in enumerate(METHODS):
        yy=[100*ep[(ep.method==method)&(ep.density==d)].collision.mean() for d in dlevels]
        ax.plot(xs,yy,marker=['o','s','^','D','v','P'][i],label=DISPLAY[method],color=PALETTE[i])
    ax.set_xticks(xs);ax.set_xticklabels([d.title() for d in dlevels])
    ax.legend(ncol=3,frameon=False,fontsize=8)
    _style(ax,'Collision rate vs. traffic density','Collisions (%)','Configured density')
    ax.set_ylim(bottom=0)
    _export(fig,figs,'fig_06_collision_by_density')
    # Fig 7: traffic behavior comparison
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    xx=np.arange(len(METHODS));w=.36
    for k,behavior in enumerate(['IDM','aggressive']):
        rates=[100*ep[(ep.method==m)&(ep.behavior==behavior)].collision.mean() for m in METHODS]
        ax.bar(xx+(-w/2 if k==0 else w/2),rates,w,label=behavior,color=['#5C8392','#C28C6C'][k])
    ax.set_xticks(xx);ax.set_xticklabels(LABELS)
    ax.legend(frameon=False,fontsize=8)
    _style(ax,'Robustness to driving-style changes','Collisions (%)')
    _export(fig,figs,'fig_07_behavior_shift')
    # Fig 8: Pareto mobility versus safety
    fig,ax=plt.subplots(figsize=(7.6,4.6))
    for i,r in summary.iterrows():
        ax.scatter(r.collision_pct,r.mean_speed_mps,color=PALETTE[i],s=100,
                   edgecolor='#333',linewidth=.5,zorder=4)
        ax.annotate(r['method'],(r.collision_pct,r.mean_speed_mps),
                    xytext=(5,5),textcoords='offset points',fontsize=8)
    _style(ax,'Safety-mobility trade-off','Mean velocity (m/s)','Collision rate (%)')
    _export(fig,figs,'fig_08_safety_mobility')
    # Fig 9: ECDF of minimum TTC per episode
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    for i,method in enumerate(METHODS):
        vals=np.sort(ep.loc[ep.method==method,'min_ttc_s_cap15'].dropna().to_numpy(float))
        if len(vals):
            ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',
                    label=DISPLAY[method],color=PALETTE[i])
    ax.axvline(2.5,color='#666',linestyle='--',linewidth=.8,label='Proxy threshold')
    _style(ax,'Minimum closing TTC per episode','Empirical CDF','TTC (s; capped at 15)')
    ax.legend(fontsize=8,ncol=3,frameon=False)
    _export(fig,figs,'fig_09_ttc_ecdf')
    # Fig 10: collision heatmap
    matrix=np.array([[100*ep[(ep.scenario==s.name)&(ep.method==m)].collision.mean()
                      for m in METHODS] for s in SCENARIOS])
    fig,ax=plt.subplots(figsize=(8.4,4))
    im=ax.imshow(matrix,cmap='Greys',vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(np.arange(len(METHODS)),LABELS)
    ax.set_yticks(np.arange(len(SCENARIOS)),[s.name for s in SCENARIOS])
    for row in range(len(SCENARIOS)):
        for col in range(len(METHODS)):
            v=matrix[row,col]
            ax.text(col,row,f'{v:.0f}%',ha='center',va='center',
                    color='white' if v>55 else 'black',fontsize=8)
    fig.colorbar(im,ax=ax,label='Collision rate (%)',fraction=.04)
    ax.set_title('Scenario-controller collision matrix',fontsize=10)
    _export(fig,figs,'fig_10_collision_heatmap')
    # Fig 11: empirical initial conditions, by density (sanity indicator)
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    base=ep[ep.method=='baseline']
    datasets=[base[base.density==d].initial_nearby_count.to_numpy() for d in ['low','medium','high']]
    ax.boxplot(datasets,tick_labels=['Low','Medium','High'],showmeans=True)
    _style(ax,'Initial local traffic (scenario differentiation check)',
           'Vehicles within 60 m at episode reset','Density setting')
    _export(fig,figs,'fig_11_initial_traffic')
    # Fig 12: memory-specific and common interventions
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    series=[int(((step.method=='hybrid')&(step.intervention_reason==key)).sum())
            for key in ['ttc','memory_additive','ttc_memory']]
    ax.bar(['TTC only','Memory only','Both'],series,
           color=['#6E8795','#AE8B79','#627C6C'],edgecolor='#444')
    _style(ax,'Hybrid intervention attribution','Number of executed overrides')
    _export(fig,figs,'fig_12_hybrid_attribution')
    # 2 IEEE-friendly two-panel and four-panel composite figures
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.5))
    _columns(axs[0],summary.collision_pct.to_numpy(),' (a) Safety','Collision rate (%)')
    _columns(axs[1],summary.near_miss_exposure_pct.to_numpy(),'(b) Risk exposure','Steps (%)')
    fig.tight_layout()
    _export(fig,figs,'panel_A_safety')
    fig,axs=plt.subplots(1,2,figsize=(10.2,3.5))
    _columns(axs[0],summary.mean_speed_mps.to_numpy(),'(a) Mobility','Speed (m/s)')
    _columns(axs[1],summary.latency_p95_ms.to_numpy(),'(b) CPU cost','P95 decision (ms)')
    fig.tight_layout()
    _export(fig,figs,'panel_B_efficiency')

    best=summary.sort_values('collision_pct').iloc[0]
    lines=[
        'HIGHWAY-ENV SAFETY BENCHMARK V3 — RESULTS (NOT AN ARTICLE)',
        f'Simulated evaluation episodes: {len(ep)} | decisions: {len(step)}',
        f'Best observed collision rate: {best["method"]}, {best["collision_pct"]:.2f}%',
        '', 'METHOD SUMMARY:',summary.to_string(index=False),'',
        'PAIRED COLLISION RISK DIFFERENCES VS TTC (percentage points; 95% seed-bootstrap CI):',
        pd.DataFrame(paired).to_string(index=False),'',
        'Important: small pilot samples have broad uncertainty. No automatic significance claim.',
        'RSS* = approximate longitudinal RSS; CBF* = discrete one-step barrier heuristic.',
        'All measures use oracle simulator state, not real sensor measurements.',
        'The 6 scenarios are ONE highway map with 3 densities x 2 vehicle behaviors.',
        'Inspect QUALITY_CHECK.txt before using any result in a submission.',
    ]
    (out/'RESULTS_SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Snippet only, not a fabricated full paper.
    paper='% Generated from real execution. Add captions / labels in IEEE paper.\n'
    paper+='\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{figures/panel_A_safety.pdf}\n\\caption{Collision and risk-exposure comparison.}\n\\label{fig:safety}\n\\end{figure}\n'
    paper+='\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{figures/panel_B_efficiency.pdf}\n\\caption{Velocity and CPU decision latency.}\n\\label{fig:efficiency}\n\\end{figure}\n'
    (out/'paper_assets.tex').write_text(paper,encoding='utf-8')
    (out/'report_manifest.json').write_text(json.dumps({
        'figures_png':len(list(figs.glob('fig_*.png'))),
        'figures_pdf':len(list(figs.glob('fig_*.pdf'))),
        'panels_pdf':len(list(figs.glob('panel_*.pdf'))),
        'tables_tex':len(list(tables.glob('table_*.tex'))),
    },indent=2),encoding='utf-8')
    print(f'[REPORT] 12 scientific plots + 2 panels + 9 tables in {out}',flush=True)
    return {'summary':summary,'paired':pd.DataFrame(paired)}
