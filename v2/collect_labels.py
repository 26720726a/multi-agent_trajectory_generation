#!/usr/bin/env python3
"""S5-3 pilot label collection: one row per (cluster state, candidate mode)."""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.generate import axis_from_config, buildable, grid
from planning.execute import WMConfig, interaction_clusters, run_wm_planner
from planning.worldmodel import WorldModel
from config import PHYSICS, PLANNER

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=20);a=p.parse_args()
 cfg=json.load(open('bench/configs/difficulty.json')); all_inst=list(grid(axis_from_config(cfg['axis'])))
 by={n:[x for x in all_inst if x.n_agents==n] for n in (2,3,4)}; q,r=divmod(a.limit,3)
 inst=[x for n,take in ((2,q+(r>0)),(3,q+(r>1)),(4,q)) for x in by[n][:take]]
 os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True); rows=[]
 for ii,x in enumerate(inst,1):
  prob=buildable(x); res=run_wm_planner(prob,WMConfig(seed=0)); wm=WorldModel(prob,seed=0)
  for ri,st in enumerate(res.states):
   cs=interaction_clusters(st.pos,st.phase,PHYSICS.interact_cluster_radius)
   modes=wm.sample_modes(max_modes=PLANNER.max_modes)
   for ci,c in enumerate(cs):
    if len(c)<2: continue
    # S5-3″: 전역 진행방향으로 R_interact 앞의 로컬 서브골. 이미 그 경계를
    # 지난 구성원은 이 상태를 수집하지 않는다.
    goals=[]; skip=False
    for i in c:
     d=prob.agents[i].goal-st.pos[i]; L=np.linalg.norm(d)
     if L <= PHYSICS.interact_cluster_radius: skip=True; break
     goals.append(st.pos[i]+d/L*PHYSICS.interact_cluster_radius)
    if skip: continue
    # S5-3′: one cached solo baseline per cluster state.  For each member,
    # freeze every other agent at its original position and ask when this agent
    # leaves every other member's original 3m neighbourhood.  The cluster
    # baseline is the slowest member's time, matching the joint all-member exit.
    solo=[]
    for i in c:
     ss=st.copy(); ss.phase[:]=3; ss.phase[i]=st.phase[i]
     rr=wm.rollout(ss,modes[0],horizon=PLANNER.horizon_steps(prob.dt))
     ti=PLANNER.horizon_s
     for tt in range(1,len(rr.traj.pos)):
      if np.linalg.norm(rr.traj.pos[tt,i]-goals[c.index(i)]) <= 0.3:
       ti=tt*prob.dt; break
     solo.append(ti)
    base=max(solo)
    for mi,m in enumerate(modes):
     ro=wm.rollout(st,m,horizon=PLANNER.horizon_steps(prob.dt)); exit_t=None
     for t in range(1,len(ro.traj.pos)):
      if all(np.linalg.norm(ro.traj.pos[t,i]-goals[c.index(i)]) <= 0.3 for i in c): exit_t=t*prob.dt;break
     y=int(exit_t is not None and not ro.stalled); et=exit_t if exit_t is not None else PLANNER.horizon_s
     center=st.pos[c].mean(0); agent=[]
     for i in c: agent.append([*(st.pos[i]-center),*st.vel[i],*np.eye(4)[int(st.phase[i])]])
     obst=[[(o.x0+o.x1)/2-center[0],(o.y0+o.y1)/2-center[1],(o.x1-o.x0)/2,(o.y1-o.y0)/2] for o in prob.world.obstacles[:8]]
     rows.append({'uid':x.uid,'planner_seed':0,'replan_idx':ri,'cluster_id':ci,'mode_idx':mi,'members':','.join(map(str,c)),'agent_tokens':json.dumps(agent),'obstacle_tokens':json.dumps(obst),'mode':m.label(prob),'k':len(c),'density':len(c)/(np.pi*PHYSICS.interact_cluster_radius**2),'y_exit':y,'dt_delay':et-base if y else -1.0,'solo_exit_s':base,'stalled':int(ro.stalled)})
  print(ii,x.uid,len(rows),flush=True)
 with open(a.out,'w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
