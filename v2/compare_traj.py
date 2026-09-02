#!/usr/bin/env python3
"""묶음 4 대조: 원본 executor 와 v2 executor 의 궤적을 비트 단위로 비교한다.

두 저장소를 한 프로세스 안에서 번갈아 import 하면 모듈 캐시가 섞이므로,
각각을 **별도 프로세스**에서 돌려 .npz 로 떨어뜨린 뒤 비교한다.
"""
import os, subprocess, sys, tempfile
import numpy as np

ORIG = "/home/kjs/Desktop/kuaicv/cvpr2027/Mahoi-WM/mahoi-wm"
V2 = "/home/kjs/Desktop/kuaicv/cvpr2027/MAHOI-WM_v2"
PY = os.path.join(ORIG, ".venv/bin/python")

RUN = '''
import sys, numpy as np
root, which, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
if which == "orig":
    from mahoi.world import SCENARIOS, get_scenario
    from mahoi.wm.execute import WMConfig, run_wm_planner
else:
    from planning.world import SCENARIOS, get_scenario
    from planning.execute import WMConfig, run_wm_planner
d = {}
for name in SCENARIOS:
    for seed in range(3):
        res = run_wm_planner(get_scenario(name), WMConfig(seed=seed))
        d[f"{name}.{seed}.pos"] = res.traj.pos
        d[f"{name}.{seed}.vel"] = res.traj.vel
        d[f"{name}.{seed}.done"] = res.traj.done
        d[f"{name}.{seed}.wp_in"] = res.traj.wp_in
        d[f"{name}.{seed}.wp_out"] = res.traj.wp_out
        d[f"{name}.{seed}.tt"] = np.array([res.traj.team_time])
        d[f"{name}.{seed}.feas"] = np.array([res.traj.feasible])
np.savez(out, **d)
print(which, "done", len(d) // 7, "runs")
'''

tmp = tempfile.mkdtemp()
script = os.path.join(tmp, "run_one.py")
open(script, "w").write(RUN)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")
for root, which in ((ORIG, "orig"), (V2, "v2")):
    r = subprocess.run([PY, script, root, which, os.path.join(tmp, which + ".npz")],
                       env=env, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-2000:])
    r.check_returncode()

a = np.load(os.path.join(tmp, "orig.npz"))
b = np.load(os.path.join(tmp, "v2.npz"))
assert set(a.files) == set(b.files)
bad = [k for k in a.files if not np.array_equal(a[k], b[k])]
for k in sorted({k.rsplit(".", 1)[0] for k in a.files}):
    tt_a, tt_b = float(a[k + ".tt"][0]), float(b[k + ".tt"][0])
    keys = [x for x in a.files if x.startswith(k + ".")]
    ok = all(np.array_equal(a[x], b[x]) for x in keys)
    print(f"  {k:<14} team_time orig={tt_a:6.2f}  v2={tt_b:6.2f}  "
          f"|d|={abs(tt_a-tt_b):.1e}  arrays={'IDENTICAL' if ok else 'DIFF'}")
print(f"\nmismatched arrays: {len(bad)}")
print("RESULT:", "IDENTICAL" if not bad else "DIFFERENT " + str(bad[:5]))
sys.exit(0 if not bad else 1)
