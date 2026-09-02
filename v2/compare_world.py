#!/usr/bin/env python3
"""묶음 3 대조: 원본 world/paths 와 v2 planning/world, planning/paths.

config 참조로 바꾼 기본값이 정말 같은 값인지, 그리고 생성된 문제/경로가
비트 단위로 같은지 본다.
"""
import sys, numpy as np
ORIG = "/home/kjs/Desktop/kuaicv/cvpr2027/Mahoi-WM/mahoi-wm"
V2 = "/home/kjs/Desktop/kuaicv/cvpr2027/MAHOI-WM_v2"


def load(path, name):
    import importlib, types
    saved = dict(sys.modules)
    sys.path.insert(0, path)
    for m in list(sys.modules):
        if m.split(".")[0] in ("mahoi", "planning", "safety", "config"):
            del sys.modules[m]
    mod = importlib.import_module(name)
    sys.path.pop(0)
    return mod


ow = load(ORIG, "mahoi.world"); op = load(ORIG, "mahoi.paths")
vw = load(V2, "planning.world"); vp = load(V2, "planning.paths")

bad = 0
# 1) 기본값 대조
for f in ("dwell", "radius", "v_max", "interact_radius"):
    a = getattr(ow.AgentSpec.__dataclass_fields__[f], "default")
    b = getattr(vw.AgentSpec.__dataclass_fields__[f], "default")
    print(f"  AgentSpec.{f:<16} orig={a!r:<8} v2={b!r:<8} {'OK' if a == b else 'DIFF'}")
    bad += a != b
for f in ("gap", "safety", "dt"):
    a = ow.Problem.__dataclass_fields__[f].default
    b = vw.Problem.__dataclass_fields__[f].default
    print(f"  Problem.{f:<18} orig={a!r:<8} v2={b!r:<8} {'OK' if a == b else 'DIFF'}")
    bad += a != b

# 2) 고정 시나리오 5종
for name in ow.SCENARIOS:
    a, b = ow.get_scenario(name), vw.get_scenario(name)
    same = (a.name == b.name and a.dt == b.dt and a.safety == b.safety
            and a.deps == b.deps and a.gap == b.gap
            and [(x.name, x.start, x.waypoint, x.goal, x.dwell, x.radius,
                  x.v_max, x.interact_radius) for x in a.agents] ==
                [(x.name, x.start, x.waypoint, x.goal, x.dwell, x.radius,
                  x.v_max, x.interact_radius) for x in b.agents]
            and [(r.x0, r.y0, r.x1, r.y1) for r in a.world.obstacles] ==
                [(r.x0, r.y0, r.x1, r.y1) for r in b.world.obstacles])
    print(f"  scenario {name:<11} {'OK' if same else 'DIFF'}")
    bad += (not same)

# 3) random_problem + build_prior  (100 시드)
tr_bad = 0
for seed in range(100):
    n = 2 + seed % 3
    a = ow.random_problem(seed, n_agents=n)
    b = vw.random_problem(seed, n_agents=n)
    if [(x.start, x.waypoint, x.goal, x.dwell, x.radius, x.v_max) for x in a.agents] != \
       [(x.start, x.waypoint, x.goal, x.dwell, x.radius, x.v_max) for x in b.agents] or \
       [(r.x0, r.y0, r.x1, r.y1) for r in a.world.obstacles] != \
       [(r.x0, r.y0, r.x1, r.y1) for r in b.world.obstacles] or a.deps != b.deps:
        bad += 1
        print(f"  random_problem seed={seed} DIFF")
    if seed < 20:
        ta, _ = op.build_prior(a)
        tb, _ = vp.build_prior(b)
        for x, y in zip(ta, tb):
            if not (np.array_equal(x.pts, y.pts) and x.wp_start == y.wp_start
                    and x.wp_end == y.wp_end and x.length == y.length):
                tr_bad += 1
print(f"  random_problem x100 + build_prior x20 : track mismatches={tr_bad}")
bad += tr_bad
print("RESULT:", "IDENTICAL" if bad == 0 else f"DIFFERENT ({bad})")
sys.exit(0 if bad == 0 else 1)
