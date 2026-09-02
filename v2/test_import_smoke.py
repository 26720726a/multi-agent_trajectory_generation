"""이관된 모듈이 import 되고, 리프 모듈의 순수 함수가 원본과 같은 값을 내는가.

원본에는 `geometry` / `traj` / `validate` 에 대응하는 테스트가 없었다 (S0 §G).
그래서 묶음 1 에서는 "옮겨 붙는가"만 확인한다 — 값 대조는 묶음 2 의
`test_vo_extraction.py` 와 묶음 6 의 회귀 대조가 맡는다.
"""
from __future__ import annotations

import numpy as np


def test_safety_geometry_imports():
    from safety.geometry import (Rect, point_rect_dist, polyline_length,
                                 resample_polyline, round_corners,
                                 seg_hits_rect, seg_seg_min_dist)
    r = Rect(1.0, 1.0, 2.0, 2.0)
    assert r.inflate(0.5) == Rect(0.5, 0.5, 2.5, 2.5)
    assert seg_hits_rect((0.0, 1.5), (3.0, 1.5), r)
    assert not seg_hits_rect((0.0, 0.0), (3.0, 0.0), r)
    assert point_rect_dist(np.array([[0.0, 1.5]]), r)[0] == 1.0
    assert seg_seg_min_dist(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                            np.array([0.0, 1.0]), np.array([1.0, 1.0])) == 1.0


def test_planning_traj_imports():
    from planning.traj import Trajectory, concat
    n, T = 2, 5
    tr = Trajectory(method="t", pos=np.zeros((T + 1, n, 2)),
                    vel=np.zeros((T + 1, n, 2)), dt=0.1,
                    wp_in=np.full(n, -1), wp_out=np.full(n, -1),
                    done=np.full(n, -1))
    assert tr.n == n and tr.T == T
    assert tr.team_time == T * 0.1
    assert concat([tr, tr]).T == 2 * T


def test_safety_validate_imports():
    """`validate` 는 planning/ 을 **import 하지 않고** 로드되어야 한다."""
    import sys
    for mod in [m for m in sys.modules if m.startswith("planning")]:
        del sys.modules[mod]
    import safety.validate as SV
    assert callable(SV.validate)
    assert not any(m.startswith("planning") for m in sys.modules), \
        "safety.validate 를 import 하는 것만으로 planning 이 끌려 왔다"
