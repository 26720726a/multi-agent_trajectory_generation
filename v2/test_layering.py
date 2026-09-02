"""계층 방향과 상수 단일 출처를 구조로 못 박는다.

원본 MAHOI-WM 은 "안전이 계획 품질과 경쟁하지 않는다"를 docstring 으로만
주장했다.  여기서는 **import 방향**과 **리터럴 부재**를 테스트로 강제한다.

S1 시점에는 safety/ 와 planning/ 이 비어 있어 (a)(b)(c) 가 자명하게 통과한다.
그것이 목적이다 — S2 이관 때 첫 위반이 곧바로 빨간불로 나오게 하려고 미리
세워 둔 가드레일이다.
"""
from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _py_files(pkg: str):
    d = os.path.join(ROOT, pkg)
    if not os.path.isdir(d):
        return
    for base, _, files in os.walk(d):
        if "__pycache__" in base:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(base, f)


def _imports(path: str):
    """이 파일이 import 하는 최상위 패키지 이름들."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            if n.level:                      # 상대 import 는 같은 패키지 안이다
                continue
            if n.module:
                out.add(n.module.split(".")[0])
    return out


def test_safety_does_not_import_planning():
    """(a) 안전 계층은 계획 계층을 모른다.

    이 방향이 지켜져야 "안전은 계획 품질과 무관하게 성립한다"가 구조적 사실이
    된다.  반대 방향(planning -> safety)은 허용된다.
    """
    bad = [(f, sorted(_imports(f) & {"planning"})) for f in _py_files("safety")]
    bad = [(f, i) for f, i in bad if i]
    assert not bad, f"safety/ 가 planning/ 을 import 한다: {bad}"


def test_config_imports_neither_layer():
    """(b) config 는 어느 계층도 import 하지 않는다.

    config 가 상위를 참조하면 순환이 생기고, 상수가 코드에 끌려다니게 된다.
    """
    bad = []
    for f in _py_files("config"):
        hit = sorted(_imports(f) & {"safety", "planning", "bench", "learn", "viz"})
        if hit:
            bad.append((os.path.relpath(f, ROOT), hit))
    assert not bad, f"config/ 가 상위 계층을 import 한다: {bad}"


#: config 가 이름을 갖고 있는 개념들.  이 이름으로 **정의**되는 값은 리터럴이면
#: 안 된다 — config 를 참조해야 한다.  S0 §C-2 의 사고(v_max 가 world.py 에
#: 1.20, build_dataset.py 에 1.5)는 값이 아니라 *이름* 이 두 번 정의되어 났다.
CONFIG_CONCEPTS = frozenset("""
    v_max v_nominal a_max dt control_dt
    radius robot_radius safety safety_margin min_sep interact_radius map_size size
    dwell dwell_s
    horizon_s replan_s stall_window stall_patience max_modes keep_modes k_routes
    max_sim_s time_budget_s switch_margin_rel switch_margin_abs
    lookahead tau tau_obst obst_pad snap_tol dep_standoff soft_margin
    speed_levels angle_levels
    TAU TAU_OBST OBST_PAD SOFT_MARGIN SNAP_TOL LOOKAHEAD DEP_STANDOFF
    SPEED_LEVELS ANGLE_LEVELS W_TTC W_TTC_OBST W_TURN W_SIDE W_HARD
""".split())

#: config 밖에서 정의되면 안 되는 **값**.  이름을 바꿔 다시 선언하는 우회를 막는다.
FORBIDDEN_VALUES = (1.20, 0.30, 0.70, 2.6, 0.22)

#: 값은 같지만 개념이 다른 자리.  (파일, 이름, 값) -> 사유.
#: 면제는 **자리마다** 적는다 — 값 하나를 통째로 빼면 그 값의 진짜 재선언까지
#: 함께 통과해 버린다.
WAIVERS = {
    ("planning/worldmodel.py", "ALPHA_PRIVILEGED", 0.30):
        "책임 분담 비율(무차원).  robot_radius=0.30 과 값만 우연히 같다 — "
        "원본 worldmodel.py:46.",
}

#: (c1) 의 면제.  (파일, 이름) -> 사유.  이름만 같고 개념이 다른 자리다.
CONCEPT_WAIVERS = {
    ("planning/paths.py", "speed_levels"):
        "AgentTrack 의 타임스텝당 속도 단계 **개수**(정수 K).  "
        "CONTROLLER.speed_levels(v_max 에 곱하는 배율 튜플)와 이름만 같다 — "
        "원본 paths.py:190,248,263.",
}


def _definition_sites(tree):
    """상수가 *정의되는* 자리만 (이름, 식) 으로 내놓는다.

    S0 §C-2 가 잡아낸 사고는 "같은 개념이 다른 곳에서 **다시 선언**되어 값이
    어긋난 것"이었다.  그래서 보는 자리를 정의가 일어나는 세 곳으로 좁힌다.

      * 대입            `radius, margin = 0.30, 0.45`
      * 어노테이션 대입  `v_max: float = 1.20`      (dataclass 필드 기본값)
      * 인자 기본값      `def rollout(..., stall_window=45)`

    호출부의 키워드 인자(`AgentSpec("A", ..., dwell=1.4)`)는 **정의가 아니라
    인스턴스 데이터**라 보지 않는다.  시나리오마다 다른 dwell 은 시나리오
    정의의 일부이고 (S2 지시서), 그것을 config 로 빼면 오히려 틀린다.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = []
            for t in node.targets:
                names += [x.id for x in ast.walk(t) if isinstance(x, ast.Name)]
            for nm in (names or ["<assign>"]):
                yield nm, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            yield getattr(node.target, "id", "<annassign>"), node.value
        elif isinstance(node, ast.arguments):
            pos = node.posonlyargs + node.args
            for arg, d in zip(pos[len(pos) - len(node.defaults):], node.defaults):
                yield arg.arg, d
            for arg, d in zip(node.kwonlyargs, node.kw_defaults):
                if d is not None:
                    yield arg.arg, d


def _literals(expr):
    """식이 **직접** 정의하는 수치.  호출 안으로는 들어가지 않는다.

    `w = World(10.0, 10.0, obstacles=[Rect(2.6, ...)])` 의 숫자들은 World 라는
    방의 모양이지 물리 상수가 아니다.  반면 `radius, margin = 0.30, 0.45` 와
    `SPEED_LEVELS = (0.0, 0.35, 0.70, 1.0)` 는 정의다 — 그래서 튜플/리스트는
    한 겹 열어 보고 호출은 열지 않는다.
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)) \
            and not isinstance(expr.value, bool):
        yield expr, float(expr.value)
    elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        for node, v in _literals(expr.operand):
            yield node, -v
    elif isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        for e in expr.elts:
            yield from _literals(e)


def _sites(pkgs=("safety", "planning", "bench", "learn", "viz")):
    for pkg in pkgs:
        for f in _py_files(pkg):
            rel = os.path.relpath(f, ROOT)
            tree = ast.parse(open(f, encoding="utf-8").read())
            for name, expr in _definition_sites(tree):
                yield rel, name, expr


def test_config_concepts_are_never_defined_as_literals():
    """(c1) config 가 이름을 가진 개념은 코드 안에서 리터럴로 정의되지 않는다.

    `v_max: float = 1.20` 은 걸리고 `v_max: float = PHYSICS.v_max` 는 통과한다.
    값이 우연히 같은지가 아니라 **누가 그 개념을 정의하는가**를 본다.
    """
    bad = []
    for rel, name, expr in _sites():
        if name not in CONFIG_CONCEPTS or (rel, name) in CONCEPT_WAIVERS:
            continue
        for node, val in _literals(expr):
            bad.append(f"{rel}:{node.lineno} {name} = {val}")
    assert not bad, ("config 가 정의하는 개념을 코드가 다시 정의한다 "
                     "(config 참조로 바꿀 것):\n  " + "\n  ".join(bad))


def test_physics_literals_live_only_in_config():
    """(c2) 물리 상수의 **값** 이 config 밖에서 정의되지 않는다.

    (c1) 을 이름을 바꿔 우회하는 것을 막는다.  개념이 정말 다르면 WAIVERS 에
    사유와 함께 등록한다.
    """
    bad = []
    for rel, name, expr in _sites():
        for node, val in _literals(expr):
            if val in FORBIDDEN_VALUES and (rel, name, val) not in WAIVERS:
                bad.append(f"{rel}:{node.lineno} {name} = {val}")
    assert not bad, ("config 밖에서 물리 상수를 정의한다 (config 를 참조하거나, "
                     "개념이 다르다면 WAIVERS 에 사유와 함께 등록할 것):\n  "
                     + "\n  ".join(bad))


def test_every_waiver_still_applies():
    """면제가 유령이 되지 않게 — 등록된 자리가 실제로 남아 있는지 확인한다."""
    live = {(rel, name, val)
            for rel, name, expr in _sites()
            for _, val in _literals(expr)}
    stale = [k for k in WAIVERS if k not in live]
    assert not stale, f"WAIVERS 에 남았지만 코드에 없는 자리: {stale}"

    named = {(rel, name) for rel, name, _ in _sites()}
    stale = [k for k in CONCEPT_WAIVERS if k not in named]
    assert not stale, f"CONCEPT_WAIVERS 에 남았지만 코드에 없는 자리: {stale}"


def test_config_exposes_the_three_singletons():
    """로더가 세 상수 묶음을 실제로 내준다."""
    from config import CONTROLLER, PHYSICS, PLANNER
    assert PHYSICS.v_max > 0 and PHYSICS.dt > 0
    assert PLANNER.replan_s > 0
    assert CONTROLLER.n_candidates > 0


def test_frozen_dataclasses_are_immutable():
    """상수를 런타임에 바꿔치기할 수 없어야 단일 출처가 의미를 갖는다."""
    import dataclasses
    import pytest
    from config import CONTROLLER, PHYSICS, PLANNER
    for obj in (PHYSICS, PLANNER, CONTROLLER):
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.__setattr__("v_max", 99.0)
