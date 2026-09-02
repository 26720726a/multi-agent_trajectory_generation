"""main.py — 설계 문서 A 전체 파이프라인 실행

Stage 1 : scene → inflate → A* → shortcut/smooth → arc-length Path   (B와 공유)
Stage 2 : 우선순위 정렬 → space-time 속도 스케줄링 (A 고유)
검증    : 충돌 0 확인 + makespan + independent baseline 과 비교 그림/GIF
"""
import numpy as np
from scene import default_scene
from path import build_path
from planner import Agent, Params, plan_all
import sim
import viz


def main():
    prm = Params(v_max=1.0, dt=0.25, radius=0.28, delta=0.06)
    scene = default_scene()
    inflate = prm.radius + 0.10
    free = scene.build_grid(inflate)

    agents = [
        # S1 스타일 '교차' 인스턴스 — A(속도만 조정)가 풀 수 있는 형태.
        # 주의: 참고 그림의 원래 과제쌍(blue 하단→우상단)은 두 경로가
        #   우측 단일 차선 corridor 를 '반대 방향'으로 공유하게 되어,
        #   기하 고정 + 속도 조절만으로는 구조적으로 해소 불가
        #   (= 문서 §실패 모드의 실례).  그래서 blue 를 서→동 횡단으로
        #   바꿔 red 의 하강 차선과 '가로질러' 교차하게 했다.
        Agent(0, "red",  "red",  start=(8.9, 9.15), goal=(8.9, 1.1)),
        Agent(1, "blue", "blue", start=(2.6, 2.4),  goal=(9.4, 2.4)),
    ]

    # ---- Stage 1: 기하 경로 확정 ----
    for a in agents:
        a.path = build_path(scene, free, a.start, a.goal, inflate)
        print(f"[stage1] {a.name}: path length = {a.path.length:.2f} m")

    # ---- baseline: independent (협동 없음) ----
    P_ind, last_ind = sim.independent_replay(agents, prm)
    chk_ind = sim.verify(P_ind, prm, scene)
    t_close = chk_ind["t_min_sep"]
    print(f"[independent] min separation = {chk_ind['min_sep']:.2f} m "
          f"(t={t_close*prm.dt:.1f}s), agent-agent collisions = "
          f"{chk_ind['agent_agent']}")

    # ---- Stage 2: A 순차 계획 ----
    plan_all(agents, prm)
    P_a, last_a = sim.replay(agents, prm)
    chk_a = sim.verify(P_a, prm, scene)
    m = sim.metrics(agents, P_a, last_a, prm, chk_a)

    print("[A] makespan            =", f"{m['makespan']:.2f} s")
    for name, tt in m["per_agent_time"].items():
        print(f"[A] {name} arrival      = {tt:.2f} s")
    print("[A] agent-agent 충돌    =", m["collisions_agent"])
    print("[A] agent-obstacle 충돌 =", m["collisions_obstacle"])
    print("[A] min separation      =", f"{m['min_separation']:.2f} m "
          f"(요구 ≥ {prm.min_sep:.2f})")
    assert m["collisions_agent"] == 0 and m["collisions_obstacle"] == 0

    # ---- 시각화 ----
    viz.snapshot_figure(scene, agents, P_ind, P_a,
                        t_ind=t_close, t_a=chk_a["t_min_sep"], prm=prm,
                        out_png="method_a_result.png",
                        makespan_a=m["makespan"],
                        ind_collides=chk_ind["agent_agent"] > 0)
    viz.animate(scene, agents, P_ind, P_a, prm, "method_a_anim.gif", stride=1)
    print("saved: method_a_result.png / method_a_anim.gif")


if __name__ == "__main__":
    main()
