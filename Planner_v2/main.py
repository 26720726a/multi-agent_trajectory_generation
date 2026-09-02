"""main.py — 전체 파이프라인: S1~S4 실행, baseline 비교, 지표 CSV, 애니메이션 gif.

실행: python main.py
산출물: out/ 아래에 시나리오별 gif + metrics.csv
"""
import os
import copy

import scenarios
import sim
import baseline
import metrics
import viz

OUT = os.path.join(os.path.dirname(__file__), "out")


def _agents_deepcopy(build_fn, prm=None):
    """시나리오를 새로 빌드해 완전히 독립된 agent 인스턴스 세트를 얻는다
    (같은 scene/agents 객체를 sim 여러 번에 재사용하면 board 상태가 섞이므로)."""
    return build_fn(prm) if prm is not None else build_fn()


def run_scenario(name, build_fn, tie_break=True, run_baseline=True):
    scene, agents, prm = build_fn()
    print(f"\n===== {name} (tie_break={tie_break}) =====")
    for a in agents:
        print(f"  [plan] {a.name}: path length = {a.path.length:.2f} m, "
              f"hold={[ (h.predecessor_id,h.wp_name) for h in a.hold ]}, "
              f"announce={[a2.wp_name for a2 in a.announce]}")

    result = sim.run(scene, agents, dt=prm.dt, max_steps=prm.max_steps,
                      tau=prm.tau, delta=prm.delta, tie_break=tie_break,
                      w_col=prm.w_col, w_dev=prm.w_dev, n_dir=prm.n_dir,
                      n_speed=prm.n_speed, progress_window=prm.progress_window,
                      verbose=True)

    bl = None
    if run_baseline:
        scene_b, agents_b, prm_b = build_fn()
        bl = baseline.run(scene_b, agents_b, dt=prm_b.dt, max_steps=prm_b.max_steps)

    row = metrics.summarize(name, agents, result, bl)
    metrics.print_row(row)

    gif_path = os.path.join(OUT, f"{name}.gif")
    note = f"{name}  tie_break={tie_break}  success={result.success}  deadlock={result.deadlock}"
    viz.animate(scene, agents, result.history, prm.dt, gif_path, title=name,
                stride=max(1, int(0.2 / prm.dt)), extra_note=note)
    print(f"  saved: {gif_path}")

    if bl is not None:
        gif_b = os.path.join(OUT, f"{name}_baseline.gif")
        scene_b2, agents_b2, prm_b2 = build_fn()
        viz.animate(scene_b2, agents_b2, bl["history"], bl["dt"], gif_b,
                    title=f"{name} (baseline, 완전 직렬화)",
                    stride=max(1, int(0.4 / prm_b2.dt)),
                    extra_note=f"{name} baseline  makespan={bl['makespan']}")
        print(f"  saved: {gif_b}")

    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []

    rows.append(run_scenario("S1_crossing", scenarios.scenario_s1))
    rows.append(run_scenario("S2_dependency", scenarios.scenario_s2))
    rows.append(run_scenario("S3_chain", scenarios.scenario_s3))

    # S4: tie-break off → deadlock/livelock 노출, on → 해소.
    row_off = run_scenario("S4_corridor_TIEBREAK_OFF", scenarios.scenario_s4,
                            tie_break=False, run_baseline=False)
    row_on = run_scenario("S4_corridor_TIEBREAK_ON", scenarios.scenario_s4,
                           tie_break=True, run_baseline=True)
    rows.append(row_off)
    rows.append(row_on)

    csv_path = os.path.join(OUT, "metrics.csv")
    metrics.write_csv(rows, csv_path)
    print(f"\nsaved metrics: {csv_path}")

    print("\n===== 요약 =====")
    for r in rows:
        b = r.get("makespan_baseline", "-")
        ratio = r.get("makespan_ratio_B_over_baseline", "-")
        print(f"{r['scenario']:28s} success={r['success']!s:5s} "
              f"collisions(agent/obs)={r['collisions_agent']}/{r['collisions_obstacle']} "
              f"dep_violations={r['dependency_violations']} deadlock={r['deadlock']!s:5s} "
              f"makespan_B={r['makespan_B']:4d} makespan_baseline={b} ratio={ratio}")


if __name__ == "__main__":
    main()
