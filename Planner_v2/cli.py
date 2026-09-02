"""cli.py — scenario.json 기반 실행기: --validate / --preview / --run.

    python cli.py --scenario scenarios/S3.json --validate
    python cli.py --scenario scenarios/S3.json --preview
    python cli.py --scenario scenarios/S3.json --run

JSON 만 고쳐서 "장애물 수정 / 에이전트 추가 / 경로 변경"을 코드 수정 없이
하는 게 목적이다. validate → preview → run 순서로 확인하며 반복하면 된다.
"""
import argparse
import os
import sys

import config
import sim
import metrics
import viz


def main():
    ap = argparse.ArgumentParser(description="scenario.json 기반 실행기 (방법 B)")
    ap.add_argument("--scenario", required=True, help="시나리오 JSON 경로")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true", help="로드+검증만 하고 끝")
    mode.add_argument("--preview", action="store_true", help="정적 미리보기 PNG만 저장")
    mode.add_argument("--run", action="store_true", help="시뮬레이션 실행 + gif/지표 저장")
    ap.add_argument("--out", default=None,
                     help="출력 디렉터리 (기본: 시나리오 파일 옆의 out/)")
    args = ap.parse_args()

    try:
        scene, agents, prm = config.load_scenario(args.scenario)
    except config.ScenarioError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    name = os.path.splitext(os.path.basename(args.scenario))[0]
    base_dir = os.path.dirname(os.path.abspath(args.scenario))
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(base_dir, "..", "out")
    out_dir = os.path.abspath(out_dir)

    n_real_obstacles = max(0, len(scene.obstacles) - 4)   # add_boundary_walls() 가 넣은 4개 제외

    if args.validate:
        print(f"[validate] OK — {args.scenario}")
        print(f"  grid: {scene.width:.2f} x {scene.height:.2f} m, "
              f"obstacles: {n_real_obstacles}, agents: {len(agents)}")
        for a in agents:
            deps = [(h.predecessor_id, h.wp_name) for h in a.hold]
            ann = [x.wp_name for x in a.announce]
            print(f"  agent id={a.id:<3} name={a.name:<10} radius={a.radius:.2f} "
                  f"v_max={a.v_max:.2f} path_len={a.path.length:.2f}m "
                  f"hold={deps} announce={ann}")
        return

    os.makedirs(out_dir, exist_ok=True)

    if args.preview:
        out_png = os.path.join(out_dir, f"{name}_preview.png")
        viz.preview(scene, agents, out_png, title=name)
        print(f"[preview] saved: {out_png}")
        return

    # --run
    result = sim.run(scene, agents, dt=prm.dt, max_steps=prm.max_steps, tau=prm.tau,
                      delta=prm.delta, tie_break=prm.tie_break, w_col=prm.w_col,
                      w_dev=prm.w_dev, n_dir=prm.n_dir, n_speed=prm.n_speed,
                      progress_window=prm.progress_window, progress_eps=prm.progress_eps,
                      verbose=True)
    row = metrics.summarize(name, agents, result)
    metrics.print_row(row)

    gif_path = os.path.join(out_dir, f"{name}.gif")
    note = f"{name}  tie_break={prm.tie_break}  success={result.success}"
    viz.animate(scene, agents, result.history, prm.dt, gif_path, title=name,
                stride=max(1, int(0.2 / prm.dt)), extra_note=note)
    print(f"[run] saved: {gif_path}")

    csv_path = os.path.join(out_dir, f"{name}_metrics.csv")
    metrics.write_csv([row], csv_path)
    print(f"[run] saved: {csv_path}")


if __name__ == "__main__":
    main()
