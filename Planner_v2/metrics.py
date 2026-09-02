"""metrics.py — SimResult/baseline 결과를 지표 표(§7)로 정리 + CSV 저장."""
import csv


def summarize(scenario_name, agents, result, baseline_result=None):
    row = {
        "scenario": scenario_name,
        "n_agents": len(agents),
        "success": result.success,
        "makespan_B": result.steps,
        "collisions_agent": result.collisions_agent,
        "collisions_obstacle": result.collisions_obstacle,
        "dependency_violations": result.dependency_violations,
        "deadlock": result.deadlock,
        "deadlock_agents": ";".join(str(a) for a in result.deadlock_agents),
        "unsafe_steps": result.unsafe_steps,
        "tie_break_events": result.tie_break_events,
        "total_path_length_B": round(sum(result.path_length.values()), 3),
        "yields_total": sum(result.yields_per_agent.values()),
        "yields_per_agent": ";".join(f"{k}:{v}" for k, v in result.yields_per_agent.items()),
    }
    if baseline_result is not None:
        row["makespan_baseline"] = baseline_result["makespan"]
        row["total_path_length_baseline"] = round(
            sum(baseline_result["per_agent_path_len"].values()), 3)
        if result.success and baseline_result["makespan"] > 0:
            row["makespan_ratio_B_over_baseline"] = round(
                result.steps / baseline_result["makespan"], 3)
    return row


def print_row(row):
    print(f"--- {row['scenario']} ---")
    for k, v in row.items():
        if k == "scenario":
            continue
        print(f"  {k:28s}: {v}")


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
