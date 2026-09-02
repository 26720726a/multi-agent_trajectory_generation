"""S6-3 D1 — 모델이 실제로 보는 후보 특성이 그룹 안에서 충돌하는지 센다.

토큰이 후보를 구별하는 데 쓰는 칸은 learn/data.py 기준 다음이 전부다.
  * 에이전트 슬롯 10:14  route 원핫 4 = min(routes[i], 3)      (클러스터 구성원만)
  * 에이전트 슬롯 14     yield rank[i] / (n-1)                 (클러스터 구성원만)
  * CLS 2:4              cautious, split_side
같은 그룹 안에서 이 튜플이 같은 두 후보는 **모델 입력이 완전히 같다**.
"""
import csv, json, sys, itertools
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.')
from learn.data import parse_mode, N_ROUTE

CSV = 'results/labels/main_progress_dedup.csv'
OUT = 'results/s6_3'

rows = list(csv.DictReader(open(CSV, newline='', encoding='utf8')))
print(f'rows {len(rows):,}')

groups = defaultdict(list)
for r in rows:
    groups[(r['uid'], int(r['replan_idx']), int(r['cluster_id']))].append(r)

max_route = 0
stats = {s: dict(n_groups=0, n_groups_with_collision=0, n_pairs_total=0,
                 n_collision_pairs=0, n_collision_diff=0,
                 n_collision_pairs_canon=0, n_groups_canon_collides=0,
                 n_groups_canon_present=0, k_lt_n_groups=0,
                 route_outside_differs=0, rank_outside_differs=0,
                 both_outside_same=0)
         for s in ('train', 'val', 'test')}
absdiff = {s: [] for s in stats}
absdiff_all = {s: [] for s in stats}
examples = []

for gk, rs in groups.items():
    sp = rs[0]['split']
    st = stats[sp]
    st['n_groups'] += 1
    n = int(rs[0]['n_agents'])
    mem = [int(v) for v in rs[0]['members'].split(',')]
    if len(mem) < n:
        st['k_lt_n_groups'] += 1
    feats, meta = [], []
    for r in rs:
        routes, rank, cautious, splt = parse_mode(r['mode'], n)
        max_route = max(max_route, max(routes))
        key = (tuple((min(routes[i], N_ROUTE - 1), rank[i]) for i in mem),
               cautious, splt)
        feats.append(key)
        meta.append(dict(mode_idx=int(r['mode_idx']), mode=r['mode'],
                         loss=float(r['loss_max']), routes=routes, rank=rank))
    m = len(rs)
    st['n_pairs_total'] += m * (m - 1) // 2
    canon = [i for i, x in enumerate(meta) if x['mode_idx'] == 0]
    if canon:
        st['n_groups_canon_present'] += 1
    hit = False
    canon_hit = False
    for i, j in itertools.combinations(range(m), 2):
        if feats[i] != feats[j]:
            continue
        hit = True
        st['n_collision_pairs'] += 1
        a, b = meta[i], meta[j]
        d = abs(a['loss'] - b['loss'])
        absdiff_all[sp].append(d)
        if a['mode_idx'] == 0 or b['mode_idx'] == 0:
            st['n_collision_pairs_canon'] += 1
            canon_hit = True
        # 충돌 원인 분해: 클러스터 밖 route 가 다른가 / 밖 rank 가 다른가
        out = [x for x in range(n) if x not in mem]
        ro = any(a['routes'][x] != b['routes'][x] for x in out)
        rk = any(a['rank'][x] != b['rank'][x] for x in out)
        if ro:
            st['route_outside_differs'] += 1
        if rk:
            st['rank_outside_differs'] += 1
        if not ro and not rk:
            st['both_outside_same'] += 1
        if d > 0:
            st['n_collision_diff'] += 1
            absdiff[sp].append(d)
            examples.append(dict(split=sp, uid=gk[0], replan_idx=gk[1],
                                 cluster_id=gk[2], members=mem, n_agents=n,
                                 mode_a=a['mode'], mode_b=b['mode'],
                                 mode_idx_a=a['mode_idx'], mode_idx_b=b['mode_idx'],
                                 loss_a=a['loss'], loss_b=b['loss'], abs_diff=d))
    if hit:
        st['n_groups_with_collision'] += 1
    if canon_hit:
        st['n_groups_canon_collides'] += 1

print(f'mode 문자열의 최대 route 인덱스 = {max_route} (토큰 원핫 칸 {N_ROUTE})')
out_all = {}
for sp, st in stats.items():
    d = np.array(absdiff[sp]) if absdiff[sp] else np.array([0.0])
    da = np.array(absdiff_all[sp]) if absdiff_all[sp] else np.array([0.0])
    e = dict(st)
    e['frac_groups_with_collision'] = st['n_groups_with_collision'] / max(1, st['n_groups'])
    e['collision_pairs_per_group'] = st['n_collision_pairs'] / max(1, st['n_groups'])
    e['frac_collision_pairs_of_all_pairs'] = st['n_collision_pairs'] / max(1, st['n_pairs_total'])
    e['frac_collision_pairs_with_diff_loss'] = st['n_collision_diff'] / max(1, st['n_collision_pairs'])
    e['absdiff_all_pairs_median'] = float(np.median(da))
    e['absdiff_all_pairs_p90'] = float(np.percentile(da, 90))
    e['absdiff_diff_only_median'] = float(np.median(d))
    e['absdiff_diff_only_p90'] = float(np.percentile(d, 90))
    e['absdiff_diff_only_max'] = float(d.max())
    e['frac_groups_k_lt_n'] = st['k_lt_n_groups'] / max(1, st['n_groups'])
    e['max_route_index'] = int(max_route)
    out_all[sp] = e
    json.dump(e, open(f'{OUT}/collision_stats_{sp}.json', 'w'), indent=1)

examples.sort(key=lambda x: -x['abs_diff'])
json.dump(examples[:20], open(f'{OUT}/collision_examples.json', 'w'), indent=1)
print(f'\n충돌 쌍 중 loss 가 다른 예시 총 {len(examples):,}건 (상위 20건 저장)')

hdr = f"{'':44s}" + ''.join(f'{s:>12s}' for s in ('train', 'val', 'test'))
print(hdr)
def row(lbl, key, fmt='{:.4f}'):
    print(f'{lbl:44s}' + ''.join(fmt.format(out_all[s][key]).rjust(12) for s in ('train','val','test')))
row('그룹 수', 'n_groups', '{:,.0f}')
row('클러스터가 전체 부분집합인 그룹 비율', 'frac_groups_k_lt_n')
row('충돌 쌍이 1개 이상인 그룹 비율', 'frac_groups_with_collision')
row('그룹당 평균 충돌 쌍 수', 'collision_pairs_per_group', '{:.3f}')
row('전체 쌍 대비 충돌 쌍 비율', 'frac_collision_pairs_of_all_pairs')
row('충돌 쌍 중 loss_max 가 다른 비율', 'frac_collision_pairs_with_diff_loss')
row('충돌 쌍 |Δloss| 중앙값 (전체 충돌쌍)', 'absdiff_all_pairs_median', '{:.3f}')
row('충돌 쌍 |Δloss| p90 (전체 충돌쌍)', 'absdiff_all_pairs_p90', '{:.3f}')
row('  (다른 것만) 중앙값', 'absdiff_diff_only_median', '{:.3f}')
row('  (다른 것만) p90', 'absdiff_diff_only_p90', '{:.3f}')
row('  (다른 것만) 최대', 'absdiff_diff_only_max', '{:.3f}')
row('canonical 이 낀 충돌 쌍 수', 'n_collision_pairs_canon', '{:,.0f}')
row('canonical 이 충돌하는 그룹 수', 'n_groups_canon_collides', '{:,.0f}')
row('충돌 원인: 클러스터 밖 route 가 다름', 'route_outside_differs', '{:,.0f}')
row('충돌 원인: 클러스터 밖 rank 가 다름', 'rank_outside_differs', '{:,.0f}')
row('충돌 원인: 밖도 전부 같음', 'both_outside_same', '{:,.0f}')
