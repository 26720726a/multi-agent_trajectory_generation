#!/usr/bin/env python3
"""S6-3 D2 — 용량을 낮춘 대조 모델 1개 (d_model=32, 1층).

    python3 scripts/diag_smallcap.py

**learn/ 를 수정하지 않는다** (S6-3 절대 규칙).  `learn.model.ModeScorer` 의
기본 용량만 이 프로세스 안에서 바꿔 끼우고, 학습·평가는 S6-2 와 **같은
learn/train.py · learn/evaluate.py 를 그대로** 호출한다.  손실·데이터·split·
시드·lr·batch·patience 는 S6-2 의 ranking 실행과 동일하다.

목적은 순위나 게이트가 아니라 **train↔val 격차가 좁혀지는 방향인지**만 보는
것이다 (여러 크기를 스윕하지 않는다).
"""
from __future__ import annotations
import functools, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import learn.common as C
import learn.model as M
import learn.train as T
import learn.evaluate as E

D_MODEL, LAYERS = 32, 1
SMALL = functools.partial(M.ModeScorer, d_model=D_MODEL, layers=LAYERS)


#: 체크포인트가 results/s6_3 에 있으므로 탐색 경로만 앞에 하나 더 붙인다.
E.ckpt_path = functools.partial(C.ckpt_path,
                                dirs=('results/s6_3', 'results/s6_2', 'results/s6'))


def run(mod, argv):
    saved_scorer, saved_argv = mod.ModeScorer, sys.argv
    mod.ModeScorer = SMALL
    sys.argv = argv
    try:
        mod.main()
    finally:
        mod.ModeScorer, sys.argv = saved_scorer, saved_argv


if __name__ == '__main__':
    print(f'용량 축소: d_model {D_MODEL}, 층 {LAYERS} (그 외 S6-2 ranking 과 동일)')
    run(T, ['train.py', '--variant', 'A3', '--lam-rank', '1.0',
            '--tag', 'smallcap', '--out', 'results/s6_3'])
    for split in ('train', 'val', 'test'):
        run(E, ['evaluate.py', '--tag', 'smallcap', '--split', split,
                '--out-json', f'results/s6_3/eval_smallcap_{split}.json'])
