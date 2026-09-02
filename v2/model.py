#!/usr/bin/env python3
"""S6 — 2-헤드 후보 평가기 (계획서 §5.1).

원본 `Mahoi-WM/mahoi-wm/scripts/train_wm.py` 의 `ModeScorer` 를 가져와
분류 헤드의 출력 수만 바꿨다 (A3 는 3-클래스).  구조 제약은 그대로다:

  * **positional encoding 없음.** 토큰 종류(Agent/Obstacle/CLS) 임베딩만 둔다 —
    순서가 아니라 종류이므로 에이전트끼리는 모두 같은 값을 받는다.
  * **패딩 마스크.** k<8 인 빈 슬롯은 `src_key_padding_mask` 로 막고, 풀링은
    CLS 토큰에서 한다 (마스크된 토큰이 평균에 섞이지 않는다).
  * **전역 특성은 CLS 토큰으로 주입.**
  * **자기회귀 없음.** 단일 forward.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ModeScorer(nn.Module):
    def __init__(self, d_agent, d_obst, d_glob, n_cls=1, d_model=128, nhead=4,
                 layers=3, ff=256, dropout=0.1):
        super().__init__()
        self.a_in = nn.Linear(d_agent, d_model)
        self.o_in = nn.Linear(d_obst, d_model)
        self.g_in = nn.Linear(d_glob, d_model)
        self.kind = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.kind, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, n_cls))
        self.head_reg = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, 1))

    def forward(self, agent, agent_mask, obst, obst_mask, glob):
        B = agent.shape[0]
        a = self.a_in(agent) + self.kind[0]
        o = self.o_in(obst) + self.kind[1]
        g = self.g_in(glob).unsqueeze(1) + self.kind[2]            # CLS
        x = torch.cat([g, a, o], 1)
        cls_m = torch.ones(B, 1, dtype=torch.bool, device=agent.device)
        keep = torch.cat([cls_m, agent_mask, obst_mask], 1)
        h = self.enc(x, src_key_padding_mask=~keep)
        h = self.norm(h[:, 0])                                     # CLS 풀링
        return self.head_cls(h), self.head_reg(h).squeeze(-1)
