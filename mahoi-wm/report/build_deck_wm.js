// World Model + Planner deck (2026-08-19).
// Same crimson house style as build_deck.js (8/18 submission) so the two decks
// read as one series.  Run:  node report/build_deck_wm.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = path.join(__dirname, "..", "outputs", "wm");
const img = (f) => path.join(OUT, f);

// ---- crimson palette (identical to build_deck.js) -------------------------
const CRIM = "9E1B32";
const CRIM_D = "5E0C1B";
const INK = "1F2328";
const MUTED = "6E737B";
const CARD = "F4F5F6";
const TINT = "FAEEF0";
const LINE = "E2E4E7";
const ROW = "F8F8F9";
const WHITE = "FFFFFF";
const GOOD = "1A7F37";
const WARN = "B4690E";

const F = "Arial";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Multi-Agent HOI toy experiment";
pres.title = "World Model + Planner for Multi-Agent Trajectory Generation";

const W = 13.3, H = 7.5, M = 0.62;

// ---------------------------------------------------------------- helpers
function titled(s, kicker, title) {
  s.addText(kicker, {
    x: M, y: 0.34, w: 8, h: 0.28, fontFace: F, fontSize: 11, bold: true,
    color: CRIM, charSpacing: 1.4, margin: 0, valign: "top",
  });
  s.addText(title, {
    x: M, y: 0.64, w: W - 2 * M, h: 0.62, fontFace: F, fontSize: 27, bold: true,
    color: INK, margin: 0, valign: "top",
  });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill || CARD }, rectRadius: 0.05,
    line: { width: 0 },
  });
}

function badge(s, x, y, label, size) {
  const d = size || 0.30;
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d, fill: { color: CRIM }, line: { width: 0 },
  });
  s.addText(label, {
    x, y, w: d, h: d, fontFace: F, fontSize: d > 0.32 ? 12.5 : 9.5, bold: true,
    color: WHITE, align: "center", valign: "middle", margin: 0,
  });
}

function foot(s, txt) {
  s.addText(txt, {
    x: M, y: H - 0.44, w: W - 2 * M, h: 0.26, fontFace: F, fontSize: 9,
    color: MUTED, margin: 0, valign: "top",
  });
}

// Place an image inside a box, preserving its aspect ratio.
function fitImage(s, file, ar, box) {
  let w = box.w, h = w / ar;
  if (h > box.h) { h = box.h; w = h * ar; }
  s.addImage({ path: img(file), x: box.x + (box.w - w) / 2,
               y: box.y + (box.h - h) / 2, w, h });
}

function table(s, rows, opts) {
  const bodyStyle = opts.cell || (() => ({}));
  s.addTable(
    rows.map((r, i) => r.map((c, k) => ({
      text: c,
      options: i === 0
        ? { bold: true, color: WHITE, fill: { color: CRIM }, align: k ? "center" : "left" }
        : Object.assign(
            { color: INK, align: k ? "center" : "left", fill: { color: i % 2 ? ROW : WHITE } },
            bodyStyle(i, k, c)),
    }))),
    Object.assign({
      fontFace: F, fontSize: 11.5, rowH: 0.28, valign: "middle",
      border: { type: "solid", color: LINE, pt: 0.75 }, margin: [3, 8, 3, 8],
    }, opts));
}

// ================================================================ 1. title
{
  const s = pres.addSlide();
  s.background = { color: CRIM_D };
  s.addText("World Model + Planner", {
    x: M + 0.3, y: 2.05, w: 11.5, h: 0.8, fontFace: F, fontSize: 40, bold: true,
    color: WHITE, margin: 0, valign: "top",
  });
  s.addText("Multi-Agent Trajectory Generation — Toy Experiment", {
    x: M + 0.3, y: 2.95, w: 11.5, h: 0.5, fontFace: F, fontSize: 21,
    color: "F0C9D1", margin: 0, valign: "top",
  });
  s.addShape(pres.ShapeType.rect, {
    x: M + 0.3, y: 3.72, w: 1.5, h: 0.045, fill: { color: WHITE }, line: { width: 0 },
  });
  s.addText("여러 개의 future를 상상하고, 명시적 cost로 가장 좋은 하나를 고른다\n" +
            "경로를 고정하지 않고 temporal + spatial coordination을 함께 수행", {
    x: M + 0.3, y: 4.02, w: 11.5, h: 0.9, fontFace: F, fontSize: 14,
    color: "E8B9C3", margin: 0, valign: "top", lineSpacingMultiple: 1.35,
  });
  s.addText("진행 보고  ·  2026. 08. 19  ·  KUAICV  ·  류현우", {
    x: M + 0.3, y: 5.9, w: 11.5, h: 0.4, fontFace: F, fontSize: 13,
    color: "D79FAC", margin: 0, valign: "top",
  });
}

// ================================================================ 2. summary
{
  const s = pres.addSlide();
  titled(s, "SUMMARY", "이번 보고의 요지");

  card(s, M, 1.42, W - 2 * M, 1.28, TINT);
  s.addText([
    { text: "한 줄 결론 — " , options: { bold: true, color: CRIM } },
    { text: "8/18에 “반드시 기다려야 하는 병목”으로 분류했던 상황(corridor2 +50.6 %, chain3 +39.1 %)은 사실 " },
    { text: "경로를 고정했기 때문", options: { bold: true, color: CRIM } },
    { text: "이었습니다. 경로를 풀어주자 각각 " },
    { text: "+4.0 % · +8.4 %", options: { bold: true, color: CRIM } },
    { text: "로 떨어집니다. 반대로 교차형(crossing2)은 여전히 기존 A*가 낫습니다 — 두 방법은 대체재가 아니라 " },
    { text: "서로 다른 충돌 유형의 전문가", options: { bold: true, color: CRIM } },
    { text: "입니다." },
  ], { x: M + 0.34, y: 1.62, w: W - 2 * M - 0.68, h: 0.95, fontFace: F,
       fontSize: 13.5, color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.3 });

  const items = [
    ["1", "무엇을 만들었나", "현재 상태에서 여러 multi-agent future를 rollout하는 World Model과, 명시적 cost로 그중 하나를 고르는 Planner. 강화학습 아님 — 목적함수 전문이 코드 한 곳에 적혀 있음"],
    ["2", "무엇이 달라졌나", "각 rollout은 대기 시간만 다른 것이 아니라 이동 경로 자체와 timing이 함께 다름. 실행 중에는 다른 agent 상황에 따라 매초 어느 future를 탈지 다시 고름"],
    ["3", "결과", "5종 중 병목형 2종에서 −30.9 % / −22.1 %, 교차형 1종에서 +7.2 %. 전 시나리오·전 시드에서 충돌 0 · 의존성 위반 0 (8/18 독립 검증기 그대로 사용)"],
    ["4", "남은 것", "mode 샘플링이 균등해 chain3에서 시드 편차 2.71 s. soft cost 항이 아직 tie-break로만 작동. 최적성 보장 없음"],
  ];
  let yy = 2.92;
  items.forEach(([n, h1, h2]) => {
    badge(s, M + 0.02, yy + 0.04, n, 0.32);
    s.addText(h1, { x: M + 0.52, y: yy, w: 2.5, h: 0.3, fontFace: F, fontSize: 14,
                    bold: true, color: INK, margin: 0, valign: "top" });
    s.addText(h2, { x: M + 3.1, y: yy + 0.01, w: W - M - 3.1 - M, h: 0.85,
                    fontFace: F, fontSize: 12, color: MUTED, margin: 0, valign: "top" });
    yy += 0.94;
  });
}

// ================================================================ 3. motivation
{
  const s = pres.addSlide();
  titled(s, "MOTIVATION", "8/18에서 남긴 한계 — 경로를 고정했다는 것");

  s.addText("Stage 2는 각 agent의 경로를 고정하고 “자기 경로 위 진행도 kᵢ” 하나만 변수로 남겼습니다. " +
    "그 덕에 충돌·의존성이 모두 격자 위 금지 영역이 되어 최적해를 보장할 수 있었지만, 대가가 있었습니다.", {
    x: M, y: 1.4, w: W - 2 * M, h: 0.6, fontFace: F, fontSize: 12.5, color: MUTED,
    margin: 0, valign: "top",
  });

  const cols = [
    ["문제 1", "병목이 과장된다",
      "corridor2 +50.6 %, chain3 +39.1 %. 경로가 하나뿐이니 한 명이 통로를 비울 때까지 기다리는 수밖에 없음 — 하지만 그건 통로가 좁아서가 아니라 경로가 고정돼서일 수 있음"],
    ["문제 2", "아예 못 푸는 경우",
      "deadlock2 — B의 최단 경로가 A의 Goal 위를 지남. A가 도착해 정지하면 B는 영원히 통과 불가. 순차·우선순위 모두 실패"],
    ["문제 3", "데이터셋 편향",
      "못 푸는 인스턴스를 폐기 — 2-agent 40 %, 3-agent 80 %. 하필 agent끼리 강하게 얽힌 케이스가 빠져 조정이 덜 필요한 쪽으로 편향"],
  ];
  const cw = (W - 2 * M - 0.56) / 3;
  cols.forEach(([tag, nm, body], i) => {
    const x = M + i * (cw + 0.28);
    card(s, x, 2.15, cw, 2.28, i === 0 ? TINT : CARD);
    s.addText(tag, { x: x + 0.26, y: 2.32, w: 2, h: 0.28, fontFace: F, fontSize: 10.5,
                     bold: true, color: CRIM, charSpacing: 1.1, margin: 0, valign: "top" });
    s.addText(nm, { x: x + 0.26, y: 2.62, w: cw - 0.52, h: 0.34, fontFace: F,
                    fontSize: 15, bold: true, color: INK, margin: 0, valign: "top" });
    s.addText(body, { x: x + 0.26, y: 3.02, w: cw - 0.52, h: 1.3, fontFace: F,
                      fontSize: 11.5, color: MUTED, margin: 0, valign: "top" });
  });

  card(s, M, 4.72, W - 2 * M, 1.5, TINT);
  s.addText("이번 확장의 가설", { x: M + 0.34, y: 4.9, w: 4, h: 0.3, fontFace: F,
    fontSize: 13.5, bold: true, color: CRIM, margin: 0, valign: "top" });
  s.addText([
    { text: "세 문제 모두 같은 뿌리 — " },
    { text: "“어디로 갈지”를 미리 정하고 “언제 갈지”만 조정한 것", options: { bold: true } },
    { text: ". 그렇다면 경로와 timing을 함께 바꾸는 미래를 여러 개 만들어 비교하면 셋 다 완화될 것이다. " +
            "다만 최적성 보장은 잃는다 — 그 교환이 값하는지가 이번 실험의 질문." },
  ], { x: M + 0.34, y: 5.26, w: W - 2 * M - 0.68, h: 0.85, fontFace: F,
       fontSize: 12.5, color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.25 });
  foot(s, "8/18 제출본 슬라이드 10(LIMITATION) · 슬라이드 11(DATASET)에서 이미 보고한 내용입니다");
}

// ================================================================ 4. task
{
  const s = pres.addSlide();
  titled(s, "TASK", "이번 확장의 지시사항과 설계 결정");

  card(s, M, 1.4, 5.55, 2.5, TINT);
  s.addText("지시사항", { x: M + 0.28, y: 1.56, w: 3, h: 0.3, fontFace: F,
    fontSize: 13.5, bold: true, color: CRIM, margin: 0, valign: "top" });
  s.addText([
    { text: "현재 상태 + task condition에서 여러 future trajectory를 rollout하고, Planner가 비교해 가장 좋은 미래를 선택", options: { bullet: true, breakLine: true } },
    { text: "RL이 아닌 명시적 cost / loss 방식", options: { bullet: true, breakLine: true } },
    { text: "경로를 고정하고 대기만 조정하는 방식 금지 — temporal + spatial coordination을 함께", options: { bullet: true, breakLine: true } },
    { text: "시나리오 분류는 8/18 pptx 그대로, 실시간 경로 수정은 방법B pdf 참고", options: { bullet: true } },
  ], { x: M + 0.28, y: 1.94, w: 5.0, h: 1.85, fontFace: F, fontSize: 11.5,
       color: INK, paraSpaceAfter: 5, margin: 0, valign: "top" });

  const dec = [
    ["World Model", "절차적 생성기 (학습 X)",
     "GT 358개는 폐기율 40~80 %로 편향돼 있음 — 지금 학습하면 그 편향이 World Model에 굳음. API를 좁혀 두어 나중에 학습 생성기로 교체 가능"],
    ["구조", "중앙집중 joint rollout",
     "방법B의 정체성은 “미래 궤적을 버린다”라서 이번 과제와 정면 충돌. B에서는 경로를 고정하지 않는 성질만 가져옴"],
    ["재계획", "라이브러리 + 실시간 스위칭",
     "시작 전 mode 집합 생성 → 매초 재평가·스위칭 (hysteresis 적용)"],
    ["산출물", "코드 + 5시나리오 실험",
     "PPT·대량 평가는 결과 확인 후 결정"],
  ];
  const x0 = M + 5.85, wc = W - M - x0;
  s.addText("설계 결정", { x: x0, y: 1.5, w: 3, h: 0.3, fontFace: F, fontSize: 13.5,
    bold: true, color: CRIM, margin: 0, valign: "top" });
  let yy = 1.9;
  dec.forEach(([k, v, why]) => {
    s.addText(k, { x: x0, y: yy, w: 1.6, h: 0.28, fontFace: F, fontSize: 12,
                   bold: true, color: MUTED, margin: 0, valign: "top" });
    s.addText(v, { x: x0 + 1.65, y: yy, w: wc - 1.65, h: 0.28, fontFace: F,
                   fontSize: 13, bold: true, color: INK, margin: 0, valign: "top" });
    s.addText(why, { x: x0 + 1.65, y: yy + 0.29, w: wc - 1.65, h: 0.72, fontFace: F,
                     fontSize: 10.5, color: MUTED, margin: 0, valign: "top" });
    yy += 1.11;
  });

  card(s, M, 4.15, 5.55, 2.1, CARD);
  s.addText("바뀌지 않은 것", { x: M + 0.28, y: 4.3, w: 3, h: 0.3, fontFace: F,
    fontSize: 13, bold: true, color: INK, margin: 0, valign: "top" });
  s.addText([
    { text: "시나리오 5종 (crossing2 · corridor2 · deadlock2 · chain3 · fork3)", options: { bullet: true, breakLine: true } },
    { text: "제약 C1~C5 · dependency 정의 · Δt = 0.1 s · 반지름 0.30 m · safety 0.10 m", options: { bullet: true, breakLine: true } },
    { text: "임계 경로 하한 · 순차 baseline · 협응 A* — 비교 기준으로 그대로 유지", options: { bullet: true, breakLine: true } },
    { text: "독립 검증기 validate.py — 손대지 않고 재사용", options: { bullet: true } },
  ], { x: M + 0.28, y: 4.66, w: 5.0, h: 1.5, fontFace: F, fontSize: 11,
       color: MUTED, paraSpaceAfter: 4, margin: 0, valign: "top" });
}

// ================================================================ 5. overview
{
  const s = pres.addSlide();
  titled(s, "OVERVIEW", "구조 — 상상하고, 값매기고, 조금만 실행한다");
  fitImage(s, "wm_pipeline.png", 2.979, { x: M, y: 1.35, w: W - 2 * M, h: 3.05 });

  const cols = [
    ["WORLD MODEL", "여러 미래를 만든다",
      "같은 현재 상태에서 서로 다른 “팀 조직 방식”을 골라 각각 forward simulate. 경로 후보 · 양보 순서 · 속도/회피 변형이 조합되어 경로와 timing이 함께 다른 rollout이 나옴"],
    ["PLANNER", "명시적 cost로 고른다",
      "hard(충돌·의존성 위반·미완·livelock)를 먼저 걸러내고, 통과한 것들 사이에서만 soft(makespan·flow·대기·거리·여유·거칠기)로 순위. RL 아님 — 목적함수가 코드에 전문으로 적혀 있음"],
    ["EXECUTOR", "1초만 믿는다",
      "고른 미래를 끝까지 실행하지 않고 앞 1초만 커밋한 뒤 실제 상태에서 다시 상상. 상황이 바뀌면 다른 미래로 갈아탐 — 명확히 더 좋을 때만(hysteresis)"],
  ];
  const cw2 = (W - 2 * M - 0.56) / 3;
  cols.forEach(([tag, nm, body], i) => {
    const x = M + i * (cw2 + 0.28);
    card(s, x, 4.6, cw2, 2.1, i === 1 ? TINT : CARD);
    s.addText(tag, { x: x + 0.26, y: 4.76, w: 3, h: 0.28, fontFace: F, fontSize: 10.5,
                     bold: true, color: CRIM, charSpacing: 1.1, margin: 0, valign: "top" });
    s.addText(nm, { x: x + 0.26, y: 5.06, w: cw2 - 0.52, h: 0.32, fontFace: F,
                    fontSize: 14, bold: true, color: INK, margin: 0, valign: "top" });
    s.addText(body, { x: x + 0.26, y: 5.44, w: cw2 - 0.52, h: 1.2, fontFace: F,
                      fontSize: 11, color: MUTED, margin: 0, valign: "top" });
  });
}

// ================================================================ 6. A vs B vs ours
{
  const s = pres.addSlide();
  titled(s, "DESIGN", "방법 A · 방법 B에서 무엇을 가져오고 무엇을 버렸나");

  const tb = [
    ["항목", "방법 A (중앙집중)", "방법 B (분산 반응)", "이번 구조"],
    ["경로의 역할", "고정 — 속도만 조정 (1D)", "guide — 매 스텝 2D 자유 반응", "guide + 후보 k개"],
    ["이웃 정보", "미래 궤적 전부", "현재 상태만 (board)", "미래를 상상 (유한 horizon)"],
    ["정적 장애물", "inflate로 by-construction", "런타임 반응 회피", "런타임 반응 + swept 하드 검사"],
    ["agent 충돌", "space-time 탐색", "RVO / ORCA", "일반화 RVO + 안전 투영"],
    ["makespan", "순열 탐색으로 최적화", "규칙에서 창발 (보장 X)", "명시적 cost로 비교 선택"],
    ["실패 모드", "prioritized planning 불완전성", "deadlock / livelock", "둘 다 cost의 hard block으로 노출"],
  ];
  table(s, tb, {
    x: M, y: 1.42, w: W - 2 * M, colW: [2.1, 3.0, 3.0, 3.96],
    rowH: 0.34, fontSize: 12,
    cell: (i, k) => (k === 3 ? { bold: true, color: CRIM, fill: { color: TINT } } : {}),
  });

  card(s, M, 4.32, W - 2 * M, 1.95, TINT);
  s.addText("방법 B의 “탈중앙화”는 채택하지 않았습니다 — 그리고 그게 유일한 의도적 이탈입니다", {
    x: M + 0.34, y: 4.5, w: W - 2 * M - 0.68, h: 0.32, fontFace: F, fontSize: 13.5,
    bold: true, color: CRIM, margin: 0, valign: "top" });
  s.addText([
    { text: "B 문서의 정체성은 슬라이드 3에 명시돼 있습니다 — " },
    { text: "“중앙집중 A가 알던 두 가지 중 미래 궤적을 버린다”", options: { bold: true } },
    { text: ". 그런데 이번 지시는 “여러 future를 만들어 비교한다”, 즉 미래를 예측해서 평가하는 구조입니다. 그대로 융합하면 B가 아니라 sampling-based MPC가 됩니다.\n" },
    { text: "그래서 B에서 가져온 것은 탈중앙화가 아니라 " },
    { text: "“경로를 고정하지 않고 실시간으로 비켜간다”는 성질", options: { bold: true, color: CRIM } },
    { text: " (D2 sampling RVO · D3 정적 장애물 VO · D4 pure-pursuit · D5 dependency event-hold · D6 대칭 깨기 · D7 동기 스냅샷)이고, 구조는 중앙집중 receding-horizon입니다." },
  ], { x: M + 0.34, y: 4.86, w: W - 2 * M - 0.68, h: 1.3, fontFace: F, fontSize: 11.5,
       color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.22 });
  foot(s, "Stage 1 (장애물 inflate + visibility graph + arc-length 파라미터화)은 A·B와 동일하게 8/18 코드를 그대로 재사용");
}

// ================================================================ 7. world model
{
  const s = pres.addSlide();
  titled(s, "WORLD MODEL", "하나의 상태에서 서로 다른 미래가 나오는 네 개의 축");
  fitImage(s, "deadlock2/fan_t0.png", 0.96, { x: M, y: 1.32, w: 5.0, h: 5.0 });

  const x0 = M + 5.3, wc = W - M - x0;
  const tb = [
    ["축", "무엇이 바뀌나", "차이"],
    ["routes[i]", "agent i가 탈 reference 경로", "공간 (homotopy)"],
    ["yield_rank", "누가 누구에게 양보하는가", "공간 + 시간"],
    ["cautious", "양보측 preferred 속도 75 %", "시간"],
    ["split_side", "우선/양보가 반대쪽으로", "공간 (대칭 깨기)"],
  ];
  table(s, tb, { x: x0, y: 1.35, w: wc, colW: [1.6, 3.3, 2.28], rowH: 0.32,
                 fontSize: 11.5 });

  s.addText([
    { text: "경로 후보는 8/18의 " },
    { text: "VisibilityGraph.k_paths", options: { fontFace: "Consolas" } },
    { text: " — 이미 쓴 edge에 penalty를 주며 재탐색해 위상이 다른 경로를 뽑습니다. 따라서 각 rollout은 " },
    { text: "대기 시간만 다른 것이 아니라 이동 경로 자체와 timing이 함께 다릅니다.", options: { bold: true, color: CRIM } },
  ], { x: x0, y: 3.28, w: wc, h: 1.0, fontFace: F, fontSize: 12, color: INK,
       margin: 0, valign: "top", lineSpacingMultiple: 1.25 });

  card(s, x0, 4.24, wc, 2.02, TINT);
  s.addText("왜 학습 모델이 아닌가", { x: x0 + 0.28, y: 4.4, w: 4, h: 0.3, fontFace: F,
    fontSize: 13, bold: true, color: CRIM, margin: 0, valign: "top" });
  s.addText([
    { text: "8/18 GT 358개는 " },
    { text: "고정 경로 방식으로 풀리는 인스턴스만 남긴 것", options: { bold: true } },
    { text: "입니다 (2-agent 40 % · 3-agent 80 % 폐기). 지금 그것으로 생성 모델을 학습하면 편향이 World Model에 그대로 굳습니다.\n" },
    { text: "그래서 공개 API를 sample_modes + rollouts 둘로만 좁혀 두었습니다 — 편향을 해결한 뒤 " },
    { text: "Planner를 건드리지 않고 학습 생성기로 교체", options: { bold: true, color: CRIM } },
    { text: "할 수 있습니다." },
  ], { x: x0 + 0.28, y: 4.76, w: wc - 0.56, h: 1.4, fontFace: F, fontSize: 11,
       color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.2 });
  foot(s, "그림: deadlock2에서 t = 0에 상상한 16개 미래 (전 구간 rollout). 굵은 선이 Planner가 고른 것 — B가 먼저 통과하는 미래");
}

// ================================================================ 8. RVO / safety
{
  const s = pres.addSlide();
  titled(s, "WORLD MODEL", "다양성의 원천 — 회피 책임을 비대칭으로 나눈다");
  fitImage(s, "wm_alpha.png", 3.170, { x: M, y: 1.4, w: W - 2 * M, h: 2.35 });

  s.addText([
    { text: "고전 RVO는 회피 책임을 50:50으로 나눕니다. 결정론적이라 좋지만, 그래서 " },
    { text: "같은 상태에서 항상 같은 미래 하나만", options: { bold: true } },
    { text: " 나옵니다. 여기서는 yield_rank가 정한 α로 비대칭을 줍니다:", },
  ], { x: M, y: 3.85, w: W - 2 * M, h: 0.5, fontFace: F, fontSize: 12.5,
       color: MUTED, margin: 0, valign: "top" });

  card(s, M, 4.35, 6.1, 1.95, CARD);
  s.addText("u  =  v_i  +  ( v_cand − v_i ) / α_i  −  v_j", {
    x: M + 0.3, y: 4.55, w: 5.5, h: 0.34, fontFace: "Consolas", fontSize: 14.5,
    bold: true, color: INK, margin: 0, valign: "top" });
  s.addText([
    { text: "α → 1.0   내가 전부 피한다 (양보)", options: { breakLine: true } },
    { text: "α → 0.5   고전 RVO (중립)", options: { breakLine: true } },
    { text: "α → 0.3   상대가 피해줄 것을 기대 (우선)" },
  ], { x: M + 0.3, y: 5.0, w: 5.5, h: 1.1, fontFace: "Consolas", fontSize: 11.5,
       color: MUTED, margin: 0, valign: "top", lineSpacingMultiple: 1.35 });

  card(s, M + 6.4, 4.35, W - M - (M + 6.4), 1.95, TINT);
  s.addText("충돌은 “드문 일”이 아니라 “불가능한 일”", {
    x: M + 6.68, y: 4.53, w: 5.4, h: 0.3, fontFace: F, fontSize: 13, bold: true,
    color: CRIM, margin: 0, valign: "top" });
  s.addText([
    { text: "매 스텝 모든 agent가 속도를 고른 뒤 " },
    { text: "reciprocal safety projection", options: { bold: true } },
    { text: "을 돌립니다. swept 거리가 rᵢ+rⱼ+safety 미만인 쌍의 속도를 절반씩 줄이며 반복 — 안전한 배치에서 정지는 항상 안전하므로 수렴이 보장됩니다.\n" },
    { text: "투영의 대가(잃은 속도)는 Planner의 대기 항이 이미 과금하므로 " },
    { text: "안전이 목적함수와 몰래 경쟁하지 않습니다.", options: { bold: true, color: CRIM } },
  ], { x: M + 6.68, y: 4.88, w: W - M - (M + 6.68) - 0.28, h: 1.35, fontFace: F,
       fontSize: 11, color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.2 });
}

// ================================================================ 9. planner
{
  const s = pres.addSlide();
  titled(s, "PLANNER", "명시적 cost — 강화학습이 아닌 이유");

  card(s, M, 1.38, 6.35, 3.15, CARD);
  s.addText([
    { text: "J = 1e4 × ( agent충돌 + 장애물충돌\n              + dependency위반 + 미도달 + livelock )", options: { breakLine: true } },
    { text: "  + 1.00 × makespan            [s]   ← 주 목적", options: { breakLine: true } },
    { text: "  + 0.06 × flow time           [s]", options: { breakLine: true } },
    { text: "  + 0.14 × 포기한 속도          [s]", options: { breakLine: true } },
    { text: "  + 0.03 × 총 이동거리          [m]", options: { breakLine: true } },
    { text: "  + 0.60 × 안전여유 부족량      [m²s]", options: { breakLine: true } },
    { text: "  + 0.02 × 명령 거칠기          [m²/s]", options: { breakLine: true } },
    { text: "  + 0.015 × reference 이탈      [m s]" },
  ], { x: M + 0.3, y: 1.58, w: 5.8, h: 2.8, fontFace: "Consolas", fontSize: 11,
       color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.28 });

  const notes = [
    ["hard 가중치는 크지만 유한", "좁은 통로에서 일시적으로 모든 rollout이 infeasible해질 때 예외를 던지는 대신 “가장 덜 깨진” 미래를 반환하고, executor가 다음 replan에서 회복할 기회를 갖습니다"],
    ["feasible끼리는 오직 soft로", "hard block이 정확히 0이므로 선택은 soft 항만으로 결정됩니다"],
    ["horizon 만료는 위반이 아님", "아직 도착 못 한 agent는 미완일 뿐이고, terminal cost가 나머지 여정을 이미 값매김하고 있습니다"],
    ["terminal cost는 과소추정만", "임계 경로 하한(충돌 완화, dependency 유지)을 실행 도중 상태에서 재계산 — 잘린 미래가 완주한 미래보다 유리해 보이는 일이 없습니다"],
  ];
  const x0 = M + 6.65, wc = W - M - x0;
  let yy = 1.42;
  notes.forEach(([h1, h2]) => {
    s.addText(h1, { x: x0, y: yy, w: wc, h: 0.28, fontFace: F, fontSize: 13,
                    bold: true, color: CRIM, margin: 0, valign: "top" });
    s.addText(h2, { x: x0, y: yy + 0.29, w: wc, h: 0.86, fontFace: F, fontSize: 11,
                    color: MUTED, margin: 0, valign: "top" });
    yy += 1.2;
  });

  const tb = [
    ["deadlock2 · t=0 · 전 구간 rollout", "mode", "makespan", "대기", "판정"],
    ["Planner가 고른 미래", "r11|B>A", "15.5 s", "2.4 s", "feasible"],
    ["최단 경로 + A 우선 (소박한 선택)", "r00|A>B", "24.7 s", "10.8 s", "feasible"],
    ["livelock에 빠진 미래", "r12|B>A|cs", "—", "11.5 s", "hard 1e4×2"],
  ];
  table(s, tb, {
    x: M, y: 4.75, w: W - 2 * M, colW: [4.6, 1.9, 1.7, 1.5, 2.36], rowH: 0.3,
    fontSize: 11.5,
    cell: (i, k, c) => Object.assign(
      k === 1 ? { fontFace: "Consolas" } : {},
      i === 1 ? { bold: true, color: CRIM, fill: { color: TINT } } : {},
      c === "hard 1e4×2" ? { color: "B00000", bold: true } : {}),
  });
  foot(s, "16개 중 feasible: crossing2 3 · chain3 3 · fork3 5 · deadlock2 9 · corridor2 15 — 대부분의 상상은 나쁘고, cost가 그걸 정확히 걸러냅니다");
}

// ================================================================ 10. executor
{
  const s = pres.addSlide();
  titled(s, "EXECUTOR", "실시간 조정 — 왜 재선택이 연극이 아닌가");
  fitImage(s, "chain3/switches.png", 2.597, { x: M, y: 1.35, w: W - 2 * M, h: 2.85 });

  s.addText("chain3 — 20번의 결정, 3번의 실제 스위칭. 아래 패널: 예측한 잔여 시간(파랑)이 실제(빨강)에 아래에서 수렴 — terminal cost가 하한이라는 설계가 그림으로 확인됩니다.", {
    x: M, y: 4.3, w: W - 2 * M, h: 0.42, fontFace: F, fontSize: 11, color: MUTED,
    margin: 0, valign: "top",
  });

  const cols = [
    ["1", "상상이 유한하다",
      "rollout은 결정론적이라 매번 끝까지 상상하면 t=0의 선택이 개선될 여지가 없습니다. horizon(4초) 이후를 낙관적 하한으로 대체하면 잘린 시야의 순위 ≠ 완전 시야의 순위 — horizon이 밀리며 Planner는 실제로 새로운 것을 알게 됩니다."],
    ["2", "상태가 library를 벗어난다",
      "반응 회피가 agent를 reference 경로 밖으로 밀어내므로, 같은 mode label이 t=0과 t=6에서 서로 다른 미래를 가리킵니다."],
    ["3", "Hysteresis",
      "mode들은 앞 몇 초를 공유해 cost가 거의 동률입니다. replan마다 왔다갔다 하는 건 조정이 아니라 잡음이므로, 도전자가 명확히 더 좋을 때만(상대 1 % + 절대 0.05) 갈아탑니다. 덕분에 스위칭 로그가 실제로 의미 있었던 결정의 기록으로 남습니다."],
  ];
  const cw = (W - 2 * M - 0.56) / 3;
  cols.forEach(([n, nm, body], i) => {
    const x = M + i * (cw + 0.28);
    card(s, x, 4.78, cw, 1.85, i === 0 ? TINT : CARD);
    badge(s, x + 0.26, 4.94, n, 0.28);
    s.addText(nm, { x: x + 0.62, y: 4.93, w: cw - 0.9, h: 0.3, fontFace: F,
                    fontSize: 13.5, bold: true, color: INK, margin: 0, valign: "top" });
    s.addText(body, { x: x + 0.26, y: 5.3, w: cw - 0.52, h: 1.25, fontFace: F,
                      fontSize: 10.5, color: MUTED, margin: 0, valign: "top" });
  });
}

// ================================================================ 11. result 1
{
  const s = pres.addSlide();
  titled(s, "RESULT 1", "정량 비교 — 다섯 개 시나리오");
  fitImage(s, "team_time.png", 2.320, { x: M, y: 1.32, w: 7.5, h: 3.3 });

  const x0 = M + 7.8, wc = W - M - x0;
  card(s, x0, 1.32, wc, 3.3, TINT);
  s.addText("읽는 법", { x: x0 + 0.28, y: 1.48, w: wc - 0.56, h: 0.3, fontFace: F,
    fontSize: 13.5, bold: true, color: CRIM, margin: 0, valign: "top" });
  s.addText([
    { text: "검은 선 = 임계 경로 하한. dependency는 지키되 충돌은 무시한 최소 시간 — 어떤 해도 이보다 빠를 수 없음", options: { bullet: true, breakLine: true } },
    { text: "× = 그 방법으로는 해를 찾지 못함 (deadlock2 · 순차)", options: { bullet: true, breakLine: true } },
    { text: "병목형(corridor2 · chain3)에서 초록이 파랑을 크게 이깁니다 — 경로를 풀어준 효과", options: { bullet: true, breakLine: true } },
    { text: "교차형(crossing2)에서는 파랑이 여전히 낫습니다 — 타이밍만 조정하면 되는 문제이고 A*는 그걸 최적으로 풉니다", options: { bullet: true } },
  ], { x: x0 + 0.28, y: 1.88, w: wc - 0.56, h: 2.6, fontFace: F, fontSize: 11.5,
       color: INK, paraSpaceAfter: 5, margin: 0, valign: "top" });

  const tb = [
    ["시나리오", "하한", "순차", "협응 A* (경로 고정)", "World Model + Planner", "하한 대비", "A* 대비"],
    ["crossing2", "14.3", "24.5", "15.3", "16.4", "+14.7 %", "+7.2 %"],
    ["corridor2", "17.4", "32.3", "26.2", "18.1", "+4.0 %", "−30.9 %"],
    ["deadlock2", "14.0", "해 없음", "15.1", "15.5", "+10.7 %", "+2.6 %"],
    ["chain3", "17.9", "48.4", "24.9", "19.4", "+8.4 %", "−22.1 %"],
    ["fork3", "18.1", "49.8", "20.0", "19.4", "+7.2 %", "−3.0 %"],
  ];
  table(s, tb, {
    x: M, y: 4.82, w: W - 2 * M, colW: [1.85, 1.25, 1.35, 2.5, 2.9, 1.55, 1.66],
    rowH: 0.3, fontSize: 11.5,
    cell: (i, k, c) => Object.assign(
      k === 4 ? { bold: true, color: CRIM, fill: { color: TINT } } : {},
      k === 6 && c.startsWith("−") ? { bold: true, color: GOOD } : {},
      c === "해 없음" ? { color: MUTED } : {}),
  });
  foot(s, "단위: 초 (team completion time). 모든 결과는 8/18의 독립 검증기에서 충돌 0건 · 의존성 위반 0건 · 속도 제한 준수");
}

// ================================================================ 12. result 2
{
  const s = pres.addSlide();
  titled(s, "RESULT 2", "결론이 하나 뒤집혔다 — corridor2의 “병목”은 병목이 아니었다");
  fitImage(s, "corridor2/compare.png", 2.579, { x: M, y: 1.3, w: W - 2 * M, h: 3.55 });

  card(s, M, 5.0, W - 2 * M, 1.62, TINT);
  s.addText([
    { text: "8/18 보고 — " , options: { bold: true, color: MUTED } },
    { text: "“문이 하나뿐이라 속도를 줄이는 것으로는 피할 수 없고 실제로 기다려야만 하는 상황” (+50.6 %, 급정지 100→98회)\n", options: { color: MUTED } },
    { text: "그런데 " },
    { text: "문 폭은 1.8 m", options: { bold: true, color: CRIM } },
    { text: "이고 두 agent가 필요한 간격은 0.7 m입니다 — 나란히 지나갈 수 있습니다. 가운데 패널에서 B는 t=4.8 s에 아직 출발선에 서 있지만(A가 문을 비우기를 기다림), 오른쪽에서는 " },
    { text: "둘이 동시에 통로 안에 있고 서로를 향해 경로를 휘게 만듭니다", options: { bold: true, color: CRIM } },
    { text: ". 26.2 s → 18.1 s, 하한 대비 +4.0 %." },
  ], { x: M + 0.34, y: 5.18, w: W - 2 * M - 0.68, h: 1.3, fontFace: F, fontSize: 12,
       color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.22 });
  foot(s, "chain3도 같은 성격입니다 (24.9 s → 19.4 s, +39.1 % → +8.4 %). deadlock2는 순차·우선순위가 아예 실패하던 케이스를 15.5 s에 해결");
}

// ================================================================ 13. ablation
{
  const s = pres.addSlide();
  titled(s, "RESULT 3", "실시간 재선택은 얼마나 값하는가");
  fitImage(s, "wm_ablation.png", 2.574, { x: M, y: 1.4, w: 7.3, h: 3.1 });

  const x0 = M + 7.6, wc = W - M - x0;
  s.addText("ablation", { x: x0, y: 1.4, w: 3, h: 0.28, fontFace: F, fontSize: 11,
    bold: true, color: CRIM, charSpacing: 1.2, margin: 0, valign: "top" });
  s.addText([
    { text: "2-agent에서는 차이가 없습니다", options: { bold: true } },
    { text: " (≤ 0.1 s). agent가 둘뿐이면 t = 0에 이미 옳은 선택이 가능하기 때문입니다.\n\n" },
    { text: "3-agent에서만 값합니다", options: { bold: true, color: CRIM } },
    { text: " — chain3 −1.1 s, fork3 −1.3 s (약 5~6 %). 세 명이 얽히면 누가 먼저 갈지가 t = 0에 결정되지 않고, 앞선 agent가 실제로 어디쯤 있는지를 본 뒤에야 판단할 수 있습니다.\n\n" },
    { text: "즉 실시간 조정의 가치는 agent 수에 따라 커집니다 — 최종 연구 방향(다중 humanoid)에 유리한 신호입니다.", options: { color: CRIM } },
  ], { x: x0, y: 1.78, w: wc, h: 2.75, fontFace: F, fontSize: 11.5, color: INK,
       margin: 0, valign: "top", lineSpacingMultiple: 1.22 });

  const tb = [
    ["시나리오", "제안 (H=4 s)", "H=8 s", "전 구간 상상", "재선택 없음", "계산시간"],
    ["crossing2", "16.4", "16.4", "16.4", "16.4", "6.8 s"],
    ["corridor2", "18.1", "18.2", "18.2", "18.2", "7.5 s"],
    ["deadlock2", "15.5", "15.4", "15.4", "15.5", "6.9 s"],
    ["chain3", "19.4", "19.6", "19.0", "20.5", "14.5 s"],
    ["fork3", "19.4", "19.6", "20.7", "20.7", "13.5 s"],
  ];
  table(s, tb, {
    x: M, y: 4.75, w: W - 2 * M, colW: [2.2, 2.1, 1.75, 2.2, 2.1, 1.71],
    rowH: 0.3, fontSize: 11.5,
    cell: (i, k) => (k === 1 ? { bold: true, color: CRIM, fill: { color: TINT } } : {}),
  });
  foot(s, "horizon을 8초·전 구간으로 늘려도 이득이 없습니다 — 안전은 controller가 담당하고 Planner의 일은 mode 선택이기 때문. 4초가 계산 대비 가장 효율적");
}

// ================================================================ 14. validation
{
  const s = pres.addSlide();
  titled(s, "VALIDATION", "검증 — 제약 만족과 시드 견고성");

  s.addText("모든 궤적은 8/18의 mahoi/validate.py가 그대로 검사합니다. 이 검증기는 플래너 내부를 전혀 보지 않고 실행된 좌표만 봅니다 (지난 라운드에서 실제로 격자 버그를 한 건 잡아냈습니다).", {
    x: M, y: 1.38, w: W - 2 * M, h: 0.5, fontFace: F, fontSize: 12.5, color: MUTED,
    margin: 0, valign: "top" });

  const tb = [
    ["시나리오", "seed 0", "seed 1", "seed 2", "평균", "표준편차", "최소 agent 간격 (m)", "필요 (m)", "판정"],
    ["crossing2", "16.4", "16.4", "16.4", "16.4", "0.00", "0.922", "0.70", "OK"],
    ["corridor2", "18.1", "18.2", "18.2", "18.2", "0.05", "0.931", "0.70", "OK"],
    ["deadlock2", "15.5", "15.4", "15.4", "15.4", "0.05", "0.921", "0.70", "OK"],
    ["chain3", "19.4", "19.5", "25.2", "21.4", "2.71", "0.746", "0.70", "OK"],
    ["fork3", "19.4", "20.2", "19.9", "19.8", "0.33", "0.934", "0.70", "OK"],
  ];
  table(s, tb, {
    x: M, y: 2.0, w: W - 2 * M, colW: [1.7, 1.1, 1.1, 1.1, 1.1, 1.32, 2.3, 1.2, 1.14],
    rowH: 0.32, fontSize: 11.5,
    cell: (i, k, c) => Object.assign(
      c === "OK" ? { bold: true, color: GOOD } : {},
      (i === 4 && (k === 3 || k === 5)) ? { bold: true, color: WARN } : {}),
  });

  card(s, M, 4.1, 5.85, 2.15, TINT);
  s.addText("전 시나리오 · 전 시드에서", { x: M + 0.3, y: 4.28, w: 5.2, h: 0.3,
    fontFace: F, fontSize: 12, color: MUTED, margin: 0, valign: "top" });
  s.addText("agent 충돌 0 · 장애물 충돌 0\ndependency 위반 0 · 속도 제한 준수", {
    x: M + 0.3, y: 4.6, w: 5.2, h: 0.75, fontFace: F, fontSize: 16, bold: true,
    color: CRIM, margin: 0, valign: "top", lineSpacingMultiple: 1.25 });
  s.addText("충돌은 controller의 안전 투영으로 구조적으로 막히므로, 검증기는 그 설계가 실제로 성립하는지를 확인하는 역할입니다.", {
    x: M + 0.3, y: 5.5, w: 5.2, h: 0.6, fontFace: F, fontSize: 10.5, color: MUTED,
    margin: 0, valign: "top" });

  const x0 = M + 6.15, wc = W - M - x0;
  s.addText("이 검증이 드러낸 것", { x: x0, y: 4.1, w: wc, h: 0.3, fontFace: F,
    fontSize: 13, bold: true, color: CRIM, margin: 0, valign: "top" });
  s.addText([
    { text: "chain3는 시드에 민감합니다 (sd 2.71 s).", options: { bold: true, breakLine: true } },
    { text: "16개 mode는 훨씬 큰 공간의 결정론적 부분추출입니다. seed 2에서는 그 추출이 좋은 양보 순서를 놓쳐 팀이 5.8 s를 더 씁니다. 나머지 네 시나리오는 0.05~0.33 s 이내로 안정적입니다.", options: { breakLine: true, color: MUTED } },
    { text: "\nsoft 항은 한 번도 결정적이지 않았습니다.", options: { bold: true, breakLine: true } },
    { text: "5종 모두에서 Planner의 선택 = 가장 빠른 feasible rollout. flow·거리·여유 가중치는 tie-break로만 작동했고 trade-off 항으로는 아직 검증되지 않았습니다.", options: { color: MUTED } },
  ], { x: x0, y: 4.46, w: wc, h: 1.85, fontFace: F, fontSize: 11, color: INK,
       margin: 0, valign: "top", lineSpacingMultiple: 1.18 });
  foot(s, "재현: python scripts/check_wm.py --seeds 3   ·   결과 원본: outputs/wm/verification.txt");
}

// ================================================================ 15. limitation
{
  const s = pres.addSlide();
  titled(s, "LIMITATION", "알려진 한계 — 정직하게");

  const rows = [
    ["1", "mode 샘플링이 균등", "16개는 훨씬 큰 공간의 결정론적 부분추출. chain3 seed 2에서 좋은 양보 순서를 놓쳐 5.8 s 손해 (sd 2.71 s). 유용성 기대값 기반 샘플링 또는 적응적 확장 필요 — 최우선", true],
    ["2", "soft 항이 tie-break로만 작동", "5종 모두에서 선택 = 가장 빠른 feasible rollout. flow·거리·여유 가중치가 trade-off 항으로는 검증되지 않음", true],
    ["3", "최적성 보장 없음", "mode 집합 밖의 미래는 볼 수 없음. A*의 “고정 경로 위 최적” 보장이 사라진 대신 경로 자유도를 얻은 교환", false],
    ["4", "계산량", "2-agent 7 s · 3-agent 14 s (16 modes, H=4 s). 전 구간 상상은 3배. rollout 수 × horizon에 선형", false],
    ["5", "중앙집중", "방법B의 탈중앙 실행은 구현하지 않음 (구조적 충돌 때문). 실제 로봇 배치에는 추가 확장 필요", false],
    ["6", "여전히 시연 수준의 평가", "시나리오 5종은 직접 설계한 것 — 난이도별 성공률 곡선 없음 (8/18 후속과제 ②와 동일하게 미해결)", false],
  ];
  let yy = 1.45;
  rows.forEach(([n, h1, h2, hot]) => {
    card(s, M, yy, W - 2 * M, 0.82, hot ? TINT : CARD);
    badge(s, M + 0.22, yy + 0.25, n, 0.32);
    s.addText(h1, { x: M + 0.68, y: yy + 0.13, w: 3.5, h: 0.3, fontFace: F,
                    fontSize: 13.5, bold: true, color: hot ? CRIM : INK,
                    margin: 0, valign: "top" });
    s.addText(h2, { x: M + 0.68, y: yy + 0.44, w: W - 2 * M - 1.0, h: 0.34,
                    fontFace: F, fontSize: 11, color: MUTED, margin: 0, valign: "top" });
    yy += 0.9;
  });
  foot(s, "8/18에서 지시로 제외된 항목(humanoid 물리 · 가속도 제약 · 8-agent 확장)은 여기 포함하지 않았습니다");
}

// ================================================================ 16. next
{
  const s = pres.addSlide();
  titled(s, "NEXT", "8/19 이후 계획");

  const tb = [
    ["", "지금의 문제", "무엇을 할 것인가", "무엇으로 확인하는가"],
    ["1", "mode 샘플링 균등 → chain3 시드 편차 2.71 s", "유용성 기대값 기반 샘플링 / 좋은 mode 주변 적응적 확장 / 첫 replan 후 재샘플링", "시드 간 표준편차 0.5 s 이하"],
    ["2", "soft 항이 tie-break로만 작동", "makespan과 flow·거리가 실제로 상충하는 시나리오를 설계하고 가중치 sweep", "가중치를 바꾸면 선택이 바뀌는가"],
    ["3", "두 방법이 서로 다른 유형의 전문가", "교차형/병목형 자동 판별 후 선택적 적용, 또는 A*를 mode 하나로 편입", "5종 전부에서 각 방법의 최선 이상"],
    ["4", "폐기율 40~80 % (8/18 한계 ①)", "WM+Planner로 GT 재생성 — 경로가 자유로우니 폐기율이 크게 낮아질 것", "폐기율 10 % 이하"],
    ["5", "시연 수준 평가 (8/18 한계 ③)", "waypoint 근접도 · 통로폭 · dependency 밀도 축으로 대량 실행", "난이도별 성공률 곡선"],
    ["6", "World Model 학습 미착수", "④로 편향 없는 데이터셋 확보 후 조건부 생성 모델을 그대로 드롭인", "절차적 생성기 대비 rollout 품질"],
  ];
  table(s, tb, {
    x: M, y: 1.45, w: W - 2 * M, colW: [0.5, 3.35, 4.6, 3.61], rowH: 0.5,
    fontSize: 11,
    cell: (i, k) => Object.assign(
      k === 0 ? { bold: true, color: CRIM, align: "center" } : {},
      i <= 2 ? { fill: { color: TINT } } : {}),
  });
  foot(s, "①②는 이번 확장이 새로 만든 숙제, ④⑤⑥은 8/18에서 이어지는 숙제입니다");
}

// ================================================================ 17. discussion
{
  const s = pres.addSlide();
  s.background = { color: CRIM_D };
  s.addText("여쭙고 싶은 것", {
    x: M + 0.3, y: 1.1, w: 11.5, h: 0.65, fontFace: F, fontSize: 32, bold: true,
    color: WHITE, margin: 0, valign: "top",
  });
  const qs = [
    ["1", "Dependency의 형태",
      "동시 수행(같이 들기) · 물체 상태 변화 조건 · 상대 시간 간격 중 어느 것을 우선 지원할까요"],
    ["2", "협동 효율의 정의",
      "현재는 makespan 위주(flow 0.06)입니다. flow time·이동거리 비중을 올리면 궤적 성격이 달라집니다. 학습 supervision 기준으로 무엇이 적절할까요 — 한계 ②와 직결됩니다"],
    ["3", "A*와 WM+Planner의 관계",
      "대체할지, 충돌 유형별로 선택할지, A*를 mode 하나로 흡수할지"],
    ["4", "물리와의 연결 시점",
      "humanoid motion / 시뮬레이터 전환 시점과 인터페이스. 현재 controller는 속도 명령 기반이라 가속도 제약을 넣을 자리는 마련돼 있습니다"],
  ];
  let yy = 2.15;
  qs.forEach(([n, h1, h2]) => {
    s.addShape(pres.ShapeType.ellipse, {
      x: M + 0.3, y: yy + 0.04, w: 0.32, h: 0.32, fill: { color: WHITE }, line: { width: 0 },
    });
    s.addText(n, { x: M + 0.3, y: yy + 0.04, w: 0.32, h: 0.32, fontFace: F,
                   fontSize: 12.5, bold: true, color: CRIM_D, align: "center",
                   valign: "middle", margin: 0 });
    s.addText(h1, { x: M + 0.82, y: yy, w: 4.0, h: 0.34, fontFace: F, fontSize: 16,
                    bold: true, color: WHITE, margin: 0, valign: "top" });
    s.addText(h2, { x: M + 4.95, y: yy + 0.02, w: W - M - (M + 4.95), h: 1.0,
                    fontFace: F, fontSize: 12, color: "E8B9C3", margin: 0, valign: "top" });
    yy += 1.12;
  });
  s.addText("코드 · 결과 원본:  mahoi_toy_wm.zip  ·  outputs/wm/summary.md  ·  outputs/wm/verification.txt\n" +
            "재현:  python scripts/run_wm_experiments.py   /   python scripts/check_wm.py --seeds 3", {
    x: M + 0.3, y: 6.5, w: 11.5, h: 0.7, fontFace: F, fontSize: 10.5,
    color: "C98D9C", margin: 0, valign: "top", lineSpacingMultiple: 1.3,
  });
}

// ---------------------------------------------------------------- write
const outFile = path.join(__dirname, "MultiAgent_WorldModel_Planner_류현우_0819.pptx");
pres.writeFile({ fileName: outFile })
  .then(() => console.log("wrote " + outFile))
  .catch((e) => { console.error(e); process.exit(1); });
