"""③ 위험 예측 모델(공간 × 시간) 학습과 검증.

공간 모델  "어느 읍면이 위험한가"
  - 어떤 시점 T까지의 기록만 보고 만든 특징(과거 발생률, 최근 추세, 이웃 읍면·시군의 발생률, 면적 등)으로
    그 뒤 3년(T+1~T+3)의 산불 건수를 맞히도록 학습한다. (포아송 그래디언트 부스팅)
  - 학습용 표는 T=2014~2020 을 쌓아서 만든다. 2024년 이후 기록은 학습에 전혀 쓰지 않는다.
시간 모델  "언제 위험한가"
  - 원인 유형별 월 분포(시도 단위, 전국 분포로 보정)와 읍면의 원인 구성(시군 구성으로 보정)을 곱해 읍면별 월 곡선을 만든다.
검증
  - 시간 검증: 2011~2023으로 순위를 매기고 2024~2026.9 산불 1,164건으로 채점.
  - 공간 검증: 시군구를 5묶음으로 나눠, 학습에 쓰지 않은 지역만 예측해 같은 방식으로 채점.
  - 기준선: "과거 건수 순위"(모델 없이 통계표만 본 것)와 비교.

출력: out/predictions.csv, out/monthly.json, out/metrics.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
TRAIN_END, H = 2023, 3           # 2023년까지 학습, 3년 뒤까지 예측
FIRST_YEAR, LAST_DATE = 2011, "2026-09"
TRAIN_T = list(range(2014, 2021))   # 특징 기준 연도 T (목표: T+1~T+3, 마지막 목표가 2021~2023)
GROUPS = ["입산자 실화", "논밭·영농 소각", "쓰레기 소각", "성묘객 실화", "담뱃불 실화", "건축물 화재 비화",
          "산업현장 실화", "원인 미상", "기타"]
GRADE_CUTS = [0.05, 0.15, 0.35, 0.65]   # 상위 5% / 15% / 35% / 65% 경계 → 1~5등급

fires = pd.read_csv(OUT / "fires.csv", dtype={"uid": str, "code5": str}, parse_dates=["date"])
units = pd.read_csv(OUT / "units_all.csv", dtype={"uid": str, "code5": str})
adj = json.loads((OUT / "adjacency.json").read_text(encoding="utf-8"))
# '시군' 묶음: 구가 나뉜 시는 시 전체, 특별·광역시의 구는 그 광역시 전체를 한 묶음으로 본다 (군은 따로)
METRO = ["11", "26", "27", "28", "29", "30", "31"]
is_metro_gu = units.code5.str[:2].isin(METRO) & units.si.str.endswith("구")
units["city"] = np.where(is_metro_gu, units.code5.str[:2], units.code5.str[:2] + units.si)
units["city_label"] = np.where(is_metro_gu, units.sido, units.sido_s + " " + units.si)
units["label"] = np.where(units.kind != "동", (units.gu.fillna("") + " " + units.name).str.strip(),
                          np.where(is_metro_gu, units.si, (units.gu.fillna("") + " 동 지역").str.strip()))
units["sido2"] = units.code5.str[:2]
U = units.uid.values
uix = {u: i for i, u in enumerate(U)}
NB = [[uix[v] for v in adj.get(u, {})] for u in U]

# 시도 단위 지목(임야·전·답) 비율 — 올려주신 지목 파일은 시도 단위라 읍면 특징으로는 못 쓰고 시도 특징으로만 쓴다
try:
    import openpyxl
    ws = openpyxl.load_workbook(next((ROOT / "raw/읍면동별지목면적").glob("*.xlsx")))["데이터"]
    rows = list(ws.iter_rows(values_only=True))
    jm = {r[0]: {"sido_forest": r[11] / r[1], "sido_farm": (r[3] + r[5]) / r[1]} for r in rows[4:]}
    units["sido_forest"] = units.sido.map(lambda s: jm.get(s, {}).get("sido_forest", np.nan))
    units["sido_farm"] = units.sido.map(lambda s: jm.get(s, {}).get("sido_farm", np.nan))
except Exception as e:   # 파일이 없어도 돌아가게
    print("지목 파일 생략:", e)
    units["sido_forest"] = units["sido_farm"] = np.nan

# 읍면 단위 산림(임야)·농지(전+답) 비율 — 01b_landuse.py 가 토지임야정보로 만든 것. 없으면 이 특징 없이 돈다
LAND_COLS = ["forest_ratio", "farm_ratio"]
try:
    land = pd.read_csv(OUT / "landuse.csv", dtype={"uid": str}).set_index("uid")
    for c in LAND_COLS:
        units[c] = units.uid.map(land[c])
    HAS_LAND = bool(units.forest_ratio.notna().any())
except FileNotFoundError:
    print("토지 비율 파일 없음(out/landuse.csv) — 토지 특징 없이 진행")
    HAS_LAND = False
    for c in LAND_COLS:
        units[c] = np.nan
DROP = []   # features() 에서 뺄 열. 토지 특징을 쓸지는 아래 내부 검증으로 정한다

fu = fires[fires.uid.notna()].copy()
fu["ui"] = fu.uid.map(uix)
Y = np.zeros((len(U), 2027 - FIRST_YEAR))                  # 단위 × 연도 건수
np.add.at(Y, (fu.ui.values, fu.year.values - FIRST_YEAR), 1)
YC = {g: np.zeros_like(Y) for g in GROUPS}                 # 원인 유형별
for g in GROUPS:
    s = fu[fu.cause_group == g]
    np.add.at(YC[g], (s.ui.values, s.year.values - FIRST_YEAR), 1)
HA = np.zeros_like(Y)
np.add.at(HA, (fu.ui.values, fu.year.values - FIRST_YEAR), fu.area_ha.values)
city_ix = pd.factorize(units.city)[0]
sido_ix = pd.factorize(units.sido2)[0]


def group_mean_excl_self(v, gix):
    s = np.bincount(gix, weights=v); n = np.bincount(gix)
    return (s[gix] - v) / np.maximum(n[gix] - 1, 1)


def features(T):
    """T년 말까지의 기록만으로 만든 단위별 특징."""
    k = T - FIRST_YEAR + 1
    y = Y[:, :k]
    n_years = k
    f = pd.DataFrame(index=U)
    f["rate_all"] = y.sum(1) / n_years
    f["rate_5y"] = y[:, -5:].sum(1) / min(5, k)
    f["rate_3y"] = y[:, -3:].sum(1) / min(3, k)
    w = 0.5 ** ((k - 1 - np.arange(k)) / 4.0)            # 반감기 4년 가중
    f["rate_ewm"] = (y * w).sum(1) / w.sum()
    f["active_years"] = (y > 0).sum(1) / n_years
    last = np.where((y > 0).any(1), k - 1 - np.argmax(y[:, ::-1] > 0, axis=1), 20)
    f["since_last"] = np.minimum(last, 20)
    f["max_year"] = y.max(1)
    f["log_ha"] = np.log1p(HA[:, :k].sum(1))
    for g in GROUPS[:7]:
        f["rate_" + g] = YC[g][:, :k].sum(1) / n_years
    ra, r5 = f.rate_all.values, f.rate_5y.values
    f["nb_rate"] = [ra[n].mean() if n else np.nan for n in NB]
    f["nb_rate_5y"] = [r5[n].mean() if n else np.nan for n in NB]
    f["nb_max"] = [ra[n].max() if n else np.nan for n in NB]
    f["city_rate"] = group_mean_excl_self(ra, city_ix)    # 같은 시군의 다른 읍면 평균
    f["city_rate_5y"] = group_mean_excl_self(r5, city_ix)
    f["sido_rate"] = group_mean_excl_self(ra, sido_ix)
    f["log_area"] = np.log(units.area_km2.values)
    f["density"] = ra / units.area_km2.values * 100      # 100㎢당 연 발생
    f["kind"] = units.kind.map({"읍": 0, "면": 1, "동": 2}).values
    f["x"], f["y"] = units.cx.values / 1000, units.cy.values / 1000
    f["n_nb"] = units.n_neighbors.values
    f["sido_forest"], f["sido_farm"] = units.sido_forest.values, units.sido_farm.values
    f["forest_ratio"], f["farm_ratio"] = units.forest_ratio.values, units.farm_ratio.values
    f["hist_years"] = n_years
    return f.drop(columns=DROP)


def target(T, h=H):
    a = T - FIRST_YEAR + 1
    return Y[:, a:a + h].sum(1)


def coverage(score, actual, q):
    """점수 상위 q 비율의 단위가 실제 산불의 몇 %를 담았나 (동점은 공평하게 나눠 계산)."""
    score, actual = np.asarray(score, float), np.asarray(actual, float)
    k = q * len(score)
    order = np.argsort(-score, kind="stable")
    s, a = score[order], actual[order]
    got, i = 0.0, 0
    while i < len(s) and i < k:
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        take = min(j, k) - i
        got += a[i:j].sum() * take / (j - i)
        i = j
    return got / actual.sum()


def within_city_coverage(score, actual, q=0.3, min_units=5):
    """실사용 기준: 각 시군 안에서 상위 q 읍면을 골랐을 때 그 시군 산불의 몇 %를 담나 (전국 합산)."""
    got = tot = 0.0
    for c in np.unique(city_ix):
        m = city_ix == c
        if m.sum() < min_units or actual[m].sum() == 0:
            continue
        got += coverage(score[m], actual[m], q) * actual[m].sum()
        tot += actual[m].sum()
    return got / tot


MONO_UP = ["rate_all", "rate_5y", "rate_3y", "rate_ewm", "active_years", "max_year", "log_ha", "nb_rate", "nb_rate_5y",
           "nb_max", "city_rate", "city_rate_5y", "sido_rate"] + ["rate_" + g for g in GROUPS[:7]]


def make_model(columns):
    """포아송 부스팅. '과거에 많이 났을수록 위험 점수가 내려가지는 않는다'는 단조 제약을 걸어 과적합을 줄인다."""
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0, monotonic_cst=[1 if c in MONO_UP else 0 for c in columns])


def fit(train_T):
    X = pd.concat([features(T) for T in train_T])
    y = np.concatenate([target(T) for T in train_T])
    return make_model(X.columns).fit(X, y), X, y


def smooth_score(T):
    """모델 없는 비교군 2: 자기 발생률과 이웃·시군 발생률을 섞은 점수(공식 하나)."""
    f = features(T)
    nb = f.nb_rate.fillna(f.city_rate)
    return (f.rate_ewm * 0.6 + f.rate_all * 0.4) * 0.7 + nb * 0.2 + f.city_rate * 0.1


# ---------- A. 모델 고르기용 내부 검증 (학습 기간 안에서만 굴려 보기; 시험 기간 2024~는 건드리지 않음)
# 기준 연도 T0 = 2017~2020 네 번: 매번 T0 이전 자료로만 학습하고 T0+1~T0+3 을 맞힌 뒤 평균
# 토지 특징(읍면 산림·농지 비율)도 여기서만 비교한다: 넣은 모델의 상위 20% 포착률이 더 높을 때만 쓴다
VARIANTS = {"학습 모델(토지 특징 없음)": LAND_COLS, "학습 모델(토지 특징 포함)": []} if HAS_LAND else {"학습 모델": LAND_COLS}
rows = {k: [] for k in ["기준선(과거 건수)", "이웃 보정 공식", *VARIANTS]}
for T0 in (2017, 2018, 2019, 2020):
    y0 = target(T0)
    scs = {"기준선(과거 건수)": features(T0).rate_all.values, "이웃 보정 공식": smooth_score(T0).values}
    for name, drop in VARIANTS.items():
        DROP[:] = drop
        m0, _, _ = fit(list(range(2013, T0 - H + 1)))
        scs[name] = m0.predict(features(T0))
    for name, sc in scs.items():
        rows[name].append([coverage(sc, y0, q) for q in (0.1, 0.2, 0.3)] + [within_city_coverage(sc, y0)])
inner = {k: np.mean(v, 0) for k, v in rows.items()}
print("[내부 검증: 2018~2023 구간 4회 평균] 상위 10/20/30% 포착률 / 시군 안 상위 30%")
for k, v in inner.items():
    print(f"  {k}: " + " / ".join(f"{x * 100:.1f}%" for x in v))
USE_LAND = HAS_LAND and inner["학습 모델(토지 특징 포함)"][1] > inner["학습 모델(토지 특징 없음)"][1]
DROP[:] = [] if USE_LAND else LAND_COLS
print("  → 토지 특징", "사용" if USE_LAND else "사용 안 함", "(내부 검증 상위 20% 기준)")

# ---------- B. 시간 검증 (최종): 2023년까지 학습 → 2024~2026.9 채점
model, Xtr, ytr = fit(TRAIN_T)
fT = features(TRAIN_END)
y_test = Y[:, TRAIN_END - FIRST_YEAR + 1:].sum(1)
pred = model.predict(fT)
scores = {"기준선(과거 건수)": fT.rate_all.values, "이웃 보정 공식": smooth_score(TRAIN_END).values, "학습 모델": pred}
QS = [0.05, 0.1, 0.2, 0.3, 0.5]
test = {k: {f"top{int(q * 100)}": round(coverage(v, y_test, q) * 100, 1) for q in QS} for k, v in scores.items()}
for k, v in scores.items():
    test[k]["within_city_top30"] = round(within_city_coverage(v, y_test) * 100, 1)
print(f"\n[시간 검증 2024~2026.9] 시험 산불 {int(y_test.sum())}건 (단위 매칭된 것)")
for k, v in test.items():
    print(f"  {k}: {v}")

# 부트스트랩으로 '모델 - 기준선' 차이의 불확실성 (시험 산불을 복원추출)
rng = np.random.default_rng(0)
ids = np.repeat(np.arange(len(U)), y_test.astype(int))
diffs = []
for _ in range(1000):
    yb = np.bincount(rng.choice(ids, len(ids)), minlength=len(U)).astype(float)
    diffs.append(coverage(pred, yb, 0.2) - coverage(fT.rate_all.values, yb, 0.2))
ci = np.percentile(diffs, [2.5, 97.5]) * 100
print(f"  상위 20% 포착률 차이(모델-기준선): {np.mean(diffs) * 100:+.1f}%p, 95% 구간 [{ci[0]:+.1f}, {ci[1]:+.1f}]")

# ---------- C. 공간 검증: 학습에 쓰지 않은 시군만 예측 (시군구 5묶음)
cities = np.unique(city_ix)
fold_of_city = {c: i % 5 for i, c in enumerate(np.random.default_rng(1).permutation(cities))}
fold = np.array([fold_of_city[c] for c in city_ix])
sp_pred = np.zeros(len(U))
for k in range(5):
    tr_mask = np.tile(fold != k, len(TRAIN_T))
    mk = make_model(Xtr.columns).fit(Xtr[tr_mask], ytr[tr_mask])
    sp_pred[fold == k] = mk.predict(fT[fold == k])
spatial = {f"top{int(q * 100)}": round(coverage(sp_pred, y_test, q) * 100, 1) for q in QS}
spatial["within_city_top30"] = round(within_city_coverage(sp_pred, y_test) * 100, 1)
print(f"\n[공간+시간 검증] 처음 보는 지역만 예측: {spatial}")

# 포착률 곡선 (그래프용)
curve_q = np.round(np.arange(0.0, 1.0001, 0.025), 3)
curves = {k: [round(coverage(v, y_test, q) * 100, 1) if q > 0 else 0.0 for q in curve_q] for k, v in scores.items()}
curves["공간 검증"] = [round(coverage(sp_pred, y_test, q) * 100, 1) if q > 0 else 0.0 for q in curve_q]

# 특징 중요도 (순열 중요도: 그 특징을 섞었을 때 시험 기간 예측이 얼마나 나빠지나)
pi = permutation_importance(model, fT, y_test, n_repeats=10, random_state=0, scoring="neg_mean_poisson_deviance")
importance = sorted(zip(fT.columns, pi.importances_mean), key=lambda x: -x[1])[:12]
print("\n[중요 특징]", [(a, round(b, 4)) for a, b in importance[:8]])

# ---------- D. 시간 모델: 원인 유형별 월 곡선 × 읍면의 원인 구성
tr = fires[fires.year <= TRAIN_END]
te = fires[fires.year > TRAIN_END]


def time_model(src):
    nat = src.groupby(["cause_group", "month"]).size().unstack(fill_value=0).reindex(index=GROUPS, columns=range(1, 13), fill_value=0)
    nat_p = (nat.T / nat.sum(1)).T.values                                   # 전국: 원인별 월 분포
    src = src.assign(sido2=src.code5.str[:2])
    sd = src.groupby(["sido2", "cause_group", "month"]).size()
    K = 30.0
    month_p = {}
    for s2 in units.sido2.unique():
        m = np.zeros((len(GROUPS), 12))
        for gi, g in enumerate(GROUPS):
            c = np.array([sd.get((s2, g, mm), 0) for mm in range(1, 13)], float)
            m[gi] = (c + K * nat_p[gi]) / (c.sum() + K)                     # 시도 분포를 전국 분포로 보정
        month_p[s2] = m
    nat_mix = src.cause_group.value_counts().reindex(GROUPS, fill_value=0).values.astype(float)
    nat_mix /= nat_mix.sum()
    src = src.assign(city=src.code5.map(units_city))
    cm = src.groupby(["city", "cause_group"]).size().unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0)
    um = src[src.uid.notna()].groupby(["uid", "cause_group"]).size().unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0)
    mix = np.zeros((len(U), len(GROUPS)))
    for i, r in enumerate(units.itertuples()):
        c = cm.loc[r.city].values.astype(float) if r.city in cm.index else np.zeros(len(GROUPS))
        cmix = (c + 10 * nat_mix) / (c.sum() + 10)                          # 시군 구성을 전국 구성으로 보정
        u = um.loc[r.uid].values.astype(float) if r.uid in um.index else np.zeros(len(GROUPS))
        mix[i] = (u + 4 * cmix) / (u.sum() + 4)                             # 읍면 구성을 시군 구성으로 보정
    curve = np.stack([mix[i] @ month_p[r.sido2] for i, r in enumerate(units.itertuples())])
    return mix, curve, nat_p, nat_mix


units_city = dict(zip(units.code5, units.city))
mix_tr, curve_tr, nat_p_tr, nat_mix_tr = time_model(tr)
nat_month = tr.month.value_counts(normalize=True).reindex(range(1, 13), fill_value=0).values
# 시험: (1) 월 맞히기 — 시험 산불이 난 달에 곡선이 준 확률의 평균 로그값
tu = te[te.uid.notna()]
ti, tm = tu.uid.map(uix).values, tu.month.values - 1
ll_nat = np.log(nat_month[tm]).mean()
ll_unit = np.log(curve_tr[ti, tm]).mean()
# (2) 읍면×월 칸 포착률: 전체 칸 중 상위 q만 지켰을 때
cell_test = np.zeros((len(U), 12)); np.add.at(cell_test, (ti, tm), 1)
cells = {
    "기준선(과거 건수 × 전국 월 분포)": np.outer(fT.rate_all.values, nat_month),
    "학습 모델 × 전국 월 분포": np.outer(pred, nat_month),
    "학습 모델 × 읍면별 월 곡선": pred[:, None] * curve_tr,
}
cell_cov = {k: {f"top{int(q * 100)}": round(coverage(v.ravel(), cell_test.ravel(), q) * 100, 1) for q in (0.05, 0.1, 0.2)}
            for k, v in cells.items()}
print(f"\n[시간 모델] 월 로그우도: 전국 곡선 {ll_nat:.3f} → 읍면별 곡선 {ll_unit:.3f}")
for k, v in cell_cov.items():
    print(f"  읍면×월 칸 포착률 {k}: {v}")

# ---------- E. 최종 점수: 전체 기록(2026.9까지)으로 다시 계산해 '다음 시기' 예측
# 2026년은 9월까지만 있으므로 0.75년으로 환산하지 않고, 마지막 완결 연도(2025)까지를 특징으로 쓴다.
FINAL_T = 2025
final_model, _, _ = fit(list(range(2014, FINAL_T - H + 1)))
fF = features(FINAL_T)
final = final_model.predict(fF) / H                       # 연간 예상 건수
mix, curve, nat_p, nat_mix = time_model(fires)
rank = (-final).argsort().argsort()
pct = (rank + 0.5) / len(U)
grade = 1 + np.searchsorted(GRADE_CUTS, pct)
hist_n = Y.sum(1)
out = units[["uid", "code5", "sido", "sido_s", "sigungu", "si", "gu", "name", "kind", "city", "city_label", "label",
             "has_shape", "area_km2", "lon", "lat", "cx", "cy"]].copy()
out["risk_annual"] = np.round(final, 4)
out["grade"] = grade
out["rank_all"] = rank + 1
out["rank_city"] = out.assign(r=final).groupby(units.city.values).r.rank(ascending=False, method="first").astype(int)
out["n_city"] = out.groupby(units.city.values).uid.transform("size").values
out["fires_total"] = hist_n.astype(int)
out["fires_5y"] = Y[:, 2021 - FIRST_YEAR:2026 - FIRST_YEAR].sum(1).astype(int)
out["ha_total"] = np.round(HA.sum(1), 2)
out["base_rank"] = (-Y[:, :FINAL_T - FIRST_YEAR + 1].sum(1)).argsort(kind="stable").argsort() + 1
out["forest_ratio"], out["farm_ratio"] = units.forest_ratio.values, units.farm_ratio.values   # 화면 표시용 (모델 사용 여부와 무관)
out.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
(OUT / "monthly.json").write_text(json.dumps({
    "uids": list(U), "groups": GROUPS,
    "mix": np.round(mix, 4).tolist(), "curve": np.round(curve, 4).tolist(),
    "nat_month_by_group": np.round(nat_p, 4).tolist(), "nat_mix": np.round(nat_mix, 4).tolist(),
}, ensure_ascii=False), encoding="utf-8")

metrics = {
    "train_period": f"{FIRST_YEAR}~{TRAIN_END}", "test_period": f"{TRAIN_END + 1}~{LAST_DATE}",
    "n_units": int(len(U)), "n_test_fires": int(y_test.sum()), "n_test_fires_all": int(len(te)),
    "n_train_fires": int(len(tr)), "inner_validation": {k: [round(x * 100, 1) for x in v] for k, v in inner.items()},
    "test": test, "spatial": spatial,
    "diff_top20": {"mean": round(float(np.mean(diffs)) * 100, 1), "ci95": [round(float(ci[0]), 1), round(float(ci[1]), 1)]},
    "curve_q": [float(q) for q in curve_q], "curves": curves,
    "importance": [[a, round(float(b), 4)] for a, b in importance],
    "time_model": {"loglik_national": round(float(ll_nat), 3), "loglik_unit": round(float(ll_unit), 3), "cell_coverage": cell_cov},
    "recur_pct": round(float((y_test[Y[:, :TRAIN_END - FIRST_YEAR + 1].sum(1) > 0].sum()) / y_test.sum() * 100), 1),
    "grade_cuts": GRADE_CUTS, "final_feature_year": FINAL_T, "has_land": HAS_LAND, "use_land": bool(USE_LAND),
    "grade_share_test": {str(g): round(float(y_test[(1 + np.searchsorted(GRADE_CUTS, ((-pred).argsort().argsort() + 0.5) / len(U))) == g].sum() / y_test.sum() * 100), 1)
                         for g in range(1, 6)},
}
(OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n등급별 시험 산불 비중(2023년까지 학습한 모델 기준):", metrics["grade_share_test"])
print("저장 완료: predictions.csv, monthly.json, metrics.json")
