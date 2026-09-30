"""④ 대시보드 만들기: 분석 결과를 한 개의 HTML 파일에 넣는다.

입력: out/predictions.csv, out/monthly.json, out/fires.csv, out/topo.json, out/metrics.json, out/match_summary.json
출력: dashboard/index.html          (내 컴퓨터에서 바로 열거나 app.py 로 띄우는 완성본)
      dashboard/artifact_body.html  (Claude Artifact 게시용: doctype·head 없는 본문)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, DASH = ROOT / "out", ROOT / "dashboard"

pred = pd.read_csv(OUT / "predictions.csv", dtype={"uid": str, "code5": str, "city": str})
monthly = json.loads((OUT / "monthly.json").read_text(encoding="utf-8"))
fires = pd.read_csv(OUT / "fires.csv", dtype={"uid": str, "code5": str}, parse_dates=["date"])
topo = json.loads((OUT / "topo.json").read_text(encoding="utf-8"))
metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
match = json.loads((OUT / "match_summary.json").read_text(encoding="utf-8"))

assert list(pred.uid) == monthly["uids"]
assert list(pred.uid[:len(topo["uids"])]) == topo["uids"]      # 지도 있는 단위가 앞쪽에 같은 순서로

CAUSES = list(fires.cause.value_counts().index)                 # 원인 14종 (많은 순)
GROUPS = monthly["groups"]
cause_ix = {c: i for i, c in enumerate(CAUSES)}
group_of_cause = [GROUPS.index(fires[fires.cause == c].cause_group.iloc[0]) for c in CAUSES]

by_unit = {u: g for u, g in fires[fires.uid.notna()].sort_values("date", ascending=False).groupby("uid")}
units = []
for i, r in enumerate(pred.itertuples()):
    fl = []
    if r.uid in by_unit:
        for f in by_unit[r.uid].itertuples():
            fl.append([int(f.date.strftime("%Y%m%d")), cause_ix[f.cause], round(float(f.area_ha), 2),
                       f.ri if isinstance(f.ri, str) else "", int(f.hour) if pd.notna(f.hour) else -1])
    units.append({
        "id": r.uid, "sd": r.sido_s, "ct": r.city, "cl": r.city_label, "nm": r.label, "kd": r.kind, "sh": int(bool(r.has_shape)),
        "ar": None if pd.isna(r.area_km2) else round(float(r.area_km2), 1),
        "fo": None if pd.isna(r.forest_ratio) else round(float(r.forest_ratio) * 100),   # 임야 비율 %
        "fa": None if pd.isna(r.farm_ratio) else round(float(r.farm_ratio) * 100),       # 농지(전+답) 비율 %
        "rk": round(float(r.risk_annual), 3), "gr": int(r.grade), "ra": int(r.rank_all), "rc": int(r.rank_city), "nc": int(r.n_city),
        "cx": None if pd.isna(r.cx) else round((r.cx - topo["x0"]) / topo["grid"], 1),
        "cy": None if pd.isna(r.cy) else round((r.cy - topo["y0"]) / topo["grid"], 1),
        "cv": [round(x, 4) for x in monthly["curve"][i]],
        "f": fl,
    })

# 원인 유형별 발생 시각 (2022~2025 보충 자료가 붙은 건만): 11~17시 비중
fh = fires[fires.hour.notna()]
hours = {}
for g in GROUPS:
    s = fh[fh.cause_group == g]
    if len(s) >= 30:
        hours[g] = {"n": int(len(s)), "share_11_17": round(float(s.hour.between(11, 16).mean() * 100)),
                    "peak": int(s.hour.round().astype(int).value_counts().idxmax())}

city_unknown = fires[fires.uid.isna()]
data = {
    "meta": {
        "period": "2011~2026.9", "n_fires": int(len(fires)), "n_units": int(len(units)),
        "n_shape": int(pred.has_shape.sum()), "metrics": metrics, "match": match,
        "last_date": fires.date.max().strftime("%Y-%m-%d"), "hours": hours,
        "n_supp": int(fh.shape[0]),
        "month_share": [round(float(x), 4) for x in fires.month.value_counts(normalize=True).reindex(range(1, 13), fill_value=0)],
    },
    "causes": CAUSES, "groups": GROUPS, "groupOfCause": group_of_cause,
    "topo": {k: topo[k] for k in ("grid", "arcs", "arcUnits", "rings")},
    "units": units,
}
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
template = (DASH / "template.html").read_text(encoding="utf-8")
body = template.replace("/*__DATA__*/null", payload)
(DASH / "artifact_body.html").write_text(body.replace("/*__STANDALONE__*/false", "false"), encoding="utf-8")
full = ('<!doctype html>\n<html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        "<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style></head><body>\n"
        + body.replace("/*__STANDALONE__*/false", "true") + "\n</body></html>")
(DASH / "index.html").write_text(full, encoding="utf-8")
print(f"대시보드 저장: index.html {len(full) / 1e6:.2f}MB (단위 {len(units)}개, 산불 {sum(len(u['f']) for u in units)}건)")
