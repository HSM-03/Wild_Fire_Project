"""①-2 읍면별 산림·농지 비율: 국토교통부 토지임야정보(필지별 지목·면적)를 분석 단위로 합친다.

입력: raw/토지임야정보/*.csv   브이월드 '전체데이터' 시도별 CSV (cp949). 쓰는 열: 법정동코드, 법정동명, 지목명, 면적
      out/units_all.csv       01_boundaries.py 가 만든 분석 단위
출력: out/landuse_by_dong.csv  법정동(10자리)별 지목 면적 합계 — 5GB 원자료를 다시 읽지 않기 위한 중간 결과
      out/landuse.csv          단위별 전체 면적, 임야 비율(forest_ratio), 농지(전+답) 비율(farm_ratio)
      out/landuse_summary.json 매칭 요약

단위에 붙이는 방법
  1) 번호로: 읍면은 법정동코드 앞 8자리 = uid, 동은 앞 5자리 + "D" = uid
  2) 이름으로: 2026년에 번호가 바뀐 곳(전남·광주 통합 46·29→12, 화성시 구 신설, 양지면·대소면 읍 승격)과
     경계 파일에 없는 접경지역 읍면(uid에 X)은 '시도 + 시군 + 읍면 이름'으로 찾는다. 읍/면 글자는 떼고 비교한다.
비율의 분모는 경계 면적이 아니라 그 단위의 지적 면적 합계다(바다·미등록지 제외). 그래서 도형이 없는 접경지역 단위도 비율이 나온다.
토지 이용은 2026-09 한 시점 값을 전 기간에 쓴다(15년 사이 변화가 작다).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SRC = sorted((ROOT / "raw/토지임야정보").glob("*.csv"))
CACHE = OUT / "landuse_by_dong.csv"
JIMOK = {"임야": "forest", "전": "field", "답": "paddy"}
# 토지임야정보의 시도 이름 → 경계 파일(units)의 시도 이름. 통합된 곳은 후보가 여러 개
SIDO_ALIAS = {"전남광주통합특별시": ["전라남도", "광주광역시"]}
# 2026 인천 구 개편: 새 구 → 옛 구의 동 지역 (두 옛 구에 같은 동 이름이 있을 때 고르는 데 씀. 02_fires.py 의 SGG_ALIAS 참고)
NEW_GU = {"영종구": ["28110D"], "제물포구": ["28110D", "28140D"], "검단구": ["28260D"], "서해구": ["28260D"]}

if not SRC:
    print("토지임야정보 파일이 없어 건너뜀 (raw/토지임야정보/*.csv)")
    sys.exit(0)

# ---------- 1. 법정동별 합계 (원자료가 바뀌었을 때만 다시 계산, 약 3~4분)
if not CACHE.exists() or CACHE.stat().st_mtime < max(f.stat().st_mtime for f in SRC):
    parts, names, dates = [], {}, set()
    for f in SRC:
        print("읽는 중:", f.name, flush=True)
        for ch in pd.read_csv(f, encoding="cp949", dtype=str, chunksize=2_000_000,
                              usecols=["법정동코드", "법정동명", "지목명", "면적", "데이터기준일자"]):
            ch["면적"] = pd.to_numeric(ch["면적"], errors="coerce").fillna(0)
            ch["g"] = ch["지목명"].map(JIMOK).fillna("other")
            parts.append(ch.groupby(["법정동코드", "g"])["면적"].sum())
            for c, n in ch.drop_duplicates("법정동코드")[["법정동코드", "법정동명"]].values:
                names.setdefault(c, n)
            dates.update(ch["데이터기준일자"].unique())
    dong = pd.concat(parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    dong = dong.reindex(columns=["forest", "field", "paddy", "other"], fill_value=0)
    dong["total"] = dong.sum(axis=1)
    dong.insert(0, "name", pd.Series(names))
    dong.index.name = "code10"
    dong.to_csv(CACHE, encoding="utf-8-sig")
    print("기준일:", sorted(dates))
dong = pd.read_csv(CACHE, dtype={"code10": str})

# ---------- 2. 단위에 붙이기
units = pd.read_csv(OUT / "units_all.csv", dtype={"uid": str, "code5": str})
uids = set(units.uid)
si_of = dict(zip(units.uid, units.si))


def base(n):
    """'양지면' → '양지' (면→읍 승격을 같은 곳으로 보기 위해)"""
    return n[:-1] if n and n[-1] in "읍면" else n


# 이름으로 찾는 표: (시도, 시군, 읍면 이름) → uid,  (시도, 시, 구) → 동 지역 uid,  (시도, 동 이름) → 동 지역 uid
by_name, dong_by_name, dong_by_dname = {}, {}, {}
for r in units.itertuples():
    si = r.si if isinstance(r.si, str) else ""
    gu = r.gu if isinstance(r.gu, str) else ""
    if r.kind == "동":
        dong_by_name[(r.sido, si, gu)] = r.uid
        for d in (r.dongs.split("|") if isinstance(r.dongs, str) else []):
            dong_by_dname.setdefault((r.sido, d), []).append(r.uid)
    else:
        by_name[(r.sido, si, base(r.name))] = r.uid


def parse(name):
    """'경기도 화성시 효행구 봉담읍 동화리' → 시도, 시, 구, 읍면 (없으면 '')"""
    p = name.split()
    sido, rest = p[0], p[1:]
    if sido.startswith("세종"):
        rest = ["세종시"] + rest
    si = rest[0] if rest else ""
    gu = rest[1] if len(rest) > 1 and si.endswith("시") and rest[1].endswith("구") else ""
    emd = next((t for t in rest[1:] if t[-1] in "읍면"), "")
    return sido, si, gu, emd


def find_uid(code, name):
    if code[:8] in uids:
        return code[:8], "번호"
    sido, si, gu, emd = parse(name)
    if not emd and code[:5] + "D" in uids:
        return code[:5] + "D", "번호"
    for sd in SIDO_ALIAS.get(sido, [sido]):
        if emd:
            u = by_name.get((sd, si, base(emd)))
        else:
            # 시·구 이름이 그대로 있으면 그것으로 (광주 동구 → 29110D, 목포시 → 46110D)
            u = dong_by_name.get((sd, si, gu))
            if not u:   # 인천처럼 구가 쪼개지고 합쳐진 곳: 옛 구 후보 안에서 동 이름으로. 화성처럼 구가 새로 생긴 시: 같은 시 안에서
                c = [x for x in dong_by_dname.get((sd, name.split()[-1]), [])
                     if x in NEW_GU.get(si, []) or si_of[x] == si]
                u = (c[0] if len(c) == 1 else None) or dong_by_name.get((sd, si, ""))
        if u:
            return u, "이름"
    return None, "못 찾음"


dong[["uid", "how"]] = [find_uid(c, n) for c, n in zip(dong.code10, dong.name)]
agg = dong[dong.uid.notna()].groupby("uid")[["forest", "field", "paddy", "other", "total"]].sum()
land = pd.DataFrame({
    "uid": units.uid,
    "land_km2": units.uid.map(agg.total / 1e6).round(3),
    "forest_ratio": units.uid.map(agg.forest / agg.total).round(4),
    "farm_ratio": units.uid.map((agg.field + agg.paddy) / agg.total).round(4),
})
land.to_csv(OUT / "landuse.csv", index=False, encoding="utf-8-sig")

miss = dong[dong.uid.isna()]
summary = {
    "n_dong": int(len(dong)), "by_number": int((dong.how == "번호").sum()), "by_name": int((dong.how == "이름").sum()),
    "unmatched": int(len(miss)), "unmatched_area_pct": round(float(miss.total.sum() / dong.total.sum() * 100), 3),
    "unmatched_examples": miss.name.head(20).tolist(),
    "units_with_land": int(land.forest_ratio.notna().sum()), "units_total": int(len(land)),
    "units_without_land": units.loc[land.forest_ratio.isna().values, "uid"].tolist(),
}
(OUT / "landuse_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"법정동 {summary['n_dong']:,}곳: 번호로 {summary['by_number']:,}, 이름으로 {summary['by_name']:,}, "
      f"못 찾음 {summary['unmatched']} (면적 {summary['unmatched_area_pct']}%)")
print(f"토지 비율이 있는 단위 {summary['units_with_land']} / {summary['units_total']}")
if summary["unmatched"]:
    print("못 찾은 예:", summary["unmatched_examples"][:10])
