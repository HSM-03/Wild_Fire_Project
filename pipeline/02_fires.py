"""② 산불통계 정리 → 분석 단위(읍·면·동 지역)에 붙이기.

하는 일
1. 산림청 엑셀(2011~2026.9)을 읽고 날짜·요일·원인·피해면적을 정리한다.
2. "경기 양평 단월", "전남광주 순천 서", "충북 청주 상당 문의"처럼 제각각인 장소 표기를
   시군구 → 읍면 순서로 해석해 경계 파일의 분석 단위에 맞춘다.
   - 바뀐 이름(예: 홍천 동면 → 영귀미면)은 ALIASES 표로,
   - 오타로 보이는 것(예: 포항 '홍해' → '흥해')은 "한 글자만 다르고 후보가 하나뿐일 때"만 고친다.
   - 경계 파일에 빠져 있는 접경지역 읍면(파주 파평면 등)은 '지도 없는 단위'로 따로 만든다.
3. 공공데이터포털 파일(2022~2025.9)의 발생 시각·진화 시각·동리를 날짜+장소로 붙인다.

출력: out/fires.csv, out/units_all.csv(지도 없는 단위 포함), out/match_log.csv, out/match_summary.json
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sigungu_codes import SIGUNGU

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
XLSX = ROOT / "raw/산불통계/산불통계 2011~2026.9.xlsx"
SUPP = ROOT / "raw/산불통계/산불통계 2022~2025.9.txt"

# 산불통계의 시도 표기 → 경계 파일의 시도 코드
SIDO_CODES = {"서울": ["11"], "부산": ["26"], "대구": ["27"], "인천": ["28"], "광주": ["29"], "대전": ["30"],
              "울산": ["31"], "세종": ["36"], "경기": ["41"], "충북": ["43"], "충남": ["44"], "전남": ["46"],
              "전남광주": ["46", "29"], "경북": ["47"], "경남": ["48"], "제주": ["50"], "강원": ["51"], "전북": ["52"]}
EXTRA_SGG = {"경북": ["27720"]}           # 군위군: 2023.7 경북 → 대구 편입
# 2026.7 인천 행정체제 개편(영종구·검단구·서해구·제물포구) → 경계 파일의 옛 구로 연결
SGG_ALIAS = {("인천", "영종"): "28110", ("인천", "제물포"): "28110", ("인천", "검단"): "28260", ("인천", "서해"): "28260"}
# 이름이 바뀐 읍면: (시군구 코드, 옛 이름) → 지금 이름.  '동'은 그 시의 '동 지역'으로 편입된 경우
ALIASES = {
    ("51720", "동"): "영귀미면", ("51800", "남"): "국토정중앙면", ("47250", "사벌"): "사벌국면",
    ("47130", "양북"): "문무대왕면", ("51750", "수주"): "무릉도원면", ("51750", "중동"): "산솔면",
    ("27720", "고로"): "삼국유사면", ("47930", "서"): "금강송면", ("47930", "원남"): "매화면",
    ("46790", "남"): "사평면", ("46790", "북"): "백아면", ("46710", "남"): "가사문학면",
    ("41670", "능서"): "세종대왕면", ("47840", "금수"): "금수강산면", ("41820", "하"): "조종면",
    ("41610", "오포"): "동", ("41610", "실촌"): "곤지암읍", ("41610", "중부"): "남한산성면",
    ("43130", "가금"): "중앙탑면", ("47830", "고령"): "대가야읍", ("47750", "부동"): "주왕산면",
    ("47900", "상리"): "효자면", ("47900", "하리"): "은풍면",
    ("47113", "홍해"): "흥해읍",   # 반복되는 오타
}
# 경계 파일(연속수치지형도)에 빠져 있는 접경지역 읍면. 산불 기록에 실제로 나온 것만 '지도 없는 단위'로 만든다.
MISSING = {
    "41480": ["문산읍", "탄현면", "파평면", "군내면", "장단면", "진동면", "진서면"],
    "41800": ["백학면", "미산면", "왕징면", "중면", "장남면"],
    "41570": ["통진읍", "월곶면", "하성면"],
    "28710": ["강화읍", "선원면", "내가면", "하점면", "양사면", "송해면", "교동면", "삼산면", "서도면"],
    "28720": ["백령면", "대청면", "연평면"],
    "51780": ["철원읍", "김화읍", "근북면", "근동면", "원동면", "원남면", "임남면"],
    "51800": ["해안면"], "51820": ["거진읍", "현내면", "수동면"],
}

CAUSE_GROUP = {   # 원인 14종 → 예방 조치가 갈리는 유형
    "입산자 실화": "입산자 실화", "논,밭두렁 태우기": "논밭·영농 소각", "농산폐기물 태우기": "논밭·영농 소각",
    "쓰레기 태우기": "쓰레기 소각", "성묘객 실화": "성묘객 실화", "담뱃불 실화": "담뱃불 실화",
    "건축물 실화": "건축물 화재 비화", "산업현장 실화": "산업현장 실화", "원인 미상": "원인 미상",
    "기타": "기타", "방화": "기타", "기상적 산불": "기타", "어린이 불장난": "기타", "군사훈련": "기타",
}


def short(name):
    """'양평군'→'양평', '중구'→'중', '세종시'→'세종'"""
    return name[:-1] if len(name) >= 2 else name


def stem(name):
    """'단월면'→'단월', '청운동'→'청운', '을지로1가'→'을지로1'"""
    return re.sub(r"(읍|면|동|가|로)$", "", name) if len(name) >= 2 else name


units = pd.read_csv(OUT / "units.csv", dtype={"uid": str, "code5": str})
em = defaultdict(dict)        # 시군구 코드 → {읍면 어간: uid}
dong_unit, has_em = {}, set()
for r in units.itertuples():
    if r.kind == "동":
        dong_unit[r.code5] = r.uid
        for d in str(r.dongs).split("|"):
            em[r.code5].setdefault(stem(d), r.uid)
            em[r.code5].setdefault(re.sub(r"\d+$", "", stem(d)), r.uid)
    else:
        has_em.add(r.code5)
        em[r.code5][stem(r.name)] = r.uid     # 읍면 이름이 동 이름보다 우선
        em[r.code5][r.name] = r.uid
em_stems = defaultdict(set)   # 오타 보정 후보: 읍면 어간만 (2글자 이상)
for r in units.itertuples():
    if r.kind != "동" and len(stem(r.name)) >= 2:
        em_stems[r.code5].add(stem(r.name))
missing_em = {c: {stem(n): n for n in names} for c, names in MISSING.items()}

sgg_keys = defaultdict(list)  # 시도 코드 → [(붙여 쓴 시군구 표기, 코드, 구 생략 여부)]
city_codes = defaultdict(list)
for code, v in SIGUNGU.items():
    si, gu = short(v["si"]), short(v["gu"]) if v["gu"] else ""
    sgg_keys[code[:2]].append((si + gu, code, False))
    if gu:
        sgg_keys[code[:2]].append((si, code, True))
    city_codes[(code[:2], v["si"])].append(code)
extra_units = {}


def one_off(a, b):
    return len(a) == len(b) >= 2 and sum(x != y for x, y in zip(a, b)) == 1


def resolve_em(code, rest):
    """시군구 안에서 읍면 찾기 → (uid, 방법) 또는 None"""
    if rest in em[code]:
        return em[code][rest], "일치"
    if (code, rest) in ALIASES:
        new = ALIASES[(code, rest)]
        return (dong_unit[code] if new == "동" else em[code][new]), "개명"
    if rest in missing_em.get(code, {}):
        name = missing_em[code][rest]
        uid = f"{code}X{name}"
        extra_units[uid] = (code, name)
        return uid, "지도없음"
    return None


def match(place):
    tok = place.replace("undefined", " ").split()
    sido, body = tok[0], "".join(tok[1:])
    if sido == "세종":
        body = "세종" + body.removeprefix("세종")
    cands = []
    for sd in SIDO_CODES[sido]:
        cands += sgg_keys[sd]
    cands += [(short(SIGUNGU[c]["si"]), c, False) for c in EXTRA_SGG.get(sido, [])]
    cands += [(k[1], c, False) for k, c in SGG_ALIAS.items() if k[0] == sido]
    hits = []
    for key, code, gu_omitted in sorted(cands, key=lambda x: -len(x[0])):
        if not body.startswith(key):
            continue
        rest = body[len(key):]
        v = SIGUNGU[code]
        r = resolve_em(code, rest) if rest else None
        if r:                                  # "가평 가평"은 가평읍 → 읍면 해석이 먼저
            hits.append((0, code, r[0], r[1]))
        elif rest == "" or rest in (short(v["gu"]) if v["gu"] else short(v["si"]), v["gu"], v["si"]):
            hits.append((1 if not gu_omitted else 3, code, dong_unit.get(code), "시군구까지만"))
        else:
            hits.append((5, code, rest, "읍면불명"))
    # 구가 있는 시에서 구를 틀리게 적은 경우: 마지막 토막을 같은 시의 다른 구에서 찾는다
    if not any(h[0] == 0 for h in hits) and hits:
        last = tok[-1]
        for sd_code in {h[1] for h in hits}:
            for c in city_codes[(sd_code[:2], SIGUNGU[sd_code]["si"])]:
                r = resolve_em(c, last)
                if r:
                    hits.append((2, c, r[0], "구보정"))
    # 오타 추정: 한 글자만 다르고 후보가 하나뿐일 때만
    if not any(h[0] <= 2 for h in hits):
        targets = set()
        for pr, code, rest, how in [h for h in hits if h[0] == 5]:
            names = {s for s in em_stems[code] | set(missing_em.get(code, {})) if one_off(s, rest)}
            targets |= {(code, resolve_em(code, s)[0]) for s in names}
        if len(targets) == 1:                  # 같은 시 전체에서 후보가 딱 하나일 때만
            code, uid = targets.pop()
            hits.append((4, code, uid, "오타보정"))
    if not hits:
        return None, None, "시군구불명"
    hits.sort(key=lambda h: h[0])
    pr, code, val, how = hits[0]
    if pr == 3 and len({h[1] for h in hits if h[0] == 3}) > 1:
        return None, None, "구불명"            # "전북 전주"처럼 구를 알 수 없음
    if pr == 5:
        if code not in has_em and code in dong_unit:   # 읍면이 없는 구: 동 이름이 달라도 그 구의 동 지역
            return code, dong_unit[code], "동지역"
        return code, None, "읍면불명"
    if how == "시군구까지만":
        return code, val, ("동지역" if val else "읍면불명")
    return code, val, how


# ---------- 1. 본체 읽기
df = pd.read_excel(XLSX)
df.columns = ["date_raw", "place", "cause", "area_ha"]
df["date"] = pd.to_datetime(df.date_raw.str[:10])
df["weekday"] = df.date_raw.str.extract(r"\((.)\)")[0]
df["year"], df["month"] = df.date.dt.year, df.date.dt.month
df["area_ha"] = pd.to_numeric(df.area_ha, errors="coerce").fillna(0.0)
df["cause_group"] = df.cause.map(CAUSE_GROUP)
assert df.cause_group.notna().all()
m = df.place.map(match)
df["code5"], df["uid"], df["match"] = m.str[0], m.str[1], m.str[2]
df["fid"] = range(len(df))

# ---------- 2. 2022~2025 보충 파일(발생 시각·진화 시각·동리) 붙이기
txt = SUPP.read_bytes().decode("cp949")
rows = []
for b in re.split(r"\n(?=\d+\.\r?\n)", txt.strip()):
    d, last = {}, None
    for ln in b.splitlines()[1:]:
        mm = re.match(r"\s{2}(\S+?):\s?(.*)$", ln)
        if mm and not ln.startswith("    "):
            last = mm.group(1); d[last] = mm.group(2).strip()
        elif ln.strip() and last:
            d[last] = (d.get(last, "") + " " + ln.strip()).strip()
    rows.append(d)
s = pd.DataFrame(rows).fillna("")
s["date"] = pd.to_datetime(s["발생일시_년"] + "-" + s["발생일시_월"] + "-" + s["발생일시_일"])
s["place"] = (s["발생장소_시도"] + " " + s["발생장소_시군구"] + " " + s["발생장소_읍면"]).str.split().str.join(" ")
sm = s.place.map(match)
s["uid"], s["code5"] = sm.str[1], sm.str[0]
s["area_ha"] = pd.to_numeric(s["피해면적_합계"], errors="coerce").fillna(0.0)
s["hour"] = pd.to_numeric(s["발생일시_시간"].str[:2], errors="coerce")
end = pd.to_datetime(s["진화종료시간_년"] + "-" + s["진화종료시간_월"] + "-" + s["진화종료시간_일"] + " " + s["진화종료시간_시간"], errors="coerce")
start = pd.to_datetime(s.date.dt.strftime("%Y-%m-%d") + " " + s["발생일시_시간"], errors="coerce")
s["duration_h"] = ((end - start).dt.total_seconds() / 3600).where(lambda x: (x > 0) & (x < 24 * 14))

df["hour"], df["ri"], df["duration_h"] = pd.NA, "", pd.NA
pool = defaultdict(list)
for r in df[df.year.between(2022, 2025)].itertuples():
    pool[(r.date, r.uid if r.uid else r.code5)].append(r.fid)
taken, n_join = set(), 0
for r in s.itertuples():
    ids = [f for f in pool.get((r.date, r.uid if r.uid else r.code5), []) if f not in taken]
    if not ids:
        continue
    best = min(ids, key=lambda f: abs(df.at[f, "area_ha"] - r.area_ha))   # 같은 날·같은 곳이 여러 건이면 피해면적이 가까운 것
    taken.add(best); n_join += 1
    df.at[best, "hour"], df.at[best, "ri"], df.at[best, "duration_h"] = r.hour, r._asdict().get("발생장소_동리", ""), r.duration_h
df["ri"] = [getattr(r, "ri") for r in df.itertuples()]

# ---------- 3. 지도 없는 단위 추가
units["has_shape"] = True
extra = []
for uid, (code, name) in extra_units.items():
    v = SIGUNGU[code]
    extra.append({"uid": uid, "code5": code, "kind": name[-1], "name": name, "sido": v["sido"], "sido_s": v["sido_s"],
                  "si": v["si"], "gu": v["gu"], "sigungu": v["full"], "dongs": "", "has_shape": False, "n_neighbors": 0})
units_all = pd.concat([units, pd.DataFrame(extra)], ignore_index=True)
units_all.to_csv(OUT / "units_all.csv", index=False, encoding="utf-8-sig")

# ---------- 4. 저장 + 요약
cols = ["fid", "date", "year", "month", "weekday", "hour", "place", "code5", "uid", "match", "cause", "cause_group",
        "area_ha", "ri", "duration_h"]
df[cols].to_csv(OUT / "fires.csv", index=False, encoding="utf-8-sig")
log = (df[df.match.isin(["개명", "구보정", "오타보정", "지도없음", "읍면불명", "시군구불명", "구불명"])]
       .groupby(["match", "place"]).agg(n=("fid", "size"), uid=("uid", "first")).reset_index()
       .merge(units_all[["uid", "sigungu", "name"]], on="uid", how="left").sort_values(["match", "n"], ascending=[True, False]))
log.to_csv(OUT / "match_log.csv", index=False, encoding="utf-8-sig")
summary = {
    "total": int(len(df)),
    "by_match": {k: int(v) for k, v in df.match.value_counts().items()},
    "unit_matched": int(df.uid.notna().sum()),
    "unit_matched_pct": round(100 * df.uid.notna().mean(), 2),
    "sgg_matched_pct": round(100 * df.code5.notna().mean(), 2),
    "no_shape_units": len(extra), "no_shape_fires": int((df.match == "지도없음").sum()),
    "supp_rows": int(len(s)), "supp_joined": int(n_join),
    "units_with_shape": int(len(units)), "units_total": int(len(units_all)),
}
(OUT / "match_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1))
for k in ["읍면불명", "시군구불명", "구불명", "오타보정", "구보정"]:
    x = log[log.match == k]
    print(f"\n[{k}] {x.n.sum()}건:", "; ".join(f"{p}→{nm}({n})" if isinstance(nm, str) else f"{p}({n})"
                                          for p, nm, n in zip(x.place, x.name, x.n)))
