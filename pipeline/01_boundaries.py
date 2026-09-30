"""① 읍면동 경계 → 분석 단위(읍·면 + 시군구별 '동 지역') 만들기.

하는 일
1. 경계 shapefile(법정 읍면동 5,012개)을 읽는다.
2. 분석 단위를 정한다. 읍·면은 하나씩, 동(洞)은 같은 시군구끼리 '동 지역' 한 덩어리로 묶는다.
   (산불통계가 도시 지역은 "경기 시흥"처럼 시·구까지만 적혀 있어서 동 단위로는 붙일 수 없기 때문)
3. 화면에 그릴 수 있게 경계선을 가볍게 만든다(격자 맞춤 → 같은 단위 내부선 제거 → 선 단순화).
   이웃한 두 단위가 같은 선을 공유하도록(TopoJSON 방식) 처리해서 틈이 생기지 않는다.
4. 단위별 면적·중심점, 이웃 관계(경계를 맞댄 단위)를 계산한다.

출력: out/units.csv, out/adjacency.json, out/topo.json
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shpio import read_dbf, read_polygons, ring_centroid, tm_inverse
from sigungu_codes import SIGUNGU

ROOT = Path(__file__).resolve().parents[1]
SHP = ROOT / "raw/bnd/N3A_G0110000"
OUT = ROOT / "out"
GRID = 50.0          # 격자 맞춤 간격(m)
TOL = 110.0          # 선 단순화 허용 오차(m)
MIN_ISLAND = 0.25e6  # 이보다 작은 외딴 섬(㎡)은 화면용 도형에서 뺀다 (면적 계산에는 포함)

t0 = time.time()
recs = read_dbf(str(SHP) + ".dbf")
polys = read_polygons(str(SHP) + ".shp")
print(f"읽음: 읍면동 {len(recs)}개, {time.time() - t0:.1f}s")

# ---------- 1. 분석 단위 정의
TYPE = {"HJD008": "읍", "HJD009": "면", "HJD010": "동"}
unit_of_rec, units = [], {}
for r in recs:
    code5 = r["BJCD"][:5]
    kind = TYPE[r["DIVI"]]
    sg = SIGUNGU[code5]
    if kind == "동":
        uid, name = code5 + "D", "동 지역"
    else:
        uid, name = r["BJCD"][:8], r["NAME"]
    if uid not in units:
        units[uid] = {"uid": uid, "code5": code5, "kind": kind, "name": name, "sido": sg["sido"],
                      "sido_s": sg["sido_s"], "si": sg["si"], "gu": sg["gu"], "sigungu": sg["full"],
                      "dongs": [], "area": 0.0, "cx": 0.0, "cy": 0.0}
    if kind == "동":
        units[uid]["dongs"].append(r["NAME"])
    unit_of_rec.append(uid)
uids = list(units)
uidx = {u: i for i, u in enumerate(uids)}
print(f"분석 단위 {len(uids)}개 (읍 {sum(u['kind']=='읍' for u in units.values())}, "
      f"면 {sum(u['kind']=='면' for u in units.values())}, 동 지역 {sum(u['kind']=='동' for u in units.values())})")

# ---------- 2. 면적·중심점 (원본 정밀 좌표로)
for rec_i, rings in enumerate(polys):
    u = units[unit_of_rec[rec_i]]
    for ring in rings:
        cx, cy, a = ring_centroid(ring)   # shapefile 외곽 링은 시계방향(음수), 구멍은 양수
        u["area"] += -a
        u["cx"] += cx * -a
        u["cy"] += cy * -a
for u in units.values():
    u["cx"] /= u["area"]
    u["cy"] /= u["area"]
    lon, lat = tm_inverse(u["cx"], u["cy"])
    u["lon"], u["lat"] = float(lon), float(lat)
    u["area_km2"] = u["area"] / 1e6

# ---------- 3. 격자 맞춤 + 변(edge) 수집
xs = np.concatenate([r[:, 0] for p in polys for r in p])
ys = np.concatenate([r[:, 1] for p in polys for r in p])
X0, Y0 = np.floor(xs.min() / GRID) * GRID, np.floor(ys.min() / GRID) * GRID
del xs, ys
M = 1 << 15
ea, eb, eu = [], [], []
for rec_i, rings in enumerate(polys):
    ui = uidx[unit_of_rec[rec_i]]
    for ring in rings:
        ix = np.rint((ring[:, 0] - X0) / GRID).astype(np.int64)
        iy = np.rint((ring[:, 1] - Y0) / GRID).astype(np.int64)
        k = ix * M + iy
        keep = np.ones(len(k), bool)
        keep[1:] = k[1:] != k[:-1]
        k = k[keep]
        if len(k) < 3:
            continue
        a, b = k[:-1], k[1:]
        ea.append(np.minimum(a, b)); eb.append(np.maximum(a, b)); eu.append(np.full(len(a), ui, np.int64))
ea, eb, eu = np.concatenate(ea), np.concatenate(eb), np.concatenate(eu)
print(f"격자 맞춤 후 변 {len(ea):,}개")

# 같은 단위가 짝수 번 쓴 변 = 내부선(또는 접힌 선) → 제거
order = np.lexsort((eu, eb, ea))
ea, eb, eu = ea[order], eb[order], eu[order]
new = np.ones(len(ea), bool)
new[1:] = (ea[1:] != ea[:-1]) | (eb[1:] != eb[:-1]) | (eu[1:] != eu[:-1])
starts = np.flatnonzero(new)
counts = np.diff(np.append(starts, len(ea)))
odd = counts % 2 == 1
ea, eb, eu = ea[starts][odd], eb[starts][odd], eu[starts][odd]
# 변마다 소유 단위 묶기
new = np.ones(len(ea), bool)
new[1:] = (ea[1:] != ea[:-1]) | (eb[1:] != eb[:-1])
starts = np.flatnonzero(new)
ends = np.append(starts[1:], len(ea))
E_a, E_b = ea[starts], eb[starts]
E_lab = [tuple(eu[s:e]) for s, e in zip(starts, ends)]
n_owner = np.array([len(l) for l in E_lab])
print(f"남은 변 {len(E_a):,}개 (바깥선 {np.sum(n_owner==1):,}, 공유선 {np.sum(n_owner==2):,}, 3개 이상 {np.sum(n_owner>2):,})")

# ---------- 4. 이웃 관계 (공유선 길이)
def pt(k):
    return (k // M) * GRID + X0, (k % M) * GRID + Y0

ax, ay = pt(E_a); bx, by = pt(E_b)
elen = np.hypot(ax - bx, ay - by)
shared = defaultdict(float)
for i in np.flatnonzero(n_owner >= 2):
    l = E_lab[i]
    for p in range(len(l)):
        for q in range(p + 1, len(l)):
            shared[(l[p], l[q])] += elen[i]
adj = defaultdict(dict)
for (p, q), L in shared.items():
    if L >= 200:   # 200m 이상 맞닿은 경우만 이웃으로
        adj[uids[p]][uids[q]] = round(L)
        adj[uids[q]][uids[p]] = round(L)

# ---------- 5. 호(arc) 만들기: 교차점에서 교차점까지 이어진 선
lab_id, lab_list = {}, []
E_l = np.empty(len(E_a), np.int64)
for i, l in enumerate(E_lab):
    if l not in lab_id:
        lab_id[l] = len(lab_list); lab_list.append(l)
    E_l[i] = lab_id[l]
vadj = defaultdict(list)
for i in range(len(E_a)):
    vadj[int(E_a[i])].append(i)
    vadj[int(E_b[i])].append(i)


def is_junction(v):
    es = vadj[v]
    return len(es) != 2 or E_l[es[0]] != E_l[es[1]]


used = np.zeros(len(E_a), bool)
arcs = []   # (label id, [vertex keys])


def walk(v, e):
    path = [v]
    while True:
        used[e] = True
        v = int(E_b[e]) if int(E_a[e]) == v else int(E_a[e])
        path.append(v)
        if is_junction(v) or v == path[0]:
            return path
        es = vadj[v]
        e = es[0] if es[1] == e else es[1]
        if used[e]:
            return path


for v in list(vadj):
    if is_junction(v):
        for e in vadj[v]:
            if not used[e]:
                arcs.append((int(E_l[e]), walk(v, e)))
for e in range(len(E_a)):   # 교차점이 없는 닫힌 고리(섬 등)
    if not used[e]:
        arcs.append((int(E_l[e]), walk(int(E_a[e]), e)))
print(f"호 {len(arcs):,}개, 꼭짓점 {sum(len(a[1]) for a in arcs):,}개")


# ---------- 6. 선 단순화 (Douglas-Peucker)
def simplify(P, tol):
    n = len(P)
    if n <= 2:
        return P
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = P[i], P[j]
        seg = b - a
        L = np.hypot(*seg)
        mid = P[i + 1:j]
        if L == 0:
            d = np.hypot(mid[:, 0] - a[0], mid[:, 1] - a[1])
        else:
            d = np.abs(seg[0] * (mid[:, 1] - a[1]) - seg[1] * (mid[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            k += i + 1
            keep[k] = True
            stack += [(i, k), (k, j)]
    return P[keep]


out_arcs, out_lab = [], []
dropped = 0
for lid, path in arcs:
    k = np.array(path, np.int64)
    P = np.stack([k // M, k % M], 1).astype(float)   # 격자 단위
    closed = path[0] == path[-1]
    if closed:
        area = 0.5 * abs(np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1])) * GRID * GRID
        if len(lab_list[lid]) == 1 and area < MIN_ISLAND:
            dropped += 1
            continue
        # 닫힌 고리는 가장 먼 점에서 둘로 나눠 단순화
        far = int(np.argmax(np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1])))
        S = np.vstack([simplify(P[:far + 1], TOL / GRID), simplify(P[far:], TOL / GRID)[1:]])
        if len(S) < 4:
            dropped += 1
            continue
    else:
        S = simplify(P, TOL / GRID)
    out_arcs.append(S.astype(int))
    out_lab.append(lab_list[lid])
print(f"단순화 후 호 {len(out_arcs):,}개, 꼭짓점 {sum(len(a) for a in out_arcs):,}개 (작은 섬 {dropped:,}개 제외)")

# ---------- 7. 단위별 고리 조립 (호를 이어 닫힌 도형으로)
unit_arcs = defaultdict(list)
for ai, lab in enumerate(out_lab):
    for ui in lab:
        unit_arcs[ui].append(ai)
unit_rings = {}
for ui, alist in unit_arcs.items():
    ends_at = defaultdict(list)
    for ai in alist:
        A = out_arcs[ai]
        ends_at[tuple(A[0])].append((ai, False))
        ends_at[tuple(A[-1])].append((ai, True))
    done, rings = set(), []
    for ai in alist:
        if ai in done:
            continue
        ring, cur, rev = [], ai, False
        start = tuple(out_arcs[ai][0])
        while True:
            done.add(cur)
            ring.append(~cur if rev else cur)
            end = tuple(out_arcs[cur][0] if rev else out_arcs[cur][-1])
            if end == start:
                break
            nxt = [(a, r) for a, r in ends_at[end] if a not in done]
            if not nxt:
                break
            cur, at_end = nxt[0]
            rev = at_end
        rings.append(ring)
    unit_rings[ui] = rings

topo = {
    "grid": GRID, "x0": float(X0), "y0": float(Y0),
    "arcs": [],
    "arcUnits": [[int(u) for u in lab] for lab in out_lab],
    "rings": [unit_rings.get(i, []) for i in range(len(uids))],
    "uids": uids,
}
for A in out_arcs:   # 첫 점 + 차분(delta) 저장
    d = np.diff(A, axis=0)
    topo["arcs"].append([int(A[0, 0]), int(A[0, 1])] + d.reshape(-1).astype(int).tolist())
OUT.mkdir(exist_ok=True)
(OUT / "topo.json").write_text(json.dumps(topo, separators=(",", ":")), encoding="utf-8")
(OUT / "adjacency.json").write_text(json.dumps(adj, ensure_ascii=False), encoding="utf-8")
df = pd.DataFrame(units.values())
df["n_neighbors"] = df.uid.map(lambda u: len(adj.get(u, {})))
df["dongs"] = df.dongs.map(lambda d: "|".join(d))
df.drop(columns=["area"]).to_csv(OUT / "units.csv", index=False, encoding="utf-8-sig")
print(f"완료 {time.time() - t0:.1f}s · topo.json {(OUT / 'topo.json').stat().st_size / 1e6:.2f}MB · "
      f"도형 없는 단위 {sum(1 for r in topo['rings'] if not r)}개 · 이웃 없는 단위 {(df.n_neighbors == 0).sum()}개")
print(f"전체 면적 {df.area_km2.sum():,.0f}㎢")
