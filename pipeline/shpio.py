"""외부 라이브러리 없이 shapefile(.shp/.dbf)을 읽는 최소 도구.

geopandas/shapely 없이도 파이프라인이 돌아가도록 직접 구현했다.
- read_dbf: 속성 표(읍면동 이름·법정동 코드)
- read_polygons: 폴리곤 도형(레코드별 링 목록, 좌표는 shapefile 원본 좌표계)
- tm_inverse: 국토지리정보원 UTM-K(EPSG:5179) → 경위도(WGS84와 사실상 동일한 GRS80)
"""
import struct
import numpy as np


def read_dbf(path, encodings=("cp949", "utf8")):
    with open(path, "rb") as f:
        head = f.read(32)
        n, hlen, rlen = struct.unpack("<xxxxIHH", head[:12])
        fields = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            fields.append((d[:11].split(b"\0")[0].decode(), d[16]))
        f.seek(hlen)
        recs = []
        for _ in range(n):
            r = f.read(rlen)
            o, row = 1, {}
            for name, ln in fields:
                raw = r[o:o + ln]
                o += ln
                for enc in encodings:
                    try:
                        row[name] = raw.decode(enc).strip()
                        break
                    except UnicodeDecodeError:
                        continue
            recs.append(row)
    return recs


def read_polygons(path):
    """레코드마다 [ring(N,2) ...] 목록을 돌려준다 (shape type 5/15/25)."""
    with open(path, "rb") as f:
        buf = f.read()
    pos, out = 100, []
    while pos < len(buf):
        clen = struct.unpack(">i", buf[pos + 4:pos + 8])[0] * 2
        rec = buf[pos + 8:pos + 8 + clen]
        pos += 8 + clen
        stype = struct.unpack("<i", rec[:4])[0]
        if stype == 0:
            out.append([])
            continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = np.frombuffer(rec, dtype="<i4", count=nparts, offset=44)
        pts = np.frombuffer(rec, dtype="<f8", count=npts * 2, offset=44 + 4 * nparts).reshape(-1, 2)
        bounds = list(parts) + [npts]
        out.append([pts[bounds[i]:bounds[i + 1]] for i in range(nparts)])
    return out


def ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def ring_centroid(r):
    x, y = r[:, 0], r[:, 1]
    cr = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = 0.5 * cr.sum()
    if a == 0:
        return float(x.mean()), float(y.mean()), 0.0
    cx = ((x[:-1] + x[1:]) * cr).sum() / (6 * a)
    cy = ((y[:-1] + y[1:]) * cr).sum() / (6 * a)
    return float(cx), float(cy), float(a)


# ---- UTM-K (EPSG:5179) 역변환: GRS80, 원점 38N 127.5E, k0 0.9996, FE 1,000,000 FN 2,000,000
_A, _F = 6378137.0, 1 / 298.257222101
_K0, _LAT0, _LON0, _FE, _FN = 0.9996, np.radians(38.0), np.radians(127.5), 1_000_000.0, 2_000_000.0
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)


def _meridian(phi):
    e2 = _E2
    return _A * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
                 - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * phi)
                 + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * phi)
                 - (35 * e2**3 / 3072) * np.sin(6 * phi))


def tm_inverse(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = _meridian(_LAT0) + (y - _FN) / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256))
    e1 = (1 - np.sqrt(1 - _E2)) / (1 + np.sqrt(1 - _E2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1**3 / 32) * np.sin(2 * mu)
          + (21 * e1**2 / 16 - 55 * e1**4 / 32) * np.sin(4 * mu)
          + (151 * e1**3 / 96) * np.sin(6 * mu) + (1097 * e1**4 / 512) * np.sin(8 * mu))
    c1 = _EP2 * np.cos(p1)**2
    t1 = np.tan(p1)**2
    n1 = _A / np.sqrt(1 - _E2 * np.sin(p1)**2)
    r1 = _A * (1 - _E2) / (1 - _E2 * np.sin(p1)**2)**1.5
    d = (x - _FE) / (n1 * _K0)
    lat = p1 - (n1 * np.tan(p1) / r1) * (d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * _EP2) * d**4 / 24
                                        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * _EP2 - 3 * c1**2) * d**6 / 720)
    lon = _LON0 + (d - (1 + 2 * t1 + c1) * d**3 / 6
                   + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * _EP2 + 24 * t1**2) * d**5 / 120) / np.cos(p1)
    return np.degrees(lon), np.degrees(lat)
