#!/usr/bin/env python3
"""단계 1-A: reference 가이드라인으로 기준(ground-truth) 데이터 생성 — GPU(torch CUDA).

원천 규칙 : screening_guideline_ko.md  →  screening_rules_v1.4.yaml
입력      : dataset/{ship}.csv  (43컬럼, UTC 포함)
산출      : dataset_baseline/{Vessel}_baseline.csv     이상치 셀 -> NaN
            dataset_baseline/{Vessel}_flags.npz        컬럼별 uint8 비트마스크
            results/outlier_baseline/{Vessel}_summary.csv
            results/outlier_baseline/calibration.csv
            results/outlier_baseline/summary_all.csv

비트마스크 (컬럼별 uint8)
    1  STATIC    정적 한계
    2  SPIKE     로컬 중앙값 편차 + 회복
    4  DELTA     단위시간당 변화율
    8  FREEZE    운항 중 값 고정 (동일값/임계이하/위치)
    16 RELATION  변수 간 정합성

실행 (프로젝트 루트):
    python code/rule_based_screening.py --vessels "Ship A" ...

GPU 는 본 파일의 GPU_ID 로 지정한다 (셸에 CUDA_VISIBLE_DEVICES 를 붙이지 않는다).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

GPU_ID = 0
DEV = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("dataset")
OUT_DATA = Path("dataset_baseline")
OUT_RES = Path("results/outlier_baseline")
RULES_PATH = Path("rules/screening_rules_v1.4.yaml")

VESSELS = [
    "Ship A", "Ship B", "Ship C", "Ship D", "Ship E", "Ship F",
    "Ship G", "Ship H", "Ship I", "Ship J", "Ship K", "Ship L",
]

# 비트
B_STATIC, B_SPIKE, B_DELTA, B_FREEZE, B_RELATION = 1, 2, 4, 8, 16
BIT_NAMES = {B_STATIC: "static", B_SPIKE: "spike", B_DELTA: "delta",
             B_FREEZE: "freeze", B_RELATION: "relation"}

# GPU unfold 청크 (행)
CHUNK = 400_000
EARTH_R = 6_371_000.0

# UTC 간격 허용 상한(분). 실측 1분 간격 98.6%, 2~5분이 대부분의 잔여.
# 이보다 큰 간격은 "진짜 결손"으로 보고 run 을 끊고 Δ 판정에서 제외한다.
MAX_DT_MIN = 5


# ---------------------------------------------------------------- GPU helpers
def to_gpu(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a).copy(), dtype=dtype, device=DEV)


def utc_to_minutes(utc: pd.Series) -> np.ndarray:
    """UTC -> 분 단위 정수. pandas 의 datetime 해상도(ns/us/s)에 의존하지 않는다."""
    return utc.to_numpy().astype("datetime64[m]").astype("int64")


def time_break(minutes: torch.Tensor, max_dt: int = MAX_DT_MIN) -> torch.Tensor:
    """UTC 간격이 max_dt 분을 넘는 지점 = run 경계 (True)."""
    brk = torch.zeros(minutes.numel(), dtype=torch.bool, device=minutes.device)
    brk[0] = True
    brk[1:] = (minutes[1:] - minutes[:-1]) > max_dt
    return brk


def rolling_median_centered(x: torch.Tensor, w: int) -> torch.Tensor:
    """중앙 정렬 rolling median (NaN 무시). w 는 홀수."""
    n = x.numel()
    pad = w // 2
    xp = F.pad(x.view(1, 1, -1), (pad, pad), mode="replicate").view(-1)
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        uf = xp[s:e + 2 * pad].unfold(0, w, 1)          # [e-s, w]
        out[s:e] = uf.nanmedian(dim=1).values
    return out


def rolling_minmax(x: torch.Tensor, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """후방(trailing) w 윈도의 min/max. NaN 은 무시(=inf/-inf 치환)."""
    n = x.numel()
    xmax_src = torch.nan_to_num(x, nan=-float("inf"))
    xmin_src = torch.nan_to_num(x, nan=float("inf"))
    pad = w - 1
    amax = torch.empty(n, dtype=x.dtype, device=x.device)
    amin = torch.empty(n, dtype=x.dtype, device=x.device)
    for src, dst, fn in ((xmax_src, amax, "max"), (xmin_src, amin, "min")):
        fill = -float("inf") if fn == "max" else float("inf")
        sp = F.pad(src.view(1, 1, -1), (pad, 0), mode="constant", value=fill).view(-1)
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            uf = sp[s:e + pad].unfold(0, w, 1)
            dst[s:e] = uf.amax(dim=1) if fn == "max" else uf.amin(dim=1)
    return amin, amax


def _group_span_minutes(gid: torch.Tensor, minutes: torch.Tensor, ngroups: int) -> torch.Tensor:
    """그룹별 (마지막 UTC - 첫 UTC + 1) = 실제 지속 '분'. 행 수가 아니다."""
    big = torch.iinfo(torch.int64).max
    mx = torch.full((ngroups,), -big, dtype=torch.int64, device=gid.device)
    mn = torch.full((ngroups,), big, dtype=torch.int64, device=gid.device)
    mx.scatter_reduce_(0, gid, minutes, reduce="amax", include_self=True)
    mn.scatter_reduce_(0, gid, minutes, reduce="amin", include_self=True)
    return mx - mn + 1


def runlen_equal(x: torch.Tensor, brk: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
    """연속 동일값 run 의 지속시간(분). NaN·시간단절에서 run 을 끊는다."""
    n = x.numel()
    nan = torch.isnan(x)
    change = torch.ones(n, dtype=torch.bool, device=x.device)
    change[1:] = (x[1:] != x[:-1]) | nan[1:] | nan[:-1] | brk[1:]
    gid = torch.cumsum(change.long(), 0) - 1
    span = _group_span_minutes(gid, minutes, int(gid[-1]) + 1)
    return span[gid]


def runlen_cond(cond: torch.Tensor, brk: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
    """cond True-run 의 지속시간(분). cond False 위치는 0."""
    n = cond.numel()
    prev = torch.zeros(n, dtype=torch.bool, device=cond.device)
    prev[1:] = cond[:-1] & ~brk[1:]
    start = cond & ~prev
    out = torch.zeros(n, dtype=torch.long, device=cond.device)
    if not bool(start.any()):
        return out
    rid = torch.cumsum(start.long(), 0) - 1          # cond 위치에서 0-based run id
    ng = int(rid[-1]) + 1
    gid_masked = torch.where(cond, rid, torch.zeros_like(rid))
    # cond=False 행이 그룹 범위를 오염시키지 않도록 cond 행만으로 span 계산
    span = _group_span_minutes(gid_masked[cond], minutes[cond], ng)
    out[cond] = span[rid[cond]]
    return out


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dp = p2 - p1
    dl = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dp / 2) ** 2 + torch.cos(p1) * torch.cos(p2) * torch.sin(dl / 2) ** 2
    return 2 * EARTH_R * torch.asin(torch.sqrt(a.clamp(0, 1)))


def circ_absdiff(x: torch.Tensor, period: float) -> torch.Tensor:
    """원형 절대차 (0/360 경계 처리). x[i] - x[i-1], 첫 원소는 NaN."""
    d = torch.full_like(x, float("nan"))
    raw = x[1:] - x[:-1]
    half = period / 2.0
    d[1:] = torch.remainder(raw + half, period) - half
    return d.abs()


def nanquantile_gpu(x: torch.Tensor, q: float) -> float:
    v = x[~torch.isnan(x)]
    if v.numel() == 0:
        return float("nan")
    # torch.quantile 은 입력 크기 제한이 있어 큰 경우 정렬 기반으로 계산
    if v.numel() > 10_000_000:
        v, _ = torch.sort(v)
        idx = min(v.numel() - 1, max(0, int(round(q * (v.numel() - 1)))))
        return float(v[idx])
    return float(torch.quantile(v.double(), q))


# ----------------------------------------------------------------- 교정 로직
def gap_cutoff(values: np.ndarray, tail_pct: float, min_gap: float):
    """분포 tail 에서 gap>=min_gap 을 찾아 cutoff 산출 (가이드 §5.2). (low, high)."""
    v = values[~np.isnan(values)]
    if v.size < 1000:
        return None, None
    lo_q = np.percentile(v, tail_pct)
    hi_q = np.percentile(v, 100.0 - tail_pct)

    cut_hi = None
    up = np.unique(v[v >= hi_q])                 # 오름차순
    if up.size >= 2:
        d = np.diff(up)
        k = np.flatnonzero(d >= min_gap)
        if k.size:
            i = k[0]                             # 정상측에서 가장 가까운 큰 틈
            cut_hi = float((up[i] + up[i + 1]) / 2.0)

    cut_lo = None
    dn = np.unique(v[v <= lo_q])[::-1]           # 내림차순(정상측 -> 극단측)
    if dn.size >= 2:
        d = np.diff(dn)                          # 음수
        k = np.flatnonzero(-d >= min_gap)
        if k.size:
            i = k[0]
            cut_lo = float((dn[i] + dn[i + 1]) / 2.0)

    return cut_lo, cut_hi


# --------------------------------------------------------------- 메인 처리
def _sog_gps_check(df: pd.DataFrame, minutes: torch.Tensor, brk: torch.Tensor,
                   cand: torch.Tensor, gc: dict) -> tuple[torch.Tensor, dict]:
    """v1.4: SOG 동일값 후보 구간을 위·경도 이동속도로 검증한다.

    cand = 기간 조건(동일값 window_min 이상)·운항 조건을 만족한 행. 동일값 run 마다
      v_gps = Σ Haversine(연속 유효 위치) / Σ 경과 분   (위치 사이 간격 1~max_step_min 분만)
    (a) 유효 위치 비율 < min_pos_coverage 또는 v_gps 계산 불가 → 이상치(위치 정보 없음)
    (b) |SOG − v_gps| > max_abs_diff_kn → 이상치(위치와 불일치)
    그 밖은 위치로 확인된 정속 항해로 보고 이상치에서 뺀다."""
    c = cand.cpu().numpy()
    info = {"gps_runs": 0, "gps_flag_nopos": 0, "gps_flag_diff": 0, "gps_cleared": 0, "gps_cleared_rows": 0}
    if not c.any():
        return cand, info
    s = df["SPEED(SOG)"].to_numpy(dtype="float64")
    la = df["Latitude (deg)"].to_numpy(dtype="float64"); lo = df["Longitude (deg)"].to_numpy(dtype="float64")
    t = minutes.cpu().numpy(); b = brk.cpu().numpy()
    n = s.size
    nan = np.isnan(s)
    ch = np.ones(n, dtype=bool)
    ch[1:] = (s[1:] != s[:-1]) | nan[1:] | nan[:-1] | b[1:]
    gid = np.cumsum(ch) - 1
    posok = np.isfinite(la) & np.isfinite(lo) & (np.abs(la) <= 90) & (np.abs(lo) <= 180)
    out = c.copy()
    idx = np.flatnonzero(c)
    gi = gid[idx]
    for r in np.split(idx, np.flatnonzero(np.diff(gi)) + 1):   # idx 는 정렬, run 별로 연속
        info["gps_runs"] += 1
        pv = r[posok[r]]
        cov = pv.size / r.size
        vg = np.nan
        if pv.size >= 2:
            dt = np.diff(t[pv]); okk = (dt >= 1) & (dt <= int(gc["max_step_min"]))
            if okk.any():
                p1, p2 = np.radians(la[pv][:-1][okk]), np.radians(la[pv][1:][okk])
                dl = np.radians(lo[pv][1:][okk] - lo[pv][:-1][okk])
                a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
                dist = 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                vg = dist.sum() / dt[okk].sum() / 60.0 / 0.514444
        if cov < float(gc["min_pos_coverage"]) or not np.isfinite(vg):
            info["gps_flag_nopos"] += 1
        elif abs(s[r[0]] - vg) > float(gc["max_abs_diff_kn"]):
            info["gps_flag_diff"] += 1
        else:
            out[r] = False; info["gps_cleared"] += 1; info["gps_cleared_rows"] += int(r.size)
    return to_gpu(out, torch.bool), info


def process_vessel(vessel: str, rules: dict, limit_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    path = DATA_DIR / f"{vessel}.csv"
    t0 = time.time()
    df = pd.read_csv(path, nrows=limit_rows)
    n = len(df)
    print(f"  load {n:,} rows x {df.shape[1]} cols  ({time.time()-t0:.1f}s)", flush=True)

    utc = pd.to_datetime(df["UTC"], format="%Y-%m-%d %H:%M:%S")
    minutes = to_gpu(utc_to_minutes(utc), torch.int64)
    brk = time_break(minutes)

    numeric_cols = [c for c in df.columns
                    if c not in ("UTC", "VESSELCODE", "ShipName", "DGPS EW", "DGPS NS")]
    for c in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 0. 선박별 단위 스케일 보정 (규칙 적용 전) ---
    scale_applied = {}
    for col, fac in (rules.get("_scale_fix", {}).get(vessel, {}) or {}).items():
        if col in df.columns:
            df[col] = df[col] * float(fac)
            scale_applied[col] = float(fac)
    if scale_applied:
        print(f"  스케일 보정 적용: {scale_applied}", flush=True)

    # --- 정규화 (삭제 아님) ---
    norm_cols = [c for c in rules.get("normalize_mod360", []) if c in df.columns]
    norm_counts = {}
    for c in norm_cols:
        before = df[c].to_numpy(dtype="float64", copy=True)
        after = np.mod(before, 360.0)
        changed = int(np.sum(~np.isnan(before) & (np.abs(after - before) > 1e-9)))
        df[c] = after
        norm_counts[c] = changed

    # --- GPU 적재 ---
    gpu: dict[str, torch.Tensor] = {}
    for c in numeric_cols:
        dt = torch.float64 if c in ("Latitude (deg)", "Longitude (deg)") else torch.float32
        gpu[c] = to_gpu(df[c].to_numpy(dtype="float64"), dt)

    u = rules["underway"]
    sog = gpu.get(u["column"])
    underway = (sog >= float(u["min_knots"])) if sog is not None else torch.zeros(n, dtype=torch.bool, device=DEV)
    underway = underway & ~torch.isnan(sog)

    flags: dict[str, torch.Tensor] = {c: torch.zeros(n, dtype=torch.uint8, device=DEV) for c in numeric_cols}
    counts: list[dict] = []
    calib: dict[str, float] = {}

    def add(col: str, mask: torch.Tensor, bit: int, rule: str):
        mask = mask & ~torch.isnan(gpu[col]) if gpu[col].dtype.is_floating_point else mask
        k = int(mask.sum())
        flags[col] |= (mask.to(torch.uint8) * bit)
        counts.append({"column": col, "axis": BIT_NAMES[bit], "rule": rule, "n_flag": k,
                       "pct": 100.0 * k / n})

    # ---------------- 1. static (v1.3: static_vessel 로 선박별 덮어쓰기) ----------------
    sv = (rules.get("static_vessel", {}) or {}).get(vessel, {}) or {}
    static_all = dict(rules.get("static", {}))
    for col, ov in sv.items():
        static_all[col] = {**static_all.get(col, {}), **ov}
    for col, spec in static_all.items():
        if col not in gpu:
            continue
        x = gpu[col]
        m = torch.zeros(n, dtype=torch.bool, device=DEV)
        if "abs_hi" in spec:
            m |= x.abs() > float(spec["abs_hi"])
        if "lo" in spec:
            m |= (x <= float(spec["lo"])) if spec.get("lo_strict") else (x < float(spec["lo"]))
        if "hi" in spec:
            m |= (x >= float(spec["hi"])) if spec.get("hi_strict") else (x > float(spec["hi"]))
        add(col, m, B_STATIC, "static_limit")

    # static_underway (v1.3, 육안 V04·V14): 운항 중(SOG>=5)에만 적용하는 정적 한계
    for col, spec in rules.get("static_underway", {}).items():
        if col not in gpu:
            continue
        x = gpu[col]
        m = torch.zeros(n, dtype=torch.bool, device=DEV)
        if "lo" in spec:
            m |= (x <= float(spec["lo"])) if spec.get("lo_strict") else (x < float(spec["lo"]))
        if "hi" in spec:
            m |= (x >= float(spec["hi"])) if spec.get("hi_strict") else (x > float(spec["hi"]))
        add(col, m & underway, B_STATIC, "static_underway")

    # static_calibrated: DEADWEIGHT / DISPLACEMENT
    sc = rules.get("static_calibrated", {})
    dwt_hi = lw_med = None
    if "DEADWEIGHT" in sc and "DEADWEIGHT" in gpu:
        dwt = gpu["DEADWEIGHT"]
        if int((~torch.isnan(dwt)).sum()) > 0:
            dwt_hi = float(torch.nanquantile(dwt[~torch.isnan(dwt)].double(), 1.0))
            calib["DEADWEIGHT_hi"] = dwt_hi
            dwt_lo = float(sc["DEADWEIGHT"].get("lo", 0.0))   # v1.3 (V09): 6만 t 미만 = 적하계산기 오류
            add("DEADWEIGHT", (dwt < dwt_lo) | (dwt > dwt_hi), B_STATIC, "static_calibrated")
    if "DISPLACEMENT" in sc and "DISPLACEMENT" in gpu and "DEADWEIGHT" in gpu:
        disp, dwt = gpu["DISPLACEMENT"], gpu["DEADWEIGHT"]
        both = ~torch.isnan(disp) & ~torch.isnan(dwt)
        if int(both.sum()) > 1000:
            lw_med = float(torch.median((disp - dwt)[both].double()))
            calib["LIGHTSHIP_median"] = lw_med
            hi = lw_med + (dwt_hi if dwt_hi is not None else 0.0)
            lo = lw_med + float(sc["DISPLACEMENT"].get("lo_offset", 0.0))   # v1.3 (V10): 경하중량+6만 미만
            calib["DISPLACEMENT_lo"] = lo
            add("DISPLACEMENT", (disp < lo) | (disp > hi), B_STATIC, "static_calibrated")

    # static_gap_cutoff: TRIM / Trim Noon
    gc = rules.get("static_gap_cutoff", {})
    for col in gc.get("columns", []):
        if col not in gpu:
            continue
        lo_c, hi_c = gap_cutoff(df[col].to_numpy(dtype="float64"),
                               float(gc["tail_pct"]), float(gc["min_gap_m"]))
        calib[f"{col}_cut_low"] = lo_c if lo_c is not None else float("nan")
        calib[f"{col}_cut_high"] = hi_c if hi_c is not None else float("nan")
        x = gpu[col]
        m = torch.zeros(n, dtype=torch.bool, device=DEV)
        if lo_c is not None:
            m |= x < lo_c
        if hi_c is not None:
            m |= x > hi_c
        add(col, m, B_STATIC, "gap_cutoff")

    # ---------------- 2. spike ----------------
    for col, spec in rules.get("spike_robust", {}).items():
        if col not in gpu:
            continue
        x = gpu[col]
        w = int(spec["window_min"])
        med = rolling_median_centered(x, w)
        dev = (x - med).abs()
        pool = dev[underway] if spec.get("require_underway") else dev
        thr = nanquantile_gpu(pool, float(spec["dev_quantile"]))
        calib[f"{col}_spike_thr"] = thr
        lag = int(spec["recovery_lag_min"])
        xm = torch.roll(x, lag)
        xp = torch.roll(x, -lag)
        xm[:lag] = float("nan")
        xp[-lag:] = float("nan")
        recov = (xm - xp).abs() < float(spec["recovery_tol_m"])
        m = (dev > thr) & recov
        if spec.get("require_underway"):
            m &= underway
        add(col, m, B_SPIKE, f"spike_P{spec['dev_quantile']}")

    # ---------------- 3. delta ----------------
    for col, spec in rules.get("delta", {}).items():
        if col not in gpu:
            continue
        x = gpu[col]
        if spec.get("circular"):
            ad = circ_absdiff(x, float(spec["period"]))
        else:
            ad = torch.full_like(x, float("nan"))
            ad[1:] = (x[1:] - x[:-1]).abs()
        dt = torch.full((n,), float("nan"), device=DEV, dtype=x.dtype)
        dt[1:] = (minutes[1:] - minutes[:-1]).to(x.dtype)
        rate = ad / dt
        # Δt 가 1~MAX_DT_MIN 분일 때만 변화율 판정 (큰 결손 구간은 의미 없음)
        dt_ok = (dt >= 1) & (dt <= MAX_DT_MIN)
        m = (rate >= float(spec["abs_rate_per_min"])) & dt_ok
        add(col, m, B_DELTA, f"delta>={spec['abs_rate_per_min']}/min")

    # ---------------- 4a. freeze_equal ----------------
    for col, spec in rules.get("freeze_equal", {}).items():
        if col not in gpu:
            continue
        rl = runlen_equal(gpu[col], brk, minutes)
        m = (rl >= int(spec["window_min"])) & underway
        rg = spec.get("resolution_guard")
        if rg:   # v1.3 (D01): 달력월별 소수점 보유 비율이 하한 미만(정수 로깅)이면 그 달의 freeze 판정을 비활성
            xv = df[col].to_numpy(dtype="float64")
            dec = (np.abs(np.round(xv * 10.0) % 10.0) >= 0.5) & ~np.isnan(xv)
            mon = utc.dt.to_period("M").astype(str).to_numpy()
            ok = np.ones(n, dtype=bool); guarded = []
            for mo in np.unique(mon):
                sel = mon == mo; nv = int((~np.isnan(xv[sel])).sum())
                if nv > 0 and dec[sel].sum() / nv < float(rg["min_decimal_frac"]):
                    ok[sel] = False; guarded.append(mo)
            m = m & to_gpu(ok, torch.bool)
            calib[f"{col}_freeze_guarded_months"] = len(guarded)
            print(f"  freeze_equal[{col}] resolution guard: {len(guarded)} months disabled ({guarded[0] if guarded else '-'}..{guarded[-1] if guarded else '-'})", flush=True)
        gc = spec.get("gps_check")
        if gc and "Latitude (deg)" in gpu and "Longitude (deg)" in gpu:
            m, info = _sog_gps_check(df, minutes, brk, m, gc)
            calib.update({f"{col}_{k}": v for k, v in info.items()})
            print(f"  freeze_equal[{col}] gps_check: {info}", flush=True)
            add(col, m, B_FREEZE, f"freeze_equal>={spec['window_min']}min&gps_check")
            continue
        add(col, m, B_FREEZE, f"freeze_equal>={spec['window_min']}min")

    # ---------------- 4b. freeze_value ----------------
    for col, spec in rules.get("freeze_value", {}).items():
        if col not in gpu:
            continue
        x = gpu[col]
        thr = float(spec["threshold"])
        cond = (x <= thr) if spec["op"] == "le" else (x >= thr)
        cond &= ~torch.isnan(x) & underway
        rl = runlen_cond(cond, brk, minutes)
        m = rl >= int(spec["window_min"])
        add(col, m, B_FREEZE, f"freeze_value {spec['op']}{thr} >={spec['window_min']}min")

    # ---------------- 4c. freeze_position ----------------
    fp = rules.get("freeze_position")
    pos_mask = None
    if fp and fp["lat_col"] in gpu and fp["lon_col"] in gpu:
        lat, lon = gpu[fp["lat_col"]], gpu[fp["lon_col"]]
        H = int(fp["horizon_min"])
        tol = int(fp["horizon_tol_min"])
        j = torch.searchsorted(minutes, minutes + H)
        ok = j < n
        jj = j.clamp(max=n - 1)
        dtj = (minutes[jj] - minutes).to(torch.int64)
        ok &= (dtj >= H) & (dtj <= H + tol)
        dist = haversine_m(lat, lon, lat[jj], lon[jj])
        m = ok & (dist <= float(fp["max_move_m"])) & ~torch.isnan(lat) & ~torch.isnan(lon)
        if fp.get("require_underway"):
            m &= underway
        pos_mask = m
        add(fp["lat_col"], m, B_FREEZE, f"freeze_position<={fp['max_move_m']:g}m/{H}min")
        add(fp["lon_col"], m, B_FREEZE, f"freeze_position<={fp['max_move_m']:g}m/{H}min")

    # ---------------- 5. relation ----------------
    rel = rules.get("relation", {})
    if "lightweight" in rel and lw_med is not None:
        sp = rel["lightweight"]
        disp, dwt = gpu["DISPLACEMENT"], gpu["DEADWEIGHT"]
        lw = disp - dwt
        tol = max(float(sp["abs_tol_t"]), float(sp["rel_tol"]) * abs(lw_med))
        calib["LW_tol"] = tol
        m = (~torch.isnan(lw)) & ((lw - lw_med).abs() > tol)
        add("DISPLACEMENT", m, B_RELATION, "lightweight")
        add("DEADWEIGHT", m, B_RELATION, "lightweight")

    if "shaft_power" in rel and all(c in gpu for c in rel["shaft_power"]["cols"]):
        sp = rel["shaft_power"]
        P, R, T = gpu["ME SHAFT POWER"], gpu["ME RPM"], gpu["ME SHAFT TORQUE"]
        pc = T * math.pi * R / 30.0
        valid = (~torch.isnan(P)) & (~torch.isnan(R)) & (~torch.isnan(T)) \
            & (R.abs() > float(sp["min_abs_rpm"])) & (pc.abs() > 1e-6)
        alpha = 1.0
        if sp.get("alpha_correction") and int(valid.sum()) > 1000:
            alpha = float(torch.median((P[valid] / pc[valid]).double()))
            if not math.isfinite(alpha) or abs(alpha) < 1e-9:
                alpha = 1.0
        calib["shaft_alpha"] = alpha
        rel_err = (alpha * pc - P).abs() / P.abs().clamp(min=1.0)
        m = valid & (rel_err > float(sp["rel_err_tol"]))
        for c in ("ME SHAFT POWER", "ME RPM", "ME SHAFT TORQUE"):
            add(c, m, B_RELATION, "shaft_power_identity")

    if "trim_consistency" in rel and all(c in gpu for c in rel["trim_consistency"]["cols"]):
        sp = rel["trim_consistency"]
        TR, A, Fw = gpu["TRIM"], gpu["AFT DRAFT"], gpu["FWD DRAFT"]
        calc = A - Fw
        valid = (~torch.isnan(TR)) & (~torch.isnan(A)) & (~torch.isnan(Fw))
        sign = 1.0
        if sp.get("auto_sign") and int(valid.sum()) > 1000:
            e1 = float(torch.nanmedian((TR[valid] - calc[valid]).abs().double()))
            e2 = float(torch.nanmedian((TR[valid] + calc[valid]).abs().double()))
            if e2 < e1:
                sign = -1.0
        calib["trim_sign"] = sign
        m = valid & ((TR - sign * calc).abs() > float(sp["abs_tol_m"]))
        for c in ("TRIM", "AFT DRAFT", "FWD DRAFT"):
            add(c, m, B_RELATION, "trim_consistency")

    if "stw_sog_consistency" in rel and all(c in gpu for c in rel["stw_sog_consistency"]["cols"]):   # v1.3 (V05)
        sp = rel["stw_sog_consistency"]
        STW, SOG = gpu[sp["cols"][0]], gpu[sp["cols"][1]]
        cond = (~torch.isnan(STW)) & (~torch.isnan(SOG)) & ((STW - SOG).abs() > float(sp["max_abs_diff_kn"])) & underway
        rl = runlen_cond(cond, brk, minutes)
        m = rl >= int(sp["window_min"])
        for c in sp.get("flag_cols", [sp["cols"][0]]):
            add(c, m, B_RELATION, f"stw_sog_consistency>{sp['max_abs_diff_kn']}kn>={sp['window_min']}min")

    # ---------------- 5b. 연료유량계 척도 결함 ----------------
    if "sfoc_consistency" in rel and all(c in gpu for c in rel["sfoc_consistency"]["cols"]):
        sp = rel["sfoc_consistency"]
        FO, P = gpu["ME FO FLOW"], gpu["ME SHAFT POWER"]
        valid = (~torch.isnan(FO)) & (~torch.isnan(P)) & (P > float(sp["min_power_kw"])) \
            & (FO > 0) & (FO <= 16000.0) & underway
        day = torch.div(minutes, 1440, rounding_mode="floor")
        dfr = pd.DataFrame({"day": day[valid].cpu().numpy(),
                            "r": (FO[valid] / P[valid]).cpu().numpy()})
        g = dfr.groupby("day").r.agg(["median", "size"])
        g = g[g["size"] >= int(sp["min_samples"])]
        lo, hi = float(sp["lo"]), float(sp["hi"])
        bad = np.sort(g.index[(g["median"] < lo) | (g["median"] > hi)].to_numpy())
        # 결함일 사이 짧은 공백(정박 등)을 메운다: 유량계 결함은 그 사이에도 지속된다
        br = int(sp.get("bridge_days", 0))
        if br > 0 and bad.size > 1:
            filled = [bad[0]]
            for a, b in zip(bad[:-1], bad[1:]):
                if 1 < b - a <= br:
                    filled.extend(range(a + 1, b))
                filled.append(b)
            bad = np.asarray(sorted(set(filled)))
        calib["sfoc_days_eval"] = int(len(g))
        calib["sfoc_days_bad"] = int(bad.size)
        calib["sfoc_daily_median"] = float(g["median"].median()) if len(g) else float("nan")
        if bad.size:
            ep = np.datetime64("1970-01-01")
            calib["sfoc_bad_first"] = str(ep + np.timedelta64(int(bad.min()), "D"))
            calib["sfoc_bad_last"] = str(ep + np.timedelta64(int(bad.max()), "D"))
            bd = torch.as_tensor(bad, dtype=torch.int64, device=DEV)
            bad_t = torch.isin(day, bd)
        else:
            bad_t = torch.zeros(n, dtype=torch.bool, device=DEV)
        for c in sp.get("flag_cols", ["ME FO FLOW"]):
            add(c, bad_t, B_RELATION, f"sfoc_consistency[{lo},{hi}]L/kWh")

    # ---------------- 7. 정박·접안 구간 (행 플래그) ----------------
    bs = rules.get("berth_segment", {})
    berth = torch.zeros(n, dtype=torch.bool, device=DEV)
    if bs and "Latitude (deg)" in gpu:
        W = int(bs["window_min"])
        lat, lon = gpu["Latitude (deg)"], gpu["Longitude (deg)"]
        low = (sog <= float(bs["sog_max_knots"])) & ~torch.isnan(sog)
        cl = torch.cumsum(low.long(), 0)
        cv = torch.cumsum((~torch.isnan(sog)).long(), 0)
        lo_idx = torch.searchsorted(minutes, minutes - (W - 1))
        prev = (lo_idx - 1).clamp(min=0)
        base_l = torch.where(lo_idx > 0, cl[prev], torch.zeros_like(cl[prev]))
        base_v = torch.where(lo_idx > 0, cv[prev], torch.zeros_like(cv[prev]))
        n_low = cl - base_l
        n_val = (cv - base_v).clamp(min=1)
        ratio = n_low.to(torch.float32) / n_val.to(torch.float32)
        # 윈도가 긴 시간단절을 넘지 않을 때만 유효
        span_ok = (minutes - minutes[lo_idx]) <= (W + 10)
        latmin, latmax = rolling_minmax(lat.to(torch.float32), W)
        lonmin, lonmax = rolling_minmax(lon.to(torch.float32), W)
        finite = torch.isfinite(latmin) & torch.isfinite(latmax) & torch.isfinite(lonmin) & torch.isfinite(lonmax)
        diam = haversine_m(latmin.double(), lonmin.double(), latmax.double(), lonmax.double())
        berth = (ratio >= float(bs["sog_ratio_min"])) & span_ok & finite \
            & (diam <= float(bs["max_diameter_m"]))
        counts.append({"column": "__berth_segment__", "axis": "segment", "rule": "berth_or_moored",
                       "n_flag": int(berth.sum()), "pct": 100.0 * int(berth.sum()) / n})

    # ---------------- 산출 ----------------
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    flags_np = {c: flags[c].cpu().numpy() for c in numeric_cols}

    out = df.copy()
    n_nan_before = int(out[numeric_cols].isna().to_numpy().sum())
    for c in numeric_cols:
        fm = flags_np[c] > 0
        if fm.any():
            out.loc[fm, c] = np.nan
    n_nan_after = int(out[numeric_cols].isna().to_numpy().sum())
    out["berth_flag"] = berth.cpu().numpy().astype(np.int8)

    out.to_csv(OUT_DATA / f"{vessel}_baseline.csv", index=False)
    np.savez_compressed(
        OUT_DATA / f"{vessel}_flags.npz",
        berth=berth.cpu().numpy().astype(np.uint8),
        **{c.replace("/", "_"): flags_np[c] for c in numeric_cols},
    )

    summ = pd.DataFrame(counts)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    summ.insert(0, "vessel", vessel)
    summ.to_csv(OUT_RES / f"{vessel}_summary.csv", index=False)

    total_flagged = int(sum((flags_np[c] > 0).sum() for c in numeric_cols))
    print(f"  rows={n:,}  이상치 셀={total_flagged:,} "
          f"(NaN {n_nan_before:,} -> {n_nan_after:,}, +{n_nan_after-n_nan_before:,})  "
          f"berth={int(berth.sum()):,} ({100.0*int(berth.sum())/n:.2f}%)  "
          f"정규화={norm_counts}  ({time.time()-t0:.1f}s)", flush=True)

    calib_row = {"vessel": vessel, "rows": n,
                 "scale_fix": ";".join(f"{k}x{v:g}" for k, v in scale_applied.items()) or "",
                 **calib,
                 "berth_rows": int(berth.sum()),
                 "flagged_cells": total_flagged,
                 "nan_before": n_nan_before, "nan_after": n_nan_after}
    for c, k in norm_counts.items():
        calib_row[f"norm360_{c}"] = k

    del gpu, flags
    torch.cuda.empty_cache()
    return summ, calib_row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessels", nargs="*", default=VESSELS)
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="디버그용: 선박별 상위 N행만 처리")
    args = ap.parse_args()

    rules = yaml.safe_load(RULES_PATH.read_text())
    # 선박별 스케일 보정 파일 로드
    sf_path = Path(rules.get("vessel_scale_fix_file", "ship_scale_fix.yaml"))
    rules["_scale_fix"] = {}
    if sf_path.exists():
        blob = yaml.safe_load(sf_path.read_text()) or {}
        rules["_scale_fix"] = blob.get("vessel_scale_fix", {}) or {}
        print(f"스케일 보정 로드: {sf_path} -> {rules['_scale_fix']}", flush=True)
    else:
        print(f"스케일 보정 파일 없음: {sf_path} (보정 없이 진행)", flush=True)
    print(f"device={DEV}  rules={RULES_PATH} v{rules['meta']['version']}", flush=True)
    if DEV.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(GPU_ID)}", flush=True)

    all_summ, all_calib = [], []
    t0 = time.time()
    for v in args.vessels:
        print(f"\n[{v}]", flush=True)
        s, c = process_vessel(v, rules, args.limit_rows)
        all_summ.append(s)
        all_calib.append(c)

    OUT_RES.mkdir(parents=True, exist_ok=True)
    pd.concat(all_summ, ignore_index=True).to_csv(OUT_RES / "summary_all.csv", index=False)
    pd.DataFrame(all_calib).to_csv(OUT_RES / "calibration.csv", index=False)
    print(f"\n완료 ({time.time()-t0:.1f}s) -> {OUT_DATA}/ , {OUT_RES}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
