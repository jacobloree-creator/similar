import streamlit as st
import pandas as pd
import os
import altair as alt
import numpy as np
import io
import zipfile

# ============================================================
# STATIC MODEL SETTINGS (no sliders for education/recert logic)
# ============================================================

BASELINE_Q_DEFAULT = 0.10  # q for quantile-centered origin z

# Recertification probability based on NOC group digit mismatch (OPTION 1: max-level mismatch)
P_RECERT_FIRST = 0.80   # if 1st digit differs
P_RECERT_THIRD = 0.35   # else if 3rd differs
P_RECERT_FOURTH = 0.10  # else if 4th differs

# Smooth ramp in z_eff for recert months
RECERT_Z0 = 0.30   # ramp start (recert=0 at/below)
RECERT_Z1 = 1.20   # ramp full strength (recert full at/above)

# Cap expected recertification months (safety valve)
RECERT_CAP_MONTHS = 18.0

# Base recertification months by education tier (applied only when tier stays the same)
# 1=university, 2=4yr college, 3=2yr college, 4=HS, 5=no education required
RECERT_BASE_MONTHS_BY_TIER = {
    1: 48.0,  # university
    2: 24.0,  # 4-year college
    3: 12.0,  # 2-year diploma
    4: 0.0,
    5: 0.0,
}

# ============================================================
# ---------- Load Data ----------
# ============================================================

@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    # Load similarity matrix (raw Euclidean distances)
    similarity_df = pd.read_excel(os.path.join(base_path, "similarity matrix_v2.xlsx"), index_col=0)
    similarity_df.index = similarity_df.index.astype(str).str.zfill(5).str.strip()
    similarity_df.columns = similarity_df.columns.astype(str).str.zfill(5).str.strip()

    # Load NOC titles
    titles_df = pd.read_excel(os.path.join(base_path, "noc title.xlsx"))
    titles_df.columns = titles_df.columns.str.strip().str.lower()
    titles_df["noc"] = titles_df["noc"].astype(str).str.zfill(5).str.strip()

    # Load monthly wages
    wages_df = pd.read_excel(os.path.join(base_path, "monthly_wages.xlsx"))
    wages_df["noc"] = wages_df["noc"].astype(str).str.zfill(5).str.strip()
    code_to_wage = dict(zip(wages_df["noc"], wages_df["monthly_wage"]))

    def _norm(series):
        return series.astype(str).str.zfill(5).str.strip()

    # ---- Load automation risk (CSV or XLSX). If missing/invalid, leave empty dict. ----
    code_to_roa = {}
    roa_csv = os.path.join(base_path, "automation_risk.csv")
    roa_xlsx = os.path.join(base_path, "automation_risk.xlsx")

    try:
        df = None
        if os.path.exists(roa_csv):
            df = pd.read_csv(roa_csv)
        elif os.path.exists(roa_xlsx):
            df = pd.read_excel(roa_xlsx)

        if df is not None and not df.empty:
            cols = {c.lower().strip(): c for c in df.columns}
            noc_col = cols.get("noc") or cols.get("code") or list(df.columns)[0]
            prob_col = (
                cols.get("roa_prob")
                or cols.get("automation_probability")
                or cols.get("probability")
                or list(df.columns)[1]
            )
            df[noc_col] = _norm(df[noc_col])
            code_to_roa = dict(zip(df[noc_col], df[prob_col]))
    except Exception:
        code_to_roa = {}

    # ---- Load job-by-province share data (optional) ----
    jobprov_df = pd.DataFrame()
    jp_csv = os.path.join(base_path, "job_province_share.csv")
    jp_xlsx = os.path.join(base_path, "job_province_share.xlsx")

    try:
        jp = None
        if os.path.exists(jp_csv):
            jp = pd.read_csv(jp_csv)
        elif os.path.exists(jp_xlsx):
            jp = pd.read_excel(jp_xlsx)

        if jp is not None and not jp.empty:
            cols = {c.lower().strip(): c for c in jp.columns}
            noc_col = cols.get("noc") or cols.get("code") or list(jp.columns)[0]
            prov_col = cols.get("province") or cols.get("prov") or list(jp.columns)[1]
            share_col = (
                cols.get("share")
                or cols.get("share_in_job_province")
                or cols.get("pct")
                or list(jp.columns)[2]
            )

            jp[noc_col] = _norm(jp[noc_col])
            jp[prov_col] = jp[prov_col].astype(str).str.strip()
            jp = jp[[noc_col, prov_col, share_col]].rename(
                columns={noc_col: "noc", prov_col: "province", share_col: "share"}
            )
            jobprov_df = jp
    except Exception:
        jobprov_df = pd.DataFrame()

    # ---- Load shortage/surplus projections (optional) ----
    # file: future_shortage.(csv|xlsx) with columns: noc, status, magnitude
    # signed balance: +magnitude for shortage; -magnitude for surplus; 0 for balanced
    code_to_balance = {}
    code_to_status = {}

    ss_csv = os.path.join(base_path, "future_shortage.csv")
    ss_xlsx = os.path.join(base_path, "future_shortage.xlsx")

    try:
        ss = None
        if os.path.exists(ss_csv):
            ss = pd.read_csv(ss_csv)
        elif os.path.exists(ss_xlsx):
            ss = pd.read_excel(ss_xlsx)

        if ss is not None and not ss.empty:
            ss.columns = ss.columns.astype(str).str.strip().str.lower()

            if {"noc", "status", "magnitude"}.issubset(set(ss.columns)):
                ss["noc"] = _norm(ss["noc"])
                ss["status"] = ss["status"].astype(str).str.strip().str.lower()
                ss["magnitude"] = pd.to_numeric(ss["magnitude"], errors="coerce")

                def _signed_balance(row):
                    stt = row["status"]
                    mag = row["magnitude"]
                    if pd.isna(mag):
                        return np.nan
                    if "short" in stt:
                        return float(mag)
                    if "surp" in stt:
                        return -float(mag)
                    if "bal" in stt or "neutral" in stt:
                        return 0.0
                    return np.nan

                ss["_balance"] = ss.apply(_signed_balance, axis=1)
                ss = ss[pd.notnull(ss["_balance"])]

                code_to_balance = dict(zip(ss["noc"], ss["_balance"]))
                for k, v in code_to_balance.items():
                    if float(v) > 0:
                        code_to_status[k] = "shortage"
                    elif float(v) < 0:
                        code_to_status[k] = "surplus"
                    else:
                        code_to_status[k] = "balanced"
    except Exception:
        code_to_balance = {}
        code_to_status = {}

    # Create mappings
    code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
    title_to_code = {v.lower(): k for k, v in code_to_title.items()}

    return (
        similarity_df,
        code_to_title,
        title_to_code,
        code_to_wage,
        code_to_roa,
        jobprov_df,
        code_to_balance,
        code_to_status,
    )


(
    similarity_df,
    code_to_title,
    title_to_code,
    code_to_wage,
    code_to_roa,
    jobprov_df,
    code_to_balance,
    code_to_status,
) = load_data()

# ============================================================
# ---------- Helper Functions ----------
# ============================================================

def noc_str(code: str) -> str:
    return str(code).zfill(5).strip()


def get_education_level(noc_code):
    """
    2nd digit of 5-digit code (index 1) is your education tier:
      1=university, 2=4yr college, 3=2yr college, 4=high school, 5=no education required
    """
    try:
        s = noc_str(noc_code)
        return int(s[1])
    except Exception:
        return None

def make_zip_bundle(files: dict) -> bytes:
    """
    files: { "filename.csv": dataframe, "notes.txt": str, ... }
    returns: zipped bytes for st.download_button
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, obj in files.items():
            if obj is None:
                continue
            if isinstance(obj, pd.DataFrame):
                z.writestr(name, obj.to_csv(index=False))
            else:
                z.writestr(name, str(obj))
    buf.seek(0)
    return buf.getvalue()
    
# --- Education-months matrix (tier upgrading only) ---
EDU_MONTHS_MATRIX = np.array([
    # dest:  1   2   3   4   5
    [   0,  0,  0,  0,  0],   # origin 1 (uni)
    [  18,  0,  0,  0,  0],   # origin 2 -> uni
    [  30, 14,  0,  0,  0],   # origin 3 -> uni / 4yr college
    [  48, 30, 18,  0,  0],   # origin 4 -> uni / 4yr / 2yr
    [  60, 36, 24, 12,  0],   # origin 5 -> uni / 4yr / 2yr / HS
], dtype=float)

EDU_MONTHS_CAP = 60.0


def expected_tier_upgrade_months(origin_code: str, dest_code: str) -> float:
    eo = get_education_level(origin_code)
    ed = get_education_level(dest_code)
    if eo is None or ed is None:
        return 0.0
    if not (1 <= eo <= 5 and 1 <= ed <= 5):
        return 0.0
    m = float(EDU_MONTHS_MATRIX[eo - 1, ed - 1])
    return float(min(m, EDU_MONTHS_CAP))


# --- NOC group digits for recertification logic ---
def group_digits(code: str):
    """
    1st, 3rd, and 4th digit denote group.
    (2nd digit is education tier in your convention.)
    """
    s = noc_str(code)
    return s[0], s[2], s[3]


def group_mismatch_flags(origin: str, dest: str):
    o1, o3, o4 = group_digits(origin)
    d1, d3, d4 = group_digits(dest)
    return {
        "diff_1st": int(o1 != d1),
        "diff_3rd": int(o3 != d3),
        "diff_4th": int(o4 != d4),
    }


def recert_probability_from_digits(origin: str, dest: str) -> float:
    """
    OPTION 1 (max-level mismatch):
      - if 1st differs => high probability
      - else if 3rd differs => medium probability
      - else if 4th differs => low probability
      - else 0
    """
    f = group_mismatch_flags(origin, dest)
    if f["diff_1st"]:
        return float(P_RECERT_FIRST)
    if f["diff_3rd"]:
        return float(P_RECERT_THIRD)
    if f["diff_4th"]:
        return float(P_RECERT_FOURTH)
    return 0.0


def recert_ramp(z_eff: float) -> float:
    """
    Smooth linear ramp:
      0 at z<=RECERT_Z0
      1 at z>=RECERT_Z1
      linear in between
    """
    z = float(z_eff)
    if z <= float(RECERT_Z0):
        return 0.0
    if z >= float(RECERT_Z1):
        return 1.0
    denom = float(RECERT_Z1) - float(RECERT_Z0)
    if denom <= 0:
        return 1.0
    return float((z - float(RECERT_Z0)) / denom)


def expected_recert_months_same_tier(origin: str, dest: str, z_eff: float) -> float:
    """
    Same-tier only (1->1, 2->2, 3->3, ...):
      min(CAP, base_months[tier] × P(digit mismatch) × ramp(z_eff))
    """
    eo = get_education_level(origin)
    ed = get_education_level(dest)
    if eo is None or ed is None:
        return 0.0
    if eo != ed:
        return 0.0

    base_months = float(RECERT_BASE_MONTHS_BY_TIER.get(eo, 0.0))
    if base_months <= 0:
        return 0.0

    p = recert_probability_from_digits(origin, dest)
    r = recert_ramp(z_eff)
    m = base_months * p * r
    return float(min(float(RECERT_CAP_MONTHS), max(0.0, m)))


# --- Per-origin quantile-centered standardization ---
def origin_standardized_z(origin_code: str, dest_code: str, q: float):
    """
    z_q = (d_od - Q_q(o)) / std_o
    """
    if origin_code not in similarity_df.index or dest_code not in similarity_df.columns:
        return None

    row = similarity_df.loc[origin_code].dropna()
    if row.empty or dest_code not in row.index:
        return None

    std = float(row.std())
    if std == 0.0 or np.isnan(std):
        return None

    baseline = float(row.quantile(float(q)))
    d = float(row.loc[dest_code])
    return float((d - baseline) / std)


def training_multiplier(z_eff):
    z = float(z_eff)
    if z < 0.5:
        return 1.0
    elif z < 1.0:
        return 1.2
    elif z < 1.5:
        return 1.5
    else:
        return 2.0


def geographic_cost(dest_code, province, C_move=20000.0):
    if province is None:
        return 0.0
    if jobprov_df is None or jobprov_df.empty:
        return 0.0

    sub = jobprov_df[jobprov_df["noc"] == dest_code]
    if sub.empty:
        return 0.0

    a_p0 = float(sub.loc[sub["province"] == province, "share"].sum())
    a_max = float(sub["share"].max())
    if a_max <= 0:
        return 0.0

    p_move = 1.0 - a_p0 / a_max
    p_move = max(0.0, min(1.0, p_move))
    return float(C_move) * p_move


def compare_two_jobs(code1, code2):
    if code1 not in similarity_df.index or code2 not in similarity_df.index:
        return None
    scores = similarity_df.loc[code1].drop(code1).dropna()
    scores = scores[scores != 0]
    scores = scores.sort_values()
    if code2 not in scores.index:
        return None
    rank = scores.index.get_loc(code2) + 1
    total = len(scores)
    score = similarity_df.loc[code1, code2]
    if pd.isna(score):
        score = similarity_df.loc[code2, code1]
    if pd.isna(score):
        return None
    return score, rank, total


def get_most_and_least_similar(code, n=5):
    if code not in similarity_df.index:
        return None, None, None
    origin_level = get_education_level(code)

    scores = similarity_df.loc[code].drop(code).dropna()

    # Restrict by education distance (output restriction only)
    allowed = []
    for c in scores.index:
        lev = get_education_level(c)
        if origin_level is not None and lev is not None:
            if abs(lev - origin_level) <= EDU_GAP:
                allowed.append(c)
    scores = scores.loc[allowed] if allowed else scores.iloc[0:0]

    if scores.empty:
        return [], [], scores

    top_matches = scores.nsmallest(n)
    bottom_matches = scores.nlargest(n)
    top_results = [(occ, code_to_title.get(occ, "Unknown Title"), score) for occ, score in top_matches.items()]
    bottom_results = [(occ, code_to_title.get(occ, "Unknown Title"), score) for occ, score in bottom_matches.items()]
    return top_results, bottom_results, scores


def payback_period_months(cost_usd: float, w_origin: float, w_dest: float):
    """
    Months/years needed for destination wage premium to recoup switching cost.
    If wage gap <= 0, payback is None (no payback via wages).
    """
    try:
        C = float(cost_usd)
        wo = float(w_origin)
        wd = float(w_dest)
    except Exception:
        return {
            "wage_gap_monthly": np.nan,
            "payback_months": None,
            "payback_years": None,
            "status": "Missing/invalid wage or cost inputs.",
        }

    gap = wd - wo

    if C is None or np.isnan(C):
        return {
            "wage_gap_monthly": gap,
            "payback_months": None,
            "payback_years": None,
            "status": "Switching cost unavailable.",
        }

    if C <= 0:
        return {
            "wage_gap_monthly": gap,
            "payback_months": 0.0,
            "payback_years": 0.0,
            "status": "No switching cost to recoup.",
        }

    if gap <= 0:
        return {
            "wage_gap_monthly": gap,
            "payback_months": None,
            "payback_years": None,
            "status": "No wage payback (destination wage not higher).",
        }

    m = C / gap
    return {
        "wage_gap_monthly": gap,
        "payback_months": float(m),
        "payback_years": float(m / 12.0),
        "status": "Payback computed from wage gap.",
    }

# ============================================================
# Calibration (k) — BEFORE adding education effects
# ============================================================

@st.cache_data(show_spinner=False)
def compute_calibration_k_cached(
    risky_codes_tuple,
    safe_codes_tuple,
    target_usd=24000.0,
    beta=0.14,
    alpha=1.2,
    q=0.10,
):
    """
    IMPORTANT: Calibration uses ONLY the 2-month baseline (no tier-upgrade months and no recert months).
    """
    risky_codes = list(risky_codes_tuple)
    safe_codes = list(safe_codes_tuple)

    pairs = []
    for r in risky_codes:
        w_origin = code_to_wage.get(r)
        if w_origin is None:
            continue
        w_origin = float(w_origin)
        if w_origin <= 0:
            continue

        for s in safe_codes:
            if s == r:
                continue

            z_raw = origin_standardized_z(r, s, q=q)
            if z_raw is None:
                continue

            z_eff = max(float(z_raw), 0.0)
            base = 2.0 * w_origin
            dist_term = 1 + float(beta) * (z_eff ** float(alpha))
            mult = float(training_multiplier(z_eff))
            raw_cost = base * dist_term * mult
            pairs.append(raw_cost)

    if not pairs:
        return 1.0, 0

    mean_raw = float(np.mean(pairs))
    if mean_raw <= 0:
        return 1.0, 0

    return float(target_usd) / mean_raw, len(pairs)

# ============================================================
# Switching cost functions (tier-upgrade + recert same-tier with ramp+cap)
# ============================================================

def calculate_switching_cost(code1, code2, beta=0.14, alpha=1.2, q=0.10):
    level1 = get_education_level(code1)
    level2 = get_education_level(code2)
    if level1 is None or level2 is None:
        return None
    if abs(level1 - level2) > EDU_GAP:
        return None

    z_raw = origin_standardized_z(code1, code2, q=q)
    if z_raw is None:
        return None

    z_eff = max(float(z_raw), 0.0)

    w_origin = code_to_wage.get(code1)
    if w_origin is None or float(w_origin) <= 0:
        return None
    w_origin = float(w_origin)

    tier_months = expected_tier_upgrade_months(code1, code2)
    recert_months = expected_recert_months_same_tier(code1, code2, z_eff)

    # This is the total "months until prepared" additive months term in the model
    prep_months = float(2.0 + float(tier_months) + float(recert_months))

    base = prep_months * w_origin
    dist_term = 1 + float(beta) * (z_eff ** float(alpha))
    mult = float(training_multiplier(z_eff))

    skill_cost = float(CALIB_K) * base * dist_term * mult

    if USE_GEO:
        geo = geographic_cost(code2, USER_PROVINCE, GEO_C_MOVE)
        return float(skill_cost) + float(GEO_LAMBDA) * float(geo)
    else:
        return float(skill_cost)


def switching_cost_components(origin_code, dest_code, beta=0.14, alpha=1.2, q=0.10):
    level1 = get_education_level(origin_code)
    level2 = get_education_level(dest_code)
    if level1 is None or level2 is None:
        return None
    if abs(level1 - level2) > EDU_GAP:
        return None

    z_raw = origin_standardized_z(origin_code, dest_code, q=q)
    if z_raw is None:
        return None

    z_eff = max(float(z_raw), 0.0)

    w_origin = code_to_wage.get(origin_code)
    if w_origin is None or float(w_origin) <= 0:
        return None
    w_origin = float(w_origin)

    w_dest = code_to_wage.get(dest_code)
    w_dest = float(w_dest) if (w_dest is not None and float(w_dest) > 0) else np.nan

    tier_months = float(expected_tier_upgrade_months(origin_code, dest_code))
    recert_months = float(expected_recert_months_same_tier(origin_code, dest_code, z_eff))
    prep_months = float(2.0 + tier_months + recert_months)

    base = prep_months * w_origin
    dist_term = 1 + float(beta) * (z_eff ** float(alpha))
    mult = float(training_multiplier(z_eff))

    skill_uncalibrated = base * dist_term * mult
    skill_cost = float(CALIB_K) * skill_uncalibrated

    geo_raw = 0.0
    geo_add = 0.0
    if USE_GEO:
        geo_raw = float(geographic_cost(dest_code, USER_PROVINCE, GEO_C_MOVE))
        geo_add = float(GEO_LAMBDA) * geo_raw

    total = float(skill_cost) + float(geo_add)

    months_equiv = total / w_origin
    years_equiv = total / (12.0 * w_origin)

    # Payback period
    pb = None
    if pd.notnull(w_dest) and float(w_origin) > 0:
        pb = payback_period_months(total, w_origin, w_dest)

    # Diagnostics for recert model
    eo = get_education_level(origin_code)
    ed = get_education_level(dest_code)
    f = group_mismatch_flags(origin_code, dest_code)
    same_tier = (eo is not None and ed is not None and eo == ed)
    base_recert = float(RECERT_BASE_MONTHS_BY_TIER.get(eo, 0.0)) if same_tier else 0.0
    p_digit = recert_probability_from_digits(origin_code, dest_code) if (same_tier and base_recert > 0) else 0.0
    ramp = recert_ramp(z_eff) if p_digit > 0 else 0.0

    return {
        "Origin": origin_code,
        "Destination": dest_code,
        "Title": code_to_title.get(dest_code, "Unknown Title"),

        "Origin wage ($/mo)": float(w_origin),
        "Destination wage ($/mo)": float(w_dest) if pd.notnull(w_dest) else np.nan,
        "Wage gap ($/mo)": float(w_dest - w_origin) if pd.notnull(w_dest) else np.nan,

        "Baseline quantile q": float(q),
        "z_raw (quantile-centered)": float(z_raw),
        "z_eff (hinged for cost)": float(z_eff),

        "Education tier (origin)": int(eo) if eo is not None else None,
        "Education tier (dest)": int(ed) if ed is not None else None,

        "Tier upgrade months": float(tier_months),
        "Recert base months (same-tier)": float(base_recert),
        "P(recert | digits)": float(p_digit),
        "Recert ramp(z_eff)": float(ramp),
        "Recert cap (months)": float(RECERT_CAP_MONTHS),
        "Group diff (1st)": int(f["diff_1st"]),
        "Group diff (3rd)": int(f["diff_3rd"]),
        "Group diff (4th)": int(f["diff_4th"]),
        "Recert months (expected)": float(recert_months),

        "Preparation months (2 + tier + recert)": float(prep_months),

        "Base (prep_months × origin wage)": float(base),
        "Distance term": float(dist_term),
        "Training mult": float(mult),

        "k (calibration)": float(CALIB_K),
        "Skill cost": float(skill_cost),

        "Geo raw": float(geo_raw),
        "λ (geo weight)": float(GEO_LAMBDA) if USE_GEO else 0.0,
        "Geo add": float(geo_add),

        "Total cost": float(total),
        "Months of origin wages": float(months_equiv),
        "Years of origin wages": float(years_equiv),

        "Payback months (wage gap)": (float(pb["payback_months"]) if (pb and pb["payback_months"] is not None) else np.nan),
        "Payback years (wage gap)": (float(pb["payback_years"]) if (pb and pb["payback_years"] is not None) else np.nan),
        "Payback status": (pb["status"] if pb else "Payback not available."),
    }


def compute_switching_costs_from_origin(origin_code, beta, alpha, q):
    rows = []
    origin_level = get_education_level(origin_code)
    w_origin = code_to_wage.get(origin_code)

    for dest in similarity_df.columns:
        if dest == origin_code:
            continue
        lev = get_education_level(dest)
        if origin_level is None or lev is None:
            continue
        if abs(lev - origin_level) > EDU_GAP:
            continue

        cost = calculate_switching_cost(origin_code, dest, beta=beta, alpha=alpha, q=q)
        if pd.notnull(cost):
            years = np.nan
            if w_origin is not None and float(w_origin) > 0:
                years = float(cost) / (12.0 * float(w_origin))
            rows.append({
                "code": dest,
                "title": code_to_title.get(dest, "Unknown Title"),
                "cost": float(cost),
                "years": float(years) if pd.notnull(years) else np.nan,
            })
    return pd.DataFrame(rows)

# ---- Histograms with tooltips listing occupations per bin ----
def similarity_hist_with_titles(all_scores, maxbins=30, max_titles=50):
    if all_scores is None or len(all_scores) == 0:
        hist_df = pd.DataFrame({"score": (all_scores.values if all_scores is not None else [])})
        return (
            alt.Chart(hist_df)
            .mark_bar(opacity=0.7, color="steelblue")
            .encode(
                alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Similarity Score (Euclidean distance)"),
                alt.Y("count()", title="Number of Occupations"),
                tooltip=["count()"],
            )
            .properties(width=600, height=400)
        )

    df = pd.DataFrame({"code": all_scores.index, "score": all_scores.values})
    df["title"] = df["code"].map(lambda c: code_to_title.get(c, "Unknown Title"))
    df["label"] = df["code"] + " – " + df["title"]

    edges = np.histogram_bin_edges(df["score"], bins=maxbins)
    df["bin_interval"] = pd.cut(df["score"], bins=edges, include_lowest=True)

    rows = []
    for iv, g in df.groupby("bin_interval"):
        if iv is None or g.empty:
            continue
        labels = g["label"].tolist()
        extra = ""
        if len(labels) > max_titles:
            extra = f"\n... (+{len(labels) - max_titles} more)"
            labels = labels[:max_titles]
        titles_str = "\n".join(labels) + extra

        rows.append({
            "bin_start": float(iv.left),
            "bin_end": float(iv.right),
            "count": int(len(g)),
            "titles_str": titles_str,
        })

    bins_df = pd.DataFrame(rows)
    if bins_df.empty:
        return (
            alt.Chart(df)
            .mark_bar(opacity=0.7, color="steelblue")
            .encode(
                alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Similarity Score (Euclidean distance)"),
                alt.Y("count()", title="Number of Occupations"),
            )
            .properties(width=600, height=400)
        )

    return (
        alt.Chart(bins_df)
        .mark_bar(opacity=0.7, color="steelblue")
        .encode(
            x=alt.X("bin_start:Q", bin=alt.Bin(binned=True), title="Similarity Score (Euclidean distance)"),
            x2="bin_end:Q",
            y=alt.Y("count:Q", title="Number of Occupations"),
            tooltip=[
                alt.Tooltip("count:Q", title="Number of occupations"),
                alt.Tooltip("bin_start:Q", format=".2f", title="Score from"),
                alt.Tooltip("bin_end:Q", format=".2f", title="Score to"),
                alt.Tooltip("titles_str:N", title="Occupations in this bin"),
            ],
        )
        .properties(width=600, height=400)
    )


def cost_hist_with_titles(cost_df, value_col, x_title, fmt_start, fmt_end, maxbins=30, max_titles=50):
    if cost_df is None or cost_df.empty or value_col not in cost_df.columns:
        return (
            alt.Chart(pd.DataFrame({value_col: [0]}))
            .mark_bar()
            .encode(alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=1), title=x_title))
            .properties(width=600, height=400)
        )

    df = cost_df.copy()
    df = df[pd.notnull(df[value_col])]
    if df.empty:
        return (
            alt.Chart(pd.DataFrame({value_col: [0]}))
            .mark_bar()
            .encode(alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=1), title=x_title))
            .properties(width=600, height=400)
        )

    df["label"] = df["code"] + " – " + df["title"]

    edges = np.histogram_bin_edges(df[value_col], bins=maxbins)
    df["bin_interval"] = pd.cut(df[value_col], bins=edges, include_lowest=True)

    rows = []
    for iv, g in df.groupby("bin_interval"):
        if iv is None or g.empty:
            continue
        labels = g["label"].tolist()
        extra = ""
        if len(labels) > max_titles:
            extra = f"\n... (+{len(labels) - max_titles} more)"
            labels = labels[:max_titles]
        titles_str = "\n".join(labels) + extra

        rows.append({
            "bin_start": float(iv.left),
            "bin_end": float(iv.right),
            "count": int(len(g)),
            "titles_str": titles_str,
        })

    bins_df = pd.DataFrame(rows)
    if bins_df.empty:
        return (
            alt.Chart(df)
            .mark_bar(opacity=0.7, color="seagreen")
            .encode(
                alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=30), title=x_title),
                alt.Y("count()", title="Number of Occupations"),
            )
            .properties(width=600, height=400)
        )

    return (
        alt.Chart(bins_df)
        .mark_bar(opacity=0.7, color="seagreen")
        .encode(
            x=alt.X("bin_start:Q", bin=alt.Bin(binned=True), title=x_title),
            x2="bin_end:Q",
            y=alt.Y("count:Q", title="Number of Occupations"),
            tooltip=[
                alt.Tooltip("count:Q", title="Number of occupations"),
                alt.Tooltip("bin_start:Q", format=fmt_start, title="From"),
                alt.Tooltip("bin_end:Q", format=fmt_end, title="To"),
                alt.Tooltip("titles_str:N", title="Occupations in this bin"),
            ],
        )
        .properties(width=600, height=400)
    )

# ============================================================
# Allocation simulation: surplus -> shortage greedy fill
# Uses cost model, and reports:
#   - weighted-average switching cost, wages
#   - weighted-average preparation months (baseline 2 + tier + recert)
# ============================================================

def allocate_surplus_to_shortage(
    origin_code: str,
    beta: float,
    alpha: float,
    q: float,
    shortages_balance: dict,
    surplus_amount: float,
    only_wage_gain: bool = False,
):
    """
    Greedy allocation from ONE surplus origin to shortage destinations by lowest switching cost.
    Updates shortages_balance in-place (remaining shortage capacity).

    Returns:
      flows_df: per-destination flows and weighted totals
      summary: dict with totals and weighted averages (including avg preparation months)
    """
    origin_code = noc_str(origin_code)
    S = float(surplus_amount)

    w_origin = code_to_wage.get(origin_code)
    w_origin = float(w_origin) if (w_origin is not None and float(w_origin) > 0) else np.nan

    candidates = []
    for d, H in shortages_balance.items():
        if H is None:
            continue
        H = float(H)
        if H <= 0:
            continue

        w_dest = code_to_wage.get(d)
        w_dest = float(w_dest) if (w_dest is not None and float(w_dest) > 0) else np.nan
        if only_wage_gain and (pd.isna(w_origin) or pd.isna(w_dest) or w_dest <= w_origin):
            continue

        cost = calculate_switching_cost(origin_code, d, beta=beta, alpha=alpha, q=q)
        if cost is None or pd.isna(cost):
            continue

        candidates.append((d, float(cost), H, w_dest))

    # Sort by lowest cost first
    candidates.sort(key=lambda x: x[1])

    rows = []
    total_moved = 0.0
    total_cost = 0.0
    total_dest_wage = 0.0
    total_origin_wage = 0.0
    total_prep_months = 0.0

    for d, cost, H, w_dest in candidates:
        if S <= 0:
            break

        x = min(S, H)
        if x <= 0:
            continue

        # preparation months (baseline 2 + tier upgrade + expected recert)
        tier_months = float(expected_tier_upgrade_months(origin_code, d))
        z_raw = origin_standardized_z(origin_code, d, q=q)
        z_eff = max(float(z_raw), 0.0) if (z_raw is not None and not np.isnan(z_raw)) else 0.0
        recert_months = float(expected_recert_months_same_tier(origin_code, d, z_eff))
        prep_months = float(2.0 + tier_months + recert_months)

        # update balances
        S -= x
        shortages_balance[d] = H - x

        # totals
        total_moved += x
        total_cost += x * cost
        total_prep_months += x * prep_months
        if pd.notna(w_dest):
            total_dest_wage += x * w_dest
        if pd.notna(w_origin):
            total_origin_wage += x * w_origin

        rows.append({
            "Origin": origin_code,
            "Origin title": code_to_title.get(origin_code, "Unknown Title"),
            "Destination": d,
            "Destination title": code_to_title.get(d, "Unknown Title"),
            "Shortage magnitude (dest)": float(code_to_balance.get(d, np.nan)),
            "Flow moved": x,
            "Switching cost per worker ($)": cost,
            "Flow-weighted switching cost ($)": x * cost,
            "Origin wage ($/mo)": w_origin,
            "Destination wage ($/mo)": w_dest,
            "Wage gap ($/mo)": (w_dest - w_origin) if (pd.notna(w_dest) and pd.notna(w_origin)) else np.nan,
            "Tier upgrade months": tier_months,
            "Recert months (expected)": recert_months,
            "Preparation months (expected)": prep_months,
        })

    flows_df = pd.DataFrame(rows)

    avg_cost = (total_cost / total_moved) if total_moved > 0 else np.nan
    avg_dest_wage = (total_dest_wage / total_moved) if total_moved > 0 else np.nan
    avg_origin_wage = (total_origin_wage / total_moved) if total_moved > 0 else np.nan
    avg_prep_months = (total_prep_months / total_moved) if total_moved > 0 else np.nan

    summary = {
        "origin": origin_code,
        "origin_title": code_to_title.get(origin_code, "Unknown Title"),
        "surplus_input": float(surplus_amount),
        "moved_total": float(total_moved),
        "unallocated_surplus": float(S),
        "total_switching_cost": float(total_cost),
        "avg_switching_cost_per_worker": float(avg_cost) if pd.notna(avg_cost) else np.nan,
        "avg_origin_wage": float(avg_origin_wage) if pd.notna(avg_origin_wage) else np.nan,
        "avg_destination_wage": float(avg_dest_wage) if pd.notna(avg_dest_wage) else np.nan,
        "avg_wage_change": float(avg_dest_wage - avg_origin_wage) if (pd.notna(avg_dest_wage) and pd.notna(avg_origin_wage)) else np.nan,
        "avg_preparation_months": float(avg_prep_months) if pd.notna(avg_prep_months) else np.nan,
    }

    return flows_df, summary


def simulate_all_surplus_to_shortage(
    beta: float,
    alpha: float,
    q: float,
    only_wage_gain: bool = False,
    origin_order: str = "Largest surplus first",
):
    """
    Greedy-by-origin simulation across all surplus occupations.
    Updates a shared shortage balance dict as each origin is allocated.
    NOTE: Order matters (greedy, not global optimization).
    """
    if not code_to_balance:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    # Remaining shortage capacity (positive)
    shortages_balance = {
        noc: float(bal)
        for noc, bal in code_to_balance.items()
        if bal is not None and not (isinstance(bal, float) and np.isnan(bal)) and float(bal) > 0
    }

    surplus_items = []
    for noc, bal in code_to_balance.items():
        if bal is None or (isinstance(bal, float) and np.isnan(bal)):
            continue
        if float(bal) < 0:
            surplus_items.append((noc, abs(float(bal))))

    if not surplus_items or not shortages_balance:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    if origin_order == "Largest surplus first":
        surplus_items.sort(key=lambda x: x[1], reverse=True)
    else:
        surplus_items.sort(key=lambda x: x[1], reverse=False)

    all_flows = []
    origin_summaries = []

    total_moved = 0.0
    total_cost = 0.0
    total_dest_wage = 0.0
    total_origin_wage = 0.0
    total_prep_months = 0.0

    for origin_code, surplus_amount in surplus_items:
        if sum(shortages_balance.values()) <= 0:
            break

        flows_df, summary = allocate_surplus_to_shortage(
            origin_code=origin_code,
            beta=beta,
            alpha=alpha,
            q=q,
            shortages_balance=shortages_balance,
            surplus_amount=surplus_amount,
            only_wage_gain=only_wage_gain,
        )

        origin_summaries.append(summary)

        if not flows_df.empty:
            all_flows.append(flows_df)

            total_moved += float(flows_df["Flow moved"].sum())
            total_cost += float(flows_df["Flow-weighted switching cost ($)"].sum())

            flow = flows_df["Flow moved"]
            if "Destination wage ($/mo)" in flows_df.columns:
                total_dest_wage += float((flows_df["Destination wage ($/mo)"] * flow).dropna().sum())
            if "Origin wage ($/mo)" in flows_df.columns:
                total_origin_wage += float((flows_df["Origin wage ($/mo)"] * flow).dropna().sum())
            if "Preparation months (expected)" in flows_df.columns:
                total_prep_months += float((flows_df["Preparation months (expected)"] * flow).dropna().sum())

    flows_all = pd.concat(all_flows, ignore_index=True) if all_flows else pd.DataFrame()
    origin_summary_df = pd.DataFrame(origin_summaries)

    avg_cost = (total_cost / total_moved) if total_moved > 0 else np.nan
    avg_dest_wage = (total_dest_wage / total_moved) if total_moved > 0 else np.nan
    avg_origin_wage = (total_origin_wage / total_moved) if total_moved > 0 else np.nan
    avg_prep_months = (total_prep_months / total_moved) if total_moved > 0 else np.nan

    totals = {
        "total_moved": float(total_moved),
        "total_switching_cost": float(total_cost),
        "avg_switching_cost_per_worker": float(avg_cost) if pd.notna(avg_cost) else np.nan,
        "avg_origin_wage": float(avg_origin_wage) if pd.notna(avg_origin_wage) else np.nan,
        "avg_destination_wage": float(avg_dest_wage) if pd.notna(avg_dest_wage) else np.nan,
        "avg_wage_change": float(avg_dest_wage - avg_origin_wage) if (pd.notna(avg_dest_wage) and pd.notna(avg_origin_wage)) else np.nan,
        "avg_preparation_months": float(avg_prep_months) if pd.notna(avg_prep_months) else np.nan,
        "shortage_remaining_total": float(sum(shortages_balance.values())) if shortages_balance else 0.0,
    }

    shortage_remaining_df = pd.DataFrame(
        [{"Destination": k, "Remaining shortage": v, "Title": code_to_title.get(k, "Unknown Title")}
         for k, v in shortages_balance.items() if v > 0]
    ).sort_values("Remaining shortage", ascending=False)

    return flows_all, totals, origin_summary_df, shortage_remaining_df


# ============================================================
# ---------- Streamlit App ----------
# ============================================================

st.set_page_config(page_title="APOLLO", layout="wide")
st.title("Welcome to the Analysis Platform for Occupational Linkages and Labour Outcomes (APOLLO)")

# Sidebar parameters
st.sidebar.subheader("Switching Cost Parameters")
beta = st.sidebar.slider("Skill distance scaling (beta)", min_value=0.0, max_value=0.5, value=0.14, step=0.01)
alpha = st.sidebar.slider("Non-linear exponent (alpha)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

q = st.sidebar.slider(
    "Origin baseline quantile q (Q_q(o) => z=0)",
    min_value=0.0,
    max_value=0.5,
    value=float(BASELINE_Q_DEFAULT),
    step=0.05,
)

EDU_GAP = st.sidebar.slider(
    "Max education distance allowed (0 = same level only)",
    min_value=0,
    max_value=4,
    value=0,
    step=1,
)

USE_GEO = st.sidebar.checkbox("Include geographic mobility cost", value=False)

USER_PROVINCE = None
GEO_C_MOVE = 0.0
GEO_LAMBDA = 0.0

if USE_GEO:
    if jobprov_df is not None and not jobprov_df.empty:
        province_options = sorted(jobprov_df["province"].dropna().unique())
        USER_PROVINCE = st.sidebar.selectbox("Worker's province of origin:", province_options)
        GEO_C_MOVE = st.sidebar.number_input(
            "Relocation cost if move required ($)",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
        )
        GEO_LAMBDA = st.sidebar.slider("Weight on geographic cost (λ)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    else:
        st.sidebar.info(
            "Geographic cost selected, but no job_province_share file found. Geographic component will be treated as zero."
        )

HIST_UNIT = st.sidebar.radio(
    "Switching cost histogram units",
    options=["Dollars ($)", "Years of origin wages"],
    index=0,
)

n_results = st.sidebar.slider("Number of results to show:", min_value=3, max_value=20, value=5)
menu = st.sidebar.radio("Choose an option:", [
    "Look up by code",
    "Look up by title",
    "Compare two jobs",
    "Surplus → Shortage pathways",
])

# ---------- Risky/Safe sets ----------
RISKY_THRESHOLD = 0.70
SAFE_THRESHOLD = 0.70

available_nocs = set(similarity_df.index) & set(code_to_wage.keys())
if code_to_roa:
    RISKY_CODES = {noc for noc in available_nocs if code_to_roa.get(noc) is not None and code_to_roa[noc] >= RISKY_THRESHOLD}
    SAFE_CODES = {noc for noc in available_nocs if code_to_roa.get(noc) is not None and code_to_roa[noc] < SAFE_THRESHOLD}
else:
    RISKY_CODES, SAFE_CODES = set(), set()

# ---------- Calibration (k) computed BEFORE education effects ----------
CALIB_K, CALIB_PAIRS = compute_calibration_k_cached(
    tuple(sorted(RISKY_CODES)),
    tuple(sorted(SAFE_CODES)),
    target_usd=24000.0,
    beta=beta,
    alpha=alpha,
    q=q,
)

with st.sidebar.expander("Calibration status", expanded=False):
    st.markdown(
        f"""
- **ROA records loaded:** {'Yes' if code_to_roa else 'No'}
- **Risky codes (ROA ≥ 0.70):** {len(RISKY_CODES)}
- **Safe codes (ROA < 0.70):** {len(SAFE_CODES)}
- **Risky→Safe pairs used for k:** {CALIB_PAIRS}
- **Calibration k (applied in all costs):** {CALIB_K:.3f}
- **Standardization:** per-origin quantile-centered z (q = {q:.2f})
- **IMPORTANT:** k is calibrated using **baseline-only** (2 months, excludes tier-upgrade + recert months)
- **Future shortage data loaded:** {'Yes' if (code_to_balance is not None and len(code_to_balance) > 0) else 'No'}
"""
    )

with st.expander("Methodology"):
    st.markdown(
        """
### Overview
This app estimates the **cost of transitioning workers between occupations** by combining information on
skill similarity, education requirements, expected credential adjustment, wages, geography, and projected
labour market imbalances.

The goal is not to predict individual behaviour, but to provide a **transparent, internally consistent
approximation of transition frictions** that can be used for scenario analysis and policy exploration.

---

### Skill similarity
- Occupations are represented by vectors of skills, abilities, and knowledge derived from O*NET data.
- Similarity between occupations is measured using **Euclidean distance** in this multidimensional skill space.
- Smaller distances indicate more similar skill requirements.

---

### Origin-specific standardization
- Raw skill distances are standardized **within each origin occupation’s distribution of destinations**.
- Distances are centered at an **origin-specific baseline** defined by the *q-th quantile* of that distribution
  (rather than the mean).
- This avoids implying that roughly half of all possible transitions are “easier than average,” while still
  preserving a standard deviation of one within each origin.
- A standardized distance of \(z=0\) indicates **no additional skill-distance penalty**, not zero cost.

---

### Switching cost structure
The switching cost from origin occupation \(o\) to destination \(d\) is modeled as:

\[
\text{Cost}_{o\to d}
= k \cdot \left( M_{o\to d} \cdot w_o \right)
\cdot \left( 1 + \beta \, z_{o\to d}^{\alpha} \right)
\cdot m(z_{o\to d})
\;+\; \lambda \cdot \text{GeoCost}_d
\]

where:

- \(w_o\) is the origin occupation’s monthly wage.
- \(M_{o\to d}\) is the expected **months until prepared** for the destination job.
- \(z_{o\to d}\) is the origin-specific standardized skill distance (hinged at zero).
- \(\beta\) and \(\alpha\) control how costs increase with skill dissimilarity.
- \(m(z)\) is a discrete training-intensity multiplier.
- \(k\) is a global calibration factor.
- The geographic term is optional.

---

### Months until prepared
The additive preparation-time component is:

\[
M_{o\to d}
= 2
+ \text{Tier-upgrade months}
+ \text{Expected recertification months}
\]

- **2 months** represents baseline job search and adjustment time.
- **Tier-upgrade months** are based on differences in formal education requirements
  (e.g. high school → college → university).
- **Recertification months** apply only when the education tier stays the same and reflect
  the *expected* time needed to re-credential or retrain within a field.

---

### Expected recertification
- Recertification risk is proxied using differences in **NOC group digits**
  (first, third, and fourth digits).
- Larger group differences imply a higher probability that recertification is required.
- Expected recertification months are:
  - scaled by this probability,
  - smoothly ramped up with skill distance,
  - and capped to prevent extreme values.
- This captures **field mismatch risk**, not formal licensing requirements per se.

---

### Calibration
- The global scaling factor \(k\) is calibrated so that the average cost of transitions
  from **high-automation-risk occupations to lower-risk occupations** is approximately **$24,000**.
- Calibration is performed **before** adding education or recertification months,
  using only the 2-month baseline, ensuring comparability across specifications.

---

### Geographic mobility (optional)
- If enabled, an additional expected relocation cost is added.
- This cost depends on how geographically concentrated the destination occupation is
  relative to the worker’s province of origin.

---

### Surplus–shortage simulations
- When future labour market projections are provided, the app simulates reallocations
  from surplus occupations to shortage occupations.
- Workers are assigned using a **greedy, lowest-cost-first heuristic**:
  surplus workers fill the cheapest available shortage until that shortage is exhausted,
  then proceed to the next.
- The simulation reports:
  - total and average switching costs,
  - resulting average wages,
  - and the average months until workers are prepared for their new jobs.
- Results should be interpreted as **approximate, scenario-based indicators**, not forecasts.
        """
    )

# ============================================================
# Utility for payback columns (tables)
# ============================================================

def add_payback_columns(df: pd.DataFrame, origin_code: str, cost_col_numeric: str):
    w_origin = code_to_wage.get(origin_code)
    w_origin_val = float(w_origin) if (w_origin is not None and float(w_origin) > 0) else np.nan

    df["_w_origin"] = w_origin_val
    df["_w_dest"] = df["Code"].map(lambda c: code_to_wage.get(noc_str(c)))
    df["_gap"] = df["_w_dest"] - df["_w_origin"]

    def _pb_row(r):
        if pd.isnull(r["_w_origin"]) or pd.isnull(r["_w_dest"]):
            return {"wage_gap_monthly": np.nan, "payback_months": None, "payback_years": None, "status": "Missing wages."}
        return payback_period_months(r[cost_col_numeric], r["_w_origin"], r["_w_dest"])

    df["_pb"] = df.apply(_pb_row, axis=1)

    df["Destination wage ($/mo)"] = df["_w_dest"].map(lambda v: f"{float(v):,.0f}" if pd.notnull(v) else "N/A")
    df["Wage gap ($/mo)"] = df["_gap"].map(lambda v: f"{float(v):,.0f}" if pd.notnull(v) else "N/A")
    df["Payback (months)"] = df["_pb"].map(lambda d: f"{d['payback_months']:.1f}" if (d and d["payback_months"] is not None) else "N/A")
    df["Payback (years)"] = df["_pb"].map(lambda d: f"{d['payback_years']:.2f}" if (d and d["payback_years"] is not None) else "N/A")

    return df.drop(columns=["_w_origin", "_w_dest", "_gap", "_pb"])


# ============================================================
# ---------- Pages ----------
# ============================================================

# ---------- Look up by code ----------
if menu == "Look up by code":
    code = st.text_input("Enter 5-digit occupation code:")
    if code:
        code = noc_str(code)
        if code in similarity_df.index:
            top_results, bottom_results, all_scores = get_most_and_least_similar(code, n=n_results)
            w_origin = code_to_wage.get(code)

            st.subheader(f"Most Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            df_top = pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])
            df_top["_sc_numeric"] = df_top["Code"].apply(lambda x: calculate_switching_cost(code, x, beta=beta, alpha=alpha, q=q))
            df_top["Switching Cost ($)"] = df_top["_sc_numeric"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            df_top["Years of origin wages"] = df_top["_sc_numeric"].map(
                lambda x: (x / (12.0 * float(w_origin))) if (pd.notnull(x) and w_origin is not None and float(w_origin) > 0) else np.nan
            ).map(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")

            df_top = add_payback_columns(df_top, origin_code=code, cost_col_numeric="_sc_numeric")
            df_top = df_top.drop(columns=["_sc_numeric"])
            st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

            st.subheader(f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
            df_bottom["_sc_numeric"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(code, x, beta=beta, alpha=alpha, q=q))
            df_bottom["Switching Cost ($)"] = df_bottom["_sc_numeric"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            df_bottom["Years of origin wages"] = df_bottom["_sc_numeric"].map(
                lambda x: (x / (12.0 * float(w_origin))) if (pd.notnull(x) and w_origin is not None and float(w_origin) > 0) else np.nan
            ).map(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")

            df_bottom = add_payback_columns(df_bottom, origin_code=code, cost_col_numeric="_sc_numeric")
            df_bottom = df_bottom.drop(columns=["_sc_numeric"])
            st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

            with st.expander("Switching cost decomposition (details)", expanded=False):
                shown_codes = (
                    list(pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])["Code"].astype(str))
                    + list(pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])["Code"].astype(str))
                )
                decomp_rows = []
                for dest in shown_codes:
                    d = switching_cost_components(code, dest, beta=beta, alpha=alpha, q=q)
                    if d is not None:
                        decomp_rows.append(d)

                if decomp_rows:
                    st.dataframe(pd.DataFrame(decomp_rows), use_container_width=True)
                else:
                    st.info("No decomposition rows available (likely filtered out by education distance or missing data).")

            st.subheader(f"Similarity Score Distribution for {code} – {code_to_title.get(code,'Unknown')}")
            st.caption("Tip: hover on a bar to see which occupations fall in that similarity range.")
            st.altair_chart(similarity_hist_with_titles(all_scores), use_container_width=True)

            costs_df = compute_switching_costs_from_origin(code, beta=beta, alpha=alpha, q=q)
            if HIST_UNIT == "Dollars ($)":
                st.subheader(f"Switching Cost Distribution (Dollars) from {code} – {code_to_title.get(code,'Unknown')}")
                st.caption("Tip: hover on a bar to see which occupations fall in that cost range.")
                st.altair_chart(
                    cost_hist_with_titles(costs_df, value_col="cost", x_title="Switching Cost ($)", fmt_start=",.0f", fmt_end=",.0f"),
                    use_container_width=True,
                )
            else:
                st.subheader(f"Switching Cost Distribution (Years of origin wages) from {code} – {code_to_title.get(code,'Unknown')}")
                st.caption("Tip: hover on a bar to see which occupations fall in that cost range.")
                st.altair_chart(
                    cost_hist_with_titles(costs_df, value_col="years", x_title="Years of origin wages", fmt_start=".2f", fmt_end=".2f"),
                    use_container_width=True,
                )

        else:
            st.error("❌ Code not found in similarity matrix.")


# ---------- Look up by title ----------
elif menu == "Look up by title":
    available_codes = [c for c in code_to_title if c in similarity_df.index]
    title_options = [f"{c} – {code_to_title[c]}" for c in available_codes]

    selected_item = st.selectbox("Select an occupation:", sorted(title_options))
    if selected_item:
        selected_code, selected_title = selected_item.split(" – ")
        top_results, bottom_results, all_scores = get_most_and_least_similar(selected_code, n=n_results)

        w_origin = code_to_wage.get(selected_code)

        st.subheader(f"Most Similar Occupations for {selected_code} – {selected_title}")
        df_top = pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])
        df_top["_sc_numeric"] = df_top["Code"].apply(lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha, q=q))
        df_top["Switching Cost ($)"] = df_top["_sc_numeric"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        df_top["Years of origin wages"] = df_top["_sc_numeric"].map(
            lambda x: (x / (12.0 * float(w_origin))) if (pd.notnull(x) and w_origin is not None and float(w_origin) > 0) else np.nan
        ).map(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")

        df_top = add_payback_columns(df_top, origin_code=selected_code, cost_col_numeric="_sc_numeric")
        df_top = df_top.drop(columns=["_sc_numeric"])
        st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        st.subheader(f"Least Similar Occupations for {selected_code} – {selected_title}")
        df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
        df_bottom["_sc_numeric"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha, q=q))
        df_bottom["Switching Cost ($)"] = df_bottom["_sc_numeric"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        df_bottom["Years of origin wages"] = df_bottom["_sc_numeric"].map(
            lambda x: (x / (12.0 * float(w_origin))) if (pd.notnull(x) and w_origin is not None and float(w_origin) > 0) else np.nan
        ).map(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")

        df_bottom = add_payback_columns(df_bottom, origin_code=selected_code, cost_col_numeric="_sc_numeric")
        df_bottom = df_bottom.drop(columns=["_sc_numeric"])
        st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        with st.expander("Switching cost decomposition (details)", expanded=False):
            shown_codes = (
                list(pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])["Code"].astype(str))
                + list(pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])["Code"].astype(str))
            )
            decomp_rows = []
            for dest in shown_codes:
                d = switching_cost_components(selected_code, dest, beta=beta, alpha=alpha, q=q)
                if d is not None:
                    decomp_rows.append(d)

            if decomp_rows:
                st.dataframe(pd.DataFrame(decomp_rows), use_container_width=True)
            else:
                st.info("No decomposition rows available (likely filtered out by education distance or missing data).")

        st.subheader(f"Similarity Score Distribution for {selected_code} – {selected_title}")
        st.caption("Tip: hover on a bar to see which occupations fall in that similarity range.")
        st.altair_chart(similarity_hist_with_titles(all_scores), use_container_width=True)

        costs_df = compute_switching_costs_from_origin(selected_code, beta=beta, alpha=alpha, q=q)
        if HIST_UNIT == "Dollars ($)":
            st.subheader(f"Switching Cost Distribution (Dollars) from {selected_code} – {selected_title}")
            st.caption("Tip: hover on a bar to see which occupations fall in that cost range.")
            st.altair_chart(
                cost_hist_with_titles(costs_df, value_col="cost", x_title="Switching Cost ($)", fmt_start=",.0f", fmt_end=",.0f"),
                use_container_width=True,
            )
        else:
            st.subheader(f"Switching Cost Distribution (Years of origin wages) from {selected_code} – {selected_title}")
            st.caption("Tip: hover on a bar to see which occupations fall in that cost range.")
            st.altair_chart(
                cost_hist_with_titles(costs_df, value_col="years", x_title="Years of origin wages", fmt_start=".2f", fmt_end=".2f"),
                use_container_width=True,
            )


# ---------- Compare two jobs ----------
elif menu == "Compare two jobs":
    available_codes = [c for c in code_to_title if c in similarity_df.index]
    title_options = [f"{c} – {code_to_title[c]}" for c in available_codes]

    job1_item = st.selectbox("Select first occupation:", sorted(title_options), key="job1")
    job2_item = st.selectbox("Select second occupation:", sorted(title_options), key="job2")

    job1_code, job1_title = job1_item.split(" – ")
    job2_code, job2_title = job2_item.split(" – ")

    if st.button("Compare"):
        result = compare_two_jobs(job1_code, job2_code)
        if result:
            score, rank, total = result
            cost = calculate_switching_cost(job1_code, job2_code, beta=beta, alpha=alpha, q=q)

            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({job1_title}) vs {job2_code} ({job2_title})\n"
                f"- Similarity score (raw distance): `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations"
            )

            if cost is not None:
                w_origin = code_to_wage.get(job1_code)
                months_equiv = None
                years_equiv = None
                if w_origin is not None and float(w_origin) > 0:
                    w_origin = float(w_origin)
                    months_equiv = cost / w_origin
                    years_equiv = cost / (12.0 * w_origin)

                extra_geo = ""
                if USE_GEO and USER_PROVINCE is not None:
                    extra_geo = f" (includes geographic mobility component for {USER_PROVINCE})"

                msg = (
                    f"💰 **Estimated Switching Cost** (from {job1_code} to {job2_code}): "
                    f"`${cost:,.2f}` (education distance ≤ {EDU_GAP}){extra_geo}\n\n"
                )
                if years_equiv is not None:
                    msg += (
                        f"📆 **Equivalent to:** {months_equiv:.1f} months "
                        f"({years_equiv:.2f} years) of origin wages"
                    )
                st.info(msg)

                st.subheader("Wage payback period (recoup switching cost via higher earnings)")
                w_dest = code_to_wage.get(job2_code)

                if w_origin is None or w_dest is None or float(w_origin) <= 0 or float(w_dest) <= 0:
                    st.info("Payback not available (missing or invalid wage data for one or both occupations).")
                else:
                    pb = payback_period_months(cost, float(w_origin), float(w_dest))
                    st.write(
                        f"- Origin monthly wage: **${float(w_origin):,.0f}**\n"
                        f"- Destination monthly wage: **${float(w_dest):,.0f}**\n"
                        f"- Monthly wage gap (dest − origin): **${pb['wage_gap_monthly']:,.0f}**"
                    )
                    if pb["payback_months"] is None:
                        st.info(f"ℹ️ {pb['status']}")
                    else:
                        st.success(
                            f"✅ Estimated payback time: **{pb['payback_months']:.1f} months** "
                            f"(**{pb['payback_years']:.2f} years**) assuming the wage gap persists."
                        )

                with st.expander("Switching cost decomposition (details)", expanded=False):
                    d = switching_cost_components(job1_code, job2_code, beta=beta, alpha=alpha, q=q)
                    if d is None:
                        st.info("No decomposition available (filtered out or missing data).")
                    else:
                        st.dataframe(pd.DataFrame([d]), use_container_width=True)

            else:
                st.info(
                    "ℹ️ Switching cost is not reported because the two occupations are further apart in education "
                    f"than the allowed maximum distance ({EDU_GAP})."
                )
        else:
            st.error("❌ Could not compare occupations.")


# ---------- Surplus → Shortage pathways ----------
elif menu == "Surplus → Shortage pathways":
    st.header("Surplus → Shortage pathways")
    st.caption(
        "Overlays surplus/shortage projections on the switching cost model and simulates reallocations."
    )

    if not code_to_balance:
        st.warning("No projection file found. Add future_shortage.xlsx (columns: noc, status, magnitude).")
    else:
        # Build surplus origins
        origins = []
        for c in similarity_df.index:
            bal = code_to_balance.get(c)
            if bal is None or (isinstance(bal, float) and np.isnan(bal)):
                continue
            if float(bal) < 0:
                origins.append(c)

        origins = sorted(set(origins))
        if not origins:
            st.info("No occupations classified as surplus in the projection data.")
        else:
            origin_labels = [f"{c} – {code_to_title.get(c,'Unknown Title')}" for c in origins]
            origin_item = st.selectbox("Select a surplus origin occupation:", origin_labels)

            origin_code = origin_item.split(" – ")[0]
            origin_bal = float(code_to_balance.get(origin_code, np.nan))
            origin_status = code_to_status.get(origin_code, "unknown")
            w_origin = code_to_wage.get(origin_code)

            c1, c2, c3 = st.columns(3)
            c1.metric("Origin status", origin_status)
            c2.metric("Origin magnitude (surplus < 0)", f"{origin_bal:,.2f}" if pd.notna(origin_bal) else "N/A")
            c3.metric("Origin wage ($/mo)", f"{float(w_origin):,.0f}" if (w_origin is not None and float(w_origin) > 0) else "N/A")

            st.subheader("B) Where can these workers go? Cheapest shortage destinations")
            st.caption("Workers are matched to shortage occupations in order of lowest estimated switching cost.")
            top_n = st.slider("Number of pathways to show", 5, 100, 25, 5)
            min_short = st.number_input("Minimum shortage magnitude", value=0.0, min_value=0.0)
            only_wage_gain = st.checkbox(
                "Only show shortage destinations with higher wage than origin",
                value=False
            )

            # Build temporary shortages dict and generate candidates by running allocation with huge surplus
            shortages_tmp = {
                noc: float(bal)
                for noc, bal in code_to_balance.items()
                if bal is not None and not (isinstance(bal, float) and np.isnan(bal)) and float(bal) > 0
            }

            flows_tmp, _ = allocate_surplus_to_shortage(
                origin_code=origin_code,
                beta=beta,
                alpha=alpha,
                q=q,
                shortages_balance=shortages_tmp,
                surplus_amount=1e18,
                only_wage_gain=only_wage_gain,
            )

            df = flows_tmp.copy()
            if not df.empty:
                df["Destination balance (shortage +)"] = df["Destination"].map(lambda d: float(code_to_balance.get(d, np.nan)))
                df = df[pd.notnull(df["Destination balance (shortage +)"]) & (df["Destination balance (shortage +)"] >= float(min_short))]

                df = df.sort_values("Switching cost per worker ($)").head(int(top_n))

                out = df[[
                    "Destination",
                    "Destination title",
                    "Destination balance (shortage +)",
                    "Destination wage ($/mo)",
                    "Wage gap ($/mo)",
                    "Switching cost per worker ($)",
                    "Preparation months (expected)",
                ]].copy()

                out["Destination balance (shortage +)"] = out["Destination balance (shortage +)"].map(lambda v: f"{float(v):,.2f}" if pd.notna(v) else "N/A")
                out["Destination wage ($/mo)"] = out["Destination wage ($/mo)"].map(lambda v: f"{float(v):,.0f}" if pd.notna(v) else "N/A")
                out["Wage gap ($/mo)"] = out["Wage gap ($/mo)"].map(lambda v: f"{float(v):,.0f}" if pd.notna(v) else "N/A")
                out["Switching cost per worker ($)"] = out["Switching cost per worker ($)"].map(lambda v: f"{float(v):,.0f}" if pd.notna(v) else "N/A")
                out["Preparation months (expected)"] = out["Preparation months (expected)"].map(lambda v: f"{float(v):.1f}" if pd.notna(v) else "N/A")

                st.dataframe(out, use_container_width=True)
            else:
                st.info("No eligible shortage destinations found (may be filtered by education gap or missing wage data).")

            st.divider()
            st.subheader("C) What happens if we reallocate all workers from this surplus occupation?")
            st.caption("Allocates the full surplus of the selected origin across shortages using a lowest-cost-first heuristic.")
            sim_only_wage_gain = st.checkbox(
                "Simulation: require destination wage > origin wage",
                value=only_wage_gain,
                key="sim_only_wage_gain_origin",
            )

            if st.button("Run origin allocation simulation"):
                shortages_balance = {
                    noc: float(bal)
                    for noc, bal in code_to_balance.items()
                    if bal is not None and not (isinstance(bal, float) and np.isnan(bal)) and float(bal) > 0
                }
                surplus_amount = abs(float(code_to_balance.get(origin_code, 0.0)))

                flows_df, summary = allocate_surplus_to_shortage(
                    origin_code=origin_code,
                    beta=beta,
                    alpha=alpha,
                    q=q,
                    shortages_balance=shortages_balance,
                    surplus_amount=surplus_amount,
                    only_wage_gain=sim_only_wage_gain,
                )

                if summary["moved_total"] <= 0:
                    st.warning("No allocation was possible (check education gap filter, wage filter, or missing wage data).")
                else:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Moved (total)", f"{summary['moved_total']:,.0f}")
                    m2.metric("Unallocated surplus", f"{summary['unallocated_surplus']:,.0f}")
                    m3.metric("Total switching cost", f"${summary['total_switching_cost']:,.0f}")
                    m4.metric("Avg cost per moved worker", f"${summary['avg_switching_cost_per_worker']:,.0f}")
                    m5.metric(
                        "Avg preparation months",
                        f"{summary['avg_preparation_months']:.1f}" if pd.notna(summary["avg_preparation_months"]) else "N/A"
                    )

                    n1, n2, n3 = st.columns(3)
                    n1.metric("Avg origin wage ($/mo)", f"${summary['avg_origin_wage']:,.0f}" if pd.notna(summary["avg_origin_wage"]) else "N/A")
                    n2.metric("Avg new wage ($/mo)", f"${summary['avg_destination_wage']:,.0f}" if pd.notna(summary["avg_destination_wage"]) else "N/A")
                    n3.metric("Avg wage change ($/mo)", f"${summary['avg_wage_change']:,.0f}" if pd.notna(summary["avg_wage_change"]) else "N/A")

                    st.subheader("Allocation results (origin → multiple destinations)")
                    st.dataframe(flows_df, use_container_width=True)

                    # ---- Download bundle: origin simulation ----
                    bundle = make_zip_bundle({
                        f"origin_{origin_code}_flows.csv": flows_df,
                        f"origin_{origin_code}_summary.csv": pd.DataFrame([summary]),
                        "readme.txt": (
                            "Bundle contents:\n"
                            "- origin_*_flows.csv: destination-level flows and costs for the selected origin\n"
                            "- origin_*_summary.csv: totals and averages for the selected origin\n"
                        ),
                    })
                    st.download_button(
                        label="Download origin simulation bundle (ZIP)",
                        data=bundle,
                        file_name=f"origin_{origin_code}_simulation_bundle.zip",
                        mime="application/zip",
                    )

                    # Bar chart: flows by destination
                    bar_df = flows_df.copy()
                    bar_df["label"] = bar_df["Destination"] + " – " + bar_df["Destination title"]
                    bar = (
                        alt.Chart(bar_df)
                        .mark_bar()
                        .encode(
                            y=alt.Y("label:N", sort="-x", title="Destination"),
                            x=alt.X("Flow moved:Q", title="Flow moved (weighted workers)"),
                            tooltip=[
                                "Destination:N",
                                "Destination title:N",
                                alt.Tooltip("Flow moved:Q", format=",.0f"),
                                alt.Tooltip("Switching cost per worker ($):Q", format=",.0f"),
                                alt.Tooltip("Flow-weighted switching cost ($):Q", format=",.0f"),
                                alt.Tooltip("Preparation months (expected):Q", format=".1f"),
                                alt.Tooltip("Destination wage ($/mo):Q", format=",.0f"),
                            ],
                        )
                        .properties(height=min(900, 30 * max(5, len(bar_df))))
                    )
                    st.altair_chart(bar, use_container_width=True)

            st.divider()
            st.subheader("D) What happens system-wide if we reallocate across all surplus occupations?")
            st.caption("Runs the same allocation across all surplus origins; ordering matters because shortages are filled as we go.")
            st.caption(
                "This is a greedy simulation (not a global optimizer). "
                "Order matters: earlier origins consume shortage capacity first."
            )

            origin_order = st.radio(
                "Order of surplus origins",
                options=["Largest surplus first", "Smallest surplus first"],
                index=0,
                horizontal=True,
            )
            sim_only_wage_gain_all = st.checkbox(
                "Require destination wage > origin wage (all-origins simulation)",
                value=False,
                key="sim_only_wage_gain_all",
            )

            if st.button("Run full system simulation"):
                flows_all, totals, origin_summary_df, shortage_remaining_df = simulate_all_surplus_to_shortage(
                    beta=beta,
                    alpha=alpha,
                    q=q,
                    only_wage_gain=sim_only_wage_gain_all,
                    origin_order=origin_order,
                )

                if totals.get("total_moved", 0.0) <= 0:
                    st.warning("No allocation was possible. Check missing wage data, education gap, or wage filter.")
                else:
                    t1, t2, t3, t4, t5 = st.columns(5)
                    t1.metric("Total moved", f"{totals['total_moved']:,.0f}")
                    t2.metric("Total switching cost", f"${totals['total_switching_cost']:,.0f}")
                    t3.metric("Avg cost per moved worker", f"${totals['avg_switching_cost_per_worker']:,.0f}")
                    t4.metric("Shortage remaining (total)", f"{totals['shortage_remaining_total']:,.0f}")
                    t5.metric(
                        "Avg preparation months",
                        f"{totals['avg_preparation_months']:.1f}" if pd.notna(totals["avg_preparation_months"]) else "N/A"
                    )

                    u1, u2, u3 = st.columns(3)
                    u1.metric("Avg origin wage ($/mo)", f"${totals['avg_origin_wage']:,.0f}" if pd.notna(totals["avg_origin_wage"]) else "N/A")
                    u2.metric("Avg new wage ($/mo)", f"${totals['avg_destination_wage']:,.0f}" if pd.notna(totals["avg_destination_wage"]) else "N/A")
                    u3.metric("Avg wage change ($/mo)", f"${totals['avg_wage_change']:,.0f}" if pd.notna(totals["avg_wage_change"]) else "N/A")

                    st.subheader("Flows (all origins)")
                    st.dataframe(flows_all, use_container_width=True)

                    st.subheader("Origin summaries")
                    st.dataframe(origin_summary_df, use_container_width=True)

                    st.subheader("Remaining shortage after simulation")
                    st.dataframe(shortage_remaining_df, use_container_width=True)

                # ---- Download bundle: system simulation ----
                bundle = make_zip_bundle({
                    "system_flows_all.csv": flows_all,
                    "system_totals.csv": pd.DataFrame([totals]),
                    "system_origin_summaries.csv": origin_summary_df,
                    "system_shortage_remaining.csv": shortage_remaining_df,
                    "readme.txt": (
                        "Bundle contents:\n"
                        "- system_flows_all.csv: all origin->destination flows (greedy allocation)\n"
                        "- system_totals.csv: aggregate totals and averages\n"
                        "- system_origin_summaries.csv: per-origin moved/cost/wage/prep-month summaries\n"
                        "- system_shortage_remaining.csv: shortage capacity remaining after simulation\n"
                        "Notes:\n"
                        "- This is a greedy heuristic; ordering of surplus origins affects results.\n"
                    ),
                })
                st.download_button(
                    label="Download system simulation bundle (ZIP)",
                    data=bundle,
                    file_name="system_simulation_bundle.zip",
                    mime="application/zip",
                )
