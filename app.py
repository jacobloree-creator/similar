import streamlit as st
import pandas as pd
import os
import altair as alt
import numpy as np
import io
import zipfile

# ============================================================
# CONSTANTS  (all magic numbers in one place)
# ============================================================

# --- Baseline quantile for origin-standardized z ---
BASELINE_Q_DEFAULT = 0.10

# --- Calibration target ---
CALIB_TARGET_USD = 24_000.0
CALIB_BASELINE_MONTHS = 2.0          # months used *only* during calibration

# --- Automation-risk thresholds ---
ROA_RISKY_THRESHOLD = 0.70
ROA_SAFE_THRESHOLD  = 0.70

# --- Recertification probabilities (NOC digit mismatch) ---
P_RECERT_FIRST  = 0.80
P_RECERT_THIRD  = 0.35
P_RECERT_FOURTH = 0.10

# --- Recert ramp boundaries ---
RECERT_Z0 = 0.30
RECERT_Z1 = 1.20

# --- Caps ---
RECERT_CAP_MONTHS = 18.0
EDU_MONTHS_CAP    = 60.0

# --- Base recertification months by education tier (same-tier transitions only) ---
# 1=university  2=4yr college  3=2yr college  4=HS  5=no education required
RECERT_BASE_MONTHS_BY_TIER: dict[int, float] = {
    1: 48.0,
    2: 24.0,
    3: 12.0,
    4:  0.0,
    5:  0.0,
}

# --- Education upgrade months matrix (origin tier × dest tier, 1-indexed) ---
#   rows = origin tier (1-5), cols = dest tier (1-5)
#   0 = no upgrade needed / not applicable
EDU_MONTHS_MATRIX = np.array([
    [  0,  0,  0,  0,  0],   # origin 1 (university)
    [ 18,  0,  0,  0,  0],   # origin 2
    [ 30, 14,  0,  0,  0],   # origin 3
    [ 48, 30, 18,  0,  0],   # origin 4
    [ 60, 36, 24, 12,  0],   # origin 5
], dtype=float)

# --- Geographic cost default ---
GEO_C_MOVE_DEFAULT = 20_000.0

# --- Infinite-surplus sentinel used in "show all pathways" query ---
_INFINITE_SURPLUS = 1e18


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)

    def _norm(s: pd.Series) -> pd.Series:
        return s.astype(str).str.zfill(5).str.strip()

    # Similarity matrix (Euclidean distances)
    sim = pd.read_excel(os.path.join(base, "similarity matrix_v2.xlsx"), index_col=0)
    sim.index   = _norm(sim.index.to_series())
    sim.columns = _norm(pd.Series(sim.columns))

    # NOC titles
    t = pd.read_excel(os.path.join(base, "noc title.xlsx"))
    t.columns = t.columns.str.strip().str.lower()
    t["noc"]  = _norm(t["noc"])
    code_to_title = dict(zip(t["noc"], t["title"]))

    # Monthly wages
    w = pd.read_excel(os.path.join(base, "monthly_wages.xlsx"))
    w["noc"]  = _norm(w["noc"])
    code_to_wage = dict(zip(w["noc"], w["monthly_wage"]))

    # --- Helper: load an optional flat file (csv or xlsx) ---
    def _load_optional(csv_name, xlsx_name):
        for fname, reader in [(csv_name, pd.read_csv), (xlsx_name, pd.read_excel)]:
            path = os.path.join(base, fname)
            if os.path.exists(path):
                try:
                    df = reader(path)
                    return df if not df.empty else None
                except Exception:
                    pass
        return None

    # Automation risk
    code_to_roa: dict = {}
    df_roa = _load_optional("automation_risk.csv", "automation_risk.xlsx")
    if df_roa is not None:
        cols = {c.lower().strip(): c for c in df_roa.columns}
        noc_col  = cols.get("noc") or cols.get("code") or df_roa.columns[0]
        prob_col = (cols.get("roa_prob") or cols.get("automation_probability")
                    or cols.get("probability") or df_roa.columns[1])
        df_roa[noc_col] = _norm(df_roa[noc_col])
        code_to_roa = dict(zip(df_roa[noc_col], df_roa[prob_col]))

    # Job-by-province share
    jobprov_df = pd.DataFrame()
    df_jp = _load_optional("job_province_share.csv", "job_province_share.xlsx")
    if df_jp is not None:
        cols = {c.lower().strip(): c for c in df_jp.columns}
        noc_col   = cols.get("noc") or cols.get("code") or df_jp.columns[0]
        prov_col  = cols.get("province") or cols.get("prov") or df_jp.columns[1]
        share_col = (cols.get("share") or cols.get("share_in_job_province")
                     or cols.get("pct") or df_jp.columns[2])
        df_jp[noc_col]  = _norm(df_jp[noc_col])
        df_jp[prov_col] = df_jp[prov_col].astype(str).str.strip()
        jobprov_df = df_jp[[noc_col, prov_col, share_col]].rename(
            columns={noc_col: "noc", prov_col: "province", share_col: "share"}
        )

    # Shortage/surplus projections
    code_to_balance: dict = {}
    code_to_status:  dict = {}
    df_ss = _load_optional("future_shortage.csv", "future_shortage.xlsx")
    if df_ss is not None:
        df_ss.columns = df_ss.columns.astype(str).str.strip().str.lower()
        if {"noc", "status", "magnitude"}.issubset(df_ss.columns):
            df_ss["noc"]       = _norm(df_ss["noc"])
            df_ss["status"]    = df_ss["status"].astype(str).str.strip().str.lower()
            df_ss["magnitude"] = pd.to_numeric(df_ss["magnitude"], errors="coerce")

            def _signed(row):
                s, m = row["status"], row["magnitude"]
                if pd.isna(m):
                    return np.nan
                if "short" in s: return  float(m)
                if "surp"  in s: return -float(m)
                if "bal"   in s or "neutral" in s: return 0.0
                return np.nan

            df_ss["_bal"] = df_ss.apply(_signed, axis=1)
            df_ss = df_ss.dropna(subset=["_bal"])
            code_to_balance = dict(zip(df_ss["noc"], df_ss["_bal"]))
            code_to_status  = {
                k: ("shortage" if float(v) > 0 else ("surplus" if float(v) < 0 else "balanced"))
                for k, v in code_to_balance.items()
            }

    return (
        sim,
        code_to_title,
        dict(zip(t["title"].str.lower(), t["noc"])),   # title_to_code
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
# SMALL UTILITIES
# ============================================================

def noc_str(code) -> str:
    return str(code).zfill(5).strip()


def get_education_level(code) -> int | None:
    """2nd digit (index 1) of the 5-digit NOC code encodes the education tier."""
    try:
        return int(noc_str(code)[1])
    except Exception:
        return None


def group_digits(code: str) -> tuple[str, str, str]:
    """Return (1st, 3rd, 4th) NOC digits used for recertification logic."""
    s = noc_str(code)
    return s[0], s[2], s[3]


def make_zip_bundle(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, obj in files.items():
            if obj is None:
                continue
            z.writestr(name, obj.to_csv(index=False) if isinstance(obj, pd.DataFrame) else str(obj))
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# EDUCATION & RECERTIFICATION MATHS
# ============================================================

def expected_tier_upgrade_months(origin: str, dest: str) -> float:
    eo, ed = get_education_level(origin), get_education_level(dest)
    if eo is None or ed is None or not (1 <= eo <= 5 and 1 <= ed <= 5):
        return 0.0
    return min(float(EDU_MONTHS_MATRIX[eo - 1, ed - 1]), EDU_MONTHS_CAP)


def group_mismatch_flags(origin: str, dest: str) -> dict[str, int]:
    o1, o3, o4 = group_digits(origin)
    d1, d3, d4 = group_digits(dest)
    return {"diff_1st": int(o1 != d1), "diff_3rd": int(o3 != d3), "diff_4th": int(o4 != d4)}


def recert_probability_from_digits(origin: str, dest: str) -> float:
    f = group_mismatch_flags(origin, dest)
    if f["diff_1st"]:  return P_RECERT_FIRST
    if f["diff_3rd"]:  return P_RECERT_THIRD
    if f["diff_4th"]:  return P_RECERT_FOURTH
    return 0.0


def recert_ramp(z_eff: float) -> float:
    z = float(z_eff)
    if z <= RECERT_Z0: return 0.0
    if z >= RECERT_Z1: return 1.0
    return (z - RECERT_Z0) / (RECERT_Z1 - RECERT_Z0)


def expected_recert_months_same_tier(origin: str, dest: str, z_eff: float) -> float:
    eo, ed = get_education_level(origin), get_education_level(dest)
    if eo is None or ed is None or eo != ed:
        return 0.0
    base = float(RECERT_BASE_MONTHS_BY_TIER.get(eo, 0.0))
    if base <= 0:
        return 0.0
    return min(RECERT_CAP_MONTHS, base * recert_probability_from_digits(origin, dest) * recert_ramp(z_eff))


# ============================================================
# SKILL-DISTANCE STANDARDIZATION & TRAINING MULTIPLIER
# ============================================================

def origin_standardized_z(origin: str, dest: str, q: float) -> float | None:
    if origin not in similarity_df.index or dest not in similarity_df.columns:
        return None
    row = similarity_df.loc[origin].dropna()
    if dest not in row.index:
        return None
    std = float(row.std())
    if std == 0.0 or np.isnan(std):
        return None
    return float((row.loc[dest] - row.quantile(q)) / std)


def training_multiplier(z_eff: float) -> float:
    z = float(z_eff)
    if z < 0.5: return 1.0
    if z < 1.0: return 1.2
    if z < 1.5: return 1.5
    return 2.0


# ============================================================
# GEOGRAPHIC COST
# ============================================================

def geographic_cost(dest: str, province: str | None, c_move: float = GEO_C_MOVE_DEFAULT) -> float:
    if province is None or jobprov_df.empty:
        return 0.0
    sub = jobprov_df[jobprov_df["noc"] == dest]
    if sub.empty:
        return 0.0
    a_max = float(sub["share"].max())
    if a_max <= 0:
        return 0.0
    a_p0  = float(sub.loc[sub["province"] == province, "share"].sum())
    return c_move * max(0.0, min(1.0, 1.0 - a_p0 / a_max))


# ============================================================
# SWITCHING COST  (single source of truth)
# ============================================================

def switching_cost_components(
    origin: str,
    dest:   str,
    beta:   float = 0.14,
    alpha:  float = 1.2,
    q:      float = 0.10,
    *,
    edu_gap:      int   = 0,
    use_geo:      bool  = False,
    user_province: str | None = None,
    geo_c_move:   float = GEO_C_MOVE_DEFAULT,
    geo_lambda:   float = 0.0,
    calib_k:      float = 1.0,
    prep_months_override: float | None = None,  # used during calibration (baseline-only)
) -> dict | None:
    """
    Full decomposition of the switching cost.
    Returns None if the transition is ineligible (education gap, missing data, etc.).
    `calculate_switching_cost` is a thin wrapper that extracts just the total.
    """
    eo = get_education_level(origin)
    ed = get_education_level(dest)
    if eo is None or ed is None or abs(eo - ed) > edu_gap:
        return None

    z_raw = origin_standardized_z(origin, dest, q)
    if z_raw is None:
        return None
    z_eff = max(float(z_raw), 0.0)

    w_origin = code_to_wage.get(origin)
    if w_origin is None or float(w_origin) <= 0:
        return None
    w_origin = float(w_origin)

    w_dest_raw = code_to_wage.get(dest)
    w_dest = float(w_dest_raw) if (w_dest_raw is not None and float(w_dest_raw) > 0) else np.nan

    # --- Preparation months ---
    if prep_months_override is not None:
        tier_months  = 0.0
        recert_months = 0.0
        prep_months  = float(prep_months_override)
    else:
        tier_months   = expected_tier_upgrade_months(origin, dest)
        recert_months = expected_recert_months_same_tier(origin, dest, z_eff)
        prep_months   = CALIB_BASELINE_MONTHS + tier_months + recert_months

    # --- Core cost ---
    base       = prep_months * w_origin
    dist_term  = 1.0 + beta * (z_eff ** alpha)
    mult       = training_multiplier(z_eff)
    skill_cost = calib_k * base * dist_term * mult

    # --- Geographic add-on ---
    geo_raw = geographic_cost(dest, user_province, geo_c_move) if use_geo else 0.0
    geo_add = geo_lambda * geo_raw

    total = skill_cost + geo_add

    # --- Payback ---
    pb = payback_period_months(total, w_origin, w_dest)

    # --- Recert diagnostics ---
    same_tier  = (eo == ed)
    base_recert = float(RECERT_BASE_MONTHS_BY_TIER.get(eo, 0.0)) if same_tier else 0.0
    p_digit     = recert_probability_from_digits(origin, dest) if (same_tier and base_recert > 0) else 0.0
    ramp        = recert_ramp(z_eff) if p_digit > 0 else 0.0
    f           = group_mismatch_flags(origin, dest)

    return {
        "Origin":          origin,
        "Destination":     dest,
        "Title":           code_to_title.get(dest, "Unknown Title"),

        "Origin wage ($/mo)":      w_origin,
        "Destination wage ($/mo)": w_dest,
        "Wage gap ($/mo)":         (w_dest - w_origin) if pd.notna(w_dest) else np.nan,

        "Baseline quantile q":      q,
        "z_raw (quantile-centered)": z_raw,
        "z_eff (hinged for cost)":  z_eff,

        "Education tier (origin)": eo,
        "Education tier (dest)":   ed,

        "Tier upgrade months":            tier_months,
        "Recert base months (same-tier)": base_recert,
        "P(recert | digits)":             p_digit,
        "Recert ramp(z_eff)":             ramp,
        "Recert cap (months)":            RECERT_CAP_MONTHS,
        "Group diff (1st)":  f["diff_1st"],
        "Group diff (3rd)":  f["diff_3rd"],
        "Group diff (4th)":  f["diff_4th"],
        "Recert months (expected)": recert_months,

        "Preparation months (2 + tier + recert)": prep_months,

        "Base (prep_months × origin wage)": base,
        "Distance term":  dist_term,
        "Training mult":  mult,

        "k (calibration)": calib_k,
        "Skill cost":      skill_cost,

        "Geo raw":      geo_raw,
        "λ (geo weight)": geo_lambda,
        "Geo add":      geo_add,

        "Total cost":             total,
        "Months of origin wages": total / w_origin,
        "Years of origin wages":  total / (12.0 * w_origin),

        "Payback months (wage gap)": pb["payback_months"] if pb["payback_months"] is not None else np.nan,
        "Payback years (wage gap)":  pb["payback_years"]  if pb["payback_years"]  is not None else np.nan,
        "Payback status": pb["status"],
    }


def calculate_switching_cost(
    origin: str,
    dest:   str,
    beta:   float = 0.14,
    alpha:  float = 1.2,
    q:      float = 0.10,
    **kwargs,
) -> float | None:
    """Thin wrapper: returns just the total cost (or None)."""
    d = switching_cost_components(origin, dest, beta, alpha, q, **kwargs)
    return d["Total cost"] if d is not None else None


# ============================================================
# CALIBRATION
# ============================================================

@st.cache_data(show_spinner=False)
def compute_calibration_k_cached(
    risky_codes_tuple: tuple,
    safe_codes_tuple:  tuple,
    target_usd: float = CALIB_TARGET_USD,
    beta:  float = 0.14,
    alpha: float = 1.2,
    q:     float = 0.10,
    edu_gap: int = 0,
) -> tuple[float, int]:
    """
    k is calibrated using ONLY the 2-month baseline (no tier-upgrade / recert months).
    """
    raw_costs = []
    for r in risky_codes_tuple:
        w_o = code_to_wage.get(r)
        if w_o is None or float(w_o) <= 0:
            continue
        for s in safe_codes_tuple:
            if s == r:
                continue
            c = switching_cost_components(
                r, s, beta, alpha, q,
                edu_gap=edu_gap,
                calib_k=1.0,
                prep_months_override=CALIB_BASELINE_MONTHS,
            )
            if c is not None:
                raw_costs.append(c["Total cost"])

    if not raw_costs:
        return 1.0, 0
    mean_raw = float(np.mean(raw_costs))
    return (target_usd / mean_raw if mean_raw > 0 else 1.0), len(raw_costs)


# ============================================================
# PAYBACK PERIOD
# ============================================================

def payback_period_months(cost: float, w_origin: float, w_dest: float) -> dict:
    try:
        C, wo, wd = float(cost), float(w_origin), float(w_dest)
    except Exception:
        return {"wage_gap_monthly": np.nan, "payback_months": None, "payback_years": None,
                "status": "Missing/invalid wage or cost inputs."}

    gap = wd - wo
    if np.isnan(C):
        return {"wage_gap_monthly": gap, "payback_months": None, "payback_years": None,
                "status": "Switching cost unavailable."}
    if C <= 0:
        return {"wage_gap_monthly": gap, "payback_months": 0.0, "payback_years": 0.0,
                "status": "No switching cost to recoup."}
    if gap <= 0:
        return {"wage_gap_monthly": gap, "payback_months": None, "payback_years": None,
                "status": "No wage payback (destination wage not higher)."}
    m = C / gap
    return {"wage_gap_monthly": gap, "payback_months": m, "payback_years": m / 12.0,
            "status": "Payback computed from wage gap."}


# ============================================================
# SIMILARITY & COST QUERIES
# ============================================================

def get_most_and_least_similar(code: str, n: int = 5, edu_gap: int = 0):
    if code not in similarity_df.index:
        return None, None, None

    origin_level = get_education_level(code)
    scores = similarity_df.loc[code].drop(code).dropna()

    if origin_level is not None:
        allowed = [
            c for c in scores.index
            if (lev := get_education_level(c)) is not None and abs(lev - origin_level) <= edu_gap
        ]
        scores = scores.loc[allowed] if allowed else scores.iloc[0:0]

    if scores.empty:
        return [], [], scores

    def _fmt(series):
        return [(c, code_to_title.get(c, "Unknown Title"), v) for c, v in series.items()]

    return _fmt(scores.nsmallest(n)), _fmt(scores.nlargest(n)), scores


def compare_two_jobs(code1: str, code2: str) -> tuple | None:
    if code1 not in similarity_df.index or code2 not in similarity_df.index:
        return None
    scores = similarity_df.loc[code1].drop(code1).dropna()
    scores = scores[scores != 0].sort_values()
    if code2 not in scores.index:
        return None
    score = similarity_df.loc[code1, code2]
    if pd.isna(score):
        score = similarity_df.loc[code2, code1]
    if pd.isna(score):
        return None
    return score, scores.index.get_loc(code2) + 1, len(scores)


def compute_switching_costs_from_origin(origin: str, beta, alpha, q, edu_gap, **geo_kwargs) -> pd.DataFrame:
    origin_level = get_education_level(origin)
    w_origin     = code_to_wage.get(origin)

    rows = []
    for dest in similarity_df.columns:
        if dest == origin:
            continue
        if origin_level is not None:
            lev = get_education_level(dest)
            if lev is None or abs(lev - origin_level) > edu_gap:
                continue
        cost = calculate_switching_cost(origin, dest, beta=beta, alpha=alpha, q=q, edu_gap=edu_gap, **geo_kwargs)
        if pd.notna(cost):
            years = (float(cost) / (12.0 * float(w_origin))
                     if (w_origin and float(w_origin) > 0) else np.nan)
            rows.append({"code": dest, "title": code_to_title.get(dest, "Unknown Title"),
                         "cost": float(cost), "years": float(years)})
    return pd.DataFrame(rows)


# ============================================================
# CHARTING  (single parameterised function for both histogram types)
# ============================================================

def _binned_chart(
    df:        pd.DataFrame,
    value_col: str,
    x_title:   str,
    fmt:       str,
    color:     str,
    maxbins:   int = 30,
    max_titles: int = 50,
) -> alt.Chart:
    """Generic binned bar chart with occupation tooltips."""
    df = df.dropna(subset=[value_col]) if not df.empty else df

    if df.empty:
        return (alt.Chart(pd.DataFrame({value_col: [0]}))
                .mark_bar(color=color)
                .encode(alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=1), title=x_title))
                .properties(width=600, height=400))

    # Build label column (works for both similarity scores and cost frames)
    label_col = "label"
    if "code" in df.columns and "title" in df.columns:
        df = df.copy()
        df[label_col] = df["code"] + " – " + df["title"]
    elif label_col not in df.columns:
        df = df.copy()
        df[label_col] = df.index.astype(str)

    edges = np.histogram_bin_edges(df[value_col], bins=maxbins)
    df["_bin"] = pd.cut(df[value_col], bins=edges, include_lowest=True)

    rows = []
    for iv, g in df.groupby("_bin"):
        if iv is None or g.empty:
            continue
        labels = g[label_col].tolist()
        extra  = f"\n... (+{len(labels) - max_titles} more)" if len(labels) > max_titles else ""
        rows.append({
            "bin_start": float(iv.left),
            "bin_end":   float(iv.right),
            "count":     len(g),
            "titles_str": "\n".join(labels[:max_titles]) + extra,
        })

    if not rows:
        return (alt.Chart(df)
                .mark_bar(opacity=0.7, color=color)
                .encode(
                    alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=maxbins), title=x_title),
                    alt.Y("count()", title="Number of Occupations"),
                ).properties(width=600, height=400))

    bins_df = pd.DataFrame(rows)
    return (
        alt.Chart(bins_df)
        .mark_bar(opacity=0.7, color=color)
        .encode(
            x=alt.X("bin_start:Q", bin=alt.Bin(binned=True), title=x_title),
            x2="bin_end:Q",
            y=alt.Y("count:Q", title="Number of Occupations"),
            tooltip=[
                alt.Tooltip("count:Q",      title="Number of occupations"),
                alt.Tooltip("bin_start:Q",  format=fmt, title="From"),
                alt.Tooltip("bin_end:Q",    format=fmt, title="To"),
                alt.Tooltip("titles_str:N", title="Occupations in this bin"),
            ],
        )
        .properties(width=600, height=400)
    )


def similarity_hist(scores: pd.Series) -> alt.Chart:
    df = pd.DataFrame({"code": scores.index, "score": scores.values})
    df["title"] = df["code"].map(lambda c: code_to_title.get(c, "Unknown Title"))
    return _binned_chart(df, "score", "Similarity Score (Euclidean distance)", ".2f", "steelblue")


def cost_hist(cost_df: pd.DataFrame, unit: str) -> alt.Chart:
    if unit == "Dollars ($)":
        return _binned_chart(cost_df, "cost",  "Switching Cost ($)",         ",.0f", "seagreen")
    return     _binned_chart(cost_df, "years", "Years of origin wages",      ".2f",  "seagreen")


# ============================================================
# PAYBACK COLUMNS HELPER
# ============================================================

def add_payback_columns(df: pd.DataFrame, origin_code: str, cost_col: str) -> pd.DataFrame:
    """
    Adds destination wage, wage gap, and payback columns to a results table.
    `cost_col` must be a numeric column already present in df.
    """
    w_o = code_to_wage.get(origin_code)
    w_o_val = float(w_o) if (w_o and float(w_o) > 0) else np.nan

    def _pb(row):
        wd = code_to_wage.get(noc_str(row["Code"]))
        wd = float(wd) if (wd and float(wd) > 0) else np.nan
        gap = wd - w_o_val if (pd.notna(wd) and pd.notna(w_o_val)) else np.nan
        pb  = payback_period_months(row[cost_col], w_o_val, wd) if pd.notna(wd) else {}
        return pd.Series({
            "Destination wage ($/mo)": f"{wd:,.0f}"     if pd.notna(wd)  else "N/A",
            "Wage gap ($/mo)":         f"{gap:,.0f}"    if pd.notna(gap) else "N/A",
            "Payback (months)": f"{pb['payback_months']:.1f}" if pb.get("payback_months") is not None else "N/A",
            "Payback (years)":  f"{pb['payback_years']:.2f}"  if pb.get("payback_years")  is not None else "N/A",
        })

    return pd.concat([df, df.apply(_pb, axis=1)], axis=1)


# ============================================================
# LOOKUP PAGE  (shared by "Look up by code" and "Look up by title")
# ============================================================

def render_lookup_page(selected_code: str, geo_kwargs: dict, sidebar_params: dict):
    """Renders the full lookup UI for a given NOC code."""
    beta    = sidebar_params["beta"]
    alpha   = sidebar_params["alpha"]
    q       = sidebar_params["q"]
    edu_gap = sidebar_params["edu_gap"]
    n       = sidebar_params["n_results"]
    hist_unit = sidebar_params["hist_unit"]

    title    = code_to_title.get(selected_code, "Unknown")
    w_origin = code_to_wage.get(selected_code)

    top_results, bottom_results, all_scores = get_most_and_least_similar(selected_code, n=n, edu_gap=edu_gap)

    for label, results in [("Most Similar", top_results), ("Least Similar", bottom_results)]:
        st.subheader(f"{label} Occupations for {selected_code} – {title}")
        df = pd.DataFrame(results, columns=["Code", "Title", "Similarity Score"])
        df["_cost"] = df["Code"].apply(
            lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha, q=q,
                                               edu_gap=edu_gap, **geo_kwargs)
        )
        df["Switching Cost ($)"] = df["_cost"].map(
            lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
        )
        df["Years of origin wages"] = df["_cost"].map(
            lambda x: f"{x / (12.0 * float(w_origin)):.2f}"
            if (pd.notna(x) and w_origin and float(w_origin) > 0) else "N/A"
        )
        df = add_payback_columns(df, selected_code, "_cost")
        df = df.drop(columns=["_cost"])
        st.dataframe(df, use_container_width=True,
                     column_config={"Title": st.column_config.Column(width="large")})

    with st.expander("Switching cost decomposition (details)", expanded=False):
        shown_codes = (
            [r[0] for r in (top_results or [])] +
            [r[0] for r in (bottom_results or [])]
        )
        decomp = [
            d for c in shown_codes
            if (d := switching_cost_components(selected_code, c, beta, alpha, q,
                                               edu_gap=edu_gap, **geo_kwargs)) is not None
        ]
        if decomp:
            st.dataframe(pd.DataFrame(decomp), use_container_width=True)
        else:
            st.info("No decomposition rows available.")

    st.subheader(f"Similarity Score Distribution for {selected_code} – {title}")
    st.caption("Tip: hover on a bar to see which occupations fall in that similarity range.")
    if all_scores is not None and not all_scores.empty:
        st.altair_chart(similarity_hist(all_scores), use_container_width=True)

    costs_df = compute_switching_costs_from_origin(
        selected_code, beta=beta, alpha=alpha, q=q, edu_gap=edu_gap, **geo_kwargs
    )
    st.subheader(f"Switching Cost Distribution ({hist_unit}) from {selected_code} – {title}")
    st.caption("Tip: hover on a bar to see which occupations fall in that cost range.")
    st.altair_chart(cost_hist(costs_df, hist_unit), use_container_width=True)


# ============================================================
# ALLOCATION SIMULATION
# ============================================================

def allocate_surplus_to_shortage(
    origin_code:       str,
    beta:              float,
    alpha:             float,
    q:                 float,
    shortages_balance: dict,
    surplus_amount:    float,
    only_wage_gain:    bool = False,
    edu_gap:           int  = 0,
    geo_kwargs:        dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    geo_kwargs = geo_kwargs or {}
    origin_code = noc_str(origin_code)
    S = float(surplus_amount)

    w_origin = code_to_wage.get(origin_code)
    w_origin = float(w_origin) if (w_origin and float(w_origin) > 0) else np.nan

    # Build candidate list
    candidates = []
    for d, H in shortages_balance.items():
        if H is None or float(H) <= 0:
            continue
        w_dest = code_to_wage.get(d)
        w_dest = float(w_dest) if (w_dest and float(w_dest) > 0) else np.nan
        if only_wage_gain and (pd.isna(w_origin) or pd.isna(w_dest) or w_dest <= w_origin):
            continue
        cost = calculate_switching_cost(origin_code, d, beta=beta, alpha=alpha, q=q,
                                        edu_gap=edu_gap, **geo_kwargs)
        if cost is None or pd.isna(cost):
            continue
        candidates.append((d, float(cost), float(H), w_dest))

    candidates.sort(key=lambda x: x[1])

    rows = []
    totals = {"moved": 0.0, "cost": 0.0, "dest_wage": 0.0,
              "origin_wage": 0.0, "prep_months": 0.0}

    for d, cost, H, w_dest in candidates:
        if S <= 0:
            break
        x = min(S, H)
        if x <= 0:
            continue

        # Prep months for this specific pair
        tier_m   = expected_tier_upgrade_months(origin_code, d)
        z_raw    = origin_standardized_z(origin_code, d, q)
        z_eff    = max(float(z_raw), 0.0) if (z_raw is not None and not np.isnan(z_raw)) else 0.0
        recert_m = expected_recert_months_same_tier(origin_code, d, z_eff)
        prep_m   = CALIB_BASELINE_MONTHS + tier_m + recert_m

        S -= x
        shortages_balance[d] = H - x

        totals["moved"]       += x
        totals["cost"]        += x * cost
        totals["prep_months"] += x * prep_m
        if pd.notna(w_dest):   totals["dest_wage"]   += x * w_dest
        if pd.notna(w_origin): totals["origin_wage"] += x * w_origin

        rows.append({
            "Origin":               origin_code,
            "Origin title":         code_to_title.get(origin_code, "Unknown Title"),
            "Destination":          d,
            "Destination title":    code_to_title.get(d, "Unknown Title"),
            "Shortage magnitude (dest)": float(code_to_balance.get(d, np.nan)),
            "Flow moved":           x,
            "Switching cost per worker ($)":      cost,
            "Flow-weighted switching cost ($)":   x * cost,
            "Origin wage ($/mo)":      w_origin,
            "Destination wage ($/mo)": w_dest,
            "Wage gap ($/mo)": (w_dest - w_origin) if (pd.notna(w_dest) and pd.notna(w_origin)) else np.nan,
            "Tier upgrade months":        tier_m,
            "Recert months (expected)":   recert_m,
            "Preparation months (expected)": prep_m,
        })

    mv = totals["moved"]

    def _wavg(key):
        return float(totals[key] / mv) if mv > 0 else np.nan

    summary = {
        "origin":             origin_code,
        "origin_title":       code_to_title.get(origin_code, "Unknown Title"),
        "surplus_input":      float(surplus_amount),
        "moved_total":        mv,
        "unallocated_surplus": S,
        "total_switching_cost":          totals["cost"],
        "avg_switching_cost_per_worker": _wavg("cost"),
        "avg_origin_wage":               _wavg("origin_wage"),
        "avg_destination_wage":          _wavg("dest_wage"),
        "avg_wage_change": (
            float(totals["dest_wage"] / mv - totals["origin_wage"] / mv)
            if mv > 0 and totals["dest_wage"] > 0 and totals["origin_wage"] > 0 else np.nan
        ),
        "avg_preparation_months": _wavg("prep_months"),
    }

    return pd.DataFrame(rows), summary


def simulate_all_surplus_to_shortage(
    beta:           float,
    alpha:          float,
    q:              float,
    only_wage_gain: bool  = False,
    origin_order:   str   = "Largest surplus first",
    edu_gap:        int   = 0,
    geo_kwargs:     dict | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    geo_kwargs = geo_kwargs or {}

    if not code_to_balance:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    shortages_balance = {
        noc: float(bal) for noc, bal in code_to_balance.items()
        if bal is not None and not (isinstance(bal, float) and np.isnan(bal)) and float(bal) > 0
    }
    surplus_items = [
        (noc, abs(float(bal))) for noc, bal in code_to_balance.items()
        if bal is not None and not (isinstance(bal, float) and np.isnan(bal)) and float(bal) < 0
    ]

    if not surplus_items or not shortages_balance:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    surplus_items.sort(key=lambda x: x[1], reverse=(origin_order == "Largest surplus first"))

    all_flows       = []
    origin_summaries = []
    agg = {"moved": 0.0, "cost": 0.0, "dest_wage": 0.0,
           "origin_wage": 0.0, "prep_months": 0.0}

    for origin_code, surplus_amount in surplus_items:
        if sum(shortages_balance.values()) <= 0:
            break

        flows_df, summary = allocate_surplus_to_shortage(
            origin_code, beta, alpha, q, shortages_balance,
            surplus_amount, only_wage_gain, edu_gap, geo_kwargs,
        )
        origin_summaries.append(summary)

        if not flows_df.empty:
            all_flows.append(flows_df)
            mv   = float(flows_df["Flow moved"].sum())
            flow = flows_df["Flow moved"]
            agg["moved"]      += mv
            agg["cost"]       += float(flows_df["Flow-weighted switching cost ($)"].sum())
            agg["dest_wage"]  += float((flows_df["Destination wage ($/mo)"] * flow).dropna().sum())
            agg["origin_wage"] += float((flows_df["Origin wage ($/mo)"]     * flow).dropna().sum())
            agg["prep_months"] += float((flows_df["Preparation months (expected)"] * flow).dropna().sum())

    flows_all        = pd.concat(all_flows, ignore_index=True) if all_flows else pd.DataFrame()
    origin_summary_df = pd.DataFrame(origin_summaries)
    mv = agg["moved"]

    def _wavg(key):
        return float(agg[key] / mv) if mv > 0 else np.nan

    totals = {
        "total_moved":                   mv,
        "total_switching_cost":          agg["cost"],
        "avg_switching_cost_per_worker": _wavg("cost"),
        "avg_origin_wage":               _wavg("origin_wage"),
        "avg_destination_wage":          _wavg("dest_wage"),
        "avg_wage_change": (
            float(agg["dest_wage"] / mv - agg["origin_wage"] / mv)
            if mv > 0 and agg["dest_wage"] > 0 and agg["origin_wage"] > 0 else np.nan
        ),
        "avg_preparation_months":        _wavg("prep_months"),
        "shortage_remaining_total":      float(sum(shortages_balance.values())),
    }

    shortage_remaining_df = pd.DataFrame([
        {"Destination": k, "Remaining shortage": v, "Title": code_to_title.get(k, "Unknown Title")}
        for k, v in shortages_balance.items() if v > 0
    ]).sort_values("Remaining shortage", ascending=False)

    return flows_all, totals, origin_summary_df, shortage_remaining_df


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="APOLLO", layout="wide")
st.title("Welcome to the Analysis Platform for Occupational Linkages and Labour Outcomes (APOLLO)")

# ---- Sidebar ----
st.sidebar.subheader("Switching Cost Parameters")
beta  = st.sidebar.slider("Skill distance scaling (beta)",  0.0, 0.5,  0.14, 0.01)
alpha = st.sidebar.slider("Non-linear exponent (alpha)",    0.5, 3.0,  1.2,  0.1)
q     = st.sidebar.slider(
    "Origin baseline quantile q (Q_q(o) => z=0)",
    0.0, 0.5, float(BASELINE_Q_DEFAULT), 0.05,
)
EDU_GAP = st.sidebar.slider(
    "Max education distance allowed (0 = same level only)",
    0, 4, 0, 1,
)

USE_GEO     = st.sidebar.checkbox("Include geographic mobility cost", value=False)
USER_PROVINCE = None
GEO_C_MOVE  = 0.0
GEO_LAMBDA  = 0.0

if USE_GEO:
    if not jobprov_df.empty:
        province_options = sorted(jobprov_df["province"].dropna().unique())
        USER_PROVINCE = st.sidebar.selectbox("Worker's province of origin:", province_options)
        GEO_C_MOVE    = st.sidebar.number_input("Relocation cost if move required ($)",
                                                 0.0, value=GEO_C_MOVE_DEFAULT, step=1000.0)
        GEO_LAMBDA    = st.sidebar.slider("Weight on geographic cost (λ)", 0.0, 1.0, 0.5, 0.1)
    else:
        st.sidebar.info("No job_province_share file found — geographic component treated as zero.")

HIST_UNIT = st.sidebar.radio(
    "Switching cost histogram units",
    ["Dollars ($)", "Years of origin wages"],
    index=0,
)
n_results = st.sidebar.slider("Number of results to show:", 3, 20, 5)
menu = st.sidebar.radio("Choose an option:", [
    "Look up by code",
    "Look up by title",
    "Compare two jobs",
    "Surplus → Shortage pathways",
])

# Bundle sidebar params so pages don't need individual globals
sidebar_params = {
    "beta": beta, "alpha": alpha, "q": q,
    "edu_gap": EDU_GAP, "n_results": n_results, "hist_unit": HIST_UNIT,
}
geo_kwargs = {
    "use_geo":       USE_GEO,
    "user_province": USER_PROVINCE,
    "geo_c_move":    GEO_C_MOVE,
    "geo_lambda":    GEO_LAMBDA,
}

# ---- Risky / Safe sets ----
available_nocs = set(similarity_df.index) & set(code_to_wage.keys())
if code_to_roa:
    RISKY_CODES = {n for n in available_nocs
                   if code_to_roa.get(n) is not None and code_to_roa[n] >= ROA_RISKY_THRESHOLD}
    SAFE_CODES  = {n for n in available_nocs
                   if code_to_roa.get(n) is not None and code_to_roa[n] <  ROA_SAFE_THRESHOLD}
else:
    RISKY_CODES, SAFE_CODES = set(), set()

# ---- Calibration ----
CALIB_K, CALIB_PAIRS = compute_calibration_k_cached(
    tuple(sorted(RISKY_CODES)),
    tuple(sorted(SAFE_CODES)),
    target_usd=CALIB_TARGET_USD,
    beta=beta, alpha=alpha, q=q, edu_gap=EDU_GAP,
)

# Inject calibration k into geo_kwargs so all cost calls pick it up
geo_kwargs["calib_k"] = CALIB_K

with st.sidebar.expander("Calibration status", expanded=False):
    st.markdown(f"""
- **ROA records loaded:** {'Yes' if code_to_roa else 'No'}
- **Risky codes (ROA ≥ {ROA_RISKY_THRESHOLD}):** {len(RISKY_CODES)}
- **Safe codes (ROA < {ROA_SAFE_THRESHOLD}):** {len(SAFE_CODES)}
- **Risky→Safe pairs used for k:** {CALIB_PAIRS}
- **Calibration k:** {CALIB_K:.3f}
- **Standardization:** per-origin quantile-centered z (q = {q:.2f})
- **NOTE:** k calibrated using baseline-only ({CALIB_BASELINE_MONTHS:.0f} months)
- **Future shortage data loaded:** {'Yes' if code_to_balance else 'No'}
""")

with st.expander("Methodology"):
    st.markdown("""
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
- Raw skill distances are standardized **within each origin occupation's distribution of destinations**.
- Distances are centered at an **origin-specific baseline** defined by the *q-th quantile* of that distribution.
- A standardized distance of z=0 indicates **no additional skill-distance penalty**, not zero cost.

---

### Switching cost structure

Cost(o→d) = k · (M(o→d) · w_o) · (1 + β · z^α) · m(z) + λ · GeoCost_d

where M(o→d) = 2 + tier-upgrade months + expected recertification months.

---

### Calibration
- k is calibrated so that the average cost of transitions from high-automation-risk occupations
  to lower-risk occupations is approximately $24,000.
- Calibration uses only the 2-month baseline (no education or recertification months).

---

### Surplus–shortage simulations
- Workers are assigned using a **greedy, lowest-cost-first heuristic**.
- Results are **approximate, scenario-based indicators**, not forecasts.
""")


# ============================================================
# PAGES
# ============================================================

if menu == "Look up by code":
    code = st.text_input("Enter 5-digit occupation code:")
    if code:
        code = noc_str(code)
        if code in similarity_df.index:
            render_lookup_page(code, geo_kwargs, sidebar_params)
        else:
            st.error("❌ Code not found in similarity matrix.")


elif menu == "Look up by title":
    available_codes  = [c for c in code_to_title if c in similarity_df.index]
    title_options    = sorted(f"{c} – {code_to_title[c]}" for c in available_codes)
    selected_item    = st.selectbox("Select an occupation:", title_options)
    if selected_item:
        selected_code = selected_item.split(" – ")[0]
        render_lookup_page(selected_code, geo_kwargs, sidebar_params)


elif menu == "Compare two jobs":
    available_codes = [c for c in code_to_title if c in similarity_df.index]
    title_options   = sorted(f"{c} – {code_to_title[c]}" for c in available_codes)

    job1_item = st.selectbox("Select first occupation:",  title_options, key="job1")
    job2_item = st.selectbox("Select second occupation:", title_options, key="job2")
    job1_code, job1_title = job1_item.split(" – ", 1)
    job2_code, job2_title = job2_item.split(" – ", 1)

    if st.button("Compare"):
        result = compare_two_jobs(job1_code, job2_code)
        if result:
            score, rank, total = result
            cost = calculate_switching_cost(
                job1_code, job2_code, beta=beta, alpha=alpha, q=q,
                edu_gap=EDU_GAP, **geo_kwargs,
            )

            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({job1_title}) vs {job2_code} ({job2_title})\n"
                f"- Similarity score (raw distance): `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations"
            )

            if cost is not None:
                w_o = code_to_wage.get(job1_code)
                w_o = float(w_o) if (w_o and float(w_o) > 0) else None
                years_str = (f" ({cost / w_o / 12:.2f} years of origin wages)" if w_o else "")
                geo_note  = (f" (includes geographic mobility for {USER_PROVINCE})"
                             if (USE_GEO and USER_PROVINCE) else "")
                st.info(
                    f"💰 **Estimated Switching Cost** (from {job1_code} to {job2_code}): "
                    f"`${cost:,.2f}`{geo_note}{years_str}"
                )

                w_d = code_to_wage.get(job2_code)
                w_d = float(w_d) if (w_d and float(w_d) > 0) else None

                st.subheader("Wage payback period")
                if w_o and w_d:
                    pb = payback_period_months(cost, w_o, w_d)
                    st.write(
                        f"- Origin monthly wage: **${w_o:,.0f}**\n"
                        f"- Destination monthly wage: **${w_d:,.0f}**\n"
                        f"- Monthly wage gap: **${pb['wage_gap_monthly']:,.0f}**"
                    )
                    if pb["payback_months"] is None:
                        st.info(f"ℹ️ {pb['status']}")
                    else:
                        st.success(
                            f"✅ Estimated payback: **{pb['payback_months']:.1f} months** "
                            f"(**{pb['payback_years']:.2f} years**)"
                        )
                else:
                    st.info("Payback not available (missing wage data).")

                with st.expander("Switching cost decomposition (details)", expanded=False):
                    d = switching_cost_components(
                        job1_code, job2_code, beta, alpha, q,
                        edu_gap=EDU_GAP, **geo_kwargs,
                    )
                    if d:
                        st.dataframe(pd.DataFrame([d]), use_container_width=True)
                    else:
                        st.info("No decomposition available.")
            else:
                st.info(
                    f"ℹ️ Switching cost not reported — occupations are further apart in education "
                    f"than the allowed maximum ({EDU_GAP})."
                )
        else:
            st.error("❌ Could not compare occupations.")


elif menu == "Surplus → Shortage pathways":
    st.header("Surplus → Shortage pathways")
    st.caption("Overlays surplus/shortage projections on the switching cost model and simulates reallocations.")

    if not code_to_balance:
        st.warning("No projection file found. Add future_shortage.xlsx (columns: noc, status, magnitude).")
    else:
        origins = sorted({
            c for c in similarity_df.index
            if (b := code_to_balance.get(c)) is not None
            and not (isinstance(b, float) and np.isnan(b))
            and float(b) < 0
        })

        if not origins:
            st.info("No occupations classified as surplus in the projection data.")
        else:
            origin_labels = [f"{c} – {code_to_title.get(c,'Unknown Title')}" for c in origins]
            origin_item   = st.selectbox("Select a surplus origin occupation:", origin_labels)
            origin_code   = origin_item.split(" – ")[0]
            origin_bal    = float(code_to_balance.get(origin_code, np.nan))
            w_origin      = code_to_wage.get(origin_code)

            c1, c2, c3 = st.columns(3)
            c1.metric("Origin status",               code_to_status.get(origin_code, "unknown"))
            c2.metric("Origin magnitude (surplus<0)", f"{origin_bal:,.2f}" if pd.notna(origin_bal) else "N/A")
            c3.metric("Origin wage ($/mo)",
                      f"{float(w_origin):,.0f}" if (w_origin and float(w_origin) > 0) else "N/A")

            # ---- B: Cheapest pathways ----
            st.subheader("B) Cheapest shortage destinations")
            top_n         = st.slider("Number of pathways to show", 5, 100, 25, 5)
            min_short     = st.number_input("Minimum shortage magnitude", value=0.0, min_value=0.0)
            only_wage_gain = st.checkbox("Only show destinations with higher wage than origin", value=False)

            shortages_tmp = {
                noc: float(bal) for noc, bal in code_to_balance.items()
                if bal is not None and not (isinstance(bal, float) and np.isnan(bal)) and float(bal) > 0
            }
            flows_tmp, _ = allocate_surplus_to_shortage(
                origin_code, beta, alpha, q, shortages_tmp,
                _INFINITE_SURPLUS, only_wage_gain, EDU_GAP, geo_kwargs,
            )

            if not flows_tmp.empty:
                df = flows_tmp.copy()
                df["Destination balance (shortage +)"] = df["Destination"].map(
                    lambda d: float(code_to_balance.get(d, np.nan))
                )
                df = df[
                    df["Destination balance (shortage +)"].notna() &
                    (df["Destination balance (shortage +)"] >= float(min_short))
                ].sort_values("Switching cost per worker ($)").head(int(top_n))

                fmt_cols = {
                    "Destination balance (shortage +)": lambda v: f"{float(v):,.2f}" if pd.notna(v) else "N/A",
                    "Destination wage ($/mo)":          lambda v: f"{float(v):,.0f}"  if pd.notna(v) else "N/A",
                    "Wage gap ($/mo)":                  lambda v: f"{float(v):,.0f}"  if pd.notna(v) else "N/A",
                    "Switching cost per worker ($)":    lambda v: f"{float(v):,.0f}"  if pd.notna(v) else "N/A",
                    "Preparation months (expected)":    lambda v: f"{float(v):.1f}"   if pd.notna(v) else "N/A",
                }
                out = df[[
                    "Destination", "Destination title", "Destination balance (shortage +)",
                    "Destination wage ($/mo)", "Wage gap ($/mo)",
                    "Switching cost per worker ($)", "Preparation months (expected)",
                ]].copy()
                for col, fn in fmt_cols.items():
                    out[col] = out[col].map(fn)
                st.dataframe(out, use_container_width=True)
            else:
                st.info("No eligible shortage destinations found.")

            # ---- C: Single origin simulation ----
            st.divider()
            st.subheader("C) Reallocate all workers from this surplus occupation")
            sim_only_wage = st.checkbox(
                "Require destination wage > origin wage", value=only_wage_gain, key="sim_wage_origin"
            )

            if st.button("Run origin allocation simulation"):
                shortages_balance = {
                    noc: float(bal) for noc, bal in code_to_balance.items()
                    if bal is not None and not (isinstance(bal, float) and np.isnan(bal)) and float(bal) > 0
                }
                surplus_amount = abs(float(code_to_balance.get(origin_code, 0.0)))

                flows_df, summary = allocate_surplus_to_shortage(
                    origin_code, beta, alpha, q, shortages_balance,
                    surplus_amount, sim_only_wage, EDU_GAP, geo_kwargs,
                )

                if summary["moved_total"] <= 0:
                    st.warning("No allocation possible — check education gap, wage filter, or missing data.")
                else:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Moved (total)",          f"{summary['moved_total']:,.0f}")
                    m2.metric("Unallocated surplus",    f"{summary['unallocated_surplus']:,.0f}")
                    m3.metric("Total switching cost",   f"${summary['total_switching_cost']:,.0f}")
                    m4.metric("Avg cost per worker",    f"${summary['avg_switching_cost_per_worker']:,.0f}")
                    m5.metric("Avg preparation months",
                              f"{summary['avg_preparation_months']:.1f}"
                              if pd.notna(summary['avg_preparation_months']) else "N/A")

                    n1, n2, n3 = st.columns(3)
                    n1.metric("Avg origin wage ($/mo)",
                              f"${summary['avg_origin_wage']:,.0f}" if pd.notna(summary['avg_origin_wage']) else "N/A")
                    n2.metric("Avg new wage ($/mo)",
                              f"${summary['avg_destination_wage']:,.0f}" if pd.notna(summary['avg_destination_wage']) else "N/A")
                    n3.metric("Avg wage change ($/mo)",
                              f"${summary['avg_wage_change']:,.0f}" if pd.notna(summary['avg_wage_change']) else "N/A")

                    st.subheader("Allocation results")
                    st.dataframe(flows_df, use_container_width=True)

                    bundle = make_zip_bundle({
                        f"origin_{origin_code}_flows.csv":   flows_df,
                        f"origin_{origin_code}_summary.csv": pd.DataFrame([summary]),
                        "readme.txt": (
                            "Bundle contents:\n"
                            "- origin_*_flows.csv   : destination-level flows and costs\n"
                            "- origin_*_summary.csv : totals and averages\n"
                        ),
                    })
                    st.download_button("Download origin simulation bundle (ZIP)", bundle,
                                       f"origin_{origin_code}_simulation_bundle.zip", "application/zip")

                    bar_df = flows_df.copy()
                    bar_df["label"] = bar_df["Destination"] + " – " + bar_df["Destination title"]
                    st.altair_chart(
                        alt.Chart(bar_df).mark_bar().encode(
                            y=alt.Y("label:N", sort="-x", title="Destination"),
                            x=alt.X("Flow moved:Q", title="Flow moved"),
                            tooltip=[
                                "Destination:N", "Destination title:N",
                                alt.Tooltip("Flow moved:Q",                        format=",.0f"),
                                alt.Tooltip("Switching cost per worker ($):Q",     format=",.0f"),
                                alt.Tooltip("Flow-weighted switching cost ($):Q",  format=",.0f"),
                                alt.Tooltip("Preparation months (expected):Q",     format=".1f"),
                                alt.Tooltip("Destination wage ($/mo):Q",           format=",.0f"),
                            ],
                        ).properties(height=min(900, 30 * max(5, len(bar_df)))),
                        use_container_width=True,
                    )

            # ---- D: System-wide simulation ----
            st.divider()
            st.subheader("D) System-wide reallocation across all surplus occupations")
            st.caption(
                "Greedy simulation — not a global optimizer. "
                "Earlier origins consume shortage capacity first."
            )

            origin_order = st.radio(
                "Order of surplus origins",
                ["Largest surplus first", "Smallest surplus first"],
                horizontal=True,
            )
            sim_only_wage_all = st.checkbox(
                "Require destination wage > origin wage (all-origins)", value=False, key="sim_wage_all"
            )

            if st.button("Run full system simulation"):
                flows_all, totals, origin_summary_df, shortage_remaining_df = simulate_all_surplus_to_shortage(
                    beta=beta, alpha=alpha, q=q,
                    only_wage_gain=sim_only_wage_all,
                    origin_order=origin_order,
                    edu_gap=EDU_GAP,
                    geo_kwargs=geo_kwargs,
                )

                if totals.get("total_moved", 0.0) <= 0:
                    st.warning("No allocation possible — check missing wage data, education gap, or wage filter.")
                else:
                    t1, t2, t3, t4, t5 = st.columns(5)
                    t1.metric("Total moved",            f"{totals['total_moved']:,.0f}")
                    t2.metric("Total switching cost",   f"${totals['total_switching_cost']:,.0f}")
                    t3.metric("Avg cost per worker",    f"${totals['avg_switching_cost_per_worker']:,.0f}")
                    t4.metric("Shortage remaining",     f"{totals['shortage_remaining_total']:,.0f}")
                    t5.metric("Avg preparation months",
                              f"{totals['avg_preparation_months']:.1f}"
                              if pd.notna(totals['avg_preparation_months']) else "N/A")

                    u1, u2, u3 = st.columns(3)
                    u1.metric("Avg origin wage ($/mo)",
                              f"${totals['avg_origin_wage']:,.0f}" if pd.notna(totals['avg_origin_wage']) else "N/A")
                    u2.metric("Avg new wage ($/mo)",
                              f"${totals['avg_destination_wage']:,.0f}" if pd.notna(totals['avg_destination_wage']) else "N/A")
                    u3.metric("Avg wage change ($/mo)",
                              f"${totals['avg_wage_change']:,.0f}" if pd.notna(totals['avg_wage_change']) else "N/A")

                    st.subheader("Flows (all origins)")
                    st.dataframe(flows_all, use_container_width=True)
                    st.subheader("Origin summaries")
                    st.dataframe(origin_summary_df, use_container_width=True)
                    st.subheader("Remaining shortage after simulation")
                    st.dataframe(shortage_remaining_df, use_container_width=True)

                bundle = make_zip_bundle({
                    "system_flows_all.csv":          flows_all,
                    "system_totals.csv":             pd.DataFrame([totals]),
                    "system_origin_summaries.csv":   origin_summary_df,
                    "system_shortage_remaining.csv": shortage_remaining_df,
                    "readme.txt": (
                        "Bundle contents:\n"
                        "- system_flows_all.csv         : all origin→destination flows\n"
                        "- system_totals.csv            : aggregate totals and averages\n"
                        "- system_origin_summaries.csv  : per-origin summaries\n"
                        "- system_shortage_remaining.csv: remaining shortage after simulation\n"
                        "Note: greedy heuristic — ordering of surplus origins affects results.\n"
                    ),
                })
                st.download_button("Download system simulation bundle (ZIP)", bundle,
                                   "system_simulation_bundle.zip", "application/zip")
