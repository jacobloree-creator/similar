import streamlit as st
import pandas as pd
import os
import altair as alt
import numpy as np

# =========================
# Defaults (overridden by sidebar)
# =========================
BASELINE_Q = 0.10  # per-origin quantile baseline (q-quantile distance => z=0)

# Recertification defaults (uni->uni expected months = 48 * P(recert))
RECERT_BASE_MONTHS_UNI = 48
P_RECERT_FIRST = 0.80
P_RECERT_THIRD = 0.35
P_RECERT_FOURTH = 0.10

# ---------- Load Data ----------
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

    # ---- Load automation risk (CSV or XLSX). If file missing/invalid, leave empty dict. ----
    code_to_roa = {}
    roa_csv = os.path.join(base_path, "automation_risk.csv")
    roa_xlsx = os.path.join(base_path, "automation_risk.xlsx")

    def _norm(series):
        return series.astype(str).str.zfill(5).str.strip()

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
    )


(
    similarity_df,
    code_to_title,
    title_to_code,
    code_to_wage,
    code_to_roa,
    jobprov_df,
) = load_data()

# =========================
# Helper Functions
# =========================

def noc_str(code: str) -> str:
    return str(code).zfill(5).strip()


def get_education_level(noc_code):
    """
    Thousands digit by your convention (second digit of 5-digit code).
    Example: 14110 -> 4

    IMPORTANT: This digit is your education tier where
      1=university, 2=4yr college, 3=2yr college, 4=high school, 5=no education required
    """
    try:
        s = noc_str(noc_code)
        return int(s[1])
    except Exception:
        return None


# --- Education-months matrix (tier upgrading only) ---
EDU_MONTHS_MATRIX = np.array([
    # dest:  1   2   3   4   5
    [   0,  0,  0,  0,  0],   # origin 1 (uni)
    [  18,  0,  0,  0,  0],   # origin 2 -> uni
    [  30, 14,  0,  0,  0],   # origin 3 -> uni / 4yr college
    [  48, 30, 18,  0,  0],   # origin 4 -> uni / 4yr / 2yr
    [  60, 36, 24, 12,  0],   # origin 5 -> uni / 4yr / 2yr / HS
], dtype=float)

EDU_MONTHS_CAP = 60.0  # cap at 5 years


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
    You stated: 1st, 3rd, and 4th digit denote group.
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


def recert_probability(origin: str, dest: str) -> float:
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


def expected_recert_months_uni_to_uni(origin: str, dest: str) -> float:
    """
    Expected recertification months for university->university switches:
      months = RECERT_BASE_MONTHS_UNI * P(recert | mismatch level)

    Applies ONLY if both tiers are university (tier==1).
    """
    if get_education_level(origin) != 1 or get_education_level(dest) != 1:
        return 0.0
    p = recert_probability(origin, dest)
    m = float(RECERT_BASE_MONTHS_UNI) * float(p)
    return float(max(0.0, m))


# --- Option 1: Per-origin shifted standardization (quantile baseline) ---
def origin_standardized_z(origin_code: str, dest_code: str, q: float = None):
    """
    Per-origin shifted z-score:
      z_q = (d_od - Q_q(o)) / std_o

    Q_q(o) is the q-quantile of the origin's destination-distance distribution.
    This keeps per-origin scaling (std_o) but moves the zero point.
    """
    if q is None:
        q = BASELINE_Q

    if origin_code not in similarity_df.index or dest_code not in similarity_df.columns:
        return None

    row = similarity_df.loc[origin_code].dropna()
    if row.empty or dest_code not in row.index:
        return None

    std = float(row.std())  # pandas default ddof=1
    if std == 0.0 or np.isnan(std):
        return None

    baseline = float(row.quantile(q))
    d = float(row.loc[dest_code])
    return float((d - baseline) / std)


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


def training_multiplier(z_eff):
    """
    One-sided bins, expects z_eff >= 0.
    (We hinge z only where needed for alpha fractional power safety.)
    """
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


# =========================
# Calibration (k) — BEFORE adding additional education effects
# =========================
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
    IMPORTANT: Calibration uses ONLY the 2-month baseline (no tier-upgrade months and no recertification months).
    Uses quantile-centered per-origin z and hinges only for alpha-power safety.
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

            z_eff = max(float(z_raw), 0.0)  # alpha may be fractional
            base = 2.0 * w_origin
            dist_term = 1 + beta * (z_eff ** alpha)
            mult = float(training_multiplier(z_eff))
            raw_cost = base * dist_term * mult
            pairs.append(raw_cost)

    if not pairs:
        return 1.0, 0

    mean_raw = float(np.mean(pairs))
    if mean_raw <= 0:
        return 1.0, 0

    return float(target_usd) / mean_raw, len(pairs)


# =========================
# Switching cost functions (with tier-upgrade + expected uni->uni recert months)
# =========================
def calculate_switching_cost(code1, code2, beta=0.14, alpha=1.2, q: float = None):
    level1 = get_education_level(code1)
    level2 = get_education_level(code2)
    if level1 is None or level2 is None:
        return None
    if abs(level1 - level2) > EDU_GAP:
        return None

    z_raw = origin_standardized_z(code1, code2, q=q)
    if z_raw is None:
        return None

    z_eff = max(float(z_raw), 0.0)  # alpha fractional safety

    w_origin = code_to_wage.get(code1)
    if w_origin is None or float(w_origin) <= 0:
        return None
    w_origin = float(w_origin)

    tier_months = expected_tier_upgrade_months(code1, code2)
    recert_months = expected_recert_months_uni_to_uni(code1, code2)
    edu_months_total = float(tier_months) + float(recert_months)

    base = (2.0 + edu_months_total) * w_origin
    dist_term = 1 + beta * (z_eff ** alpha)
    mult = float(training_multiplier(z_eff))

    skill_cost = float(CALIB_K) * base * dist_term * mult

    if USE_GEO:
        geo = geographic_cost(code2, USER_PROVINCE, GEO_C_MOVE)
        return float(skill_cost) + float(GEO_LAMBDA) * float(geo)
    else:
        return float(skill_cost)


def switching_cost_components(origin_code, dest_code, beta=0.14, alpha=1.2, q: float = None):
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

    tier_months = expected_tier_upgrade_months(origin_code, dest_code)
    recert_months = expected_recert_months_uni_to_uni(origin_code, dest_code)
    edu_months_total = float(tier_months) + float(recert_months)

    base = (2.0 + edu_months_total) * w_origin
    dist_term = 1 + beta * (z_eff ** alpha)
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

    f = group_mismatch_flags(origin_code, dest_code)
    p_recert = recert_probability(origin_code, dest_code) if (get_education_level(origin_code) == 1 and get_education_level(dest_code) == 1) else 0.0

    return {
        "Origin": origin_code,
        "Destination": dest_code,
        "Title": code_to_title.get(dest_code, "Unknown Title"),

        "Baseline quantile q": float(BASELINE_Q),
        "z_raw (quantile-centered)": float(z_raw),
        "z_eff (hinged for cost)": float(z_eff),

        "Tier upgrade months": float(tier_months),
        "Recert base months (uni→uni)": float(RECERT_BASE_MONTHS_UNI) if (get_education_level(origin_code) == 1 and get_education_level(dest_code) == 1) else 0.0,
        "P(recert | group mismatch)": float(p_recert),
        "Group diff (1st)": int(f["diff_1st"]),
        "Group diff (3rd)": int(f["diff_3rd"]),
        "Group diff (4th)": int(f["diff_4th"]),
        "Expected recert months (uni→uni)": float(recert_months),
        "Education/credential months (total)": float(edu_months_total),

        "Base ((2+edu_total)×origin wage)": float(base),
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
    }


def compute_switching_costs_from_origin(origin_code, beta, alpha, q: float = None):
    """Returns a dataframe with both $ cost and years-of-origin-wages cost for all allowed destinations."""
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


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="APOLLO", layout="wide")
st.title("Welcome to the Analysis Platform for Occupational Linkages and Labour Outcomes (APOLLO)")

# Sidebar parameters
st.sidebar.subheader("Switching Cost Parameters")
beta = st.sidebar.slider("Skill distance scaling (beta)", min_value=0.0, max_value=0.5, value=0.14, step=0.01)
alpha = st.sidebar.slider("Non-linear exponent (alpha)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

st.sidebar.subheader("Origin-standardization baseline")
BASELINE_Q = st.sidebar.slider(
    "Baseline quantile q (q-quantile distance = z=0)",
    min_value=0.0,
    max_value=0.5,
    value=0.10,
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

# Toggle for histogram units
HIST_UNIT = st.sidebar.radio(
    "Switching cost histogram units",
    options=["Dollars ($)", "Years of origin wages"],
    index=0,
)

# =========================
# Recertification controls (uni->uni only)
# =========================
st.sidebar.subheader("Recertification (uni→uni expected months)")
RECERT_BASE_MONTHS_UNI = st.sidebar.slider(
    "Base recert months if recert required (uni→uni)",
    min_value=0,
    max_value=60,
    value=int(RECERT_BASE_MONTHS_UNI),
    step=1,
)
P_RECERT_FIRST = st.sidebar.slider(
    "P(recert) if 1st digit differs",
    min_value=0.0,
    max_value=1.0,
    value=float(P_RECERT_FIRST),
    step=0.05,
)
P_RECERT_THIRD = st.sidebar.slider(
    "P(recert) if 3rd digit differs (and 1st same)",
    min_value=0.0,
    max_value=1.0,
    value=float(P_RECERT_THIRD),
    step=0.05,
)
P_RECERT_FOURTH = st.sidebar.slider(
    "P(recert) if only 4th digit differs",
    min_value=0.0,
    max_value=1.0,
    value=float(P_RECERT_FOURTH),
    step=0.05,
)

# ---------- Risky/Safe sets ----------
RISKY_THRESHOLD = 0.70
SAFE_THRESHOLD = 0.70

available_nocs = set(similarity_df.index) & set(code_to_wage.keys())
if code_to_roa:
    RISKY_CODES = {noc for noc in available_nocs if code_to_roa.get(noc) is not None and code_to_roa[noc] >= RISKY_THRESHOLD}
    SAFE_CODES = {noc for noc in available_nocs if code_to_roa.get(noc) is not None and code_to_roa[noc] < SAFE_THRESHOLD}
else:
    RISKY_CODES, SAFE_CODES = set(), set()

# ---------- Calibration (k) computed BEFORE additional education effects ----------
CALIB_K, CALIB_PAIRS = compute_calibration_k_cached(
    tuple(sorted(RISKY_CODES)),
    tuple(sorted(SAFE_CODES)),
    target_usd=24000.0,
    beta=beta,
    alpha=alpha,
    q=BASELINE_Q,
)

with st.sidebar.expander("Calibration status", expanded=False):
    st.markdown(
        f"""
- **ROA records loaded:** {'Yes' if code_to_roa else 'No'}
- **Risky codes (ROA ≥ 0.70):** {len(RISKY_CODES)}
- **Safe codes (ROA < 0.70):** {len(SAFE_CODES)}
- **Risky→Safe pairs used for k:** {CALIB_PAIRS}
- **Calibration k (applied in all costs):** {CALIB_K:.3f}
- **Geo data loaded:** {'Yes' if (jobprov_df is not None and not jobprov_df.empty) else 'No'}
- **Standardization:** per-origin quantile-centered z (q = {BASELINE_Q:.2f})
- **IMPORTANT:** k is calibrated using a **2-month baseline only** (no tier-upgrade or recert months)
"""
    )

with st.expander("Methodology"):
    st.markdown(
        """
- Similarity scores are Euclidean distances of O*NET skill/ability/knowledge vectors (smaller = more similar).  
- Distances are standardized **within each origin occupation’s distribution of destinations**, centered at an **origin-specific quantile** (baseline).  
- **Tier-upgrading months** are derived from your education-tier matrix.  
- **Recertification months (uni→uni only)** are modeled as **48 months × P(recert)**, where P(recert) depends on NOC group digit differences (1st > 3rd > 4th).  
- Switching costs are scaled by **(2 + tier_months + expected_recert_months)** of origin wages and adjusted by a distance term and training multiplier.  
- A global calibration factor **k** is chosen so that average **risky → safe** transitions are about **$24,000** under a **2-month baseline only** (calibration excludes education effects).  
        """
    )
    st.latex(r"""z_{o\to d}(q)=\frac{dist(o,d)-Q_q(o)}{\sigma_o}""")
    st.latex(
        r"""
\text{SkillCost}
= k \cdot \left((2 + T_{tier} + T_{recert})\,w_o\right)
  \cdot \left(1 + \beta\,\max(z_{o\to d}(q),0)^{\alpha}\right)
  \cdot m(\max(z_{o\to d}(q),0))
"""
    )

n_results = st.sidebar.slider("Number of results to show:", min_value=3, max_value=20, value=5)
menu = st.sidebar.radio("Choose an option:", ["Look up by code", "Look up by title", "Compare two jobs"])


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
            df_top["_sc_numeric"] = df_top["Code"].apply(lambda x: calculate_switching_cost(code, x, beta=beta, alpha=alpha, q=BASELINE_Q))
            df_top["Switching Cost ($)"] = df_top["_sc_numeric"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            df_top["Years of origin wages"] = df_top["_sc_numeric"].map(
                lambda x: (x / (12.0 * float(w_origin))) if (pd.notnull(x) and w_origin is not None and float(w_origin) > 0) else np.nan
            ).map(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")
            df_top = df_top.drop(columns=["_sc_numeric"])
            st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

            st.subheader(f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
            df_bottom["_sc_numeric"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(code, x, beta=beta, alpha=alpha, q=BASELINE_Q))
            df_bottom["Switching Cost ($)"] = df_bottom["_sc_numeric"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            df_bottom["Years of origin wages"] = df_bottom["_sc_numeric"].map(
                lambda x: (x / (12.0 * float(w_origin))) if (pd.notnull(x) and w_origin is not None and float(w_origin) > 0) else np.nan
            ).map(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")
            df_bottom = df_bottom.drop(columns=["_sc_numeric"])
            st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

            with st.expander("Switching cost decomposition (details)", expanded=False):
                shown_codes = list(pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])["Code"].astype(str)) + \
                              list(pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])["Code"].astype(str))

                decomp_rows = []
                for dest in shown_codes:
                    d = switching_cost_components(code, dest, beta=beta, alpha=alpha, q=BASELINE_Q)
                    if d is not None:
                        decomp_rows.append(d)

                if decomp_rows:
                    decomp_df = pd.DataFrame(decomp_rows)
                    preferred_cols = [
                        "Origin", "Destination", "Title",
                        "Baseline quantile q",
                        "z_raw (quantile-centered)", "z_eff (hinged for cost)",
                        "Tier upgrade months",
                        "Recert base months (uni→uni)",
                        "P(recert | group mismatch)",
                        "Group diff (1st)", "Group diff (3rd)", "Group diff (4th)",
                        "Expected recert months (uni→uni)",
                        "Education/credential months (total)",
                        "Base ((2+edu_total)×origin wage)",
                        "Distance term", "Training mult", "k (calibration)",
                        "Skill cost", "Geo add", "Total cost",
                        "Months of origin wages", "Years of origin wages",
                    ]
                    decomp_df = decomp_df[[c for c in preferred_cols if c in decomp_df.columns]]

                    money_cols = ["Base ((2+edu_total)×origin wage)", "Skill cost", "Geo add", "Total cost"]
                    for c in money_cols:
                        if c in decomp_df.columns:
                            decomp_df[c] = decomp_df[c].map(lambda v: f"{float(v):,.2f}")

                    for c in ["Baseline quantile q", "z_raw (quantile-centered)", "z_eff (hinged for cost)", "Distance term", "k (calibration)"]:
                        if c in decomp_df.columns:
                            decomp_df[c] = decomp_df[c].map(lambda v: f"{float(v):.3f}")

                    for c in ["P(recert | group mismatch)"]:
                        if c in decomp_df.columns:
                            decomp_df[c] = decomp_df[c].map(lambda v: f"{float(v):.2f}")

                    for c in ["Training mult"]:
                        if c in decomp_df.columns:
                            decomp_df[c] = decomp_df[c].map(lambda v: f"{float(v):.2f}")

                    for c in ["Tier upgrade months", "Recert base months (uni→uni)", "Expected recert months (uni→uni)", "Education/credential months (total)"]:
                        if c in decomp_df.columns:
                            decomp_df[c] = decomp_df[c].map(lambda v: f"{float(v):.0f}")

                    if "Months of origin wages" in decomp_df.columns:
                        decomp_df["Months of origin wages"] = decomp_df["Months of origin wages"].map(lambda v: f"{float(v):.1f}")
                    if "Years of origin wages" in decomp_df.columns:
                        decomp_df["Years of origin wages"] = decomp_df["Years of origin wages"].map(lambda v: f"{float(v):.2f}")

                    st.dataframe(decomp_df, use_container_width=True)
                else:
                    st.info("No decomposition rows available (likely filtered out by education distance or missing data).")

            st.subheader(f"Similarity Score Distribution for {code} – {code_to_title.get(code,'Unknown')}")
            st.caption("Tip: hover on a bar to see which occupations fall in that similarity range.")
            st.altair_chart(similarity_hist_with_titles(all_scores), use_container_width=True)

            costs_df = compute_switching_costs_from_origin(code, beta=beta, alpha=alpha, q=BASELINE_Q)
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
        df_top["_sc_numeric"] = df_top["Code"].apply(lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha, q=BASELINE_Q))
        df_top["Switching Cost ($)"] = df_top["_sc_numeric"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        df_top["Years of origin wages"] = df_top["_sc_numeric"].map(
            lambda x: (x / (12.0 * float(w_origin))) if (pd.notnull(x) and w_origin is not None and float(w_origin) > 0) else np.nan
        ).map(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")
        df_top = df_top.drop(columns=["_sc_numeric"])
        st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        st.subheader(f"Least Similar Occupations for {selected_code} – {selected_title}")
        df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
        df_bottom["_sc_numeric"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha, q=BASELINE_Q))
        df_bottom["Switching Cost ($)"] = df_bottom["_sc_numeric"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        df_bottom["Years of origin wages"] = df_bottom["_sc_numeric"].map(
            lambda x: (x / (12.0 * float(w_origin))) if (pd.notnull(x) and w_origin is not None and float(w_origin) > 0) else np.nan
        ).map(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")
        df_bottom = df_bottom.drop(columns=["_sc_numeric"])
        st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        with st.expander("Switching cost decomposition (details)", expanded=False):
            shown_codes = list(pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])["Code"].astype(str)) + \
                          list(pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])["Code"].astype(str))

            decomp_rows = []
            for dest in shown_codes:
                d = switching_cost_components(selected_code, dest, beta=beta, alpha=alpha, q=BASELINE_Q)
                if d is not None:
                    decomp_rows.append(d)

            if decomp_rows:
                decomp_df = pd.DataFrame(decomp_rows)
                st.dataframe(decomp_df, use_container_width=True)
            else:
                st.info("No decomposition rows available (likely filtered out by education distance or missing data).")

        st.subheader(f"Similarity Score Distribution for {selected_code} – {selected_title}")
        st.caption("Tip: hover on a bar to see which occupations fall in that similarity range.")
        st.altair_chart(similarity_hist_with_titles(all_scores), use_container_width=True)

        costs_df = compute_switching_costs_from_origin(selected_code, beta=beta, alpha=alpha, q=BASELINE_Q)
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
            cost = calculate_switching_cost(job1_code, job2_code, beta=beta, alpha=alpha, q=BASELINE_Q)

            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({job1_title}) vs {job2_code} ({job2_title})\n"
                f"- Similarity score (raw distance): `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations (#{rank} most similar to {job1_code})"
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

                with st.expander("Switching cost decomposition (details)", expanded=False):
                    d = switching_cost_components(job1_code, job2_code, beta=beta, alpha=alpha, q=BASELINE_Q)
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
