import streamlit as st
import pandas as pd
import os
import altair as alt
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

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

    # ---- Standardize distances (z-scores) ----
    flat_scores = similarity_df.where(~pd.isna(similarity_df)).stack().values
    mean_val, std_val = flat_scores.mean(), flat_scores.std()
    standardized_df = (similarity_df - mean_val) / std_val

    return (
        similarity_df,
        standardized_df,
        code_to_title,
        title_to_code,
        code_to_wage,
        code_to_roa,
        jobprov_df,
    )


(
    similarity_df,
    standardized_df,
    code_to_title,
    title_to_code,
    code_to_wage,
    code_to_roa,
    jobprov_df,
) = load_data()

# ---------- Helper Functions ----------

def get_education_level(noc_code):
    """
    Thousands digit by your convention (second digit of 5-digit code).
    Example: 14110 -> 4
    """
    try:
        s = str(noc_code).zfill(5)
        return int(s[1])
    except Exception:
        return None


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


def training_multiplier(z_score):
    z = abs(float(z_score))
    if z < 0.5:
        return 1.0
    elif z < 1.0:
        return 1.2
    elif z < 1.5:
        return 1.5
    else:
        return 2.0


def geographic_cost(dest_code, province, C_move=20000.0):
    """
    Expected relocation cost for moving into dest_code from given province.
    Uses jobprov_df (share of workers in each province for that occupation).
    """
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


def compute_calibration_k(risky_codes, safe_codes, target_usd=24000.0, beta=0.14, alpha=1.2):
    """
    Compute a global k so that the average cost for risky->safe pairs
    (across the full universe; no EDU restriction here) equals target_usd.

    Base cost is 2 * w_origin (two months of origin wages).
    Returns (k, n_pairs_used).
    """
    pairs = []
    for r in risky_codes:
        if r not in standardized_df.index:
            continue
        w_origin = code_to_wage.get(r)
        if w_origin is None:
            continue
        for s in safe_codes:
            if s == r or s not in standardized_df.index:
                continue

            z = standardized_df.loc[r, s]
            if pd.isna(z):
                z = standardized_df.loc[s, r]
            if pd.isna(z):
                continue

            base = 2.0 * float(w_origin)
            dist_term = 1 + beta * (abs(float(z)) ** alpha)
            mult = float(training_multiplier(z))
            raw_cost = base * dist_term * mult
            pairs.append(raw_cost)

    if not pairs:
        return 1.0, 0

    mean_raw = float(np.mean(pairs))
    if mean_raw <= 0:
        return 1.0, 0

    return float(target_usd) / mean_raw, len(pairs)


def calculate_switching_cost(code1, code2, beta=0.14, alpha=1.2):
    """
    TotalCost = SkillCost + λ * GeoCost

    SkillCost = k * [ (2*w_origin) * (1 + beta*|z|^alpha) * m(|z|) ]

    - Uses origin wages only (2 months of origin wages).
    - Applies EDU_GAP restriction for displayed outputs.
    - Geographic component is optional (controlled by USE_GEO and sidebar params).
    """
    level1 = get_education_level(code1)
    level2 = get_education_level(code2)
    if level1 is None or level2 is None:
        return None
    if abs(level1 - level2) > EDU_GAP:
        return None

    if code1 not in standardized_df.index or code2 not in standardized_df.index:
        return None

    z = standardized_df.loc[code1, code2]
    if pd.isna(z):
        z = standardized_df.loc[code2, code1]
    if pd.isna(z):
        return None

    w_origin = code_to_wage.get(code1)
    if w_origin is None:
        return None

    base = 2.0 * float(w_origin)
    dist_term = 1 + beta * (abs(float(z)) ** alpha)
    mult = float(training_multiplier(z))

    skill_cost = float(CALIB_K) * base * dist_term * mult

    if USE_GEO:
        geo = geographic_cost(code2, USER_PROVINCE, GEO_C_MOVE)
        return float(skill_cost) + float(GEO_LAMBDA) * float(geo)
    else:
        return float(skill_cost)


def switching_cost_components(origin_code, dest_code, beta=0.14, alpha=1.2):
    """
    Returns a decomposition dict for the displayed (EDU-restricted) transition:
      Base = 2*w_origin
      Distance term = (1 + beta*|z|^alpha)
      Training mult = m(|z|)
      Skill cost (before geo) = k * Base * Distance term * Training mult
      Geo add = λ * GeoCost (if enabled)
      Total cost = Skill cost + Geo add

    Also returns benchmarks:
      Months of origin wages = Total / w_origin
      Years of origin wages  = Total / (12*w_origin)
    """
    level1 = get_education_level(origin_code)
    level2 = get_education_level(dest_code)
    if level1 is None or level2 is None:
        return None
    if abs(level1 - level2) > EDU_GAP:
        return None
    if origin_code not in standardized_df.index or dest_code not in standardized_df.index:
        return None

    z = standardized_df.loc[origin_code, dest_code]
    if pd.isna(z):
        z = standardized_df.loc[dest_code, origin_code]
    if pd.isna(z):
        return None

    w_origin = code_to_wage.get(origin_code)
    if w_origin is None or float(w_origin) <= 0:
        return None
    w_origin = float(w_origin)

    base = 2.0 * w_origin
    dist_term = 1 + beta * (abs(float(z)) ** alpha)
    mult = float(training_multiplier(z))

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

    return {
        "Origin": origin_code,
        "Destination": dest_code,
        "Title": code_to_title.get(dest_code, "Unknown Title"),
        "|z|": abs(float(z)),
        "Base (2×origin wage)": base,
        "Distance term": dist_term,
        "Training mult": mult,
        "k (calibration)": float(CALIB_K),
        "Skill cost": float(skill_cost),
        "Geo raw": float(geo_raw),
        "λ (geo weight)": float(GEO_LAMBDA) if USE_GEO else 0.0,
        "Geo add": float(geo_add),
        "Total cost": float(total),
        "Months of origin wages": float(months_equiv),
        "Years of origin wages": float(years_equiv),
    }


def compute_switching_costs_from_origin(origin_code, beta, alpha):
    rows = []
    origin_level = get_education_level(origin_code)
    for dest in similarity_df.columns:
        if dest == origin_code:
            continue
        lev = get_education_level(dest)
        if origin_level is None or lev is None:
            continue
        if abs(lev - origin_level) > EDU_GAP:
            continue
        cost = calculate_switching_cost(origin_code, dest, beta=beta, alpha=alpha)
        if pd.notnull(cost):
            rows.append({"code": dest, "title": code_to_title.get(dest, "Unknown Title"), "cost": float(cost)})
    return pd.DataFrame(rows)


# ---- Histograms with tooltips listing occupations per bin (distribution like before) ----
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


def cost_hist_with_titles(cost_df, maxbins=30, max_titles=50):
    if cost_df is None or cost_df.empty:
        return (
            alt.Chart(pd.DataFrame({"cost": [0]}))
            .mark_bar()
            .encode(alt.X("cost:Q", bin=alt.Bin(maxbins=1), title="Switching Cost ($)"))
            .properties(width=600, height=400)
        )

    df = cost_df.copy()
    df["label"] = df["code"] + " – " + df["title"]

    edges = np.histogram_bin_edges(df["cost"], bins=maxbins)
    df["bin_interval"] = pd.cut(df["cost"], bins=edges, include_lowest=True)

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
                alt.X("cost:Q", bin=alt.Bin(maxbins=30), title="Switching Cost ($)"),
                alt.Y("count()", title="Number of Occupations"),
            )
            .properties(width=600, height=400)
        )

    return (
        alt.Chart(bins_df)
        .mark_bar(opacity=0.7, color="seagreen")
        .encode(
            x=alt.X("bin_start:Q", bin=alt.Bin(binned=True), title="Switching Cost ($)"),
            x2="bin_end:Q",
            y=alt.Y("count:Q", title="Number of Occupations"),
            tooltip=[
                alt.Tooltip("count:Q", title="Number of occupations"),
                alt.Tooltip("bin_start:Q", format=",.0f", title="Cost from"),
                alt.Tooltip("bin_end:Q", format=",.0f", title="Cost to"),
                alt.Tooltip("titles_str:N", title="Occupations in this bin"),
            ],
        )
        .properties(width=600, height=400)
    )


# ---- Career path ego-network helper ----
def build_and_show_ego_network(origin_code, beta, alpha, max_neighbors=15):
    df_costs = compute_switching_costs_from_origin(origin_code, beta, alpha)

    st.markdown(f"**Ego-network debug:** found {len(df_costs)} valid destinations under current settings.")
    if df_costs.empty:
        st.info(
            "Not enough valid transitions to build a network graph.\n\n"
            "- Try increasing the max education distance (EDU_GAP) in the sidebar.\n"
            "- Check that wages exist for destination occupations.\n"
            "- Check that switching costs are not all filtered out."
        )
        return

    df_costs = df_costs.sort_values("cost").head(max_neighbors)

    st.markdown("Top destinations used for the network:")
    st.dataframe(df_costs[["code", "title", "cost"]])

    G = nx.DiGraph()

    def node_attrs(code):
        title = code_to_title.get(code, "Unknown")
        roa = code_to_roa.get(code)
        if roa is None:
            color = "#999999"
        elif roa >= RISKY_THRESHOLD:
            color = "#d62728"
        else:
            color = "#1f77b4"
        wage = code_to_wage.get(code)
        label = f"{code}\n{title[:30]}"
        tooltip = f"{code} – {title}"
        if wage is not None:
            tooltip += f"<br>Wage: {wage:,.0f}"
        if roa is not None:
            tooltip += f"<br>Automation risk: {roa:.2f}"
        return dict(label=label, title=tooltip, color=color)

    G.add_node(origin_code, **node_attrs(origin_code))
    for _, row in df_costs.iterrows():
        dest = row["code"]
        G.add_node(dest, **node_attrs(dest))
        cost_val = float(row["cost"])
        G.add_edge(origin_code, dest, title=f"Cost: {cost_val:,.0f}", value=cost_val)

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#000000", directed=True)
    net.from_nx(G)
    net.repulsion(node_distance=180, spring_length=200, damping=0.85)

    components.html(net.generate_html(), height=600, scrolling=True)


# ---------- Risky/Safe sets & calibration ----------
RISKY_THRESHOLD = 0.70
SAFE_THRESHOLD = 0.70

available_nocs = set(similarity_df.index) & set(code_to_wage.keys())
if code_to_roa:
    RISKY_CODES = {noc for noc in available_nocs if code_to_roa.get(noc) is not None and code_to_roa[noc] >= RISKY_THRESHOLD}
    SAFE_CODES = {noc for noc in available_nocs if code_to_roa.get(noc) is not None and code_to_roa[noc] < SAFE_THRESHOLD}
else:
    RISKY_CODES, SAFE_CODES = set(), set()

# Calibrate k using benchmark beta/alpha (always applied)
CALIB_K, CALIB_PAIRS = compute_calibration_k(RISKY_CODES, SAFE_CODES, target_usd=24000.0, beta=0.14, alpha=1.2)

# ---------- Streamlit App ----------
st.set_page_config(page_title="APOLLO", layout="wide")
st.title("Welcome to the Analysis Platform for Occupational Linkages and Labour Outcomes (APOLLO)")

# Sidebar parameters
st.sidebar.subheader("Switching Cost Parameters")
beta = st.sidebar.slider("Skill distance scaling (beta)", min_value=0.0, max_value=0.5, value=0.14, step=0.01)
alpha = st.sidebar.slider("Non-linear exponent (alpha)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

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

with st.sidebar.expander("Calibration status", expanded=False):
    st.markdown(
        f"""
- **ROA records loaded:** {'Yes' if code_to_roa else 'No'}
- **Risky codes (ROA ≥ 0.70):** {len(RISKY_CODES)}
- **Safe codes (ROA < 0.70):** {len(SAFE_CODES)}
- **Risky→Safe pairs used for k:** {CALIB_PAIRS}
- **Calibration k (always applied):** {CALIB_K:.3f}
- **Geo data loaded:** {'Yes' if (jobprov_df is not None and not jobprov_df.empty) else 'No'}
"""
    )

with st.expander("Methodology"):
    st.markdown(
        """
- Similarity scores are Euclidean distances of O*NET skill/ability/knowledge vectors (smaller = more similar).  
- Switching costs are scaled by **two months of origin wages** and adjusted for skill distance using a non-linear term and a training multiplier.  
- A global calibration factor **k** is chosen so that average **risky → safe** transitions are about **$24,000** (using the full universe of occupations).  
- Displayed results restrict possible transitions to those within a chosen education-distance threshold (thousands digit of the NOC).  
        """
    )
    st.latex(
        r"""
\text{SkillCost}
= k \cdot \left(2\,w_o\right)
  \cdot \left(1 + \beta\,|z|^{\alpha}\right)
  \cdot m(|z|)
"""
    )
    st.markdown(
        r"""
- Here, \( m(|z|) \in \{1.0,\,1.2,\,1.5,\,2.0\} \).  
- Optionally, a geographic mobility component is added:  
  \[
    \text{TotalCost} = \text{SkillCost} + \lambda \cdot \text{GeoCost}.
  \]
        """
    )

n_results = st.sidebar.slider("Number of results to show:", min_value=3, max_value=20, value=5)
menu = st.sidebar.radio("Choose an option:", ["Look up by code", "Look up by title", "Compare two jobs"])

# ---------- Look up by code ----------
if menu == "Look up by code":
    code = st.text_input("Enter 5-digit occupation code:")
    if code:
        code = str(code).zfill(5).strip()
        if code in similarity_df.index:
            top_results, bottom_results, all_scores = get_most_and_least_similar(code, n=n_results)

            # Most similar
            st.subheader(f"Most Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            df_top = pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])
            df_top["Switching Cost ($)"] = df_top["Code"].apply(lambda x: calculate_switching_cost(code, x, beta=beta, alpha=alpha))
            df_top["Switching Cost ($)"] = df_top["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

            # Least similar
            st.subheader(f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
            df_bottom["Switching Cost ($)"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(code, x, beta=beta, alpha=alpha))
            df_bottom["Switching Cost ($)"] = df_bottom["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

            # Decomposition expander (only for shown rows)
            with st.expander("Switching cost decomposition (details)", expanded=False):
                shown_codes = list(df_top["Code"].astype(str)) + list(df_bottom["Code"].astype(str))
                decomp_rows = []
                for dest in shown_codes:
                    d = switching_cost_components(code, dest, beta=beta, alpha=alpha)
                    if d is not None:
                        decomp_rows.append(d)
                if decomp_rows:
                    decomp_df = pd.DataFrame(decomp_rows)
                    preferred_cols = [
                        "Origin", "Destination", "Title",
                        "Base (2×origin wage)", "Distance term", "Training mult", "k (calibration)",
                        "Skill cost", "Geo add", "Total cost", "Months of origin wages", "Years of origin wages", "|z|"
                    ]
                    decomp_df = decomp_df[[c for c in preferred_cols if c in decomp_df.columns]]

                    money_cols = ["Base (2×origin wage)", "Skill cost", "Geo add", "Total cost"]
                    for c in money_cols:
                        if c in decomp_df.columns:
                            decomp_df[c] = decomp_df[c].map(lambda v: f"{float(v):,.2f}")

                    if "Months of origin wages" in decomp_df.columns:
                        decomp_df["Months of origin wages"] = decomp_df["Months of origin wages"].map(
                            lambda v: f"{float(v):.1f}" if pd.notnull(v) else "N/A"
                        )
                    if "Years of origin wages" in decomp_df.columns:
                        decomp_df["Years of origin wages"] = decomp_df["Years of origin wages"].map(
                            lambda v: f"{float(v):.2f}" if pd.notnull(v) else "N/A"
                        )

                    if "Distance term" in decomp_df.columns:
                        decomp_df["Distance term"] = decomp_df["Distance term"].map(lambda v: f"{float(v):.4f}")
                    if "Training mult" in decomp_df.columns:
                        decomp_df["Training mult"] = decomp_df["Training mult"].map(lambda v: f"{float(v):.2f}")
                    if "k (calibration)" in decomp_df.columns:
                        decomp_df["k (calibration)"] = decomp_df["k (calibration)"].map(lambda v: f"{float(v):.3f}")
                    if "|z|" in decomp_df.columns:
                        decomp_df["|z|"] = decomp_df["|z|"].map(lambda v: f"{float(v):.3f}")

                    st.dataframe(decomp_df, use_container_width=True)
                else:
                    st.info("No decomposition rows available (likely filtered out by education distance or missing data).")

            # Similarity histogram
            st.subheader(f"Similarity Score Distribution for {code} – {code_to_title.get(code,'Unknown')}")
            st.caption("Tip: hover on a bar to see which occupations fall in that similarity range.")
            st.altair_chart(similarity_hist_with_titles(all_scores), use_container_width=True)

            # Switching cost histogram
            costs_df = compute_switching_costs_from_origin(code, beta=beta, alpha=alpha)
            st.subheader(f"Switching Cost Distribution from {code} – {code_to_title.get(code,'Unknown')}")
            st.caption("Tip: hover on a bar to see which occupations fall in that cost range.")
            st.altair_chart(cost_hist_with_titles(costs_df), use_container_width=True)

            # Career path network
            with st.expander("Career path network (local view)", expanded=False):
                build_and_show_ego_network(code, beta, alpha, max_neighbors=15)

# ---------- Look up by title ----------
elif menu == "Look up by title":
    available_codes = [c for c in code_to_title if c in similarity_df.index]
    title_options = [f"{c} – {code_to_title[c]}" for c in available_codes]

    selected_item = st.selectbox("Select an occupation:", sorted(title_options))
    if selected_item:
        selected_code, selected_title = selected_item.split(" – ")
        top_results, bottom_results, all_scores = get_most_and_least_similar(selected_code, n=n_results)

        st.subheader(f"Most Similar Occupations for {selected_code} – {selected_title}")
        df_top = pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])
        df_top["Switching Cost ($)"] = df_top["Code"].apply(lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha))
        df_top["Switching Cost ($)"] = df_top["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        st.subheader(f"Least Similar Occupations for {selected_code} – {selected_title}")
        df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
        df_bottom["Switching Cost ($)"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha))
        df_bottom["Switching Cost ($)"] = df_bottom["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        with st.expander("Switching cost decomposition (details)", expanded=False):
            shown_codes = list(df_top["Code"].astype(str)) + list(df_bottom["Code"].astype(str))
            decomp_rows = []
            for dest in shown_codes:
                d = switching_cost_components(selected_code, dest, beta=beta, alpha=alpha)
                if d is not None:
                    decomp_rows.append(d)
            if decomp_rows:
                decomp_df = pd.DataFrame(decomp_rows)
                preferred_cols = [
                    "Origin", "Destination", "Title",
                    "Base (2×origin wage)", "Distance term", "Training mult", "k (calibration)",
                    "Skill cost", "Geo add", "Total cost", "Months of origin wages", "Years of origin wages", "|z|"
                ]
                decomp_df = decomp_df[[c for c in preferred_cols if c in decomp_df.columns]]

                money_cols = ["Base (2×origin wage)", "Skill cost", "Geo add", "Total cost"]
                for c in money_cols:
                    if c in decomp_df.columns:
                        decomp_df[c] = decomp_df[c].map(lambda v: f"{float(v):,.2f}")

                if "Months of origin wages" in decomp_df.columns:
                    decomp_df["Months of origin wages"] = decomp_df["Months of origin wages"].map(
                        lambda v: f"{float(v):.1f}" if pd.notnull(v) else "N/A"
                    )
                if "Years of origin wages" in decomp_df.columns:
                    decomp_df["Years of origin wages"] = decomp_df["Years of origin wages"].map(
                        lambda v: f"{float(v):.2f}" if pd.notnull(v) else "N/A"
                    )

                if "Distance term" in decomp_df.columns:
                    decomp_df["Distance term"] = decomp_df["Distance term"].map(lambda v: f"{float(v):.4f}")
                if "Training mult" in decomp_df.columns:
                    decomp_df["Training mult"] = decomp_df["Training mult"].map(lambda v: f"{float(v):.2f}")
                if "k (calibration)" in decomp_df.columns:
                    decomp_df["k (calibration)"] = decomp_df["k (calibration)"].map(lambda v: f"{float(v):.3f}")
                if "|z|" in decomp_df.columns:
                    decomp_df["|z|"] = decomp_df["|z|"].map(lambda v: f"{float(v):.3f}")

                st.dataframe(decomp_df, use_container_width=True)
            else:
                st.info("No decomposition rows available (likely filtered out by education distance or missing data).")

        st.subheader(f"Similarity Score Distribution for {selected_code} – {selected_title}")
        st.caption("Tip: hover on a bar to see which occupations fall in that similarity range.")
        st.altair_chart(similarity_hist_with_titles(all_scores), use_container_width=True)

        costs_df = compute_switching_costs_from_origin(selected_code, beta=beta, alpha=alpha)
        st.subheader(f"Switching Cost Distribution from {selected_code} – {selected_title}")
        st.caption("Tip: hover on a bar to see which occupations fall in that cost range.")
        st.altair_chart(cost_hist_with_titles(costs_df), use_container_width=True)

        with st.expander("Career path network (local view)", expanded=False):
            build_and_show_ego_network(selected_code, beta, alpha, max_neighbors=15)

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
            cost = calculate_switching_cost(job1_code, job2_code, beta=beta, alpha=alpha)

            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({job1_title}) vs {job2_code} ({job2_title})\n"
                f"- Similarity score (raw distance): `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations (#{rank} most similar to {job1_code})"
            )

            if cost is not None:
                # --- NEW: benchmark cost in months/years of origin wages ---
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
                    d = switching_cost_components(job1_code, job2_code, beta=beta, alpha=alpha)
                    if d is None:
                        st.info("No decomposition available (filtered out or missing data).")
                    else:
                        dd = pd.DataFrame([d])

                        money_cols = ["Base (2×origin wage)", "Skill cost", "Geo add", "Total cost"]
                        for c in money_cols:
                            if c in dd.columns:
                                dd[c] = dd[c].map(lambda v: f"{float(v):,.2f}")

                        if "Months of origin wages" in dd.columns:
                            dd["Months of origin wages"] = dd["Months of origin wages"].map(
                                lambda v: f"{float(v):.1f}" if pd.notnull(v) else "N/A"
                            )
                        if "Years of origin wages" in dd.columns:
                            dd["Years of origin wages"] = dd["Years of origin wages"].map(
                                lambda v: f"{float(v):.2f}" if pd.notnull(v) else "N/A"
                            )

                        if "Distance term" in dd.columns:
                            dd["Distance term"] = dd["Distance term"].map(lambda v: f"{float(v):.4f}")
                        if "Training mult" in dd.columns:
                            dd["Training mult"] = dd["Training mult"].map(lambda v: f"{float(v):.2f}")
                        if "k (calibration)" in dd.columns:
                            dd["k (calibration)"] = dd["k (calibration)"].map(lambda v: f"{float(v):.3f}")
                        if "|z|" in dd.columns:
                            dd["|z|"] = dd["|z|"].map(lambda v: f"{float(v):.3f}")

                        st.dataframe(dd, use_container_width=True)

            else:
                st.info(
                    "ℹ️ Switching cost is not reported because the two occupations are further apart in education "
                    f"than the allowed maximum distance ({EDU_GAP})."
                )
        else:
            st.error("❌ Could not compare occupations.")
