import streamlit as st
import pandas as pd
import os
import altair as alt
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

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
    jobprov_df = pd.DataFrame()  # ensure it's always defined

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
        jobprov_df = pd.DataFrame()  # fallback

    # Create mappings
    code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
    title_to_code = {v.lower(): k for k, v in code_to_title.items()}

    # ---- Standardize distances (z-scores) ----
    flat_scores = similarity_df.where(~pd.isna(similarity_df)).stack().values
    mean_val, std_val = flat_scores.mean(), flat_scores.std()
    standardized_df = (similarity_df - mean_val) / std_val

    # Return all data objects (ALWAYS 7 values)
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
    Extract the 'education required' level from the 5-digit NOC.
    By your convention, this is the thousands digit, e.g. 14110 -> 4,
    i.e. the SECOND digit of the 5-digit code string.
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

    # All scores from origin, drop self and NaNs
    scores = similarity_df.loc[code].drop(code).dropna()

    # Restrict by max education distance (uses global EDU_GAP set later)
    same_or_close_codes = []
    for c in scores.index:
        lev = get_education_level(c)
        if origin_level is not None and lev is not None:
            if abs(lev - origin_level) <= EDU_GAP:
                same_or_close_codes.append(c)
    scores = scores.loc[same_or_close_codes] if same_or_close_codes else scores.iloc[0:0]

    if scores.empty:
        return [], [], scores  # no valid matches under current education gap

    top_matches = scores.nsmallest(n)
    bottom_matches = scores.nlargest(n)
    top_results = [
        (occ, code_to_title.get(occ, "Unknown Title"), score) for occ, score in top_matches.items()
    ]
    bottom_results = [
        (occ, code_to_title.get(occ, "Unknown Title"), score) for occ, score in bottom_matches.items()
    ]
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


# ---- Training multiplier based on |z| bins ----
def training_multiplier(z_score):
    z = abs(z_score)
    if z < 0.5:
        return 1.0
    elif z < 1.0:
        return 1.2
    elif z < 1.5:
        return 1.5
    else:
        return 2.0


# ---- Geographic cost helper (optional) ----
def geographic_cost(dest_code, province, C_move=20000.0):
    """
    Expected relocation cost for moving into dest_code from given province.
    Uses jobprov_df (share of workers in each province for that occupation).
    If no data, returns 0.
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

    # Probability of having to move: 1 - (local share / max share)
    p_move = 1.0 - a_p0 / a_max
    p_move = max(0.0, min(1.0, p_move))  # clamp

    return C_move * p_move


# ---- Calibration for risky -> safe-haven only (NO education restriction here) ----
def compute_calibration_k(risky_codes, safe_codes, target_usd=24000.0, beta=0.14, alpha=1.2):
    """
    Compute a global scale k so that the average cost for risky->safe pairs
    (across all education levels) equals target_usd.
    Returns (k, n_pairs_used).
    """
    pairs = []
    for r in risky_codes:
        if r not in standardized_df.index:
            continue
        for s in safe_codes:
            if s == r or s not in standardized_df.index:
                continue
            if (code_to_wage.get(r) is None) or (code_to_wage.get(s) is None):
                continue
            z = standardized_df.loc[r, s]
            if pd.isna(z):
                z = standardized_df.loc[s, r]
            if pd.isna(z):
                continue
            base = 2 * np.sqrt(code_to_wage[r] * code_to_wage[s])
            mult = training_multiplier(z)
            raw_cost = base * (1 + beta * abs(z) ** alpha) * mult
            pairs.append(raw_cost)

    if not pairs:
        return 1.0, 0
    mean_raw = float(np.mean(pairs))
    if mean_raw <= 0:
        return 1.0, 0
    return target_usd / mean_raw, len(pairs)


def calculate_switching_cost(code1, code2, beta=0.14, alpha=1.2):
    """
    Cost = k * [ 2*sqrt(w_o*w_d) * (1 + beta*|z|^alpha) * m(|z|) ]  +  λ * GeoCost
    - Calibration k is ALWAYS applied to the skill/training component.
    - Only computed for pairs whose education levels differ by at most EDU_GAP.
    - Geographic component is optional (controlled by USE_GEO and sidebar params).
    """
    level1 = get_education_level(code1)
    level2 = get_education_level(code2)
    if level1 is None or level2 is None:
        return None
    # Enforce max education distance requirement (EDU_GAP is global set from slider)
    if abs(level1 - level2) > EDU_GAP:
        return None

    if code1 not in standardized_df.index or code2 not in standardized_df.index:
        return None
    z_score = standardized_df.loc[code1, code2]
    if pd.isna(z_score):
        z_score = standardized_df.loc[code2, code1]
    if pd.isna(z_score):
        return None
    w_origin = code_to_wage.get(code1)
    w_dest = code_to_wage.get(code2)
    if w_origin is None or w_dest is None:
        return None

    base_cost = 2 * np.sqrt(w_origin * w_dest)
    multiplier = training_multiplier(z_score)
    skill_cost = CALIB_K * base_cost * (1 + beta * abs(z_score) ** alpha) * multiplier

    # Optional geographic mobility cost
    if USE_GEO:
        geo_cost = geographic_cost(code2, USER_PROVINCE, GEO_C_MOVE)
        return skill_cost + GEO_LAMBDA * geo_cost
    else:
        return skill_cost


def plot_histogram(scores, highlight_score=None):
    hist_df = pd.DataFrame({"score": scores.values})
    hist_chart = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.7, color="steelblue")
        .encode(
            alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Similarity Score (Euclidean distance)"),
            alt.Y("count()", title="Number of Occupations"),
            tooltip=["count()"],
        )
        .properties(width=600, height=400)
    )
    if highlight_score is not None:
        line = (
            alt.Chart(pd.DataFrame({"score": [highlight_score]}))
            .mark_rule(color="red", strokeWidth=2)
            .encode(x="score:Q")
        )
        hist_chart = hist_chart + line
    return hist_chart


# ---- Switching cost distribution helpers ----
def compute_switching_costs_from_origin(origin_code, beta, alpha):
    rows = []
    origin_level = get_education_level(origin_code)
    for dest in similarity_df.columns:
        if dest == origin_code:
            continue
        lev = get_education_level(dest)
        if origin_level is None or lev is None:
            continue
        # Restrict by allowed education distance
        if abs(lev - origin_level) > EDU_GAP:
            continue
        cost = calculate_switching_cost(origin_code, dest, beta=beta, alpha=alpha)
        if pd.notnull(cost):
            rows.append(
                {
                    "code": dest,
                    "title": code_to_title.get(dest, "Unknown Title"),
                    "cost": float(cost),
                }
            )
    return pd.DataFrame(rows)


def plot_cost_histogram(cost_df):
    if cost_df is None or cost_df.empty:
        return (
            alt.Chart(pd.DataFrame({"cost": [0]}))
            .mark_bar()
            .encode(
                alt.X("cost:Q", bin=alt.Bin(maxbins=1), title="Switching Cost ($)"),
            )
        )
    chart = (
        alt.Chart(cost_df)
        .mark_bar(opacity=0.7, color="seagreen")
        .encode(
            alt.X("cost:Q", bin=alt.Bin(maxbins=30), title="Switching Cost ($)"),
            alt.Y("count()", title="Number of Occupations"),
            tooltip=["count()"],
        )
        .properties(width=600, height=400)
    )
    return chart


# ---- NEW: Interactive histograms with clickable bars ----
def similarity_hist_with_table(all_scores):
    """
    Interactive similarity histogram:
    - Top: histogram over similarity scores
    - Bottom: list of occupations whose scores fall in the selected bin(s)
    """
    if all_scores is None or len(all_scores) == 0:
        return plot_histogram(all_scores or pd.Series(dtype=float))

    df = pd.DataFrame({
        "code": all_scores.index,
        "score": all_scores.values,
    })
    df["title"] = df["code"].map(lambda c: code_to_title.get(c, "Unknown Title"))
    df["label"] = (
        df["code"]
        + " – "
        + df["title"]
        + " (score="
        + df["score"].round(2).astype(str)
        + ")"
    )

    # Brush selection along the x-axis (click or drag)
    brush = alt.selection(type="interval", encodings=["x"])

    hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.7)
        .encode(
            alt.X(
                "score:Q",
                bin=alt.Bin(maxbins=30),
                title="Similarity Score (Euclidean distance)",
            ),
            alt.Y("count():Q", title="Number of Occupations"),
            tooltip=["count():Q"],
        )
        .add_selection(brush)
        .properties(height=300)
    )

    # Table of occupations in the selected bin(s)
    table = (
        alt.Chart(df)
        .transform_filter(brush)
        .transform_window(row_number="row_number()")
        .transform_filter("datum.row_number <= 50")  # cap at 50 rows for readability
        .mark_text(align="left", baseline="top")
        .encode(
            y=alt.Y("row_number:O", axis=None),
            text="label:N",
        )
        .properties(height=300)
    )

    return hist & table  # vertical concat


def cost_hist_with_table(cost_df):
    """
    Interactive switching-cost histogram:
    - Top: histogram over costs
    - Bottom: list of occupations whose costs fall in the selected bin(s)
    """
    if cost_df is None or cost_df.empty:
        return plot_cost_histogram(cost_df)

    df = cost_df.copy()
    df["label"] = (
        df["code"]
        + " – "
        + df["title"]
        + " (cost=$"
        + df["cost"].round(0).astype(int).astype(str)
        + ")"
    )

    brush = alt.selection(type="interval", encodings=["x"])

    hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.7)
        .encode(
            alt.X(
                "cost:Q",
                bin=alt.Bin(maxbins=30),
                title="Switching Cost ($)",
            ),
            alt.Y("count():Q", title="Number of Occupations"),
            tooltip=["count():Q"],
        )
        .add_selection(brush)
        .properties(height=300)
    )

    table = (
        alt.Chart(df)
        .transform_filter(brush)
        .transform_window(row_number="row_number()")
        .transform_filter("datum.row_number <= 50")
        .mark_text(align="left", baseline="top")
        .encode(
            y=alt.Y("row_number:O", axis=None),
            text="label:N",
        )
        .properties(height=300)
    )

    return hist & table


# ---- Career path ego-network helper ----
def build_and_show_ego_network(origin_code, beta, alpha, max_neighbors=15):
    """
    Build a small directed ego-network around origin_code using the lowest-cost
    destinations under current settings, and display it with pyvis.
    """
    # 1. Get all valid switching costs from this origin
    df_costs = compute_switching_costs_from_origin(origin_code, beta, alpha)

    st.markdown(
        f"**Ego-network debug:** found {len(df_costs)} valid destinations under current settings."
    )
    if df_costs.empty:
        st.info(
            "Not enough valid transitions to build a network graph.\n\n"
            "- Try increasing the max education distance (EDU_GAP) in the sidebar.\n"
            "- Check that wages exist for destination occupations.\n"
            "- Check that switching costs are not all filtered out."
        )
        return

    # 2. Keep only the lowest-cost neighbors
    df_costs = df_costs.sort_values("cost").head(max_neighbors)

    # Show the table we are actually using for the network
    st.markdown("Top destinations used for the network:")
    st.dataframe(df_costs[["code", "title", "cost"]])

    # 3. Build a directed graph
    G = nx.DiGraph()

    def node_attrs(code):
        title = code_to_title.get(code, "Unknown")
        roa = code_to_roa.get(code)
        if roa is None:
            color = "#999999"
        elif roa >= RISKY_THRESHOLD:
            color = "#d62728"  # red-ish for risky
        else:
            color = "#1f77b4"  # blue-ish for safe
        wage = code_to_wage.get(code)
        label = f"{code}\n{title[:30]}"
        tooltip = f"{code} – {title}"
        if wage is not None:
            tooltip += f"<br>Wage: {wage:,.0f}"
        if roa is not None:
            tooltip += f"<br>Automation risk: {roa:.2f}"
        return dict(label=label, title=tooltip, color=color)

    # Add origin node
    G.add_node(origin_code, **node_attrs(origin_code))

    # Add destination nodes and edges
    for _, row in df_costs.iterrows():
        dest = row["code"]
        G.add_node(dest, **node_attrs(dest))
        cost_val = float(row["cost"])
        G.add_edge(origin_code, dest, title=f"Cost: {cost_val:,.0f}", value=cost_val)

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True,
    )
    net.from_nx(G)
    net.repulsion(node_distance=180, spring_length=200, damping=0.85)

    html = net.generate_html()
    components.html(html, height=600, scrolling=True)


# ---------- Risky/Safe sets & calibration ----------
RISKY_THRESHOLD = 0.70  # ROA >= 0.70 → risky
SAFE_THRESHOLD = 0.70  # ROA < 0.70 → safe

available_nocs = set(similarity_df.index) & set(code_to_wage.keys())
if code_to_roa:
    RISKY_CODES = {
        noc
        for noc in available_nocs
        if code_to_roa.get(noc) is not None and code_to_roa[noc] >= RISKY_THRESHOLD
    }
    SAFE_CODES = {
        noc
        for noc in available_nocs
        if code_to_roa.get(noc) is not None and code_to_roa[noc] < SAFE_THRESHOLD
    }
else:
    RISKY_CODES, SAFE_CODES = set(), set()

# Calibrate k on ALL risky->safe pairs (no education restriction here)
CALIB_K, CALIB_PAIRS = compute_calibration_k(
    RISKY_CODES, SAFE_CODES, target_usd=24000.0, beta=0.14, alpha=1.2
)

# ---------- Streamlit App ----------
st.set_page_config(page_title="APOLLO", layout="wide")
st.title("Welcome to the Analysis Platform for Occupational Linkages and Labour Outcomes (APOLLO)")

# Sidebar sliders for beta and alpha
st.sidebar.subheader("Switching Cost Parameters")
beta = st.sidebar.slider(
    "Skill distance scaling (beta)", min_value=0.0, max_value=0.5, value=0.14, step=0.01
)
alpha = st.sidebar.slider(
    "Non-linear exponent (alpha)", min_value=0.5, max_value=3.0, value=1.2, step=0.1
)

# Sidebar slider for education distance
EDU_GAP = st.sidebar.slider(
    "Max education distance allowed (0 = same level only)",
    min_value=0,
    max_value=4,
    value=0,
    step=1,
)

# Optional geographic cost controls
USE_GEO = st.sidebar.checkbox("Include geographic mobility cost", value=False)

USER_PROVINCE = None
GEO_C_MOVE = 0.0
GEO_LAMBDA = 0.0

if USE_GEO:
    if jobprov_df is not None and not jobprov_df.empty:
        province_options = sorted(jobprov_df["province"].dropna().unique())
        USER_PROVINCE = st.sidebar.selectbox(
            "Worker's province of origin:", province_options
        )
        GEO_C_MOVE = st.sidebar.number_input(
            "Relocation cost if move required ($)",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
        )
        GEO_LAMBDA = st.sidebar.slider(
            "Weight on geographic cost (λ)", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )
    else:
        st.sidebar.info(
            "Geographic cost selected, but no job_province_share file found. "
            "Geographic component will be treated as zero."
        )

# Status block to verify calibration & ROA ingestion
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

# About section
with st.expander("Methodology"):
    st.markdown(
        """
- Similarity scores are based on Euclidean distances of O*NET skill, ability, and knowledge vectors.  
  Smaller scores mean occupations are more similar.  
- The global calibration factor **k** is estimated using **all risky → safe** occupational transitions,  
  regardless of education level.  
- When displaying results, switching costs are computed only between occupations whose education levels  
  (thousands digit of the 5-digit NOC) are within a chosen **maximum distance** (set by the sidebar slider).  
- The baseline switching cost combines the geometric mean of origin/destination wages, a non-linear skill-distance term,  
  a training-intensity multiplier based on standardized |z| bins, and the calibration factor **k**.  
        """
    )
    st.latex(
        r"""
\text{SkillCost}
= k \cdot \left(2\,\sqrt{w_o\,w_d}\right)
  \cdot \left(1 + \beta\,|z|^{\alpha}\right)
  \cdot m(|z|)
"""
    )
    st.markdown(
        r"""
- Here, \( m(|z|) \in \{1.0,\,1.2,\,1.5,\,2.0\} \), and \( k \) is chosen so that the **average risky → safe** transition cost  
  (across all education levels) is about **$24,000**.  
- Optionally, a **geographic mobility cost** is added:  
  \[
    \text{TotalCost} = \text{SkillCost} + \lambda \cdot \text{GeoCost},
  \]
  where \( \lambda \in [0,1] \) and GeoCost reflects the expected relocation cost given the origin province  
  and the spatial distribution of the destination occupation.  
- Adjust \( \beta \), \( \alpha \), the maximum education distance, and the geographic cost settings in the sidebar to see sensitivity.  
        """
    )

# Sidebar
n_results = st.sidebar.slider(
    "Number of results to show:", min_value=3, max_value=20, value=5
)
menu = st.sidebar.radio(
    "Choose an option:", ["Look up by code", "Look up by title", "Compare two jobs"]
)

# ---- Look up by code ----
if menu == "Look up by code":
    code = st.text_input("Enter 5-digit occupation code:")
    if code:
        code = str(code).zfill(5).strip()
        if code in similarity_df.index:
            top_results, bottom_results, all_scores = get_most_and_least_similar(
                code, n=n_results
            )

            # Most similar
            st.subheader(
                f"Most Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}"
            )
            df_top = pd.DataFrame(
                top_results, columns=["Code", "Title", "Similarity Score"]
            )
            df_top["Switching Cost ($)"] = df_top["Code"].apply(
                lambda x: calculate_switching_cost(code, x, beta=beta, alpha=alpha)
            )
            df_top["Switching Cost ($)"] = df_top["Switching Cost ($)"].map(
                lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A"
            )
            st.dataframe(
                df_top,
                use_container_width=True,
                column_config={"Title": st.column_config.Column(width="large")},
            )

            # Least similar
            st.subheader(
                f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}"
            )
            df_bottom = pd.DataFrame(
                bottom_results, columns=["Code", "Title", "Similarity Score"]
            )
            df_bottom["Switching Cost ($)"] = df_bottom["Code"].apply(
                lambda x: calculate_switching_cost(code, x, beta=beta, alpha=alpha)
            )
            df_bottom["Switching Cost ($)"] = df_bottom["Switching Cost ($)"].map(
                lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A"
            )
            st.dataframe(
                df_bottom,
                use_container_width=True,
                column_config={"Title": st.column_config.Column(width="large")},
            )

            # Similarity histogram + clickable list
            st.subheader(
                f"Similarity Score Distribution for {code} – {code_to_title.get(code,'Unknown')}"
            )
            st.caption("Tip: click or drag on the bars to see which occupations fall into that score range below.")
            st.altair_chart(similarity_hist_with_table(all_scores), use_container_width=True)

            # Switching cost histogram + clickable list
            costs_df = compute_switching_costs_from_origin(code, beta=beta, alpha=alpha)
            st.subheader(
                f"Switching Cost Distribution from {code} – {code_to_title.get(code,'Unknown')}"
            )
            st.caption("Tip: click or drag on the bars to see which occupations fall into that cost range below.")
            st.altair_chart(cost_hist_with_table(costs_df), use_container_width=True)

            # Career path ego-network
            with st.expander("Career path network (local view)", expanded=False):
                build_and_show_ego_network(code, beta, alpha, max_neighbors=15)

# ---- Look up by title ----
elif menu == "Look up by title":
    available_codes = [code for code in code_to_title if code in similarity_df.index]
    title_options = [f"{code} – {code_to_title[code]}" for code in available_codes]

    selected_item = st.selectbox("Select an occupation:", sorted(title_options))
    if selected_item:
        selected_code, selected_title = selected_item.split(" – ")
        top_results, bottom_results, all_scores = get_most_and_least_similar(
            selected_code, n=n_results
        )

        # Most similar
        st.subheader(
            f"Most Similar Occupations for {selected_code} – {selected_title}"
        )
        df_top = pd.DataFrame(
            top_results, columns=["Code", "Title", "Similarity Score"]
        )
        df_top["Switching Cost ($)"] = df_top["Code"].apply(
            lambda x: calculate_switching_cost(
                selected_code, x, beta=beta, alpha=alpha
            )
        )
        df_top["Switching Cost ($)"] = df_top["Switching Cost ($)"].map(
            lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A"
        )
        st.dataframe(
            df_top,
            use_container_width=True,
            column_config={"Title": st.column_config.Column(width="large")},
        )

        # Least similar
        st.subheader(
            f"Least Similar Occupations for {selected_code} – {selected_title}"
        )
        df_bottom = pd.DataFrame(
            bottom_results, columns=["Code", "Title", "Similarity Score"]
        )
        df_bottom["Switching Cost ($)"] = df_bottom["Code"].apply(
            lambda x: calculate_switching_cost(
                selected_code, x, beta=beta, alpha=alpha
            )
        )
        df_bottom["Switching Cost ($)"] = df_bottom["Switching Cost ($)"].map(
            lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A"
        )
        st.dataframe(
            df_bottom,
            use_container_width=True,
            column_config={"Title": st.column_config.Column(width="large")},
        )

        # Similarity histogram + clickable list
        st.subheader(
            f"Similarity Score Distribution for {selected_code} – {selected_title}"
        )
        st.caption("Tip: click or drag on the bars to see which occupations fall into that score range below.")
        st.altair_chart(similarity_hist_with_table(all_scores), use_container_width=True)

        # Switching cost histogram + clickable list
        costs_df = compute_switching_costs_from_origin(
            selected_code, beta=beta, alpha=alpha
        )
        st.subheader(
            f"Switching Cost Distribution from {selected_code} – {selected_title}"
        )
        st.caption("Tip: click or drag on the bars to see which occupations fall into that cost range below.")
        st.altair_chart(cost_hist_with_table(costs_df), use_container_width=True)

        # Career path ego-network
        with st.expander("Career path network (local view)", expanded=False):
            build_and_show_ego_network(selected_code, beta, alpha, max_neighbors=15)

# ---- Compare two jobs ----
elif menu == "Compare two jobs":
    available_codes = [code for code in code_to_title if code in similarity_df.index]
    title_options = [f"{code} – {code_to_title[code]}" for code in available_codes]

    job1_item = st.selectbox(
        "Select first occupation:", sorted(title_options), key="job1"
    )
    job2_item = st.selectbox(
        "Select second occupation:", sorted(title_options), key="job2"
    )

    job1_code, job1_title = job1_item.split(" – ")
    job2_code, job2_title = job2_item.split(" – ")

    if st.button("Compare"):
        result = compare_two_jobs(job1_code, job2_code)
        if result:
            score, rank, total = result
            cost = calculate_switching_cost(
                job1_code, job2_code, beta=beta, alpha=alpha
            )

            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({job1_title}) vs {job2_code} ({job2_title})\n"
                f"- Similarity score (raw distance): `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations "
                f"(#{rank} most similar to {job1_code})"
            )

            if cost is not None:
                extra_geo = ""
                if USE_GEO and USER_PROVINCE is not None:
                    extra_geo = f" (includes geographic mobility component for {USER_PROVINCE})"
                st.info(
                    f"💰 **Estimated Switching Cost** (from {job1_code} to {job2_code}): "
                    f"`${cost:,.2f}` (education distance ≤ {EDU_GAP}){extra_geo}"
                )
            else:
                st.info(
                    "ℹ️ Switching cost is not reported because the two occupations are further apart in education "
                    f"than the allowed maximum distance ({EDU_GAP})."
                )
        else:
            st.error("❌ Could not compare occupations.")
