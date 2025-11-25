import streamlit as st
import pandas as pd
import os
import altair as alt
import numpy as np

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
            prob_col = cols.get("roa_prob") or cols.get("automation_probability") or cols.get("probability") or list(df.columns)[1]
            df[noc_col] = _norm(df[noc_col])
            code_to_roa = dict(zip(df[noc_col], df[prob_col]))
    except Exception:
        code_to_roa = {}

    # Create mappings
    code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
    title_to_code = {v.lower(): k for k, v in code_to_title.items()}

    # ---- Standardize distances (z-scores) ----
    flat_scores = similarity_df.where(~pd.isna(similarity_df)).stack().values
    mean_val, std_val = flat_scores.mean(), flat_scores.std()
    standardized_df = (similarity_df - mean_val) / std_val

    # Return ROA mapping too
    return similarity_df, standardized_df, code_to_title, title_to_code, code_to_wage, code_to_roa

similarity_df, standardized_df, code_to_title, title_to_code, code_to_wage, code_to_roa = load_data()

# ---------- Helper Functions ----------

def get_education_level(noc_code):
    """
    Extract the 'education required' level from the 5-digit NOC.
    By your convention, this is the thousands digit, e.g. 14110 -> 4.
    That is the SECOND digit of the 5-digit code string.
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
    top_results = [(occ, code_to_title.get(occ, "Unknown Title"), score) 
                   for occ, score in top_matches.items()]
    bottom_results = [(occ, code_to_title.get(occ, "Unknown Title"), score) 
                      for occ, score in bottom_matches.items()]
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

# ---- Calibration for risky -> safe-haven only (NO education restriction here) ----
def compute_calibration_k(risky_codes, safe_codes, target_usd=20000.0, beta=0.14, alpha=1.2):
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
            raw_cost = base * (1 + beta * abs(z)**alpha) * mult
            pairs.append(raw_cost)

    if not pairs:
        return 1.0, 0
    mean_raw = float(np.mean(pairs))
    if mean_raw <= 0:
        return 1.0, 0
    return target_usd / mean_raw, len(pairs)

def calculate_switching_cost(code1, code2, beta=0.14, alpha=1.2):
    """
    Cost = k * [ 2*sqrt(w_o*w_d) * (1 + beta*|z|^alpha) * m(|z|) ]
    Calibration k is ALWAYS applied.
    Only computed for pairs whose education levels differ by at most EDU_GAP.
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
    raw_cost = base_cost * (1 + beta * abs(z_score)**alpha) * multiplier
    return CALIB_K * raw_cost  # always calibrated

def plot_histogram(scores, highlight_score=None):
    hist_df = pd.DataFrame({"score": scores.values})
    hist_chart = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.7, color="steelblue")
        .encode(
            alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Similarity Score (Euclidean distance)"),
            alt.Y("count()", title="Number of Occupations"),
            tooltip=["count()"]
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
            rows.append({
                "code": dest,
                "title": code_to_title.get(dest, "Unknown Title"),
                "cost": float(cost)
            })
    return pd.DataFrame(rows)

def plot_cost_histogram(cost_df):
    if cost_df.empty:
        return alt.Chart(pd.DataFrame({"cost":[0]})).mark_bar().encode(
            alt.X("cost:Q", bin=alt.Bin(maxbins=1), title="Switching Cost ($)")
        )
    chart = (
        alt.Chart(cost_df)
        .mark_bar(opacity=0.7, color="seagreen")
        .encode(
            alt.X("cost:Q", bin=alt.Bin(maxbins=30), title="Switching Cost ($)"),
            alt.Y("count()", title="Number of Occupations"),
            tooltip=["count()"]
        )
        .properties(width=600, height=400)
    )
    return chart

# ---------- Risky/Safe sets & calibration ----------
RISKY_THRESHOLD = 0.70   # ROA >= 0.70 → risky
SAFE_THRESHOLD = 0.70    # ROA < 0.70 → safe

available_nocs = set(similarity_df.index) & set(code_to_wage.keys())
if code_to_roa:
    RISKY_CODES = {noc for noc in available_nocs if code_to_roa.get(noc) is not None and code_to_roa[noc] >= RISKY_THRESHOLD}
    SAFE_CODES  = {noc for noc in available_nocs if code_to_roa.get(noc) is not None and code_to_roa[noc] <  SAFE_THRESHOLD}
else:
    RISKY_CODES, SAFE_CODES = set(), set()

# Calibrate k on ALL risky->safe pairs (no education restriction here)
CALIB_K, CALIB_PAIRS = compute_calibration_k(RISKY_CODES, SAFE_CODES, target_usd=20000.0, beta=0.14, alpha=1.2)

# ---------- Streamlit App ----------
st.set_page_config(page_title="APOLLO", layout="wide")
st.title("Welcome to the Analysis Platform for Occupational Linkages and Labour Outcomes (APOLLO)")

# Sidebar sliders for beta and alpha
st.sidebar.subheader("Switching Cost Parameters")
beta = st.sidebar.slider("Skill distance scaling (beta)", min_value=0.0, max_value=0.5, value=0.14, step=0.01)
alpha = st.sidebar.slider("Non-linear exponent (alpha)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

# Sidebar slider for education distance
EDU_GAP = st.sidebar.slider(
    "Max education distance allowed (0 = same level only)",
    min_value=0,
    max_value=4,
    value=0,
    step=1,
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
- The cost combines the geometric mean of origin/destination wages, a non-linear skill-distance term,  
  a training-intensity multiplier based on standardized |z| bins, and the calibration factor **k**.  
        """
    )
    st.latex(r"""
\text{SwitchingCost}
= k \cdot \left(2\,\sqrt{w_o\,w_d}\right)
  \cdot \left(1 + \beta\,|z|^{\alpha}\right)
  \cdot m(|z|)
""")
    st.markdown(
        r"""
- Here, \( m(|z|) \in \{1.0,\,1.2,\,1.5,\,2.0\} \), and \( k \) is chosen so that the **average risky → safe** transition cost  
  (across all education levels) is about **$24,000**.  
- Adjust \( \beta \), \( \alpha \), and the maximum education distance in the sidebar to see sensitivity.  
        """
    )

# Sidebar
n_results = st.sidebar.slider("Number of results to show:", min_value=3, max_value=20, value=5)
menu = st.sidebar.radio("Choose an option:", ["Look up by code", "Look up by title", "Compare two jobs"])

# ---- Look up by code ----
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

            # Similarity histogram (under current education gap)
            st.subheader(f"Similarity Score Distribution for {code} – {code_to_title.get(code,'Unknown')}")
            st.altair_chart(plot_histogram(all_scores), use_container_width=True)

            # Switching cost histogram (under current education gap)
            costs_df = compute_switching_costs_from_origin(code, beta=beta, alpha=alpha)
            st.subheader(f"Switching Cost Distribution from {code} – {code_to_title.get(code,'Unknown')}")
            st.altair_chart(plot_cost_histogram(costs_df), use_container_width=True)

# ---- Look up by title ----
elif menu == "Look up by title":
    available_codes = [code for code in code_to_title if code in similarity_df.index]
    title_options = [f"{code} – {code_to_title[code]}" for code in available_codes]

    selected_item = st.selectbox("Select an occupation:", sorted(title_options))
    if selected_item:
        selected_code, selected_title = selected_item.split(" – ")
        top_results, bottom_results, all_scores = get_most_and_least_similar(selected_code, n=n_results)

        # Most similar
        st.subheader(f"Most Similar Occupations for {selected_code} – {selected_title}")
        df_top = pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])
        df_top["Switching Cost ($)"] = df_top["Code"].apply(lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha))
        df_top["Switching Cost ($)"] = df_top["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        # Least similar
        st.subheader(f"Least Similar Occupations for {selected_code} – {selected_title}")
        df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
        df_bottom["Switching Cost ($)"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(selected_code, x, beta=beta, alpha=alpha))
        df_bottom["Switching Cost ($)"] = df_bottom["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        # Similarity histogram
        st.subheader(f"Similarity Score Distribution for {selected_code} – {selected_title}")
        st.altair_chart(plot_histogram(all_scores), use_container_width=True)

        # Switching cost histogram
        costs_df = compute_switching_costs_from_origin(selected_code, beta=beta, alpha=alpha)
        st.subheader(f"Switching Cost Distribution from {selected_code} – {selected_title}")
        st.altair_chart(plot_cost_histogram(costs_df), use_container_width=True)

# ---- Compare two jobs ----
elif menu == "Compare two jobs":
    available_codes = [code for code in code_to_title if code in similarity_df.index]
    title_options = [f"{code} – {code_to_title[code]}" for code in available_codes]

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
                f"- Ranking: `{rank}` out of `{total}` occupations "
                f"(#{rank} most similar to {job1_code})"
            )

            if cost is not None:
                st.info(
                    f"💰 **Estimated Switching Cost** (from {job1_code} to {job2_code}): "
                    f"`${cost:,.2f}` (education distance ≤ {EDU_GAP})"
                )
            else:
                st.info(
                    "ℹ️ Switching cost is not reported because the two occupations are further apart in education "
                    f"than the allowed maximum distance ({EDU_GAP})."
                )
        else:
            st.error("❌ Could not compare occupations.")
