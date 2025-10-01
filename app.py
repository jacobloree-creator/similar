import streamlit as st
import pandas as pd
import os
import altair as alt

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

    # Create mappings
    code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
    title_to_code = {v.lower(): k for k, v in code_to_title.items()}

    # ---- Standardize distances (z-scores) ----
    flat_scores = similarity_df.where(~pd.isna(similarity_df)).stack().values
    mean_val, std_val = flat_scores.mean(), flat_scores.std()
    standardized_df = (similarity_df - mean_val) / std_val

    return similarity_df, standardized_df, code_to_title, title_to_code, code_to_wage

similarity_df, standardized_df, code_to_title, title_to_code, code_to_wage = load_data()

# ---------- Helper Functions ----------
def get_most_and_least_similar(code, n=5):
    if code not in similarity_df.index:
        return None, None, None
    scores = similarity_df.loc[code].drop(code).dropna()

    # Remove zeros
    scores = scores[scores != 0]

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
    scores = scores[scores != 0]  # remove zeros
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

def calculate_switching_cost(code1, code2, beta=0.14):
    """Estimate switching cost from occupation code1 to code2."""
    if code1 not in standardized_df.index or code2 not in standardized_df.index:
        return None
    z_score = standardized_df.loc[code1, code2]
    if pd.isna(z_score):
        z_score = standardized_df.loc[code2, code1]
    if pd.isna(z_score):
        return None
    wage = code_to_wage.get(code1)
    if wage is None:
        return None
    base_cost = 2 * wage
    cost = base_cost * (1 + beta * z_score)
    return cost

def plot_histogram(scores, highlight_score=None):
    """Helper to plot histogram with optional red marker."""
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

# ---------- Streamlit App ----------
st.set_page_config(page_title="Occupation Similarity App", layout="wide")
st.title("🔍 Occupation Similarity App")

# About section
with st.expander("ℹ️ About this app"):
    st.markdown(
        """
        - Similarity scores are based on Euclidean distances of O*NET skill, ability, and knowledge vectors.
          Smaller scores mean occupations are more similar.
        - Switching costs are generated following Kambourov & Manovskii (2009) and Hawkins (2017, KC Fed) calibrations, 
          where switching between occupations costs roughly two months of origin occupation wages.  
          This penalty is scaled following Cortes and Gallipoli (2016), which finds the penalty is 16% higher per standard deviation increase in similarity score.
          Since this penalty comes from an average impact and is applied linearly, expect costs to differ from real life for very close and very far away matches, as
          costing is almost certainly non-linear.
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
            df_top["Switching Cost ($)"] = df_top["Code"].apply(lambda x: calculate_switching_cost(code, x))
            df_top["Switching Cost ($)"] = df_top["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

            # Least similar
            st.subheader(f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
            df_bottom["Switching Cost ($)"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(code, x))
            df_bottom["Switching Cost ($)"] = df_bottom["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

            # Histogram
            st.subheader(f"Similarity Score Distribution for {code} – {code_to_title.get(code,'Unknown')}")
            st.altair_chart(plot_histogram(all_scores), use_container_width=True)

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
        df_top["Switching Cost ($)"] = df_top["Code"].apply(lambda x: calculate_switching_cost(selected_code, x))
        df_top["Switching Cost ($)"] = df_top["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        st.dataframe(df_top, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        # Least similar
        st.subheader(f"Least Similar Occupations for {selected_code} – {selected_title}")
        df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
        df_bottom["Switching Cost ($)"] = df_bottom["Code"].apply(lambda x: calculate_switching_cost(selected_code, x))
        df_bottom["Switching Cost ($)"] = df_bottom["Switching Cost ($)"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        st.dataframe(df_bottom, use_container_width=True, column_config={"Title": st.column_config.Column(width="large")})

        # Histogram
        st.subheader(f"Similarity Score Distribution for {selected_code} – {selected_title}")
        st.altair_chart(plot_histogram(all_scores), use_container_width=True)

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
            cost = calculate_switching_cost(job1_code, job2_code)

            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({job1_title}) vs {job2_code} ({job2_title})\n"
                f"- Similarity score (raw distance): `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations "
                f"(#{rank} most similar to {job1_code})"
            )

            if cost is not None:
                st.info(f"💰 **Estimated Switching Cost** (from {job1_code} to {job2_code}): "
                        f"`${cost:,.2f}` (2 months wages × distance adjustment)")

            # Histogram with marker
            st.subheader(f"Similarity Score Distribution for {job1_code} – {job1_title}")
            st.altair_chart(plot_histogram(similarity_df.loc[job1_code].drop(job1_code).dropna(), highlight_score=score), use_container_width=True)
        else:
            st.error("❌ Could not compare occupations.")
