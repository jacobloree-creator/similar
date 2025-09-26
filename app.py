import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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

# ---------- Cost Function ----------
def calculate_switching_cost(code1, code2, beta=0.14):
    """Estimate switching cost from occupation code1 to code2."""
    if code1 not in standardized_df.index or code2 not in standardized_df.index:
        return None

    # Get standardized distance
    z_score = standardized_df.loc[code1, code2]
    if pd.isna(z_score):
        z_score = standardized_df.loc[code2, code1]

    if pd.isna(z_score):
        return None

    # Base cost = 2 months of wages of origin job
    wage = code_to_wage.get(code1)
    if wage is None:
        return None

    base_cost = 2 * wage
    cost = base_cost * (1 + beta * z_score)
    return cost

# ---------- Comparison Function ----------
def compare_two_jobs(code1, code2):
    """Compare two jobs by similarity and get ranking."""
    if code1 not in similarity_df.index or code2 not in similarity_df.columns:
        return None

    score = similarity_df.loc[code1, code2]

    similarities = similarity_df.loc[code1].sort_values()
    rank = similarities.reset_index().reset_index()
    rank.columns = ["rank", "code", "score"]
    rank["rank"] += 1
    rank = rank.set_index("code")
    rank_value = rank.loc[code2, "rank"]

    return score, rank_value, len(similarities)

# ---------- Streamlit App ----------
st.title("Occupation Similarity & Switching Costs")

# ---------- About the App ----------
with st.expander("ℹ️ About this app"):
    st.markdown(
        """
        - Similarity scores are based on the Euclidean distance of O*NET skill, knowledge, and ability vectors. 
          Smaller scores mean the two occupations are more similar.  
        - Switching costs are calculated Kambourov & Manovskii (2009) and Hawkins (2017, KC Fed), 
          which find the average occupation switch costs roughly two months of origin occupation wages.  
          This is scaled by how different two jobs are. We use Cortes and Gallipoli (2016)'s parameter of the 
          cost increasing by roughly 14% per standard deviation of similarity score increase.
    )

menu = ["Compare two jobs", "Look up by code", "Look up by title"]
choice = st.sidebar.radio("Select Option", menu)

# ---------- Compare Two Jobs ----------
if choice == "Compare two jobs":
    st.header("Compare Two Jobs")

    job1_code = st.text_input("Enter first job code (NOC):").zfill(5).strip()
    job2_code = st.text_input("Enter second job code (NOC):").zfill(5).strip()

    if st.button("Compare"):
        result = compare_two_jobs(job1_code, job2_code)
        if result:
            score, rank, total = result
            cost = calculate_switching_cost(job1_code, job2_code)

            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({code_to_title.get(job1_code, 'Unknown')}) vs "
                f"{job2_code} ({code_to_title.get(job2_code, 'Unknown')})\n"
                f"- Similarity score (raw distance): `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations "
                f"(#{rank} most similar to {job1_code})"
            )

            if cost is not None:
                st.info(f"💰 **Estimated Switching Cost**: "
                        f"${cost:,.2f} (2 months wages × distance adjustment)")

            # Histogram of similarities
            similarities = similarity_df.loc[job1_code].dropna()
            fig, ax = plt.subplots()
            ax.hist(similarities, bins=30, edgecolor="black")
            ax.axvline(score, color="red", linestyle="dashed", linewidth=1)
            ax.set_title("Distribution of Similarities")
            ax.set_xlabel("Similarity Score (distance)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

# ---------- Look Up by Code ----------
elif choice == "Look up by code":
    st.header("Look Up by Code")

    job_code = st.text_input("Enter job code (NOC):").zfill(5).strip()

    if job_code in similarity_df.index:
        st.subheader(f"Job: {job_code} - {code_to_title.get(job_code, 'Unknown')}")

        similarities = similarity_df.loc[job_code].dropna().sort_values()

        df = pd.DataFrame({
            "code": similarities.index,
            "title": [code_to_title.get(c, "Unknown") for c in similarities.index],
            "similarity_score": similarities.values,
            "switching_cost": [
                calculate_switching_cost(job_code, c) for c in similarities.index
            ]
        })

        df["switching_cost"] = df["switching_cost"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
        )

        st.subheader("Most Similar Occupations")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Least Similar Occupations")
        st.dataframe(df.tail(10), use_container_width=True)

# ---------- Look Up by Title ----------
elif choice == "Look up by title":
    st.header("Look Up by Title")

    job_title = st.text_input("Enter job title:").lower().strip()

    if job_title in title_to_code:
        job_code = title_to_code[job_title]
        st.subheader(f"Job: {job_code} - {job_title.title()}")

        similarities = similarity_df.loc[job_code].dropna().sort_values()

        df = pd.DataFrame({
            "code": similarities.index,
            "title": [code_to_title.get(c, "Unknown") for c in similarities.index],
            "similarity_score": similarities.values,
            "switching_cost": [
                calculate_switching_cost(job_code, c) for c in similarities.index
            ]
        })

        df["switching_cost"] = df["switching_cost"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
        )

        st.subheader("Most Similar Occupations")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Least Similar Occupations")
        st.dataframe(df.tail(10), use_container_width=True)
