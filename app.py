import streamlit as st
import pandas as pd
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


# ---------- Switching Cost Function ----------
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


# ---------- Streamlit App ----------
st.title("Occupation Similarity & Switching Cost Explorer")

st.sidebar.header("Navigation")
options = ["Compare two jobs", "Look up by code", "Look up by title"]
choice = st.sidebar.radio("Go to", options)


# ---------- Compare Two Jobs ----------
if choice == "Compare two jobs":
    st.header("Compare Two Jobs")

    job1_code = st.text_input("Enter first job code (NOC)", "").zfill(5).strip()
    job2_code = st.text_input("Enter second job code (NOC)", "").zfill(5).strip()

    if st.button("Compare"):
        if job1_code in similarity_df.index and job2_code in similarity_df.columns:
            score = similarity_df.loc[job1_code, job2_code]
            job1_title = code_to_title.get(job1_code, "Unknown Job")
            job2_title = code_to_title.get(job2_code, "Unknown Job")

            # Rank within job1 row
            row = similarity_df.loc[job1_code].sort_values()
            rank = row.index.get_loc(job2_code) + 1
            total = len(row)

            # Calculate switching cost
            cost = calculate_switching_cost(job1_code, job2_code)

            # Show results
            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({job1_title}) vs {job2_code} ({job2_title})\n"
                f"- Similarity score (raw distance): `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations "
                f"(#{rank} most similar to {job1_code})"
            )

            if cost is not None:
                st.info(
                    f"💰 **Estimated Switching Cost** "
                    f"(from {job1_code} → {job2_code}): "
                    f"`${cost:,.0f}`\n\n"
                    f"(2 months wages × distance adjustment)"
                )

            # Histogram
            fig, ax = plt.subplots()
            row.plot(kind="hist", bins=20, ax=ax, alpha=0.7)
            ax.axvline(score, color="red", linestyle="--")
            ax.set_title(f"Distribution of distances for {job1_code}")
            st.pyplot(fig)
        else:
            st.error("One or both job codes not found.")


# ---------- Look Up by Code ----------
elif choice == "Look up by code":
    st.header("Look Up by Code")

    job_code = st.text_input("Enter job code (NOC)", "").zfill(5).strip()

    if job_code in similarity_df.index:
        job_title = code_to_title.get(job_code, "Unknown Job")
        st.write(f"### {job_code}: {job_title}")

        row = similarity_df.loc[job_code].sort_values()

        # Top 10 most similar
        top_similar = row.head(10)
        results_top = []
        for other_code, score in top_similar.items():
            other_title = code_to_title.get(other_code, "Unknown Job")
            cost = calculate_switching_cost(job_code, other_code)
            results_top.append([other_code, other_title, score, cost])

        st.subheader("Most Similar Occupations")
        st.dataframe(pd.DataFrame(results_top, columns=["Code", "Title", "Distance", "Switching Cost ($)"]))

        # 10 least similar
        bottom_similar = row.tail(10)
        results_bottom = []
        for other_code, score in bottom_similar.items():
            other_title = code_to_title.get(other_code, "Unknown Job")
            cost = calculate_switching_cost(job_code, other_code)
            results_bottom.append([other_code, other_title, score, cost])

        st.subheader("Least Similar Occupations")
        st.dataframe(pd.DataFrame(results_bottom, columns=["Code", "Title", "Distance", "Switching Cost ($)"]))

    else:
        st.warning("Job code not found.")


# ---------- Look Up by Title ----------
elif choice == "Look up by title":
    st.header("Look Up by Title")

    job_title = st.text_input("Enter job title", "").lower().strip()

    if job_title in title_to_code:
        job_code = title_to_code[job_title]
        st.write(f"### {job_code}: {code_to_title[job_code]}")

        row = similarity_df.loc[job_code].sort_values()

        # Top 10 most similar
        top_similar = row.head(10)
        results_top = []
        for other_code, score in top_similar.items():
            other_title = code_to_title.get(other_code, "Unknown Job")
            cost = calculate_switching_cost(job_code, other_code)
            results_top.append([other_code, other_title, score, cost])

        st.subheader("Most Similar Occupations")
        st.dataframe(pd.DataFrame(results_top, columns=["Code", "Title", "Distance", "Switching Cost ($)"]))

        # 10 least similar
        bottom_similar = row.tail(10)
        results_bottom = []
        for other_code, score in bottom_similar.items():
            other_title = code_to_title.get(other_code, "Unknown Job")
            cost = calculate_switching_cost(job_code, other_code)
            results_bottom.append([other_code, other_title, score, cost])

        st.subheader("Least Similar Occupations")
        st.dataframe(pd.DataFrame(results_bottom, columns=["Code", "Title", "Distance", "Switching Cost ($)"]))

    else:
        st.warning("Job title not found.")
