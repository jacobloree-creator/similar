import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load Data ----------
@st.cache_data
def load_data():
    similarity = pd.read_excel("similarity_matrix.xlsx", index_col=0)
    wages = pd.read_excel("monthly_wages.xlsx")

    wages = wages.set_index("code")["monthly_wage"]
    return similarity, wages

similarity_matrix, wages = load_data()

# ---------- Standardize Distances ----------
distances = 1 - similarity_matrix  # assuming similarity ∈ [0,1]
mean_dist = distances.values[np.triu_indices_from(distances, k=1)].mean()
std_dist = distances.values[np.triu_indices_from(distances, k=1)].std()

standardized = (distances - mean_dist) / std_dist

# ---------- Switching Cost Function ----------
BETA = 0.14

def calculate_cost(origin, dest):
    if origin not in wages:
        return np.nan
    base_cost = 2 * wages[origin]
    adj = 1 + BETA * standardized.loc[origin, dest]
    return max(base_cost * adj, 0)

# Build cost matrix
cost_matrix = pd.DataFrame(index=similarity_matrix.index, columns=similarity_matrix.columns)
for i in similarity_matrix.index:
    for j in similarity_matrix.columns:
        if i != j:
            cost_matrix.loc[i, j] = calculate_cost(i, j)

cost_matrix = cost_matrix.astype(float)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Occupation Similarity & Switching Costs", layout="wide")

st.title("🔍 Occupation Similarity & Switching Costs")

# ---------- About the App ----------
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

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Look up by Code", "Look up by Title", "Compare Two Jobs"])

# ---------- Look up by Code ----------
with tab1:
    st.header("Look Up by Occupation Code")
    code = st.selectbox("Select an occupation code:", similarity_matrix.index)

    if code:
        df = pd.DataFrame({
            "code": similarity_matrix.columns,
            "similarity": similarity_matrix.loc[code],
            "standardized_distance": standardized.loc[code],
            "switching_cost": cost_matrix.loc[code]
        }).drop(index=code).dropna()

        df = df[df["similarity"] != 0]  # remove zero similarity

        df["switching_cost"] = df["switching_cost"].apply(
            lambda x: f"{x:,.2f}"
        )

        st.subheader("Most Similar Occupations")
        st.dataframe(
            df.sort_values("similarity", ascending=False).head(10),
            use_container_width=True,
            column_config={
                "code": st.column_config.Column("Code", width="small"),
                "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
                "standardized_distance": st.column_config.NumberColumn("Std Distance", format="%.2f"),
                "switching_cost": st.column_config.TextColumn("Switching Cost ($)")
            }
        )

        st.subheader("Least Similar Occupations")
        st.dataframe(
            df.sort_values("similarity", ascending=True).head(10),
            use_container_width=True,
            column_config={
                "code": st.column_config.Column("Code", width="small"),
                "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
                "standardized_distance": st.column_config.NumberColumn("Std Distance", format="%.2f"),
                "switching_cost": st.column_config.TextColumn("Switching Cost ($)")
            }
        )

# ---------- Look up by Title ----------
with tab2:
    st.header("Look Up by Occupation Title")
    title_map = pd.read_excel("occupation_titles.xlsx").set_index("code")["title"]

    code_by_title = {v: k for k, v in title_map.items()}
    title = st.selectbox("Select an occupation title:", sorted(code_by_title.keys()))

    if title:
        code = code_by_title[title]
        df = pd.DataFrame({
            "code": similarity_matrix.columns,
            "title": title_map,
            "similarity": similarity_matrix.loc[code],
            "standardized_distance": standardized.loc[code],
            "switching_cost": cost_matrix.loc[code]
        }).drop(index=code).dropna()

        df = df[df["similarity"] != 0]  # remove zero similarity

        df["switching_cost"] = df["switching_cost"].apply(
            lambda x: f"{x:,.2f}"
        )

        st.subheader("Most Similar Occupations")
        st.dataframe(
            df.sort_values("similarity", ascending=False).head(10),
            use_container_width=True,
            column_config={
                "code": st.column_config.Column("Code", width="small"),
                "title": st.column_config.Column("Title", width="large"),
                "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
                "standardized_distance": st.column_config.NumberColumn("Std Distance", format="%.2f"),
                "switching_cost": st.column_config.TextColumn("Switching Cost ($)")
            }
        )

        st.subheader("Least Similar Occupations")
        st.dataframe(
            df.sort_values("similarity", ascending=True).head(10),
            use_container_width=True,
            column_config={
                "code": st.column_config.Column("Code", width="small"),
                "title": st.column_config.Column("Title", width="large"),
                "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
                "standardized_distance": st.column_config.NumberColumn("Std Distance", format="%.2f"),
                "switching_cost": st.column_config.TextColumn("Switching Cost ($)")
            }
        )

# ---------- Compare Two Jobs ----------
with tab3:
    st.header("Compare Two Occupations")

    col1, col2 = st.columns(2)
    with col1:
        job1 = st.selectbox("Select first occupation:", similarity_matrix.index, key="job1")
    with col2:
        job2 = st.selectbox("Select second occupation:", similarity_matrix.index, key="job2")

    if job1 and job2:
        sim = similarity_matrix.loc[job1, job2]
        std = standardized.loc[job1, job2]
        cost = cost_matrix.loc[job1, job2]

        st.metric("Similarity Score", f"{sim:.3f}")
        st.metric("Standardized Distance", f"{std:.2f}")
        st.metric("Switching Cost", f"${cost:,.2f}")

# ---------- Histograms ----------
st.header("Distributions Across All Occupations")

all_sims = similarity_matrix.values.flatten()
all_sims = all_sims[all_sims != 0]

all_costs = cost_matrix.values.flatten()
all_costs = all_costs[~np.isnan(all_costs)]

fig1, ax1 = plt.subplots()
ax1.hist(all_sims, bins=30, color="skyblue", edgecolor="black")
ax1.set_title("Distribution of Similarity Scores")
ax1.set_xlabel("Similarity")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.hist(all_costs, bins=30, color="salmon", edgecolor="black")
ax2.set_title("Distribution of Switching Costs")
ax2.set_xlabel("Cost ($)")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)
