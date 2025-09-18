import streamlit as st
import pandas as pd
import os
import altair as alt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    # Load similarity matrix
    similarity_df = pd.read_excel(os.path.join(base_path, "similarity matrix_v2.xlsx"), index_col=0)
    similarity_df.index = similarity_df.index.astype(str).str.zfill(5).str.strip()
    similarity_df.columns = similarity_df.columns.astype(str).str.zfill(5).str.strip()

    # Load NOC titles
    titles_df = pd.read_excel(os.path.join(base_path, "noc title.xlsx"))
    titles_df.columns = titles_df.columns.str.strip().str.lower()
    titles_df["noc"] = titles_df["noc"].astype(str).str.zfill(5).str.strip()

    # Create mappings
    code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
    title_to_code = {v.lower(): k for k, v in code_to_title.items()}

    return similarity_df, code_to_title, title_to_code

similarity_df, code_to_title, title_to_code = load_data()

# ---------- Helper Functions ----------
def get_most_and_least_similar(code, n=5):
    if code not in similarity_df.index:
        return None, None, None
    scores = similarity_df.loc[code].drop(code).dropna()
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
    scores = similarity_df.loc[code1].drop(code1).dropna().sort_values()
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

# ---------- Streamlit App ----------
st.set_page_config(page_title="Occupation Similarity App", layout="centered")
st.title("🔍 Occupation Similarity App")

# Sidebar
n_results = st.sidebar.slider("Number of results to show:", min_value=3, max_value=20, value=5)
menu = st.sidebar.radio("Choose an option:", ["Look up by code", "Look up by title", "Compare two jobs"])

# About section
with st.expander("ℹ️ About the app"):
    st.write("""
    Similarity scores come from Euclidian distance measures of ONET skills, abilities, and knowledge required to perform
    a specific job. Each score measures the total distance between each occupation pair combo. Smaller numbers mean more similar.
    Data comes from the 2025 release.
    """)

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
            st.dataframe(df_top)
            st.download_button("📥 Download most similar results",
                               df_top.to_csv(index=False).encode("utf-8"),
                               file_name=f"{code}_most_similar.csv")

            # Least similar
            st.subheader(f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
            st.dataframe(df_bottom)
            st.download_button("📥 Download least similar results",
                               df_bottom.to_csv(index=False).encode("utf-8"),
                               file_name=f"{code}_least_similar.csv")

            # Histogram
            st.subheader(f"Similarity Score Distribution for {code} – {code_to_title.get(code,'Unknown')}")
            st.write("Placeholder: Explain histogram interpretation.")
            hist_df = pd.DataFrame({"score": all_scores.values})
            hist_chart = (
                alt.Chart(hist_df)
                .mark_bar(opacity=0.7, color="steelblue")
                .encode(
                    alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Similarity Score"),
                    alt.Y("count()", title="Number of Occupations"),
                    tooltip=["count()"]
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(hist_chart, use_container_width=True)

# ---- Look up by title with search ----
elif menu == "Look up by title":
    available_codes = [code for code in code_to_title if code in similarity_df.index]
    all_titles = [f"{code} – {code_to_title[code]}" for code in available_codes]

    search_input = st.text_input("Type occupation title (or part of it) to search:")
    if search_input:
        filtered_titles = [t for t in all_titles if search_input.lower() in t.lower()]
        if not filtered_titles:
            st.warning("No matching occupations found.")
        else:
            selected_item = st.selectbox("Select an occupation:", filtered_titles)
    else:
        selected_item = st.selectbox("Select an occupation:", all_titles)

    if selected_item:
        selected_code, selected_title = selected_item.split(" – ")
        top_results, bottom_results, all_scores = get_most_and_least_similar(selected_code, n=n_results)

        # Most similar
        st.subheader(f"Most Similar Occupations for {selected_code} – {selected_title}")
        df_top = pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"])
        st.dataframe(df_top)
        st.download_button("📥 Download most similar results",
                           df_top.to_csv(index=False).encode("utf-8"),
                           file_name=f"{selected_code}_most_similar.csv")

        # Least similar
        st.subheader(f"Least Similar Occupations for {selected_code} – {selected_title}")
        df_bottom = pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"])
        st.dataframe(df_bottom)
        st.download_button("📥 Download least similar results",
                           df_bottom.to_csv(index=False).encode("utf-8"),
                           file_name=f"{selected_code}_least_similar.csv")

        # Histogram
        st.subheader(f"Similarity Score Distribution for {selected_code} – {selected_title}")
        st.write("Placeholder: Explain histogram interpretation.")
        hist_df = pd.DataFrame({"score": all_scores.values})
        hist_chart = (
            alt.Chart(hist_df)
            .mark_bar(opacity=0.7, color="steelblue")
            .encode(
                alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Similarity Score"),
                alt.Y("count()", title="Number of Occupations"),
                tooltip=["count()"]
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(hist_chart, use_container_width=True)

        # ----- Heatmap -----
        top_n = 20
        top_codes = list(similarity_df.loc[selected_code].nsmallest(top_n).index) + [selected_code]
        heatmap_df = similarity_df.loc[top_codes, top_codes]
        st.subheader(f"Similarity Heatmap (Top {top_n}) for {selected_title}")
        st.write("Placeholder: Darker color = more similar.")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(heatmap_df.astype(float), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # ----- Network -----
        st.subheader(f"Similarity Network (Top {top_n}) for {selected_title}")
        st.write("Placeholder: Node size/color = similarity strength.")
        G = nx.Graph()
        for occ in top_codes:
            G.add_node(occ, title=code_to_title.get(occ, "Unknown"))
        for occ in top_codes:
            for neighbor in top_codes:
                if occ != neighbor:
                    weight = similarity_df.loc[occ, neighbor]
                    G.add_edge(occ, neighbor, weight=weight)
        net = Network(height="500px", width="100%", notebook=False)
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])
        net_file = "temp_network.html"
        net.save_graph(net_file)
        components.html(open(net_file,'r', encoding='utf-8').read(), height=550)

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
            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({job1_title}) vs {job2_code} ({job2_title})\n"
                f"- Similarity score: `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations "
                f"(#{rank} most similar to {job1_code})"
            )
            st.subheader("Ranking Position Visualization")
            st.write("Placeholder: Interpretation of progress bar relative to all occupations.")
            st.progress(rank/total)

            # Histogram with comparison marker
            st.subheader(f"Similarity Score Distribution for {job1_code} – {job1_title}")
            st.write("Placeholder: Vertical red line marks similarity with selected occupation.")
            hist_df = pd.DataFrame({"score": similarity_df.loc[job1_code].drop(job1_code).dropna().values})
            hist_chart = (
                alt.Chart(hist_df)
                .mark_bar(opacity=0.7, color="steelblue")
                .encode(
                    alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Similarity Score"),
                    alt.Y("count()", title="Number of Occupations"),
                    tooltip=["count()"]
                )
                .properties(width=600, height=400)
            )
            line = (
                alt.Chart(pd.DataFrame({"score":[score]}))
                .mark_rule(color="red", strokeWidth=2)
                .encode(x="score:Q")
            )
            st.altair_chart(hist_chart + line, use_container_width=True)
        else:
            st.error("❌ Could not compare occupations.")
