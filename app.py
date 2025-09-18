import streamlit as st
import pandas as pd
import os
import altair as alt
from difflib import get_close_matches  # for fuzzy title search

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    similarity_df = pd.read_excel(os.path.join(base_path, "similarity matrix_v2.xlsx"), index_col=0)
    similarity_df.index = similarity_df.index.astype(str).str.strip().str.zfill(5)
    similarity_df.columns = similarity_df.columns.astype(str).str.strip().str.zfill(5)

    titles_df = pd.read_excel(os.path.join(base_path, "noc title.xlsx"))
    titles_df.columns = titles_df.columns.str.strip().str.lower()
    titles_df["noc"] = titles_df["noc"].astype(str).str.strip().str.zfill(5)

    code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
    title_to_code = {v.lower(): k for k, v in code_to_title.items()}

    return similarity_df, code_to_title, title_to_code

similarity_df, code_to_title, title_to_code = load_data()

# ---------- Helper Functions ----------
def find_code_from_title(title_input):
    title_input = title_input.lower()
    matches = [code for code, title in code_to_title.items() if title_input in title.lower()]

    # Fuzzy match if no exact substring matches
    if not matches:
        all_titles = list(code_to_title.values())
        close_matches = get_close_matches(title_input, all_titles, n=5, cutoff=0.6)
        matches = [title_to_code[t.lower()] for t in close_matches if t.lower() in title_to_code]

    return matches

def get_top_and_bottom_similar(code, n=5):
    if code not in similarity_df.index:
        return None, None, None

    scores = similarity_df.loc[code].drop(code).dropna()

    top_matches = scores.nsmallest(n)
    bottom_matches = scores.nlargest(n)

    top_results = [(occ, code_to_title.get(occ, "Unknown"), score) for occ, score in top_matches.items()]
    bottom_results = [(occ, code_to_title.get(occ, "Unknown"), score) for occ, score in bottom_matches.items()]

    return top_results, bottom_results, top_matches

def compare_two_jobs(code1, code2):
    if code1 not in similarity_df.index or code2 not in similarity_df.index:
        return None

    scores = similarity_df.loc[code1].drop(code1).dropna().sort_values()
    if code2 not in scores.index:
        return None

    rank = scores.index.get_loc(code2) + 1
    score = similarity_df.loc[code1, code2]
    total = len(scores)

    return score, rank, total

# ---------- Streamlit App ----------
st.set_page_config(page_title="Occupation Similarity App", layout="centered")
st.title("🔍 Occupation Similarity App")

menu = st.sidebar.radio(
    "Choose an option:",
    ["Look up by code", "Look up by title", "Compare two jobs", "About the app"],
)

n_results = st.sidebar.slider("Number of results to show:", 1, 20, 5)

# ----- Look up by code -----
if menu == "Look up by code":
    code = st.text_input("Enter 5-digit occupation code:").strip().zfill(5)
    if code:
        if code not in code_to_title:
            st.error("❌ Invalid occupation code.")
        elif code not in similarity_df.index:
            st.error("❌ No similarity scores available for this occupation.")
        else:
            top_results, bottom_results, top_scores = get_top_and_bottom_similar(code, n=n_results)

            st.subheader(f"Most Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            st.dataframe(pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"]))

            st.subheader(f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            st.dataframe(pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"]))

            st.subheader(f"Similarity Score Distribution (Top {n_results}) for {code}")
            st.write("Placeholder: add interpretation guidance for this histogram.")
            hist_df = pd.DataFrame({"score": top_scores.values})
            hist_chart = (
                alt.Chart(hist_df)
                .mark_bar(opacity=0.7, color="steelblue")
                .encode(
                    alt.X("score:Q", bin=alt.Bin(maxbins=15), title="Similarity Score"),
                    alt.Y("count()", title="Number of Occupations"),
                    tooltip=["count()"]
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(hist_chart, use_container_width=True)

# ----- Look up by title -----
elif menu == "Look up by title":
    all_titles = [f"{c} – {t}" for c, t in code_to_title.items()]
    selected_item = st.selectbox("Select an occupation:", sorted(all_titles))
    selected_code = selected_item.split(" – ")[0]

    if selected_code not in similarity_df.index:
        st.error("❌ No similarity scores available for this occupation.")
    else:
        top_results, bottom_results, top_scores = get_top_and_bottom_similar(selected_code, n=n_results)

        st.subheader(f"Most Similar Occupations for {selected_code} – {code_to_title.get(selected_code,'Unknown')}")
        st.dataframe(pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"]))

        st.subheader(f"Least Similar Occupations for {selected_code} – {code_to_title.get(selected_code,'Unknown')}")
        st.dataframe(pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"]))

        st.subheader(f"Similarity Score Distribution (Top {n_results}) for {selected_code}")
        st.write("Placeholder: add interpretation guidance for this histogram.")
        hist_df = pd.DataFrame({"score": top_scores.values})
        hist_chart = (
            alt.Chart(hist_df)
            .mark_bar(opacity=0.7, color="steelblue")
            .encode(
                alt.X("score:Q", bin=alt.Bin(maxbins=15), title="Similarity Score"),
                alt.Y("count()", title="Number of Occupations"),
                tooltip=["count()"]
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(hist_chart, use_container_width=True)

# ----- Compare two jobs -----
elif menu == "Compare two jobs":
    all_titles = [f"{c} – {t}" for c, t in code_to_title.items()]
    job1_item = st.selectbox("Select first occupation:", sorted(all_titles))
    job2_item = st.selectbox("Select second occupation:", sorted(all_titles))

    job1_code = job1_item.split(" – ")[0]
    job2_code = job2_item.split(" – ")[0]

    if st.button("Compare"):
        if job1_code not in similarity_df.index or job2_code not in similarity_df.index:
            st.error("❌ One or both occupations do not have similarity scores available.")
        else:
            result = compare_two_jobs(job1_code, job2_code)
            if result:
                score, rank, total = result
                st.success(
                    f"**Comparison Result:**\n\n"
                    f"- {job1_code} ({code_to_title.get(job1_code,'Unknown')}) vs "
                    f"{job2_code} ({code_to_title.get(job2_code,'Unknown')})\n"
                    f"- Similarity score: `{score:.4f}`\n"
                    f"- Ranking: `{rank}` out of `{total}` occupations "
                    f"(#{rank} most similar to {job1_code})"
                )

                # Simple ranking visualization
                st.subheader("Ranking Position Visualization")
                st.write("Placeholder: Add interpretation guidance for ranking visualization.")
                rank_df = pd.DataFrame({"Occupation": [job2_code], "Rank": [rank]})
                rank_chart = alt.Chart(rank_df).mark_bar(color="orange").encode(
                    x="Occupation:N", y="Rank:Q", tooltip=["Rank"]
                ).properties(width=300, height=200)
                st.altair_chart(rank_chart, use_container_width=True)
            else:
                st.error("❌ Could not compare occupations.")

# ----- About -----
elif menu == "About the app":
    st.header("ℹ️ About the App")
    st.write("""
    Similarity scores come from Euclidian distance measures of ONET skills, abilities, and knowledge required to perform
    a specific job. Each score measures the total distance between each occupation pair combo. Smaller numbers mean more similar.
    Data comes from the 2025 release.
    """)
