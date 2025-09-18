import streamlit as st
import pandas as pd
import os
from difflib import get_close_matches

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    # Load similarity matrix
    similarity_df = pd.read_excel(os.path.join(base_path, "similarity matrix_v2.xlsx"), index_col=0)
    similarity_df.index = similarity_df.index.astype(str).str.strip().str.zfill(5)
    similarity_df.columns = similarity_df.columns.astype(str).str.strip().str.zfill(5)

    # Load occupation titles
    titles_df = pd.read_excel(os.path.join(base_path, "noc title.xlsx"))
    titles_df.columns = titles_df.columns.str.strip().str.lower()
    titles_df["noc"] = titles_df["noc"].astype(str).str.strip().str.zfill(5)

    # Dictionaries
    code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
    title_to_code = {v.lower(): k for k, v in code_to_title.items()}

    return similarity_df, code_to_title, title_to_code

similarity_df, code_to_title, title_to_code = load_data()

# ---------- Helper Functions ----------
def find_code_from_title(title_input):
    """Return list of codes matching the input string (with fuzzy matching)."""
    title_input = title_input.lower()
    matches = [code for code, title in code_to_title.items() if title_input in title.lower()]

    if not matches:
        all_titles = list(code_to_title.values())
        close_matches = get_close_matches(title_input, all_titles, n=5, cutoff=0.6)
        matches = [title_to_code[t.lower()] for t in close_matches if t.lower() in title_to_code]

    return matches

def get_top_and_bottom_similar(code, n=5):
    """Return top N most similar and least similar occupations for a given code."""
    if code not in similarity_df.index:
        return None, None

    row = similarity_df.loc[code].drop(code).dropna()
    top_matches = row.nsmallest(n)
    bottom_matches = row.nlargest(n)

    top_results = [(c, code_to_title.get(c, "Unknown"), score) for c, score in top_matches.items()]
    bottom_results = [(c, code_to_title.get(c, "Unknown"), score) for c, score in bottom_matches.items()]

    return top_results, bottom_results

def compare_two_jobs(code1, code2):
    """Return similarity score and ranking of code2 in code1's similarity list."""
    if code1 not in similarity_df.index or code2 not in similarity_df.index:
        return None

    scores = similarity_df.loc[code1].drop(code1).dropna().sort_values()
    if code2 not in scores.index:
        return None

    rank = scores.index.get_loc(code2) + 1
    score = scores[code2]
    total = len(scores)

    return score, rank, total

# ---------- Streamlit App ----------
st.set_page_config(page_title="Occupation Similarity App", layout="centered")
st.title("🔍 Occupation Similarity App")

menu = st.sidebar.radio(
    "Choose an option:",
    ["Look up by code", "Look up by title", "Compare two jobs", "About the app"]
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
            top_results, bottom_results = get_top_and_bottom_similar(code, n=n_results)

            st.subheader(f"Most Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            st.dataframe(pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"]))

            st.subheader(f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            st.dataframe(pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"]))

# ----- Look up by title -----
elif menu == "Look up by title":
    all_titles = [f"{c} – {t}" for c, t in code_to_title.items()]
    selected_item = st.selectbox("Select an occupation:", sorted(all_titles))
    selected_code = selected_item.split(" – ")[0]

    if selected_code not in similarity_df.index:
        st.error("❌ No similarity scores available for this occupation.")
    else:
        top_results, bottom_results = get_top_and_bottom_similar(selected_code, n=n_results)

        st.subheader(f"Most Similar Occupations for {selected_code} – {code_to_title.get(selected_code,'Unknown')}")
        st.dataframe(pd.DataFrame(top_results, columns=["Code", "Title", "Similarity Score"]))

        st.subheader(f"Least Similar Occupations for {selected_code} – {code_to_title.get(selected_code,'Unknown')}")
        st.dataframe(pd.DataFrame(bottom_results, columns=["Code", "Title", "Similarity Score"]))

# ----- Compare two jobs -----
elif menu == "Compare two jobs":
    all_titles = [f"{c} – {t}" for c, t in code_to_title.items()]
    job1_item = st.selectbox("Select first occupation:", sorted(all_titles))
    job2_item = st.selectbox("Select second occupation:", sorted(all_titles))

    job1_code = job1_item.split(" – ")[0]
    job2_code = job2_item.split(" – ")[0]

    if st.button("Compare"):
        result = compare_two_jobs(job1_code, job2_code)
        if result:
            score, rank, total = result
            st.success(
                f"**Comparison Result:**\n\n"
                f"- {job1_code} ({code_to_title.get(job1_code,'Unknown')}) vs "
                f"{job2_code} ({code_to_title.get(job2_code,'Unknown')})\n"
                f"- Similarity score: `{score:.4f}`\n"
                f"- Ranking: `{rank}` out of `{total}` occupations"
            )
        else:
            st.error("❌ Could not compare occupations or one/both codes missing similarity scores.")

# ----- About -----
elif menu == "About the app":
    st.header("ℹ️ About the App")
    st.write("""
    Similarity scores come from Euclidian distance measures of ONET skills, abilities, and knowledge required to perform
    a specific job. Each score measures the total distance between each occupation pair combo. Smaller numbers mean more similar.
    Data comes from the 2025 release.
    """)
