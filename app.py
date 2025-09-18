import streamlit as st
import pandas as pd
import os
from difflib import get_close_matches  # for fuzzy title search

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    similarity_df = pd.read_excel(os.path.join(base_path, "similarity matrix_v2.xlsx"), index_col=0)
    similarity_df.index = similarity_df.index.astype(str).str.strip()
    similarity_df.columns = similarity_df.columns.astype(str).str.strip()

    titles_df = pd.read_excel(os.path.join(base_path, "noc title.xlsx"))
    titles_df.columns = titles_df.columns.str.strip().str.lower()
    titles_df["noc"] = titles_df["noc"].astype(str).str.strip()

    code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
    title_to_code = {v.lower(): k for k, v in code_to_title.items()}

    return similarity_df, code_to_title, title_to_code

similarity_df, code_to_title, title_to_code = load_data()

# ---------- Helper Functions ----------
def find_code_from_title(title_input):
    """Find occupation codes given part of a title (case insensitive)."""
    title_input = title_input.lower()
    matches = [code for code, title in code_to_title.items() if title_input in title.lower()]

    # If no substring matches, use fuzzy matching
    if not matches:
        all_titles = list(code_to_title.values())
        close_matches = get_close_matches(title_input, all_titles, n=5, cutoff=0.6)
        matches = [title_to_code[t.lower()] for t in close_matches if t.lower() in title_to_code]

    return matches

def get_top_similar(code, n=5):
    if code not in similarity_df.index:
        return None
    scores = similarity_df.loc[code].drop(code)
    top_matches = scores.nsmallest(n)
    results = [(occ, code_to_title.get(occ, "Unknown Title"), score) for occ, score in top_matches.items()]
    return results

def get_least_similar(code, n=5):
    if code not in similarity_df.index:
        return None
    scores = similarity_df.loc[code].drop(code)
    worst_matches = scores.nlargest(n)
    results = [(occ, code_to_title.get(occ, "Unknown Title"), score) for occ, score in worst_matches.items()]
    return results

def compare_two_jobs(code1, code2):
    if code1 not in similarity_df.index or code2 not in similarity_df.index:
        return None
    scores = similarity_df.loc[code1].drop(code1).sort_values()
    if code2 not in scores.index:
        return None
    rank = scores.index.get_loc(code2) + 1
    score = similarity_df.loc[code1, code2]
    total = len(scores)
    return score, rank, total

# ---------- Streamlit App ----------
st.set_page_config(page_title="Occupation Similarity App", layout="wide")
st.title("🔍 Occupation Similarity App")

menu = st.sidebar.radio(
    "Choose an option:",
    ["Look up by code", "Look up by title", "Compare two jobs", "About the App"],
)

# ---- Look up by code ----
if menu == "Look up by code":
    code = st.text_input("Enter 5-digit occupation code:")
    if code:
        if code in similarity_df.index:
            st.subheader(f"Top Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            st.dataframe(pd.DataFrame(get_top_similar(code), columns=["Code", "Title", "Similarity Score"]))

            st.subheader(f"Least Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            st.dataframe(pd.DataFrame(get_least_similar(code), columns=["Code", "Title", "Similarity Score"]))
        else:
            st.error("❌ Invalid occupation code.")

# ---- Look up by title ----
elif menu == "Look up by title":
    title_input = st.text_input("Enter occupation title (or part of it):")
    if title_input:
        matches = find_code_from_title(title_input)
        if not matches:
            st.error("❌ No matches found.")
        else:
            selected_code = st.selectbox(
                "Select a matching occupation:",
                matches,
                format_func=lambda c: f"{c} - {code_to_title[c]}",
            )
            st.subheader(f"Top Similar Occupations for {selected_code} – {code_to_title.get(selected_code,'Unknown')}")
            st.dataframe(pd.DataFrame(get_top_similar(selected_code), columns=["Code", "Title", "Similarity Score"]))

            st.subheader(f"Least Similar Occupations for {selected_code} – {code_to_title.get(selected_code,'Unknown')}")
            st.dataframe(pd.DataFrame(get_least_similar(selected_code), columns=["Code", "Title", "Similarity Score"]))

# ---- Compare two jobs ----
elif menu == "Compare two jobs":
    job1 = st.text_input("Enter first occupation code or title:")
    job2 = st.text_input("Enter second occupation code or title:")
    if st.button("Compare"):
        if not job1.isdigit():
            matches = find_code_from_title(job1)
            job1 = matches[0] if matches else None
        if not job2.isdigit():
            matches = find_code_from_title(job2)
            job2 = matches[0] if matches else None
        if not job1 or not job2:
            st.error("❌ One or both occupations not found.")
        else:
            result = compare_two_jobs(job1, job2)
            if result:
                score, rank, total = result
                st.success(
                    f"**Comparison Result:**\n\n"
                    f"- {job1} ({code_to_title.get(job1,'Unknown')}) vs {job2} ({code_to_title.get(job2,'Unknown')})\n"
                    f"- Similarity score: `{score:.4f}`\n"
                    f"- Ranking: `{rank}` out of `{total}` occupations (# {rank} most similar)"
                )
            else:
                st.error("❌ Could not compare occupations.")

# ---- About the App ----
elif menu == "About the App":
    st.markdown("""
    ### About the App  
    Similarity scores come from Euclidian distance measures of O*NET skills, abilities, and knowledge required to perform a specific job.  
    Each score measures the total distance between each occupation pair combo. **Smaller numbers mean more similar.**  
    Data comes from the 2025 release.  
    """)
