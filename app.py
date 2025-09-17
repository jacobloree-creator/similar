import streamlit as st
import pandas as pd
import os

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)  # ensure Excel files load from repo folder

    similarity_df = pd.read_excel(os.path.join(base_path, "similarity matrix.xlsx"), index_col=0)
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
    title_input = title_input.lower()
    matches = [code for code, title in code_to_title.items() if title_input in title.lower()]
    return matches

def get_top_similar(code, n=5):
    if code not in similarity_df.index:
        return None
    scores = similarity_df.loc[code].drop(code)
    top_matches = scores.nsmallest(n)
    results = [(occ, code_to_title.get(occ, "Unknown Title"), score) for occ, score in top_matches.items()]
    return results

def compare_two_jobs(code1, code2):
    """Return similarity score and ranking position of code2 in code1’s similarity list."""
    # Ensure both codes are present
    if code1 not in similarity_df.index or code2 not in similarity_df.index:
        return None

    # Get scores for code1
    scores = similarity_df.loc[code1].drop(code1).sort_values()

    # Compute rank safely
    if code2 not in scores.index:
        return None
    try:
        rank = scores.index.get_loc(code2) + 1  # 1-based rank
    except KeyError:
        rank = None

    # Lookup similarity score, checking both directions
    score = similarity_df.loc[code1, code2]
    if pd.isna(score):  # try reverse
        score = similarity_df.loc[code2, code1]

    if pd.isna(score):
        return None  # no valid similarity

    total = len(scores)
    return score, rank, total



# ---------- Streamlit App ----------
st.set_page_config(page_title="Occupation Similarity App", layout="centered")

st.title("🔍 Occupation Similarity App")

menu = st.sidebar.radio("Choose an option:", ["Look up by code", "Look up by title", "Compare two jobs"])

if menu == "Look up by code":
    code = st.text_input("Enter 5-digit occupation code:")
    if code:
        if code in similarity_df.index:
            results = get_top_similar(code)
            st.subheader(f"Top Similar Occupations for {code} – {code_to_title.get(code,'Unknown')}")
            st.dataframe(pd.DataFrame(results, columns=["Code", "Title", "Similarity Score"]))
        else:
            st.error("❌ Invalid occupation code.")

elif menu == "Look up by title":
    title_input = st.text_input("Enter occupation title (or part of it):")
    if title_input:
        matches = find_code_from_title(title_input)
        if not matches:
            st.error("❌ No matches found.")
        else:
            selected_code = st.selectbox("Select a matching occupation:", matches, format_func=lambda c: f"{c} - {code_to_title[c]}")
            results = get_top_similar(selected_code)
            st.subheader(f"Top Similar Occupations for {selected_code} – {code_to_title.get(selected_code,'Unknown')}")
            st.dataframe(pd.DataFrame(results, columns=["Code", "Title", "Similarity Score"]))

elif menu == "Compare two jobs":
    job1 = st.text_input("Enter first occupation code or title:")
    job2 = st.text_input("Enter second occupation code or title:")

    if st.button("Compare"):
        # Resolve titles → codes
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
                st.success(f"**Comparison Result:**\n\n"
                           f"- {job1} ({code_to_title.get(job1,'Unknown')}) vs {job2} ({code_to_title.get(job2,'Unknown')})\n"
                           f"- Similarity score: `{score:.4f}`\n"
                           f"- Ranking: `{rank}` out of `{total}` occupations (# {rank} most similar)")
            else:
                st.error("❌ Could not compare occupations.")
