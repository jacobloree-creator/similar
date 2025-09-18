import streamlit as st
import pandas as pd
import os
import altair as alt

# ---------- Load Data ----------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    # Load similarity matrix
    similarity_df = pd.read_excel(os.path.join(base_path, "similarity matrix.xlsx"), index_col=0)
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
def find_code_from_title(title_input):
    """
    Return a list of occupation codes where the input matches part of the title.
    Only returns codes that exist in the similarity matrix.
    """
    title_input = title_input.lower().strip()
    matches = [code for code, title in code_to_title.items() if title_input in title.lower()]
    # Keep only codes that exist in similarity matrix
    valid_matches = [c for c in matches if c in similarity_df.index]
    return valid_matches

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
            st.write("Placeholder text: Explain here how to interpret this histogram for the selected occupation.")
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
                format_func=lambda c: f"{c} – {code_to_title[c]}"
            )
            selected_title = code_to_title[selected_code]

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
            st.write("Placeholder text: Explain here how to interpret this histogram for the selected occupation.")
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

# ---- Compare two jobs ----
elif menu == "Compare two jobs":
    job1_input = st.text_input("Enter first occupation code or title:")
    job2_input = st.text_input("Enter second occupation code or title:")

    if st.button("Compare"):
        # Resolve first job
        if job1_input.isdigit():
            job1_code = str(job1_input).zfill(5).strip()
        else:
            matches = find_code_from_title(job1_input)
            job1_code = matches[0] if matches else None

        # Resolve second job
        if job2_input.isdigit():
            job2_code = str(job2_input).zfill(5).strip()
        else:
            matches = find_code_from_title(job2_input)
            job2_code = matches[0] if matches else None

        if not job1_code or not job2_code:
            st.error("❌ One or both occupations not found.")
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

                # Ranking progress bar
                st.subheader("Ranking Position Visualization")
                st.write("Placeholder text: Explain how to interpret this progress bar relative to all occupations.")
                st.progress(rank / total)

                # Histogram with marker
                st.subheader(f"Similarity Score Distribution for {job1_code} – {code_to_title.get(job1_code,'Unknown')}")
                st.write("Placeholder text: Explain how to interpret the vertical red line for this comparison in the histogram.")
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
                    alt.Chart(pd.DataFrame({"score": [score]}))
                    .mark_rule(color="red", strokeWidth=2)
                    .encode(x="score:Q")
                )
                st.altair_chart(hist_chart + line, use_container_width=True)
            else:
                st.error("❌ Could not compare occupations.")
