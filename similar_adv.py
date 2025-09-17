import pandas as pd

# ---------- Load Data ----------

similarity_df = pd.read_excel("similarity matrix.xlsx", index_col=0)

# force index to string, strip whitespace
similarity_df.index = similarity_df.index.astype(str).str.strip()
similarity_df.columns = similarity_df.columns.astype(str).str.strip()

titles_df = pd.read_excel("noc title.xlsx")
titles_df.columns = titles_df.columns.str.strip().str.lower()

# make sure codes are strings
titles_df["noc"] = titles_df["noc"].astype(str).str.strip()

code_to_title = dict(zip(titles_df["noc"], titles_df["title"]))
title_to_code = {v.lower(): k for k, v in code_to_title.items()}

# ---------- Helper Functions ----------
def find_code_from_title(title_input):
    """Find occupation codes by title (case-insensitive, partial matches allowed)."""
    title_input = title_input.lower()
    matches = [code for code, title in code_to_title.items() if title_input in title.lower()]
    return matches

def get_top_similar(code, n=5):
    """Return the n most similar occupations (smallest scores)."""
    if code not in similarity_df.index:
        return None
    
    scores = similarity_df.loc[code].drop(code)  # exclude self
    top_matches = scores.nsmallest(n)
    results = []
    for occ, score in top_matches.items():
        title = code_to_title.get(occ, "Unknown Title")
        results.append((occ, title, score))
    return results

def compare_two_jobs(code1, code2):
    """Return similarity score and ranking position of code2 in code1’s similarity list."""
    if code1 not in similarity_df.index or code2 not in similarity_df.index:
        return None
    
    scores = similarity_df.loc[code1].drop(code1).sort_values()
    rank = scores.index.get_loc(code2) + 1  # 1-based rank
    total = len(scores)
    score = similarity_df.loc[code1, code2]
    
    return score, rank, total

def pretty_print_results(results):
    print("\nTop Similar Occupations:")
    print(f"{'Rank':<5} {'Code':<10} {'Title':<40} {'Score':<10}")
    print("-" * 70)
    for i, (occ, title, score) in enumerate(results, start=1):
        print(f"{i:<5} {occ:<10} {title:<40} {score:<10.4f}")
    print()

# ---------- Interactive Menu ----------
def main():
    while True:
        print("\n--- Occupation Similarity App ---")
        print("1. Look up by occupation code")
        print("2. Look up by occupation title")
        print("3. Compare two occupations")
        print("4. Exit")
        
        choice = input("Choose an option (1-4): ").strip()
        
        if choice == "1":
            code = input("Enter 5-digit occupation code: ").strip()
            if code in similarity_df.index:
                results = get_top_similar(code)
                pretty_print_results(results)
            else:
                print("❌ Invalid occupation code.")
        
        elif choice == "2":
            title_input = input("Enter occupation title (or part of it): ").strip()
            matches = find_code_from_title(title_input)
            if not matches:
                print("❌ No matches found.")
                continue
            elif len(matches) > 1:
                print("Multiple matches found:")
                for i, code in enumerate(matches, start=1):
                    print(f"{i}. {code} - {code_to_title[code]}")
                choice = int(input("Select the correct one: "))
                code = matches[choice - 1]
            else:
                code = matches[0]
            
            results = get_top_similar(code)
            pretty_print_results(results)
        
        elif choice == "3":
            job1 = input("Enter first occupation code or title: ").strip()
            job2 = input("Enter second occupation code or title: ").strip()
            
            # Resolve codes if titles given
            if not job1.isdigit():
                matches = find_code_from_title(job1)
                if matches:
                    job1 = matches[0]
                else:
                    print("❌ First occupation not found.")
                    continue
            if not job2.isdigit():
                matches = find_code_from_title(job2)
                if matches:
                    job2 = matches[0]
                else:
                    print("❌ Second occupation not found.")
                    continue
            
            result = compare_two_jobs(job1, job2)
            if result:
                score, rank, total = result
                print(f"\nComparison Result:")
                print(f"{job1} ({code_to_title.get(job1,'Unknown')}) vs {job2} ({code_to_title.get(job2,'Unknown')})")
                print(f"Similarity score: {score:.4f}")
                print(f"Ranking: {rank} out of {total} occupations (#{rank} most similar)\n")
            else:
                print("❌ Could not compare occupations.")
        
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("❌ Invalid option.")

# Run the program
if __name__ == "__main__":
    main()
