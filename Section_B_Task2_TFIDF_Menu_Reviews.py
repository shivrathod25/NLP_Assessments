"""
================================================================================
Assessment Code: M11-A1 | Module 11: Natural Language Processing (NLP)
Section B: Practical Coding Tasks
Task 2: TF-IDF Vectoriser for Menu Reviews
================================================================================
Requirements:
- Create a list of at least 8 short food delivery review strings as your corpus
  (mix of positive, negative, and neutral tones).
- Use sklearn's TfidfVectorizer to transform the corpus into a TF-IDF matrix.
- Display the full TF-IDF matrix as a labelled pandas DataFrame with feature names
  as column headers and review index as row labels.
- For each review, print the top 3 words with the highest TF-IDF score and their scores.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Corpus of at least 8 food delivery menu reviews (10 provided: positive, negative, neutral)
corpus = [
    # Positive Reviews
    "The spicy chicken burger was exceptionally crispy, flavorful, and hot.",           # Review 1
    "Delicious cheese pizza with a wonderfully fresh crust and generous toppings.",    # Review 2
    "Super fast delivery and the sushi rolls were remarkably fresh and authentic.",     # Review 3
    "Loved the chocolate lava cake; rich molten center and perfectly sweet.",          # Review 4

    # Negative Reviews
    "The chicken curry was completely tasteless, excessively oily, and cold.",          # Review 5
    "Terrible soggy french fries and the burger bun was hard as a rock.",              # Review 6
    "Late delivery by one hour and the garlic bread was burnt to a crisp.",             # Review 7

    # Neutral Reviews
    "The chicken sandwich was average, neither particularly bad nor impressive.",       # Review 8
    "Portion size was decent for the price, but the pasta sauce lacked seasoning.",     # Review 9
    "Standard milk tea; tasted ordinary and arrived at expected delivery time."         # Review 10
]

def main():
    print("=" * 85)
    print("MODULE 11: NLP ASSESSMENT | TASK 2: TF-IDF VECTORISER FOR MENU REVIEWS")
    print("=" * 85)

    # 2. Initialize TfidfVectorizer with standard English stop words removal
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the review corpus into a TF-IDF sparse matrix
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # 3. Display full TF-IDF matrix as a labelled pandas DataFrame
    # Row indices: Review 1, Review 2, ... Column headers: Feature names
    row_labels = [f"Review_{i+1}" for i in range(len(corpus))]
    df_tfidf = pd.DataFrame(
        data=tfidf_matrix.toarray(),
        index=row_labels,
        columns=feature_names
    )

    print(f"\nTotal Corpus Documents: {len(corpus)}")
    print(f"Total Unique Vocabulary Features: {len(feature_names)}")
    print("\n--- SAMPLE VIEW OF TF-IDF MATRIX (First 5 Rows & Selected Features) ---")
    pd.set_option('display.max_columns', 8)
    pd.set_option('display.width', 1000)
    print(df_tfidf.iloc[:5, :8])

    print("\n--- FULL SHAPE OF TF-IDF MATRIX ---")
    print(f"Matrix Dimensions: {df_tfidf.shape[0]} rows x {df_tfidf.shape[1]} columns")

    # 4. For each review, print the top 3 words with the highest TF-IDF score
    print("\n" + "=" * 85)
    print("TOP 3 MOST INFORMATIVE TERMS PER REVIEW (HIGHEST TF-IDF WEIGHTS)")
    print("=" * 85)

    for idx, (label, text) in enumerate(zip(row_labels, corpus)):
        row_vector = df_tfidf.iloc[idx]
        # Filter terms that appear in this specific document (score > 0) and sort descending
        top_terms = row_vector[row_vector > 0].sort_values(ascending=False).head(3)

        print(f"\n[{label}] \"{text}\"")
        for rank, (word, score) in enumerate(top_terms.items(), 1):
            print(f"   Rank {rank}: '{word}' -> TF-IDF Score: {score:.4f}")

    print("\n" + "=" * 85)
    print("TASK 2 COMPLETED SUCCESSFULLY: TF-IDF representations computed & inspected.")
    print("=" * 85)


if __name__ == "__main__":
    main()
