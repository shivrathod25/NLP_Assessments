"""
================================================================================
Assessment Code: M11-A1 | Module 11: Natural Language Processing (NLP)
Section B: Practical Coding Tasks
Task 3: Complaint Category Classifier – Naive Bayes
================================================================================
Requirements:
- Create a labelled dataset of at least 24 complaint strings across three classes:
  'Delivery', 'Food Quality', and 'App'.
- Preprocess each complaint using the function from Task 1, then convert the
  corpus to a TF-IDF feature matrix.
- Split into 80% train and 20% test sets, train a MultinomialNB classifier, and
  generate predictions on the test set.
- Print the full classification report (precision, recall, F1-score per class)
  and the confusion matrix.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Import reusable text preprocessor from Task 1
from Section_B_Task1_Food_Review_Preprocessor import preprocess_text

# 1. Labelled dataset: 30 complaints (10 Delivery, 10 Food Quality, 10 App)
raw_complaints = [
    # --- Category: Delivery ---
    ("The delivery driver was over an hour late and my order was left at the wrong door.", "Delivery"),
    ("Driver drove in circles on the map and never showed up with my dinner.", "Delivery"),
    ("My delivery person was extremely rude and refused to bring food to my floor.", "Delivery"),
    ("The food arrived cold because the delivery rider took multiple orders at once.", "Delivery"),
    ("Order status showed delivered on tracking, but no delivery boy arrived.", "Delivery"),
    ("Courier dropped my food package upside down and the drinks spilled in the bag.", "Delivery"),
    ("Waited for two hours for courier arrival, driver never answered my call.", "Delivery"),
    ("Driver could not locate my address despite clear street directions.", "Delivery"),
    ("Late delivery rider forgot my soda bottles in his vehicle trunk.", "Delivery"),
    ("Delivery partner marked order delivered 20 minutes before actually arriving.", "Delivery"),

    # --- Category: Food Quality ---
    ("The chicken burger was completely burnt and tasted like bitter charcoal.", "Food Quality"),
    ("The soup arrived lukewarm and the noodles were soggy and stale.", "Food Quality"),
    ("Food was spoiled, smelling foul, and caused severe stomach infection.", "Food Quality"),
    ("The cheese pizza had almost no sauce and the dough was raw and rubbery.", "Food Quality"),
    ("Found a piece of hair inside the pasta bowl, absolutely unhygienic.", "Food Quality"),
    ("The sushi was warm, fish was clearly not fresh, and smelled rotten.", "Food Quality"),
    ("French fries were dripping with old oil and completely limp.", "Food Quality"),
    ("The sandwich tasted sour and meat appeared discolored and undercooked.", "Food Quality"),
    ("Biryani was extremely salty, inedible, and chicken pieces were dried out.", "Food Quality"),
    ("Tacos were soggy because grease soaked through the paper carton.", "Food Quality"),

    # --- Category: App ---
    ("The app crashed immediately when I clicked apply promo coupon code.", "App"),
    ("Payment deducted twice from credit card but the app says order failed.", "App"),
    ("Cannot track my active delivery because the live GPS map view is frozen.", "App"),
    ("The app keeps logging me out and will not save my delivery address.", "App"),
    ("Unable to add items to cart, button is unresponsive on the checkout page.", "App"),
    ("App freezes on startup and crashes every time I open the menu screen.", "App"),
    ("Wallet balance was not updated after adding funds inside the application.", "App"),
    ("Search filters are broken and app shows restaurants that are closed.", "App"),
    ("Could not cancel order due to continuous server error 500 in the application.", "App"),
    ("Push notifications fail and payment gateway timeout appears on screen.", "App"),
]


def main():
    print("=" * 85)
    print("MODULE 11: NLP ASSESSMENT | TASK 3: COMPLAINT CATEGORY CLASSIFIER (NAIVE BAYES)")
    print("=" * 85)

    df = pd.DataFrame(raw_complaints, columns=["complaint_text", "category"])
    print(f"\nDataset Size: {len(df)} samples")
    print("Class Distribution:\n", df["category"].value_counts().to_string())

    # 2. Preprocess complaints using Task 1 function
    print("\n--- Preprocessing Text with Task 1 Function ---")
    df["cleaned_text"] = df["complaint_text"].apply(lambda text: preprocess_text(text, method='stemming'))

    # Display sample cleaned rows
    print(df[["category", "cleaned_text"]].head(6).to_string())

    # 3. Train/Test Split (80% train, 20% test) with Stratification
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["category"],
        test_size=0.20,
        random_state=42,
        stratify=df["category"]
    )

    print(f"\nTraining set size: {len(X_train_raw)} samples")
    print(f"Test set size: {len(X_test_raw)} samples")

    # Convert corpus to TF-IDF feature matrix (Fitting on train, transforming test to prevent leakage)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print(f"Vocabulary features extracted: {X_train_tfidf.shape[1]}")

    # 4. Train MultinomialNB Classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)

    # Generate predictions on the test set
    y_pred = classifier.predict(X_test_tfidf)

    # 5. Print full classification report and confusion matrix
    classes = sorted(df["category"].unique())
    print("\n" + "=" * 85)
    print("CLASSIFICATION REPORT (Multinomial Naive Bayes)")
    print("=" * 85)
    print(classification_report(y_test, y_pred, target_names=classes, digits=4, zero_division=0))

    print("=" * 85)
    print("CONFUSION MATRIX")
    print("=" * 85)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print(cm_df)

    # 6. Test on Unseen Realistic Complaints
    print("\n" + "=" * 85)
    print("PREDICTION TEST ON NEW UNSEEN COMPLAINTS")
    print("=" * 85)
    test_queries = [
        "The courier rider called 40 mins late and was lost on highway.",
        "The pizza base was undercooked dough and smells terrible.",
        "Payment got charged but order status page shows app error."
    ]

    for query in test_queries:
        cleaned_query = preprocess_text(query, method='stemming')
        vec_query = vectorizer.transform([cleaned_query])
        pred_cat = classifier.predict(vec_query)[0]
        probs = classifier.predict_proba(vec_query)[0]
        max_prob = np.max(probs) * 100
        print(f"Complaint : \"{query}\"")
        print(f"Cleaned   : \"{cleaned_query}\"")
        print(f"Predicted : [{pred_cat}] (Confidence: {max_prob:.1f}%)\n")

    print("=" * 85)
    print("TASK 3 COMPLETED SUCCESSFULLY: MultinomialNB trained and evaluated.")
    print("=" * 85)


if __name__ == "__main__":
    main()
