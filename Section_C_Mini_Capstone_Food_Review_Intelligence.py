"""
================================================================================
Assessment Code: M11-A1 | Module 11: Natural Language Processing (NLP)
Section C: Mini Capstone Project
Food Delivery Review Intelligence System
================================================================================
Objective:
Build a console-based NLP application for a food delivery platform that lets
staff add customer reviews, classifies each review by sentiment and complaint
category, and generates a summary report — combining text preprocessing,
TF-IDF vectorisation, and Naive Bayes or Logistic Regression classification
into a single working program.

Project Requirements:
1. Menu-driven with at least 3 options:
   (1) Add a new review
   (2) Classify a review and display predicted sentiment & issue category
   (3) View a summary report
2. Every review must pass through the full preprocessing pipeline (lowercase,
   punctuation removal, stopword removal, stemming or lemmatization) before
   being stored or classified.
3. Classifier outputs both:
   - Sentiment label (Positive / Negative)
   - Issue category (Delivery / Food Quality / App / General)
4. Summary report displays:
   - Total reviews added
   - Count per sentiment label
   - Count per issue category
   - 5 most frequent content words across all stored reviews
5. Graceful edge-case handling:
   - If fewer than the minimum required training samples exist, display
     a clear message instead of crashing when classification is attempted.
================================================================================
"""

import sys
import collections
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Import the reusable text preprocessor from Task 1
from Section_B_Task1_Food_Review_Preprocessor import preprocess_text

# Seed historical training data to ensure the models are primed and functional
SEED_TRAINING_REVIEWS = [
    # Delivery Issues (Negative)
    {"text": "The delivery driver was two hours late and food was completely cold.", "sentiment": "Negative", "category": "Delivery"},
    {"text": "Courier left food outside in the rain and never rang doorbell.", "sentiment": "Negative", "category": "Delivery"},
    {"text": "Driver was rude, disrespectful, hostile, and lost on the map for 45 minutes.", "sentiment": "Negative", "category": "Delivery"},
    {"text": "Delivery boy spilled the drinks and dropped my pizza box upside down.", "sentiment": "Negative", "category": "Delivery"},
    {"text": "Disrespectful courier threw the order on the floor and walked away.", "sentiment": "Negative", "category": "Delivery"},

    # Food Quality Issues (Negative)
    {"text": "The chicken burger was burnt, stale, and smelled rotten.", "sentiment": "Negative", "category": "Food Quality"},
    {"text": "Soup was completely cold and cheese had turned into rubber.", "sentiment": "Negative", "category": "Food Quality"},
    {"text": "Found a long hair in pasta sauce, totally unhygienic and disgusting.", "sentiment": "Negative", "category": "Food Quality"},
    {"text": "Fries were drenched in old grease and completely soggy and unsalted.", "sentiment": "Negative", "category": "Food Quality"},

    # App Issues (Negative)
    {"text": "The mobile app keeps crashing during payment checkout.", "sentiment": "Negative", "category": "App"},
    {"text": "Coupon discount code was not applied and card was charged twice.", "sentiment": "Negative", "category": "App"},
    {"text": "Order tracking map is completely frozen on screen in the application.", "sentiment": "Negative", "category": "App"},
    {"text": "App logged me out repeatedly and failed to process credit card payment.", "sentiment": "Negative", "category": "App"},

    # Positive Reviews (10)
    {"text": "The spicy chicken wings were extremely delicious, hot, and crispy!", "sentiment": "Positive", "category": "Food Quality"},
    {"text": "Super fast delivery within 18 minutes by a very polite courier driver.", "sentiment": "Positive", "category": "Delivery"},
    {"text": "The mobile app is wonderful, smooth, fast, and very easy to use.", "sentiment": "Positive", "category": "App"},
    {"text": "Delicious woodfired pizza with fresh basil and melted mozzarella cheese.", "sentiment": "Positive", "category": "Food Quality"},
    {"text": "Great customer experience and food arrived piping hot on time.", "sentiment": "Positive", "category": "General"},
    {"text": "Loved the chocolate dessert and fresh mango smoothie. Highly recommended!", "sentiment": "Positive", "category": "General"},
    {"text": "Driver was very polite, friendly, and delivered ahead of schedule.", "sentiment": "Positive", "category": "Delivery"},
    {"text": "Outstanding overall food quality, lovely packaging, and wonderful taste.", "sentiment": "Positive", "category": "Food Quality"},
    {"text": "App interface is fantastic, discounts work great, and checkout is super fast.", "sentiment": "Positive", "category": "App"},
    {"text": "Perfect dining experience, everything was fresh, hot, and delicious.", "sentiment": "Positive", "category": "General"},
]

MIN_SAMPLES_PER_CATEGORY = 2
MIN_TOTAL_SAMPLES = 6


class FoodReviewIntelligenceSystem:
    def __init__(self, seed_data: bool = True):
        self.stored_reviews: List[Dict[str, str]] = []
        self.sentiment_model: Optional[Pipeline] = None
        self.category_model: Optional[Pipeline] = None
        self.is_trained: bool = False

        if seed_data:
            for item in SEED_TRAINING_REVIEWS:
                self.add_review(
                    raw_text=item["text"],
                    sentiment=item["sentiment"],
                    category=item["category"],
                    retrain=False
                )
            self.train_classifiers()

    def add_review(self, raw_text: str, sentiment: Optional[str] = None, category: Optional[str] = None, retrain: bool = True) -> Dict[str, str]:
        """
        Preprocesses and stores a review. If labels are omitted, they will be predicted if models are trained.
        """
        # Requirement 2: Every review passes through full preprocessing pipeline before storing
        cleaned = preprocess_text(raw_text, method='stemming')

        # If sentiment or category are not provided, auto-classify if models are ready
        if (sentiment is None or category is None) and self.is_trained:
            pred_sent, pred_cat = self.classify_review(raw_text)
            sentiment = sentiment or pred_sent
            category = category or pred_cat
        else:
            sentiment = sentiment or "Unassigned"
            category = category or "General"

        record = {
            "id": len(self.stored_reviews) + 1,
            "raw_text": raw_text.strip(),
            "cleaned_text": cleaned,
            "sentiment": sentiment,
            "category": category
        }
        self.stored_reviews.append(record)

        if retrain and len(self.stored_reviews) >= MIN_TOTAL_SAMPLES:
            self.train_classifiers()

        return record

    def train_classifiers(self) -> Tuple[bool, str]:
        """
        Trains sentiment and issue category classifiers with graceful edge-case handling.
        """
        # Requirement 5: Edge-case validation
        valid_records = [r for r in self.stored_reviews if r["sentiment"] in ["Positive", "Negative"] and r["category"] in ["Delivery", "Food Quality", "App", "General"]]

        if len(valid_records) < MIN_TOTAL_SAMPLES:
            self.is_trained = False
            return False, f"Insufficient training data: Need at least {MIN_TOTAL_SAMPLES} labeled reviews. Currently have {len(valid_records)}."

        sentiments = [r["sentiment"] for r in valid_records]
        categories = [r["category"] for r in valid_records]

        sent_counts = collections.Counter(sentiments)
        cat_counts = collections.Counter(categories)

        if len(sent_counts) < 2 or any(cnt < MIN_SAMPLES_PER_CATEGORY for cnt in sent_counts.values()):
            self.is_trained = False
            return False, f"Insufficient class balance for sentiment: Each sentiment must have at least {MIN_SAMPLES_PER_CATEGORY} samples."

        if len(cat_counts) < 2:
            self.is_trained = False
            return False, "Insufficient categories: Need at least 2 distinct issue categories to train."

        # Build sentiment pipeline: TF-IDF + Logistic Regression (class_weight balanced)
        self.sentiment_model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', LogisticRegression(class_weight='balanced', C=2.0, random_state=42, max_iter=300))
        ])

        # Build category pipeline: TF-IDF + Multinomial Naive Bayes
        self.category_model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', MultinomialNB(alpha=0.3))
        ])

        corpus = [r["cleaned_text"] for r in valid_records]

        try:
            self.sentiment_model.fit(corpus, sentiments)
            self.category_model.fit(corpus, categories)
            self.is_trained = True
            return True, f"Successfully trained on {len(valid_records)} reviews."
        except Exception as e:
            self.is_trained = False
            return False, f"Model training failed due to: {str(e)}"

    def classify_review(self, raw_text: str) -> Tuple[str, str]:
        """
        Predicts both sentiment label and issue category for a raw review string.
        Requirement 5: Gracefully handles edge case where models are not trained.
        """
        if not self.is_trained or self.sentiment_model is None or self.category_model is None:
            return "Unable to Classify (Models Not Trained: Add at least 6 labeled samples)", "Unclassified"

        cleaned = preprocess_text(raw_text, method='stemming')
        if not cleaned:
            return "Neutral/Uncertain", "General"

        pred_sentiment = self.sentiment_model.predict([cleaned])[0]
        pred_category = self.category_model.predict([cleaned])[0]

        return pred_sentiment, pred_category

    def generate_summary_report(self) -> Dict:
        """
        Computes summary statistics required by Section C:
        - Total reviews added
        - Count per sentiment label
        - Count per issue category
        - 5 most frequent content words across all stored reviews
        """
        total_reviews = len(self.stored_reviews)
        sentiment_counts = collections.Counter(r["sentiment"] for r in self.stored_reviews)
        category_counts = collections.Counter(r["category"] for r in self.stored_reviews)

        # Extract words from cleaned_text of all stored reviews
        all_words = []
        for r in self.stored_reviews:
            tokens = r["cleaned_text"].split()
            all_words.extend([w for w in tokens if len(w) > 2])  # ignore single/double characters

        top_5_words = collections.Counter(all_words).most_common(5)

        return {
            "total_reviews": total_reviews,
            "sentiment_counts": dict(sentiment_counts),
            "category_counts": dict(category_counts),
            "top_5_words": top_5_words,
            "is_trained": self.is_trained
        }

    def print_summary_report(self):
        report = self.generate_summary_report()
        print("\n" + "=" * 70)
        print(" " * 18 + "SYSTEM SUMMARY REPORT")
        print("=" * 70)
        print(f"Total Customer Reviews Stored : {report['total_reviews']}")
        print(f"Model Training Status          : {'ACTIVE (Ready)' if report['is_trained'] else 'INACTIVE (Need more data)'}")

        print("\n[Sentiment Distribution]")
        for sent, count in report['sentiment_counts'].items():
            pct = (count / report['total_reviews'] * 100) if report['total_reviews'] else 0
            print(f"  - {sent:<15}: {count} ({pct:.1f}%)")

        print("\n[Issue Category Distribution]")
        for cat, count in report['category_counts'].items():
            pct = (count / report['total_reviews'] * 100) if report['total_reviews'] else 0
            print(f"  - {cat:<15}: {count} ({pct:.1f}%)")

        print("\n[Top 5 Most Frequent Content Words Across All Reviews]")
        for rank, (word, freq) in enumerate(report['top_5_words'], 1):
            print(f"  {rank}. '{word}' -> {freq} occurrences")
        print("=" * 70)


def run_interactive_menu():
    system = FoodReviewIntelligenceSystem(seed_data=True)

    while True:
        print("\n" + "=" * 60)
        print("   FOOD DELIVERY REVIEW INTELLIGENCE SYSTEM")
        print("   Module 11 - NLP Mini Capstone Project")
        print("=" * 60)
        print("1. Add a New Review (Store & Auto-Preprocess)")
        print("2. Classify a Review (Display Sentiment & Category)")
        print("3. View Summary Report")
        print("4. Inspect Preprocessed Data (Last 5 Reviews)")
        print("5. Exit Application")
        print("-" * 60)
        choice = input("Enter your option (1-5): ").strip()

        if choice == '1':
            print("\n--- OPTION 1: ADD NEW REVIEW ---")
            raw_text = input("Enter customer review text: ").strip()
            if not raw_text:
                print("Error: Review text cannot be empty.")
                continue

            opt_label = input("Do you want to provide manual ground truth labels? (y/n, default n): ").strip().lower()
            sent_label = None
            cat_label = None

            if opt_label == 'y':
                print("Select Sentiment: 1) Positive  2) Negative")
                s_choice = input("Choice (1/2): ").strip()
                sent_label = "Positive" if s_choice == '1' else "Negative"

                print("Select Category: 1) Delivery  2) Food Quality  3) App  4) General")
                c_choice = input("Choice (1-4): ").strip()
                cat_map = {'1': "Delivery", '2': "Food Quality", '3': "App", '4': "General"}
                cat_label = cat_map.get(c_choice, "General")

            record = system.add_review(raw_text, sentiment=sent_label, category=cat_label)
            print(f"\n[Success] Review added (ID: {record['id']})!")
            print(f"Cleaned Text : \"{record['cleaned_text']}\"")
            print(f"Sentiment    : {record['sentiment']}")
            print(f"Category     : {record['category']}")

        elif choice == '2':
            print("\n--- OPTION 2: CLASSIFY A REVIEW ---")
            review_to_test = input("Enter the review text to analyze: ").strip()
            if not review_to_test:
                print("Error: Review text cannot be empty.")
                continue

            cleaned = preprocess_text(review_to_test, method='stemming')
            sentiment, category = system.classify_review(review_to_test)

            print("\n" + "-" * 50)
            print("CLASSIFICATION RESULTS:")
            print("-" * 50)
            print(f"Raw Input        : \"{review_to_test}\"")
            print(f"Preprocessed Text: \"{cleaned}\"")
            print(f"Predicted Sentiment : {sentiment}")
            print(f"Predicted Category  : {category}")
            print("-" * 50)

        elif choice == '3':
            system.print_summary_report()

        elif choice == '4':
            print("\n--- STORED REVIEWS (LAST 5 ENTRIES) ---")
            last_entries = system.stored_reviews[-5:]
            for r in last_entries:
                print(f"ID #{r['id']} | [{r['sentiment']}] [{r['category']}]")
                print(f"  Raw     : {r['raw_text']}")
                print(f"  Cleaned : {r['cleaned_text']}\n")

        elif choice == '5':
            print("\nExiting Food Delivery Review Intelligence System. Goodbye!")
            break
        else:
            print("Invalid choice. Please select an option between 1 and 5.")


def run_demonstration():
    """
    Automated demonstration verifying all Section C project requirements
    """
    print("=" * 80)
    print("DEMONSTRATION RUN: FOOD DELIVERY REVIEW INTELLIGENCE SYSTEM")
    print("=" * 80)

    # 1. Initialize system
    system = FoodReviewIntelligenceSystem(seed_data=True)

    # 2. Test Edge Case Handling with an empty system
    print("\n[TEST 1] EDGE-CASE TEST: System with zero/insufficient training samples")
    empty_system = FoodReviewIntelligenceSystem(seed_data=False)
    pred_s, pred_c = empty_system.classify_review("The burger was totally ice cold.")
    print(f"Classification on untrained model gracefully returns:")
    print(f"  -> Sentiment: {pred_s}")
    print(f"  -> Category : {pred_c}")

    # 3. Add new reviews through preprocessing pipeline
    print("\n[TEST 2] ADDING NEW CUSTOMER REVIEWS (Preprocessing & Ingestion)")
    test_inputs = [
        "The pizza arrived 50 minutes late and was lukewarm.",
        "Loved the warm chocolate lava cake! Super rich and delicious.",
        "The checkout payment button gave a connection timeout in the mobile app."
    ]

    for text in test_inputs:
        rec = system.add_review(text)
        print(f"Added ID #{rec['id']}: \"{text}\"")
        print(f"  Preprocessed: \"{rec['cleaned_text']}\"")
        print(f"  Classified  : Sentiment=[{rec['sentiment']}], Category=[{rec['category']}]\n")

    # 4. Interactive Classification Check
    print("[TEST 3] CLASSIFYING ARBITRARY RUNTIME REVIEWS")
    runtime_queries = [
        "The driver was extremely disrespectful and threw the food on the porch.",
        "The pasta was fresh and perfectly cooked, compliments to the chef!",
        "Cannot enter my credit card security code on the payment screen."
    ]

    for q in runtime_queries:
        s, c = system.classify_review(q)
        print(f"Query: \"{q}\"")
        print(f"  -> Predicted Sentiment: {s}")
        print(f"  -> Predicted Category : {c}\n")

    # 5. Summary Report Generation
    print("[TEST 4] GENERATING SYSTEM SUMMARY REPORT")
    system.print_summary_report()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demonstration()
    else:
        # Check if stdin is interactive or piped
        if sys.stdin.isatty():
            run_interactive_menu()
        else:
            run_demonstration()
