"""
================================================================================
Assessment Code: M11-A1 | Module 11: Natural Language Processing (NLP)
Section B: Practical Coding Tasks
Task 1: Food Review Text Preprocessor
================================================================================
Requirements:
- Accept a raw review string as input and return a cleaned, processed string.
- Convert text to lowercase, remove punctuation, and strip extra whitespace.
- Tokenise the cleaned text and remove English stopwords using NLTK's stopwords corpus.
- Apply stemming using PorterStemmer (or lemmatization using WordNetLemmatizer)
  to the token list and return the result as a single space-joined string.
- Test the function on at least three different food delivery review samples.
================================================================================
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure required NLTK resources are available locally
for resource in ['stopwords', 'punkt', 'punkt_tab', 'wordnet']:
    try:
        nltk.data.find(f"corpora/{resource}" if resource in ['stopwords', 'wordnet'] else f"tokenizers/{resource}")
    except (LookupError, IndexError):
        nltk.download(resource, quiet=True)

# Initialize NLP components
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text: str, method: str = 'stemming') -> str:
    """
    Cleans and preprocesses a raw food delivery review string.

    Parameters:
    -----------
    text : str
        The raw input text from customer review.
    method : str
        Normalization technique: 'stemming' (default) or 'lemmatization'.

    Returns:
    --------
    str:
        Cleaned, tokenized, stopword-filtered, and normalized space-joined string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Convert text to lowercase
    cleaned = text.lower()

    # 2. Remove punctuation and special characters using regex and string.punctuation
    # Replace punctuation characters with spaces to avoid joining hyphenated words
    punctuation_pattern = f"[{re.escape(string.punctuation)}]"
    cleaned = re.sub(punctuation_pattern, " ", cleaned)

    # 3. Strip extra whitespace (multiple spaces, tabs, newlines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 4. Tokenise the cleaned text into individual word tokens
    tokens = word_tokenize(cleaned)

    # 5. Remove English stopwords and non-alphanumeric tokens
    filtered_tokens = [
        token for token in tokens
        if token not in stop_words and token.isalnum()
    ]

    # 6. Apply Stemming (PorterStemmer) or Lemmatization (WordNetLemmatizer)
    if method == 'lemmatization':
        normalized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    else:
        # Default PorterStemmer as specified in requirements
        normalized_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # 7. Return the result as a single space-joined string
    return " ".join(normalized_tokens)


def main():
    print("=" * 80)
    print("MODULE 11: NLP ASSESSMENT | TASK 1: FOOD REVIEW TEXT PREPROCESSOR")
    print("=" * 80)

    # Test with at least 3 distinct food delivery review samples
    test_reviews = [
        (
            "Sample 1 (Positive Review)",
            "The hot garlic noodles and cheese pizza arrived SUPER FAST in 15 mins! "
            "Everything was piping hot, freshly cooked, and incredibly delicious! 5/5 stars!!!"
        ),
        (
            "Sample 2 (Negative Review - Cold & Late Delivery)",
            "TERRIBLE SERVICE!! The driver was 45 minutes late, the order was completely wrong, "
            "and the soup was spilled all over the soggy box. I demand a full refund immediately!"
        ),
        (
            "Sample 3 (App & Packaging Issue)",
            "The mobile app kept crashing at checkout while applying my discount coupon. "
            "When the food finally arrived, it was lukewarm and missing the extra garlic dip."
        )
    ]

    for title, review in test_reviews:
        stemmed_output = preprocess_text(review, method='stemming')
        lemmatized_output = preprocess_text(review, method='lemmatization')

        print(f"\n--- {title} ---")
        print(f"RAW INPUT   :\n  \"{review}\"")
        print(f"PROCESSED (Stemmed)    :\n  \"{stemmed_output}\"")
        print(f"PROCESSED (Lemmatized) :\n  \"{lemmatized_output}\"")

    print("\n" + "=" * 80)
    print("TASK 1 COMPLETED SUCCESSFULLY: Preprocessor is reusable and ready for downstream NLP.")
    print("=" * 80)


if __name__ == "__main__":
    main()
