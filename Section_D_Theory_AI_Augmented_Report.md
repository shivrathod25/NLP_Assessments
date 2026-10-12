# Module 11: Natural Language Processing (NLP)
## Assessment Code: M11-A1 — Data Science
### Section D: AI-Augmented Learning (Report & Technical Audit)

---

### Step 1: Prompt Given to the AI Tool

#### Exact Prompt Submitted to AI Tool:
```text
Write a complete Python 3 program using NLTK and Scikit-Learn that does the following:
1. Accepts a list of food delivery customer reviews with binary sentiment labels (Positive and Negative).
2. Preprocesses each review using lowercase conversion, tokenisation, English stopword removal using NLTK, and stemming using PorterStemmer.
3. Converts the text to a TF-IDF matrix using sklearn's TfidfVectorizer and trains a Logistic Regression model to classify reviews as Positive or Negative.
4. Splits the dataset into train and test sets, prints the model's test accuracy, and predicts the sentiment for three new review strings entered at runtime.
```

---

### Step 2: Critical Bugs & Limitations Identified in the AI's Original Code

During rigorous manual code review and testing without AI, four significant bugs and architectural flaws were discovered in the AI's solution:

1. **Critical Data Leakage (Vectorizer Fit Before Train/Test Split):**
   * *Flaw:* The AI's code called `TfidfVectorizer.fit_transform(all_reviews)` on the entire dataset *before* calling `train_test_split()`.
   * *Impact:* Computing Inverse Document Frequency (IDF) statistics and constructing the vocabulary on the entire corpus allows information from the held-out test set to leak into the training process. In real-world ML engineering, this produces artificially inflated, overly optimistic test evaluation scores and fails to simulate true out-of-sample generalization.
2. **Destructive Blind Stopword Removal (Negation Erasure):**
   * *Flaw:* The AI stripped all words in `stopwords.words('english')` without an exclusion list for negation particles.
   * *Impact:* Critical sentiment shifters such as `"not"`, `"no"`, `"never"`, `"neither"`, and `"without"` were stripped. Consequently, a scathing review like *"Food was not good and never arrived"* preprocessed into `"food good arriv"`, reversing the true sentiment from negative to positive.
3. **Absence of Stratified Splitting on Small Datasets:**
   * *Flaw:* The AI called `train_test_split(X, y, test_size=0.25)` without setting `stratify=y`.
   * *Impact:* On modest-sized training sets, unstratified splitting risks severe distribution shifts between train and test splits, occasionally yielding zero positive or negative instances in the test fold.
4. **Lack of Pipeline Encapsulation & Runtime Preprocessing Vulnerability:**
   * *Flaw:* The AI manually transformed training data and then separately attempted to preprocess and transform runtime user inputs without an `sklearn.pipeline.Pipeline` wrapper.
   * *Impact:* If a user entered punctuation-heavy, empty, or novel vocabulary at runtime, the unencapsulated model lacked robust error handling and preprocessing parity.

---

### Step 3: Required 3–4 Line Explanatory Submission Note

> **Note on Corrections:**
> I eliminated data leakage by preventing vectorizer fitting on the full dataset, ensuring TF-IDF fits strictly on training data within an atomic `sklearn.pipeline.Pipeline`. I also modified the stopword cleaner to preserve sentiment-critical negation tokens (`not`, `no`, `never`, `without`), and added stratified train-test splitting to guarantee class distribution balance. These changes prevent test set leakage, stop false sentiment reversals on negated complaints, and ensure reliable runtime inference on new customer reviews.

---

### Code Artifacts Created for Practical Evaluation:
- **AI Original Code:** [`Section_D_Step1_AI_Original_Code.py`](file:///e:/TOPS%20Classes/NLP/NLP%20Assessments/Section_D_Step1_AI_Original_Code.py)
- **Corrected & Debugged Production Code:** [`Section_D_Step2_Corrected_Code.py`](file:///e:/TOPS%20Classes/NLP/NLP%20Assessments/Section_D_Step2_Corrected_Code.py)

---
*Assessment Document: M11-A1 — Module 11 (Natural Language Processing)*
