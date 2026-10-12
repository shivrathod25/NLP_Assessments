# Module 11: Natural Language Processing (NLP)
## Assessment Code: M11-A1 — Data Science
### Section A: Concept Application (Theoretical Analysis & Real-World Scenarios)

---

### Scenario 1 (S1): Automated Customer Support Chatbot Routing
**Scenario:**
> You are building an automated customer support chatbot for a food delivery platform that handles queries like:
> - *"Where is my order?"*
> - *"I want to cancel my order"*
> - *"My food arrived cold and late"*

#### Question:
> Identify two NLP techniques the chatbot must use to correctly understand and route these customer messages. Explain why each technique is necessary and describe what would happen to chatbot accuracy if either technique were removed.

#### Answer:

#### 1. Identification of the Two NLP Techniques
To understand, process, and automatically fulfill customer requests, the conversational pipeline requires:
1. **Intent Classification (Intent Recognition):** Categorizing the semantic goal or purpose behind a customer's message into predefined business categories (e.g., `TRACK_ORDER`, `CANCEL_ORDER`, `COMPLAINT_FOOD_QUALITY`, `REFUND_REQUEST`).
2. **Named Entity Recognition (NER) / Slot Filling:** Identifying and extracting structured, variable domain entities from the unstructured text (e.g., `Order_ID`, `Item_Name`, `Time_Period`, `Delivery_Address`).

---

#### 2. Why Each Technique is Necessary
* **Why Intent Classification is Necessary:**
  Customer support queries exhibit high syntactic variety. Users express the same objective using vastly different phrasing (e.g., *"Where is my food?"*, *"Check delivery status"*, *"Track order #4928"* all share the intent `TRACK_ORDER`). Intent classification maps this diverse natural language input into a single deterministic business action, directing the conversation to the correct workflow, backend API, or human support team.
* **Why Named Entity Recognition (NER) is Necessary:**
  Recognizing the user's intent alone is insufficient to fulfill an action. If a user states *"I want to cancel order #98234 and remove the garlic bread"*, the system needs the exact attributes (`Order_ID: 98234`, `Item: garlic bread`). NER extracts these parameter slots directly from the text so backend microservices can query the database or execute the refund without manual form filling.

---

#### 3. Impact on Chatbot Accuracy if Either Technique Were Removed
* **If Intent Classification Were Removed:**
  - **Routing Failure:** The chatbot cannot categorize what the customer wants to accomplish.
  - **Fallback to Ineffective Keyword Matching:** The system would rely on naive keyword rules. For instance, *"Where is my order?"* and *"I want to cancel my order"* both contain the tokens *"my"* and *"order"*. A keyword matcher could confuse cancellation with tracking, leading to disastrous routing errors.
  - **Resulting Accuracy:** Overall conversation resolution rate and task routing accuracy would collapse significantly (often dropping below 30–40% on real-world queries).
* **If NER / Slot Filling Were Removed:**
  - **Information Extraction Bottleneck:** The chatbot might know the customer wants to track or cancel an order, but cannot parse parameters present directly in the query.
  - **Frustrating Multi-Turn Friction:** The bot would be forced into repetitive interrogative loops (e.g., *"Please provide your order number"* even when the customer already provided it).
  - **Resulting Accuracy:** Slot-filling accuracy drops to 0%, causing automated fulfillment rates to decline and customer drop-off rates to spike.

---

### Scenario 2 (S2): Text Preprocessing Pipeline for Large-Scale App Reviews
**Scenario:**
> You are processing 50,000 raw customer reviews scraped from a food delivery app. The reviews contain mixed letter cases, special characters, emojis, numbers, and common filler words such as 'the', 'is', and 'a'.

#### Question:
> Describe the text preprocessing pipeline you would apply to this dataset before feeding it to any NLP model. Justify the order in which you would apply each step, and explain specifically why skipping stopword removal could hurt downstream model performance.

#### Answer:

#### 1. Complete Preprocessing Pipeline Architecture & Order
To preprocess 50,000 raw reviews efficiently without destroying sentiment-bearing information, the pipeline should execute in the following sequential order:

```
[Raw Scraped Text]
       │
       ▼
1. HTML/URL & Noise Stripping
       │
       ▼
2. Emoji & Emoticon Conversion (e.g., 😡 -> 'angry_face')
       │
       ▼
3. Lowercasing (e.g., 'BURGER' -> 'burger')
       │
       ▼
4. Contraction Expansion (e.g., "didn't" -> "did not")
       │
       ▼
5. Punctuation & Special Character Removal (Selective)
       │
       ▼
6. Number Normalization / Removal
       │
       ▼
7. Tokenization
       │
       ▼
8. Context-Aware Stopword Removal (Retaining negations: 'not', 'no', 'never')
       │
       ▼
9. Stemming or Lemmatization (e.g., 'delivered', 'delivering' -> 'deliver')
       │
       ▼
[Cleaned Token Stream / Normalized Text]
```

---

#### 2. Justification of the Step Ordering
1. **Emoji Conversion Before Punctuation Removal:** Emojis (e.g., 🍕, 😋, 😡) convey strong sentiment in food delivery reviews. If punctuation/regex cleaning is executed first, emojis may be stripped or corrupted into garbled unicode byte characters. Converting emojis into semantic strings (e.g., `:angry:` or `emoji_angry`) preserves sentiment tokens.
2. **Contraction Expansion Before Punctuation Cleaning:** Words like *"didn't"*, *"couldn't"*, and *"wasn't"* contain apostrophes. If punctuation is stripped before expanding contractions, *"didn't"* becomes *"didnt"* or *"did n t"*, preventing standard stopword or negation parsers from recognizing the negation. Expanding to *"did not"* ensures the critical negation token *"not"* is preserved.
3. **Lowercasing Before Stopword Filtering & Lemmatization:** Normalizing to lowercase ensures that standard lexicons (NLTK stopwords, WordNet) match tokens regardless of sentence position (e.g., *"The"* vs *"the"*).
4. **Tokenization Before Lemmatization:** Lemmatizers and stemmers operate at the word level; the text must be split into individual linguistic tokens first.
5. **Lemmatization After Stopword Removal:** Filtering stopwords before lemmatization saves substantial CPU cycles, as the lemmatizer does not spend time looking up grammatical lemmas for meaningless filler words.

---

#### 3. Why Skipping Stopword Removal Could Hurt Downstream Model Performance
* **Explosion of Dimensionality & Sparsity:** Filler words like *"the"*, *"is"*, *"a"*, *"at"*, *"of"* occur in almost every review. Including them inflates the vocabulary size by thousands of redundant columns in Bag-of-Words (BoW) or TF-IDF matrices, increasing memory footprint and computational cost.
* **Dilution of Informative Signal:** Standard unweighted frequency counters (BoW, CountVectorizer) will assign the highest feature weights to stopwords simply because they appear frequently. This drowns out discriminative sentiment-bearing words like *"stale"*, *"soggy"*, *"late"*, *"delicious"*, or *"overpriced"*.
* **Overfitting on Spurious Correlations:** In linear classifiers (such as Logistic Regression or Naive Bayes), non-informative stopwords that coincidentally appear slightly more often in one class can receive non-zero weights, degrading model generalization on unseen test data.

---

### Scenario 3 (S3): Text Representation for Menu Item Search & Ranking
**Scenario:**
> You are building a food item search feature that must rank menu descriptions by relevance to a customer's typed query, for example:
> *'spicy chicken without onion'*

#### Question:
> Compare Bag of Words and TF-IDF as text representation methods for this ranking task. Which would you choose and why? Identify one limitation that your chosen method still has even after it is applied.

#### Answer:

#### 1. Comparison: Bag of Words (BoW) vs. TF-IDF

| Feature / Aspect | Bag of Words (BoW) | Term Frequency - Inverse Document Frequency (TF-IDF) |
| :--- | :--- | :--- |
| **Mathematical Formulation** | Raw term frequency count: $\text{Count}(t, d)$ | $\text{TF}(t, d) \times \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$ |
| **Weighting Mechanism** | All words are weighted purely by their occurrence frequency in the document. | Balances local frequency (TF) with corpus-wide rarity (IDF). Frequent menu words are discounted. |
| **Handling of Common Menu Terms** | Generic menu words like *"food"*, *"dish"*, *"served"*, *"fresh"* dominate representation vectors. | Penalizes ubiquitous words across all restaurant menus, assigning them an IDF near zero. |
| **Discriminative Ability** | Low; a menu with 5 mentions of *"served"* outscores a menu with 1 mention of *"spicy"*. | High; rare and distinctive terms like *"spicy"* or *"tandoori"* receive high weights. |
| **Relevance Ranking Performance** | Poor ranking relevance due to lack of document specificity penalization. | Significantly superior ranking relevance for information retrieval and cosine similarity search. |

---

#### 2. Method Selection and Justification
**Choice:** **TF-IDF (or BM25, an advanced TF-IDF extension)**.

**Justification:**
In a food search engine, menu descriptions frequently contain repetitive culinary terms (e.g., *"freshly prepared"*, *"served with"*, *"sauce"*, *"plate"*). Under BoW, these common terms dominate similarity scores. In contrast, TF-IDF discounts terms that appear across all menu items and boosts rare, highly descriptive query words like *"spicy"* and *"chicken"*. When computing cosine similarity between query vector $Q$ and menu vectors $D_i$, TF-IDF ensures items explicitly featuring the specific culinary attributes rank at the top.

---

#### 3. Inherent Limitation of TF-IDF for the Query: *'spicy chicken without onion'*
Even with TF-IDF, the method suffers from a severe fundamental limitation: **Lack of Negation and Semantic Context Awareness**.

* **Why it fails here:**
  TF-IDF treats documents and queries as an unordered collection ("bag") of independent tokens.
  - In the query *"spicy chicken without onion"*, the words *"without"* and *"onion"* are both vectorized as positive numerical weights.
  - If a menu item states: *"Spicy grilled chicken burger served with caramelized onion"*, TF-IDF will compute an **extremely high relevance score** because all four words (*"spicy"*, *"chicken"*, *"with/without"*, *"onion"*) match.
  - TF-IDF cannot comprehend that the preposition *"without"* negates the noun *"onion"*. As a result, customers allergic to onions will be served dishes packed with onions.
* **Additional Limitation:** TF-IDF cannot recognize semantic synonyms (e.g., matching *"poultry"* with *"chicken"* or *"fiery"* with *"spicy"*).

---

### Scenario 4 (S4): Probabilistic Assumptions in Naive Bayes Classification
**Scenario:**
> You are training a Naive Bayes classifier to automatically label incoming food delivery support tickets as:
> - `'Delivery Issue'`
> - `'Food Quality Issue'`
> - `'App Issue'`
> using a dataset of 10,000 historical labelled tickets.

#### Question:
> Explain the core probabilistic assumption that Naive Bayes makes about its input features. Describe a realistic food delivery example where this assumption is violated, and explain how that violation would affect the model's classification accuracy.

#### Answer:

#### 1. The Core Probabilistic Assumption of Naive Bayes
The core assumption is the **Conditional Independence Assumption**. 

Given a class label $C \in \{\text{Delivery}, \text{Food Quality}, \text{App}\}$, Naive Bayes assumes that all feature words $w_1, w_2, \dots, w_n$ are **conditionally independent** of each other given the class:

$$P(w_1, w_2, \dots, w_n \mid C) = \prod_{i=1}^n P(w_i \mid C)$$

In plain terms, it assumes that the presence or absence of one word in a ticket provides zero information about the likelihood of another word appearing, conditioned on the ticket category.

---

#### 2. Realistic Food Delivery Violation Example
In human language and food delivery complaints, words exhibit strong syntactic and semantic co-dependencies.

* **Example 1: Strong Bigram Collocation ("not delivered")**
  Consider the ticket: *"My order was not delivered."*
  - The token *"not"* and the token *"delivered"* are strongly correlated.
  - Under conditional independence, the model calculates $P(\text{"not"} \mid C) \times P(\text{"delivered"} \mid C)$. 
  - However, in tickets where *"delivered"* appears alone, it often indicates successful delivery or food condition issues (e.g., *"Order was delivered cold"*). The meaning completely hinges on their pair-wise dependence.
* **Example 2: Correlated Feature Clusters ("cold food" vs. "driver delayed")**
  In `'Food Quality Issue'`, words like *"cold"*, *"soggy"*, *"raw"*, and *"taste"* occur together in clusters. In `'Delivery Issue'`, words like *"driver"*, *"map"*, *"traffic"*, *"late"*, and *"location"* are inextricably linked. The presence of *"driver"* makes the presence of *"late"* or *"location"* drastically more probable than their individual unconditioned rates.

---

#### 3. How the Violation Affects Model Classification Accuracy
1. **Double-Counting Evidence (Overconfident Probability Estimates):**
   When multiple highly correlated words appear in a ticket (e.g., *"driver"*, *"delivery partner"*, *"courier"*), Naive Bayes treats each word as independent isolated pieces of evidence. It multiplies their probabilities together, artificially inflating the posterior probability toward extreme values ($0$ or $1$).
2. **Misclassification of Nuanced Complaints:**
   Consider a customer writing: *"The driver was very polite, but the food was completely cold and spoiled."*
   - Correlated delivery terms (*"driver"*, *"polite"*) might outweigh the single quality indicator if quality terms are sparse, misrouting a genuine `'Food Quality Issue'` to the `'Delivery Issue'` queue.
3. **Inability to Model Phrasal Semantics:**
   A ticket stating *"App crashed while checking delivery status"* contains both *"app"* and *"delivery"*. Without capturing word order and dependency, the classifier may assign equal probability to both classes or misclassify the primary issue.

---

### Scenario 5 (S5): Model Evaluation & Class Imbalance Handling
**Scenario:**
> You are comparing a Logistic Regression model and a Naive Bayes model for classifying food delivery reviews as Positive, Neutral, or Negative. Your dataset is heavily imbalanced:
> - **80%** Positive
> - **15%** Neutral
> - **5%** Negative

#### Question:
> Describe two evaluation metrics you would prioritise over accuracy when comparing these models on this imbalanced dataset. Explain one technique you would apply to address the class imbalance before training.

#### Answer:

#### 1. Why Standard Accuracy is Misleading
In an 80% / 15% / 5% class distribution, a trivial majority-class classifier that blindly predicts *"Positive"* for every single review achieves an **80.0% overall accuracy**. Yet, it fails to detect 100% of negative reviews and 100% of neutral reviews, making accuracy an invalid metric for business decision-making.

---

#### 2. Two Prioritized Evaluation Metrics

##### Metric 1: Macro-Averaged F1-Score (Macro F1)
* **Definition:** The arithmetic mean of the F1-scores calculated independently for each individual class:
  $$\text{Macro F1} = \frac{\text{F1}_{\text{Positive}} + \text{F1}_{\text{Neutral}} + \text{F1}_{\text{Negative}}}{3}$$
* **Why Prioritize It:** Macro F1 assigns equal weight (33.3%) to all three classes regardless of their sample size. If a model collapses on the 5% Negative class (achieving an $\text{F1}_{\text{Negative}} = 0$), its maximum possible Macro F1 is capped at $\approx 0.66$, immediately penalizing poor performance on critical minority groups.

##### Metric 2: Per-Class Recall for the Negative Class ($\text{Recall}_{\text{Negative}}$)
* **Definition:** 
  $$\text{Recall}_{\text{Negative}} = \frac{\text{True Negatives (Identified)}}{\text{True Negatives} + \text{False Positives/Neutrals (Missed Negatives)}}$$
* **Why Prioritize It:** In a customer service and food delivery app, **identifying unhappy customers is the highest priority business requirement**. Failing to detect a 1-star review (a False Positive where bad service goes unnoticed) leads to customer churn, lost revenue, and unaddressed restaurant hygiene issues. Maximizing $\text{Recall}_{\text{Negative}}$ ensures negative feedback is systematically caught.

---

#### 3. Technique to Address Class Imbalance Before Training

##### Chosen Technique: Class-Weighted Loss (Cost-Sensitive Learning)
* **Mechanism:** 
  During training, modify the loss function by assigning penalties inversely proportional to class frequencies:
  $$w_c = \frac{N_{\text{total}}}{K \times N_c}$$
  Where $K$ is the number of classes (3), $N_c$ is the number of samples in class $c$.
  - For Positive (80%): Weight $w \approx \frac{100}{3 \times 80} = 0.42$
  - For Neutral (15%): Weight $w \approx \frac{100}{3 \times 15} = 2.22$
  - For Negative (5%): Weight $w \approx \frac{100}{3 \times 5} = 6.67$
* **Implementation in Scikit-Learn:**
  Set `class_weight='balanced'` in `LogisticRegression(class_weight='balanced')`.
* **Advantage:**
  Does not require synthesizing artificial text or discarding majority reviews (unlike naive undersampling). Every error on a minority Negative review incurs $\approx 16\times$ higher gradient penalty than an error on a Positive review, forcing the decision boundary to protect minority classes.

*(Alternative Pre-processing Method: Synthetic Minority Over-sampling via SMOTE on TF-IDF vectors or stratified random oversampling).*

---

### Scenario 6 (S6): Attention Mechanisms in Transformers vs. RNNs & Production Scalability
**Scenario:**
> You are integrating a Hugging Face transformer pipeline into your food delivery app to automatically generate polite responses to customer reviews in real time, at a rate of several hundred requests per minute.

#### Question:
> Explain what the attention mechanism in transformers enables that traditional RNN-based models could not do effectively with long text sequences. Describe one practical challenge of running a transformer model in a high-traffic production environment and one common way to mitigate it.

#### Answer:

#### 1. What the Attention Mechanism Enables Over Traditional RNNs
In traditional Recurrent Neural Networks (RNNs, LSTMs, GRUs):
1. **Sequential Bottleneck & Vanishing Gradients:** Tokens are processed sequentially ($h_t = \sigma(W x_t + U h_{t-1})$). Information from early tokens must travel through every intermediate time step, leading to exponential decay or explosion of gradients over long customer reviews (e.g., 200+ words).
2. **Information Bottleneck in Fixed-Size State:** The entire semantic history of the paragraph is compressed into a single static vector $h_t$. Early details (e.g., a specific food allergy mentioned in sentence 1) are forgotten by the time the model generates sentence 4.
3. **No Parallel Computation:** Step $t$ cannot be computed until step $t-1$ completes, prohibiting efficient GPU tensor parallelization.

**How Self-Attention Solves This:**
* **Direct Token-to-Token Connections ($O(1)$ Path Length):** In a Transformer, the Scaled Dot-Product Attention mechanism computes direct affinity matrices between queries, keys, and values:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  Every word can attend directly to any other word across the entire document in a single matrix multiplication step, regardless of distance.
* **Preservation of Long-Range Context:** Specific details mentioned at the beginning of a multi-paragraph review remain instantly accessible when formulating the ending response.
* **Massive Parallelization:** All tokens are processed concurrently during training and encoding, maximizing GPU compute core utilization.

---

#### 2. Practical Production Challenge in High-Traffic Environments
* **Challenge:** **High Inference Latency and GPU VRAM / Compute Saturation (Inference Bottleneck)**.
  - Autoregressive text generation requires generating output tokens sequentially (one forward pass per token).
  - Running hundreds of concurrent requests per minute with multi-billion parameter transformer models causes high queue delays, excessive GPU memory consumption (due to KV-cache allocation), and potential out-of-memory (OOM) crashes, leading to unacceptable response latencies (>3–5 seconds per reply).

---

#### 3. Common Production Mitigation Strategy
* **Mitigation:** **Model Quantization and High-Throughput Inference Engines (e.g., vLLM / TensorRT-LLM / ONNX Runtime + INT8/FP4 Quantization)**.
  1. **Quantization (Weight & Activation Precision Reduction):** Converting model weights from 16-bit floating-point (FP16) to 8-bit or 4-bit integers (e.g., INT8 with AWQ or GPTQ) reduces VRAM memory requirements by 50–75% and accelerates matrix multiplication via dedicated Tensor Cores with negligible degradation in linguistic output quality.
  2. **Paged Attention & Continuous Batching:** Utilizing inference serving engines like **vLLM** implements PagedAttention, eliminating KV-cache fragmentation and dynamically batching incoming customer review requests on the fly.
  3. **Model Distillation / Smaller Architectures:** Using a compact task-distilled model (such as DistilGPT2, TinyLlama, or a fine-tuned LoRA adapter on a 3B parameter model) instead of a heavyweight general model allows sub-100ms response latencies at hundreds of queries per minute.

---
*Assessment Document: M11-A1 — Module 11 (Natural Language Processing)*
