# Spam Email Detection Using a Deep Neural Network (DNN)

## 1. Problem
Classify an email/message as **spam** or **ham** (legitimate) automatically.

## 2. Approach
```
Raw text -> cleaning -> TF-IDF (5000 uni+bigrams) -> DNN (512-256-64-1) -> spam probability
```
- **Cleaning:** lowercase, replace URLs/numbers with tokens, strip punctuation.
- **Features:** TF-IDF fitted on training data only (no leakage).
- **Model:** 3 hidden Dense layers + BatchNorm + Dropout, sigmoid output, binary cross-entropy, Adam.
- **Training tricks:** class weights for imbalance, EarlyStopping, ReduceLROnPlateau.
- **Split:** 70% train / 15% validation / 15% test (stratified).
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

## 3. Run
```bash
pip install -r requirements.txt
python make_sample_data.py     # creates data/spam.csv (synthetic demo data)
python train.py                # trains + evaluates, saves to outputs/
python predict.py "Congratulations! You won a free iPhone, click http://bit.ly/x"
```

## 4. Use a real dataset (do this for your submission)
Download one and save as `data/spam.csv` with two columns `label` (spam/ham) and `text`:
- **SMS Spam Collection** (UCI / Kaggle) - ~5.5k messages
- **Enron Spam** or **SpamAssassin** - real emails
- **Kaggle "Spam Email Classification Dataset"**

If the CSV has columns `v1,v2` (SMS Spam Collection), rename them:
`df.rename(columns={"v1":"label","v2":"text"})`.

## 5. Outputs
`outputs/` contains `spam_dnn.keras`, `tfidf.joblib`, `training_curves.png`, `confusion_matrix.png`, `roc_curve.png`, `report.txt`.

## 6. Ideas to extend (for extra marks)
- Compare against Naive Bayes / Logistic Regression baselines.
- Replace TF-IDF with an Embedding + LSTM/BiLSTM or a fine-tuned BERT.
- Build a Flask/Streamlit web UI around `predict.py`.
- Add precision-recall tuning of the decision threshold to minimise false positives (ham marked as spam).
- Explain predictions using LIME/SHAP.

## 7. Limitations
TF-IDF ignores word order; the model can be fooled by obfuscated words (e.g. "fr33"); results depend on dataset quality.
