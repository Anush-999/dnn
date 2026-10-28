"""Classify an email/message:  python predict.py "Your text here"   (or run with no args for interactive mode)"""
import sys
import re
import joblib
import tensorflow as tf

model = tf.keras.models.load_model("outputs/spam_dnn.keras")
vec = joblib.load("outputs/tfidf.joblib")


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\$\d+|\d+", " numtoken ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def classify(text):
    x = vec.transform([clean_text(text)]).toarray()
    p = float(model.predict(x, verbose=0)[0][0])
    return ("SPAM" if p >= 0.5 else "HAM"), p


if __name__ == "__main__":
    if len(sys.argv) > 1:
        label, p = classify(" ".join(sys.argv[1:]))
        print(f"{label} (spam probability {p:.3f})")
    else:
        while True:
            t = input("Email text (blank to quit): ").strip()
            if not t:
                break
            label, p = classify(t)
            print(f"-> {label} (spam probability {p:.3f})")
