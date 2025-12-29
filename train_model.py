import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Load datasets
true_df = pd.read_csv("dataset/True.csv")
fake_df = pd.read_csv("dataset/Fake.csv")

# 2. Add labels
true_df["label"] = 1   # REAL
fake_df["label"] = 0   # FAKE

# 3. Combine datasets
df = pd.concat([true_df, fake_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle

# 4. Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# 5. Features and labels
X = df["text"]
y = df["label"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)

# 8. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 9. Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully with real dataset")
