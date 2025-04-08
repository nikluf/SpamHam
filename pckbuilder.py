import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text) 
    text = re.sub(r'\s+', ' ', text)  
    words = text.split()
    words = [word for word in words if word not in stop_words]  
    return ' '.join(words)

def predict_spam(text):
    text = clean_text(text)
    vectorized_text = vectorizer.transform([text]).toarray()
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"



df = pd.read_csv("spam.csv", encoding="ISO-8859-1")  



df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df['cleaned_message'] = df['v2'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000) 
X = vectorizer.fit_transform(df['cleaned_message']).toarray()
y = df['v1']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "spam_classifier.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")
