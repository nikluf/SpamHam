import re
import joblib
import nltk
from nltk.corpus import stopwords


model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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

stop_words = set(stopwords.words('english'))

spam_text = input("Enter the text you recieved:")
#print(predict_spam("Congratulations! You won a free iPhone! Click here to claim now."))
#print(predict_spam("Hey, are we meeting tomorrow?"))
print(predict_spam(spam_text))
