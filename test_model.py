import joblib
from underthesea import word_tokenize

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_sentiment(text):
    # Tách từ tiếng Việt đúng cách, giống hệt lúc train
    text = word_tokenize(text.lower(), format="text")
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

while True:
    user_input = input()
    if user_input.lower() == 'thoat':
        break
    
    result = predict_sentiment(user_input)
    print(f"{result.upper()}\n")
