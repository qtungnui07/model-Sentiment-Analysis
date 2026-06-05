import joblib
from underthesea import word_tokenize
import sys
sys.stdout.reconfigure(encoding='utf-8')

model = joblib.load('sentiment_model.pkl')
vec = joblib.load('vectorizer.pkl')

tests = [
    'trường học này đẹp quá',
    'trường học này xấu quá', 
    'thầy giáo dạy rất hay',
    'phòng học nóng quá',
    'cơ sở vật chất tốt',
    'giảng viên dạy chán quá',
]

for t in tests:
    tokenized = word_tokenize(t.lower(), format="text")
    pred = model.predict(vec.transform([tokenized]))[0]
    print(f"{t} -> {pred.upper()}")
