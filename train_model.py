import pandas as pd 
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib
from tqdm import tqdm
from underthesea import word_tokenize

sys.stdout.reconfigure(encoding='utf-8')

# Hàm tách từ tiếng Việt đúng cách
def tokenize_vietnamese(text):
    return word_tokenize(str(text).lower(), format="text")

# Đọc dữ liệu
df_train = pd.read_csv("dataset/synthetic_train.csv")
df_val = pd.read_csv("dataset/synthetic_val.csv")

# Tách từ tiếng Việt cho tập train
tqdm.pandas(desc="Tách từ 8145 câu Train")
x_train = df_train['sentence'].progress_apply(tokenize_vietnamese)
y_train = df_train['sentiment']

# Tách từ tiếng Việt cho tập val
tqdm.pandas(desc="Tách từ tập Val")
x_test = df_val['sentence'].progress_apply(tokenize_vietnamese)
y_test = df_val['sentiment']

# Chuyển chữ thành số
vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),     # Chỉ dùng từ đơn
    sublinear_tf=True,      # Giảm ảnh hưởng của từ xuất hiện quá nhiều
    min_df=2,               # Bỏ từ quá hiếm
    max_df=0.95,            # Bỏ từ xuất hiện trong >95% câu (từ vô nghĩa)
)
x_train_vectorized = vectorizer.fit_transform(x_train) 
x_test_vectorized = vectorizer.transform(x_test) 

# Huấn luyện - Logistic Regression với class_weight balanced
print("Đang huấn luyện mô hình...")
model = LogisticRegression(
    max_iter=2000,
    C=1.0,
    class_weight='balanced',
    solver='lbfgs',
)
model.fit(x_train_vectorized, y_train)

# Đánh giá trên tập val
y_pred = model.predict(x_test_vectorized)
print("test val")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Lưu model
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Đã lưu mô hình!")
