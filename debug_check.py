"""Debug: kiểm tra model predict trên chính dataset train để xác nhận label mapping đúng"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

class PhoBERTSentiment(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
checkpoint = torch.load('phobert_sentiment.pth', map_location=device, weights_only=False)
print(f"Checkpoint keys: {list(checkpoint.keys())}")
print(f"label_map: {checkpoint['label_map']}")
print(f"label_map_reverse: {checkpoint['label_map_reverse']}")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Val acc: {checkpoint.get('val_acc', 'N/A')}")

label_map = checkpoint['label_map']
label_map_reverse = checkpoint['label_map_reverse']
MAX_LEN = checkpoint['max_len']

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = PhoBERTSentiment(num_classes=3)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()

def predict(text):
    encoding = tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(output, dim=1).item()
    return pred_idx, probs

# Test trên CHÍNH DATA TRAIN để xem label mapping có đúng không
df = pd.read_csv("dataset/clean_train.csv")

print("\n=== TEST TRÊN DATA TRAIN ===")
for sentiment in ['positive', 'negative', 'neutral']:
    samples = df[df['sentiment'] == sentiment].head(3)
    print(f"\n--- Nhãn gốc: {sentiment.upper()} ---")
    for _, row in samples.iterrows():
        pred_idx, probs = predict(row['sentence'])
        pred_label = label_map_reverse[pred_idx]
        match = "✅" if pred_label == sentiment else "❌"
        print(f"  {match} \"{row['sentence'][:60]}...\"")
        print(f"     → Pred: {pred_label} | Raw logits idx: {pred_idx}")
        print(f"     → Neg[0]:{probs[0]:.2%} | Neu[1]:{probs[1]:.2%} | Pos[2]:{probs[2]:.2%}")

# Test specific sentences
print("\n\n=== TEST CÂU CỤ THỂ ===")
test_sentences = [
    ("Thầy dạy rất hay và nhiệt tình", "positive"),
    ("Trường tôi tuyệt vời lắm", "positive"),
    ("Dịch vụ quá tệ tôi rất thất vọng", "negative"),
    ("Giảng viên dạy chán quá", "negative"),
    ("Hôm nay là thứ hai", "neutral"),
]

for text, expected in test_sentences:
    pred_idx, probs = predict(text)
    pred_label = label_map_reverse[pred_idx]
    match = "✅" if pred_label == expected else "❌"
    print(f"\n{match} \"{text}\"")
    print(f"   Expected: {expected} | Got: {pred_label} (idx={pred_idx})")
    print(f"   Neg[0]:{probs[0]:.2%} | Neu[1]:{probs[1]:.2%} | Pos[2]:{probs[2]:.2%}")

# Check raw output
print("\n\n=== RAW OUTPUT CHECK ===")
text = "Thầy dạy rất hay và nhiệt tình"
encoding = tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)
with torch.no_grad():
    output = model(input_ids, attention_mask)
    print(f"Raw logits: {output[0].cpu().tolist()}")
    probs = torch.softmax(output, dim=1)[0]
    print(f"Probs: {probs.cpu().tolist()}")
    print(f"Argmax: {torch.argmax(output, dim=1).item()}")
    print(f"label_map_reverse: {label_map_reverse}")
