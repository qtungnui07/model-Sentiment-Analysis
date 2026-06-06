import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import sys
import os

# Fix encoding cho Windows console
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ============================================================
# LOAD MODEL
# ============================================================

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
checkpoint = torch.load('phobert_sentiment.pth', map_location=device, weights_only=False)
label_map_reverse = checkpoint['label_map_reverse']
MAX_LEN = checkpoint['max_len']

print(f"Device: {device}")
print(f"Model epoch: {checkpoint.get('epoch', 'N/A')} | Val acc: {checkpoint.get('val_acc', 'N/A'):.4f}")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = PhoBERTSentiment(num_classes=3)
model.load_state_dict(checkpoint['model_state'], strict=True)
model = model.to(device)
model.eval()

print("Model loaded successfully!")

# ============================================================
# HÀM DỰ ĐOÁN
# ============================================================

def predict(text):
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(output, dim=1).item()
    
    return label_map_reverse[pred], probs

# ============================================================
# EMOJI MAPPING
# ============================================================

EMOJI_MAP = {
    'positive': '😊 POSITIVE',
    'negative': '😞 NEGATIVE', 
    'neutral': '😐 NEUTRAL',
}

# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("  PHOBERT SENTIMENT ANALYSIS - Vietnamese")
print("=" * 60)
print("Gõ câu tiếng Việt để phân tích cảm xúc")
print("Gõ 'thoat' để dừng\n")

while True:
    try:
        user_input = input("📝 Nhập câu: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    
    if not user_input:
        continue
    if user_input.lower() == 'thoat':
        break
    
    # Debug: show raw bytes to verify encoding
    # print(f"   [DEBUG] Raw input bytes: {user_input.encode('utf-8')[:50]}")
    
    sentiment, probs = predict(user_input)
    
    print(f"\n   Kết quả: {EMOJI_MAP.get(sentiment, sentiment)}")
    print(f"   ┌─────────────────────────────────────┐")
    print(f"   │ Negative: {probs[0]:6.2%}                  │")
    print(f"   │ Neutral:  {probs[1]:6.2%}                  │")
    print(f"   │ Positive: {probs[2]:6.2%}                  │")
    print(f"   └─────────────────────────────────────┘\n")

print("\nĐã thoát. Cảm ơn!")
