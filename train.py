import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# BƯỚC 1: CHUẨN BỊ
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Đọc dữ liệu ĐÃ LÀM SẠCH
print("\nĐang đọc dữ liệu đã làm sạch...")
df_train = pd.read_csv("dataset/clean_train.csv")
df_val = pd.read_csv("dataset/clean_val.csv")
print(f"Train: {len(df_train)} câu | Val: {len(df_val)} câu")
print(f"Phân bố train:\n{df_train['sentiment'].value_counts()}")

# Map nhãn
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
label_map_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}

# ============================================================
# BƯỚC 2: LOAD PHOBERT TOKENIZER
# ============================================================
# PhoBERT có tokenizer riêng, nó đã biết cách tách từ tiếng Việt
# Không cần dùng underthesea nữa!

print("\nĐang tải PhoBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# ============================================================
# BƯỚC 3: TẠO DATASET
# ============================================================

MAX_LEN = 128

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # PhoBERT tokenizer tự động chuyển text thành token IDs
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

BATCH_SIZE = 16  # Nhỏ hơn vì PhoBERT nặng hơn LSTM

train_dataset = SentimentDataset(
    df_train['sentence'].tolist(),
    [label_map[s] for s in df_train['sentiment']],
    tokenizer, MAX_LEN
)
val_dataset = SentimentDataset(
    df_val['sentence'].tolist(),
    [label_map[s] for s in df_val['sentiment']],
    tokenizer, MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ============================================================
# BƯỚC 4: XÂY DỰNG MODEL
# ============================================================
# Fine-tune = Lấy PhoBERT đã biết tiếng Việt + gắn thêm 1 lớp phân loại

class PhoBERTSentiment(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Load PhoBERT đã được train sẵn trên 20GB text tiếng Việt
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        
        # Thêm lớp phân loại của riêng mình
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)  # 768 = kích thước output PhoBERT
    
    def forward(self, input_ids, attention_mask):
        # Cho PhoBERT đọc câu văn bản
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy vector đại diện cho cả câu (token [CLS])
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Phân loại
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

print("\nĐang tải PhoBERT model...")
model = PhoBERTSentiment(num_classes=3)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tổng tham số: {total_params:,}")
print(f"Tham số trainable: {trainable_params:,}")

# ============================================================
# BƯỚC 5: HUẤN LUYỆN (FINE-TUNE)
# ============================================================

# Learning rate nhỏ hơn bình thường vì PhoBERT đã biết tiếng Việt rồi
# Mình chỉ cần "chỉnh nhẹ" chứ không phải dạy từ đầu
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Tính class weights để cân bằng các lớp (sau khi làm sạch, các lớp không đều nhau)
class_counts = df_train['sentiment'].value_counts()
total = len(df_train)
weights = torch.tensor([
    total / (3 * class_counts.get('negative', 1)),
    total / (3 * class_counts.get('neutral', 1)),
    total / (3 * class_counts.get('positive', 1)),
], dtype=torch.float).to(device)
print(f"\nClass weights: neg={weights[0]:.3f}, neu={weights[1]:.3f}, pos={weights[2]:.3f}")
criterion = nn.CrossEntropyLoss(weight=weights)

NUM_EPOCHS = 5  # PhoBERT hội tụ nhanh, không cần nhiều epoch

# Learning rate scheduler với warmup
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(0.1 * total_steps)  # 10% warmup
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
best_val_acc = 0.0

print(f"\n{'='*60}")
print(f"BẮT ĐẦU FINE-TUNE PHOBERT - {NUM_EPOCHS} epochs")
print(f"{'='*60}\n")

for epoch in range(NUM_EPOCHS):
    # --- TRAINING ---
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping: giới hạn gradient để tránh "nổ gradient"
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Cập nhật learning rate
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{train_correct/train_total:.4f}"
        })
    
    train_acc = train_correct / train_total
    
    # --- VALIDATION ---
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Lưu model tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state': model.state_dict(),
            'label_map': label_map,
            'label_map_reverse': label_map_reverse,
            'max_len': MAX_LEN,
            'epoch': epoch + 1,
            'val_acc': val_acc,
        }, 'phobert_sentiment.pth')
        print(f"  -> Lưu model mới! (best val acc: {best_val_acc:.4f})")
    
    # Luôn lưu model cuối cùng để đảm bảo
    torch.save({
        'model_state': model.state_dict(),
        'label_map': label_map,
        'label_map_reverse': label_map_reverse,
        'max_len': MAX_LEN,
        'epoch': epoch + 1,
        'val_acc': val_acc,
    }, 'phobert_sentiment_latest.pth')

print(f"\n{'='*60}")
print(f"HOÀN TẤT! Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Model lưu tại: phobert_sentiment.pth")
print(f"{'='*60}")
