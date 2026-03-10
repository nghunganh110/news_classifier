import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import TextPreprocessor
from src.evaluate import plot_confusion_matrix, print_classification_report

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sample_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

MAX_SEQ_LEN = 128
VOCAB_SIZE = 10000
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
PAD_IDX = 0
UNK_IDX = 1


class Vocabulary:
    def __init__(self, max_size=VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
        self.idx2word = {PAD_IDX: '<PAD>', UNK_IDX: '<UNK>'}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=MAX_SEQ_LEN):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        ids = ids + [PAD_IDX] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2idx):
        self.encodings = [vocab.encode(t) for t in texts]
        self.labels = [label2idx[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, ff_dim=FF_DIM, max_len=MAX_SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len)
        padding_mask = (x == PAD_IDX)  # (batch, seq_len)
        # Scale embeddings by sqrt(d_model) to prevent positional encodings from
        # dominating the learned embeddings early in training (Vaswani et al., 2017)
        emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        emb = self.pos_encoding(emb)
        out = self.transformer(emb, src_key_padding_mask=padding_mask)
        # Mean pool over non-padding tokens
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        out = (out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.classifier(out)


class TransformerTrainer:
    def __init__(self, num_classes, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.vocab = Vocabulary()
        self.label2idx = {}
        self.idx2label = {}
        self.model = None

    def train(self, train_texts, train_labels, val_texts, val_labels):
        self.vocab.build(train_texts)
        all_labels = sorted(set(train_labels))
        self.label2idx = {l: i for i, l in enumerate(all_labels)}
        self.idx2label = {i: l for l, i in self.label2idx.items()}

        train_ds = NewsDataset(train_texts, train_labels, self.vocab, self.label2idx)
        val_ds = NewsDataset(val_texts, val_labels, self.vocab, self.label2idx)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        self.model = SimpleTransformerClassifier(
            vocab_size=len(self.vocab),
            num_classes=self.num_classes,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        for epoch in range(1, EPOCHS + 1):
            self.model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    preds = self.model(X).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            val_acc = correct / total
            print(f"Epoch {epoch}/{EPOCHS}  Loss: {total_loss/len(train_loader):.4f}  Val Acc: {val_acc:.4f}")

    def predict(self, texts):
        self.model.eval()
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                encoded = [self.vocab.encode(t) for t in batch]
                X = torch.tensor(encoded, dtype=torch.long).to(self.device)
                logits = self.model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend([self.idx2label[p] for p in preds])
                all_probs.extend(probs.tolist())
        return all_preds, all_probs

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'vocab': self.vocab,
            'label2idx': self.label2idx,
            'idx2label': self.idx2label,
            'num_classes': self.num_classes,
        }, path)
        print(f"Transformer model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        self.label2idx = checkpoint['label2idx']
        self.idx2label = checkpoint['idx2label']
        self.num_classes = checkpoint['num_classes']
        self.model = SimpleTransformerClassifier(
            vocab_size=len(self.vocab),
            num_classes=self.num_classes,
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH).dropna(subset=['text', 'category'])
    texts, labels = df['text'].tolist(), df['category'].tolist()
    label_list = sorted(set(labels))
    num_classes = len(label_list)

    print("Preprocessing...")
    preprocessor = TextPreprocessor()
    processed = preprocessor.preprocess_batch(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )

    trainer = TransformerTrainer(num_classes=num_classes)
    print(f"Training Transformer on {len(X_train)} samples...")
    trainer.train(X_train, y_train, X_test, y_test)

    y_pred, _ = trainer.predict(X_test)
    acc = np.mean(np.array(y_pred) == np.array(y_test))
    print(f"\nTest Accuracy: {acc:.4f}")
    print_classification_report(y_test, y_pred, label_list)
    plot_confusion_matrix(
        y_test, y_pred, label_list,
        "Transformer Confusion Matrix",
        os.path.join(OUTPUTS_DIR, 'transformer_confusion_matrix.png')
    )

    trainer.save(os.path.join(MODELS_DIR, 'transformer_model.pt'))


if __name__ == '__main__':
    main()
