import os
import math
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Configuration
# ----------------------------
SRC_LANG = 'ja'
TGT_LANG = 'en'
MAX_LEN = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# Step 1: Train SentencePiece Tokenizers
# ----------------------------
def train_tokenizers():
    if not os.path.exists(f"{SRC_LANG}.model"):
        spm.SentencePieceTrainer.Train(
            input=f"{SRC_LANG}.txt", model_prefix=SRC_LANG, vocab_size=8000)
    if not os.path.exists(f"{TGT_LANG}.model"):
        spm.SentencePieceTrainer.Train(
            input=f"{TGT_LANG}.txt", model_prefix=TGT_LANG, vocab_size=8000)

# ----------------------------
# Step 2: Dataset
# ----------------------------
class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, src_tokenizer, tgt_tokenizer):
        self.src = src_lines
        self.tgt = tgt_lines
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer

    def __len__(self):
        return len(self.src)

    def encode(self, line, tokenizer):
        tokens = [tokenizer.bos_id()] + tokenizer.encode(line) + [tokenizer.eos_id()]
        tokens = tokens[:MAX_LEN]
        pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0
        tokens += [pad_id] * (MAX_LEN - len(tokens))
        return torch.tensor(tokens)

    def __getitem__(self, idx):
        src_tensor = self.encode(self.src[idx], self.src_tok)
        tgt_tensor = self.encode(self.tgt[idx], self.tgt_tok)
        return src_tensor, tgt_tensor

# ----------------------------
# Step 3: Transformer Model
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=MAX_LEN):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return self.dropout(x + self.pos_embedding[:x.size(0), :])

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=8,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.src_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.pos_enc = PositionalEncoding(emb_size, dropout)
        self.fc_out = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.pos_enc(self.src_emb(src))
        tgt = self.pos_enc(self.tgt_emb(tgt))
        outs = self.transformer(src, tgt, src_mask, tgt_mask)
        return self.fc_out(outs)

# ----------------------------
# Step 4: Training
# ----------------------------
def train_model(model, dataloader, optimizer, loss_fn, pad_idx, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(10):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.transpose(0, 1).to(DEVICE), tgt.transpose(0, 1).to(DEVICE)
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            src_mask = model.transformer.generate_square_subsequent_mask(src.size(0)).to(DEVICE)
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_input.size(0)).to(DEVICE)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            optimizer.zero_grad()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation loss (optional, can be added later)
        # val_loss = validate_model(model, valid_dataloader, loss_fn)
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def validate_model(model, dataloader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation during validation
        for src, tgt in dataloader:
            src, tgt = src.transpose(0, 1).to(DEVICE), tgt.transpose(0, 1).to(DEVICE)
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            src_mask = model.transformer.generate_square_subsequent_mask(src.size(0)).to(DEVICE)
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_input.size(0)).to(DEVICE)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ----------------------------
# Step 5: Main
# ----------------------------
def main():
    # 1. Train or load tokenizers
    train_tokenizers()
    src_tok = spm.SentencePieceProcessor(model_file=f'{SRC_LANG}.model')
    tgt_tok = spm.SentencePieceProcessor(model_file=f'{TGT_LANG}.model')

    # 2. Load parallel data
    with open(f"{SRC_LANG}.txt", encoding='utf8') as f:
        src_lines = [line.strip() for line in f]
    with open(f"{TGT_LANG}.txt", encoding='utf8') as f:
        tgt_lines = [line.strip() for line in f]

    # 3. Prepare data
    dataset = TranslationDataset(src_lines, tgt_lines, src_tok, tgt_tok)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4. Initialize model
    src_vocab_size = src_tok.get_piece_size()
    tgt_vocab_size = tgt_tok.get_piece_size()
    model = Seq2SeqTransformer(3, 3, 512, src_vocab_size, tgt_vocab_size).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4)
    pad_idx = tgt_tok.pad_id() if tgt_tok.pad_id() >= 0 else 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 5. Train
    train_model(model, dataloader, optimizer, loss_fn, pad_idx)

    # 6. Save model
    torch.save(model.state_dict(), "transformer_ja_en.pth")

if __name__ == "__main__":
    main()
