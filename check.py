import torch
import sentencepiece as spm
from micronmt import Seq2SeqTransformer, MAX_LEN, VOCAB_SIZE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizers
src_tok = spm.SentencePieceProcessor(model_file='ja.model')
tgt_tok = spm.SentencePieceProcessor(model_file='en.model')

# Load model
model = Seq2SeqTransformer(3, 3, 512, VOCAB_SIZE, VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load("transformer_ja_en.pth"))
model.eval()

def encode_input(text, tokenizer):
    tokens = tokenizer.encode(text)
    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))
    return torch.tensor(tokens).unsqueeze(1).to(DEVICE)

def greedy_decode(model, src_tensor, max_len):
    src_mask = model.transformer.generate_square_subsequent_mask(src_tensor.size(0)).to(DEVICE)
    memory = model.transformer.encoder(model.pos_enc(model.src_emb(src_tensor)), src_mask)
    ys = torch.zeros(1, 1).type(torch.long).to(DEVICE)  # BOS token (assumed 0)

    for i in range(max_len):
        tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(0)).to(DEVICE)
        out = model.transformer.decoder(model.pos_enc(model.tgt_emb(ys)), memory, tgt_mask)
        out = model.fc_out(out)
        prob = out[-1, 0].argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[prob]], device=DEVICE)], dim=0)
        if prob == 2:  # EOS token (SentencePiece usually uses ID 2 for </s>)
            break
    return ys.squeeze().tolist()

def translate(text):
    src_tensor = encode_input(text, src_tok).transpose(0, 1)
    tgt_tokens = greedy_decode(model, src_tensor, MAX_LEN)
    translation = tgt_tok.decode([t for t in tgt_tokens if t > 2])  # skip special tokens
    return translation

# Example
if __name__ == "__main__":
    jp_text = "これはペンです。"
    en_translation = translate(jp_text)
    print(f"Japanese: {jp_text}")
    print(f"English: {en_translation}")
