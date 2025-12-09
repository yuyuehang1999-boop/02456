import os, io, json, time, math, gc
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
import sentencepiece as spm
ART = "artifacts"
SPM_DIR = os.path.join(ART, "spm")
CHUNKS_DIR = os.path.join(ART, "chunks")
OUT_DIR = os.path.join(ART, "runs")
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_chunks(split):
    base = os.path.join(CHUNKS_DIR, split)
    with io.open(os.path.join(base, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    paths = [os.path.join(base, f"{m['id']}.txt") for m in meta]
    bytelens = [m['byte_len'] for m in meta]
    return paths, bytelens
class TextChunkDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        with io.open(self.paths[i], "r", encoding="utf-8") as f:
            return f.read()
class SPMTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
    @property
    def vocab_size(self):
        return self.sp.vocab_size()
    def encode(self, text):
        return self.sp.encode(text, out_type=int)
    def decode(self, ids):
        return self.sp.decode(ids)
class ByteTokenizer:

    def __init__(self):
        self.vocab_size = 256
    def encode(self, text):
        b = text.encode("utf-8")
        return list(b)
    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="ignore")
class LMDataCollator:
    def __init__(self, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len
    def __call__(self, batch_texts):
        input_ids = []
        for t in batch_texts:
            ids = self.tok.encode(t)[:self.max_len]
            if len(ids) < self.max_len:
                ids += [0]*(self.max_len-len(ids)) 
            input_ids.append(ids)
        x = torch.tensor(input_ids, dtype=torch.long)
        labels = x.clone()
        return {"input_ids": x, "labels": labels}
def make_model(vocab_size, n_layer=6, n_head=8, n_embd=512, n_positions=512):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_positions=n_positions,
        bos_token_id=None, eos_token_id=None
    )
    model = GPT2LMHeadModel(cfg)
    return model
def train_one(tokenizer, name, max_len=512, batch_size=16, steps=2000, lr=3e-4):
    paths_tr, _ = load_chunks("train")
    paths_va, _ = load_chunks("validation")
    ds_tr = TextChunkDataset(paths_tr)
    ds_va = TextChunkDataset(paths_va)
    collate = LMDataCollator(tokenizer, max_len)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate)
    model = make_model(tokenizer.vocab_size, n_positions=max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95))
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.transformer.wte.weight.numel()
    embed_ratio = embed_params/total_params
    log = {"name": name, "vocab_size": tokenizer.vocab_size,
    "total_params": int(total_params), "embed_params": int(embed_params), "embed_ratio": float(embed_ratio)}
    t0 = time.time()
    model.train()
    it = iter(dl_tr)
    tokens_seen = 0
    for step in range(1, steps+1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl_tr); batch = next(it)
        batch = {k:v.to(device) for k,v in batch.items()}
        out = model(**batch)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += batch["input_ids"].numel()
        if step % 100 == 0:
            print(f"{name} step {step}/{steps} loss={loss.item():.3f}")
    wall = time.time()-t0
    log["train_seconds"] = wall
    log["tokens_seen"] = int(tokens_seen)
    if torch.cuda.is_available():
        log["max_mem_bytes"] = int(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        paths_va, bytelens_va = load_chunks("validation")
        ds_va = TextChunkDataset(paths_va)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate)
        nats_sum = 0.0
        bytes_sum = 0
        for i, batch_texts in enumerate(dl_va):
            batch = {k:v.to(device) for k,v in batch_texts.items()}
            out = model(**batch)
            nats = out.loss.item() * batch["input_ids"].numel()
            from prep_tokenizers_and_data import BYTES_WINDOW
            nats_sum += nats
            bytes_sum += BYTES_WINDOW * batch["input_ids"].shape[0]
        bits = nats_sum / math.log(2)
        bpb = bits / bytes_sum
        log["valid_bpb"] = float(bpb)
    out_path = os.path.join(OUT_DIR, f"{name}.json")
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print("Saved log to", out_path)
if __name__ == "__main__":
    byte_tok = ByteTokenizer()
    train_one(byte_tok, name="byte_256", max_len=512, batch_size=16, steps=1000)
    for vs in [8000, 32000, 128000]:
        spm_path = os.path.join(SPM_DIR, f"spm_bpe_{vs}.model")
        spm_tok = SPMTokenizer(spm_path)
        train_one(spm_tok, name=f"bpe_{vs}", max_len=512, batch_size=16, steps=1000)
