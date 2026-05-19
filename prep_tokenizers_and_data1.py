import os, io, math, json, random
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFKC
import sentencepiece as spm
random.seed(42)
OUTDIR = "artifacts"
RAW_DIR = os.path.join(OUTDIR, "raw")
SPM_DIR = os.path.join(OUTDIR, "spm")
CHUNKS_DIR = os.path.join(OUTDIR, "chunks")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SPM_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
print("Loading WikiText-103...")
ds = load_dataset("wikitext", "wikitext-103-v1")
for split in ["train","validation","test"]:
    path = os.path.join(RAW_DIR, f"{split}.txt")
    with io.open(path, "w", encoding="utf-8") as f:
        for row in ds[split]["text"]:
            if row.strip():
                f.write(row.strip()+"\n")
    print(f"Wrote {path}")
spm_sizes = [8000, 32000, 128000]
for size in spm_sizes:
    prefix = os.path.join(SPM_DIR, f"spm_bpe_{size}")
    if not (os.path.exists(prefix+".model") and os.path.exists(prefix+".vocab")):
        spm.SentencePieceTrainer.Train(
            input=os.path.join(RAW_DIR, "train.txt"),
            model_prefix=prefix,
            model_type="bpe",
            vocab_size=size,
            character_coverage=1.0,
            input_sentence_size=3000000,
            shuffle_input_sentence=True,
            hard_vocab_limit=False
        )
        print(f"Trained SentencePiece BPE {size}")
BYTES_WINDOW = 2048 
MAX_CHUNKS_PER_SPLIT = {
"train": None, 
"validation": 5000,
"test": 5000,
}
for split in ["train","validation","test"]:
    path = os.path.join(RAW_DIR, f"{split}.txt")
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    raw_bytes = text.encode("utf-8")
    n = len(raw_bytes)
    chunks_meta = []
    out_base = os.path.join(CHUNKS_DIR, split)
    os.makedirs(out_base, exist_ok=True)
    start = 0
    idx = 0
    limit = MAX_CHUNKS_PER_SPLIT[split]
    while start < n:
        end = min(start + BYTES_WINDOW, n)
        b = raw_bytes[start:end]
        chunk_txt = b.decode("utf-8", errors="ignore")
        with io.open(os.path.join(out_base, f"{idx}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk_txt)
        chunks_meta.append({"id": idx, "byte_len": len(b)})
        idx += 1
        start = end
        if limit and idx >= limit:
            break
    with io.open(os.path.join(out_base, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False)
    print(f"Split {split}: {idx} chunks, stored at {out_base}")
print("Done: tokenizers + byte-windowed chunks prepared.")