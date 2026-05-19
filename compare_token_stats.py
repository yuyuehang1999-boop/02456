import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sentencepiece as spm


ARTIFACTS_DIR = "artifacts"
CHUNKS_DIR = os.path.join(ARTIFACTS_DIR, "chunks")
SPM_DIR = os.path.join(ARTIFACTS_DIR, "spm")
FIG_DIR = os.path.join(ARTIFACTS_DIR, "figures")
OUT_CSV = os.path.join(ARTIFACTS_DIR, "token_stats_test.csv")
SPLIT = "test"
BAR_COLOR = "#1f77b4"


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))


class SPMTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)


def ensure_inputs_exist():
    split_dir = os.path.join(CHUNKS_DIR, SPLIT)
    meta_path = os.path.join(split_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Run prep_tokenizers_and_data.py first."
        )

    for vocab_size in [8000, 32000]:
        model_path = os.path.join(SPM_DIR, f"spm_bpe_{vocab_size}.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found. Run prep_tokenizers_and_data.py first."
            )


def load_tokenizers():
    tokenizers = [
        ("byte_256", 256, ByteTokenizer()),
    ]

    for vocab_size in [8000, 32000]:
        model_path = os.path.join(SPM_DIR, f"spm_bpe_{vocab_size}.model")
        tokenizers.append(
            (f"bpe_{vocab_size}", vocab_size, SPMTokenizer(model_path))
        )

    return tokenizers


def iter_chunk_texts(split):
    split_dir = os.path.join(CHUNKS_DIR, split)
    meta_path = os.path.join(split_dir, "meta.json")

    with io.open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for item in meta:
        chunk_path = os.path.join(split_dir, f"{item['id']}.txt")
        with io.open(chunk_path, "r", encoding="utf-8") as f:
            yield f.read()


def compute_stats():
    rows = []

    for name, vocab_size, tokenizer in load_tokenizers():
        total_bytes = 0
        total_tokens = 0

        for text in iter_chunk_texts(SPLIT):
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(tokenizer.encode(text))

        avg_bytes_per_token = total_bytes / total_tokens
        tokens_per_byte = total_tokens / total_bytes

        rows.append(
            {
                "name": name,
                "vocab_size": vocab_size,
                "split": SPLIT,
                "total_bytes": total_bytes,
                "total_tokens": total_tokens,
                "avg_bytes_per_token": avg_bytes_per_token,
                "tokens_per_byte": tokens_per_byte,
            }
        )

    return pd.DataFrame(rows).sort_values("vocab_size").reset_index(drop=True)


def save_bar_plot(df, column, ylabel, title, filename):
    plt.figure(figsize=(5, 4))
    x_labels = [str(v) for v in df["vocab_size"]]

    plt.bar(x_labels, df[column], color=BAR_COLOR)
    plt.xlabel("Vocabulary size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", path)


def main():
    ensure_inputs_exist()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = compute_stats()
    df.to_csv(OUT_CSV, index=False)
    print(df)
    print("Saved", OUT_CSV)

    save_bar_plot(
        df,
        "total_tokens",
        "Total tokens on test set",
        "Total Tokens vs. Vocabulary Size",
        "test_total_tokens_bar.png",
    )
    save_bar_plot(
        df,
        "avg_bytes_per_token",
        "Average bytes per token",
        "Avg Bytes per Token vs. Vocabulary Size",
        "test_avg_bytes_per_token_bar.png",
    )


if __name__ == "__main__":
    main()
