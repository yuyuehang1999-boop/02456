import os
import math

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "artifacts/results.csv"
FIG_DIR = "artifacts/figures"


def load_results(path=RESULTS_CSV):
    df = pd.read_csv(path)

    expected_cols = [
        "name",
        "vocab_size",
        "valid_bpb",
        "total_params",
        "embed_params",
        "embed_ratio",
        "train_seconds",
        "tokens_seen",
        "max_mem_bytes",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print("⚠ 警告：results.csv 中缺少以下列（部分图会画不出来）:", missing)

    if "tokens_seen" in df.columns and "train_seconds" in df.columns:
        df["tokens_per_sec"] = df["tokens_seen"] / df["train_seconds"]
    else:
        df["tokens_per_sec"] = float("nan")

    if "max_mem_bytes" in df.columns:
        df["max_mem_gb"] = df["max_mem_bytes"] / (1024**3)
    else:
        df["max_mem_gb"] = float("nan")

    df = df.sort_values(by="vocab_size").reset_index(drop=True)
    return df


def ensure_figdir():
    os.makedirs(FIG_DIR, exist_ok=True)


def save_current_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ saved:", path)

def plot_bpb_vs_vocab(df: pd.DataFrame):
    plt.figure(figsize=(5, 4))
    x = df["vocab_size"]
    y = df["valid_bpb"]

    plt.plot(x, y, marker="o")
    plt.xlabel("Vocabulary size")
    plt.ylabel("Bits-per-Byte (valid BPB)")
    plt.title("BPB vs. Vocabulary Size")
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.xscale("log", base=2)
    plt.xticks(x, [str(v) for v in x])

    save_current_fig("bpb_vs_vocab.png")



def plot_embed_ratio_vs_vocab(df: pd.DataFrame):
    plt.figure(figsize=(5, 4))
    x = df["vocab_size"]
    y = df["embed_ratio"] * 100.0  

    plt.plot(x, y, marker="o")
    plt.xlabel("Vocabulary size")
    plt.ylabel("Embedding parameters (%)")
    plt.title("Embedding Share vs. Vocabulary Size")
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.xscale("log", base=2)
    plt.xticks(x, [str(v) for v in x])

    save_current_fig("embed_ratio_vs_vocab.png")



def plot_throughput(df: pd.DataFrame):
    if df["tokens_per_sec"].isna().all():
        print("⚠ 无 tokens_per_sec，跳过吞吐图")
        return

    plt.figure(figsize=(6, 4))
    x = range(len(df))
    y = df["tokens_per_sec"]
    labels = df["name"]

    plt.bar(x, y)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Tokens per second")
    plt.title("Training Throughput (tokens/s)")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    save_current_fig("throughput_tokens_per_sec.png")


def plot_train_seconds(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    x = range(len(df))
    y = df["train_seconds"]
    labels = df["name"]

    plt.bar(x, y)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Training time (seconds)")
    plt.title("Training Time per Configuration")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    save_current_fig("train_time_per_config.png")

def plot_max_mem(df: pd.DataFrame):
    if df["max_mem_gb"].isna().all():
        print("⚠ 无 max_mem_bytes，跳过显存图")
        return

    plt.figure(figsize=(6, 4))
    x = range(len(df))
    y = df["max_mem_gb"]
    labels = df["name"]

    plt.bar(x, y)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Max GPU memory (GB)")
    plt.title("Peak GPU Memory Usage")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    save_current_fig("max_mem_gb_per_config.png")


def main():
    ensure_figdir()
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"{RESULTS_CSV} 不存在，请先运行 summarize_results.py 生成结果表。"
        )

    df = load_results(RESULTS_CSV)

    print("Loaded results:")
    print(df)

    plot_bpb_vs_vocab(df)
    plot_embed_ratio_vs_vocab(df)
    plot_throughput(df)
    plot_train_seconds(df)
    plot_max_mem(df)

    print("🎉 所有可用图像已生成，位于:", FIG_DIR)


if __name__ == "__main__":
    main()