#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

try:
    import torch
    import sentencepiece as spm
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
except Exception as exc:
    print("missing deps: torch, transformers, sentencepiece")
    print(f"import error: {exc}")
    raise SystemExit(1)


def read_texts(path):
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
    return texts


def train_sentencepiece(texts, outdir, vocab_size):
    outdir.mkdir(parents=True, exist_ok=True)
    input_txt = outdir / "spm_input.txt"
    model_prefix = outdir / "tokenizer"
    if (outdir / "tokenizer.model").exists():
        return outdir / "tokenizer.model"
    with input_txt.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    spm.SentencePieceTrainer.Train(
        input=str(input_txt),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        character_coverage=1.0,
        byte_fallback=True,
    )
    return outdir / "tokenizer.model"


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tok(text, add_special_tokens=True, truncation=True, max_length=self.max_len)
        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        return ids


def collate(batch, pad_id):
    max_len = max(len(x) for x in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(batch):
        out[i, : len(ids)] = ids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=192)
    ap.add_argument("--n-ff", type=int, default=768)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_texts = read_texts(data_dir / "train.jsonl")
    val_texts = read_texts(data_dir / "val.jsonl")
    rng.shuffle(train_texts)

    tok_model = train_sentencepiece(train_texts + val_texts, out_dir, args.vocab_size)
    tokenizer = LlamaTokenizer(str(tok_model))
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.n_embd,
        intermediate_size=args.n_ff,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_heads,
        max_position_embeddings=args.max_len,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )

    model = LlamaForCausalLM(config)
    model.to(args.device)
    model.train()

    dataset = JsonlDataset(train_texts, tokenizer, args.max_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    while step < args.steps:
        for batch in loader:
            batch = batch.to(args.device)
            out = model(input_ids=batch, labels=batch)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 25 == 0:
                print(f"step {step} loss {loss.item():.4f}")
            step += 1
            if step >= args.steps:
                break

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved model to {out_dir}")
    print("convert to gguf:")
    print(f"  python3 convert_hf_to_gguf.py {out_dir} --outfile {out_dir}/tiny.gguf --outtype f16")


if __name__ == "__main__":
    main()
