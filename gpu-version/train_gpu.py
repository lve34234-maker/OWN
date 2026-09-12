"""
같은 아키텍처(토크나이저 + GPT 트랜스포머)를 PyTorch로 포팅한 버전.
NumPy/JS 버전과 다른 점: backward()를 손으로 안 짜도 된다 — PyTorch의 autograd가
자동으로 계산해준다. 대신 GPU(CUDA)가 있으면 수백 배 빠르게 돌아간다.

이건 "사전학습된 가중치를 빌리는" 게 아니다 — 여전히 네 데이터로 처음부터
학습한다. 그냥 계산 도구(NumPy -> PyTorch)와 실행 장치(CPU -> GPU)만 바뀐 것이다.

실행 방법:
  로컬에 GPU가 있다면:
    pip install torch
    python train_gpu.py

  Google Colab (무료 GPU, GPU 없어도 됨):
    1. colab.research.google.com 에서 새 노트북
    2. 상단 메뉴 런타임 -> 런타임 유형 변경 -> T4 GPU 선택
    3. 이 파일과 data/ 폴더를 업로드 (왼쪽 파일 아이콘에서 드래그)
    4. 노트북 셀에 !python train_gpu.py 입력하고 실행
"""
import os
import glob
import math
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# 기존 NumPy 버전(my-own-ai/src/tokenizer.py)과 100% 동일한 로직의 BPE 토크나이저.
from tokenizer import ByteBPETokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"장치: {DEVICE}" + (" (GPU 없음 - Colab에서 런타임을 GPU로 바꾸는 걸 추천)" if DEVICE == "cpu" else ""))

# ---- 하이퍼파라미터: 로드맵 2단계용으로 CPU 버전보다 훨씬 키웠다 ----
# 데이터가 충분히 늘어났다면 (ROADMAP.md 참고, 100MB~) 이 숫자들을 더 키워도 된다.
config = dict(
    vocab_size=4000,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    batch_size=32,
    max_iters=3000,
    eval_interval=200,
    eval_iters=50,
    learning_rate=3e-4,
    weight_decay=0.01,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.qkv = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.resid_drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        # PyTorch 2.x의 scaled_dot_product_attention이 causal mask + softmax를 한 번에,
        # 알아서 GPU에 최적화된 커널로 처리해준다 (우리가 numpy로 손수 짰던 부분).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * cfg["n_embd"], cfg["n_embd"]),
            nn.Dropout(cfg["dropout"]),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def load_corpus():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not paths:
        raise SystemExit(f"{DATA_DIR} 에 .txt 코퍼스 파일이 없습니다. my-own-ai/data 내용을 복사하거나 심볼릭 링크하세요.")
    texts = [open(p, encoding="utf-8").read() for p in paths]
    print(f"코퍼스 파일 {len(paths)}개 로드")
    return "\n".join(texts)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg["eval_iters"])
        for i in range(cfg["eval_iters"]):
            x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], DEVICE)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    corpus = load_corpus()
    print(f"코퍼스 길이: {len(corpus):,}자")

    tok_path = os.path.join(CKPT_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = ByteBPETokenizer.load(tok_path)
    else:
        tok = ByteBPETokenizer()
        tok.train(corpus, vocab_size=config["vocab_size"])
        tok.save(tok_path)

    data = torch.tensor(tok.encode(corpus), dtype=torch.long)
    n = int(len(data) * 0.95)
    train_data, val_data = data[:n], data[n:]
    print(f"토큰 개수: {len(data):,} (vocab_size={tok.vocab_size})")

    cfg = {**config, "vocab_size": tok.vocab_size}
    model = GPT(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 개수: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    t0 = time.time()
    for it in range(1, cfg["max_iters"] + 1):
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % cfg["eval_interval"] == 0 or it == 1:
            losses = estimate_loss(model, train_data, val_data, cfg)
            print(f"step {it:5d}/{cfg['max_iters']} | train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time()-t0:.1f}s")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "model.pt"))
    with open(os.path.join(CKPT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f)
    print("학습 완료. checkpoints/model.pt 에 저장됨.")

    # 짧은 생성 샘플 몇 개 찍어보기
    model.eval()
    for prompt in ["def ", "class Player", "안녕"]:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
        out = model.generate(ids, max_new_tokens=50)
        print(f"\n=== {prompt!r} ===")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
