"""
바이트 레벨 BPE(Byte Pair Encoding) 토크나이저를 순수 파이썬으로 직접 구현.
남이 학습시킨 토크나이저를 갖다 쓰는 게 아니라, 우리 코퍼스로 병합 규칙(merge rule)을
처음부터 학습시킨다. 바이트 단위로 시작하기 때문에 한글이든 영어든 코드든
어떤 유니코드 텍스트가 와도 별도 처리 없이 그대로 동작한다.
"""
import re
import json
from collections import Counter

_SPLIT_PATTERN = re.compile(r"\s+|\S+")


def _pretokenize(text):
    return _SPLIT_PATTERN.findall(text)


class ByteBPETokenizer:
    def __init__(self):
        self.merges = []  # [(id1, id2, new_id), ...] 학습된 순서대로
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size=768, verbose=True):
        chunks = [list(chunk.encode("utf-8")) for chunk in _pretokenize(text)]
        num_merges = max(0, vocab_size - 256)

        for step in range(num_merges):
            pair_counts = Counter()
            for chunk in chunks:
                for a, b in zip(chunk, chunk[1:]):
                    pair_counts[(a, b)] += 1
            if not pair_counts:
                break
            best_pair, freq = pair_counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + step
            self.merges.append((best_pair[0], best_pair[1], new_id))
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            new_chunks = []
            for chunk in chunks:
                merged = []
                i = 0
                while i < len(chunk):
                    if (
                        i < len(chunk) - 1
                        and chunk[i] == best_pair[0]
                        and chunk[i + 1] == best_pair[1]
                    ):
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(chunk[i])
                        i += 1
                new_chunks.append(merged)
            chunks = new_chunks

            if verbose and (step + 1) % 100 == 0:
                print(f"  BPE 학습 중... {step + 1}/{num_merges} 병합 완료 (vocab={self.vocab_size})")

        if verbose:
            print(f"토크나이저 학습 완료: vocab_size={self.vocab_size}")

    def encode(self, text):
        merge_rank = {(a, b): new_id for a, b, new_id in self.merges}
        out = []
        for chunk in _pretokenize(text):
            tokens = list(chunk.encode("utf-8"))
            while len(tokens) >= 2:
                pairs = list(zip(tokens, tokens[1:]))
                # 학습 순서상 가장 먼저 배운 병합을 우선 적용
                candidates = [(merge_rank[p], i) for i, p in enumerate(pairs) if p in merge_rank]
                if not candidates:
                    break
                # merge_rank의 값(new_id)이 작을수록 먼저 학습된 규칙 -> 우선 적용
                candidates.sort(key=lambda t: t[0])
                new_id, i = candidates[0]
                tokens = tokens[:i] + [new_id] + tokens[i + 2:]
            out.extend(tokens)
        return out

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "merges": self.merges,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for a, b, new_id in data["merges"]:
            tok.merges.append((a, b, new_id))
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok
