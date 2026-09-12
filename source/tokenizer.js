// 바이트 레벨 BPE 토크나이저 - 순수 JS (외부 라이브러리 없음)
// Python의 tokenizer.py와 동일한 알고리즘. UTF-8 바이트 단위로 시작해서
// 자주 나오는 바이트 쌍을 병합해나간다. 한글도 바이트 단위라 자동으로 처리됨.

function textToBytes(text) {
  return Array.from(new TextEncoder().encode(text));
}

function bytesToText(bytes) {
  return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
}

function pretokenize(text) {
  // 공백 덩어리 vs 비공백 덩어리로 분리 (병합이 공백 경계를 넘지 않도록)
  return text.match(/\s+|\S+/g) || [];
}

class ByteBPETokenizer {
  constructor() {
    this.merges = []; // [[a,b,newId], ...] 학습된 순서대로
    this.vocab = new Map(); // id -> Uint8Array(bytes)
    for (let i = 0; i < 256; i++) this.vocab.set(i, new Uint8Array([i]));
  }

  get vocabSize() {
    return this.vocab.size;
  }

  train(text, vocabSize = 800, onProgress = null) {
    let chunks = pretokenize(text).map((c) => textToBytes(c));
    const numMerges = Math.max(0, vocabSize - 256);

    for (let step = 0; step < numMerges; step++) {
      const pairCounts = new Map();
      for (const chunk of chunks) {
        for (let i = 0; i < chunk.length - 1; i++) {
          const key = chunk[i] * 100000 + chunk[i + 1];
          pairCounts.set(key, (pairCounts.get(key) || 0) + 1);
        }
      }
      if (pairCounts.size === 0) break;

      let bestKey = null;
      let bestCount = 0;
      for (const [key, count] of pairCounts) {
        if (count > bestCount) {
          bestCount = count;
          bestKey = key;
        }
      }
      if (bestCount < 2) break;

      const a = Math.floor(bestKey / 100000);
      const b = bestKey % 100000;
      const newId = 256 + step;
      this.merges.push([a, b, newId]);
      const merged = new Uint8Array(this.vocab.get(a).length + this.vocab.get(b).length);
      merged.set(this.vocab.get(a), 0);
      merged.set(this.vocab.get(b), this.vocab.get(a).length);
      this.vocab.set(newId, merged);

      const newChunks = [];
      for (const chunk of chunks) {
        const merged2 = [];
        let i = 0;
        while (i < chunk.length) {
          if (i < chunk.length - 1 && chunk[i] === a && chunk[i + 1] === b) {
            merged2.push(newId);
            i += 2;
          } else {
            merged2.push(chunk[i]);
            i += 1;
          }
        }
        newChunks.push(merged2);
      }
      chunks = newChunks;

      if (onProgress && (step + 1) % 100 === 0) onProgress(step + 1, numMerges, this.vocabSize);
    }
  }

  encode(text) {
    const mergeRank = new Map();
    for (const [a, b, newId] of this.merges) mergeRank.set(a * 100000 + b, newId);

    const out = [];
    for (const chunkText of pretokenize(text)) {
      let tokens = textToBytes(chunkText);
      while (tokens.length >= 2) {
        let bestIdx = -1;
        let bestNewId = Infinity;
        for (let i = 0; i < tokens.length - 1; i++) {
          const key = tokens[i] * 100000 + tokens[i + 1];
          if (mergeRank.has(key)) {
            const newId = mergeRank.get(key);
            if (newId < bestNewId) {
              bestNewId = newId;
              bestIdx = i;
            }
          }
        }
        if (bestIdx === -1) break;
        tokens = tokens.slice(0, bestIdx).concat([bestNewId], tokens.slice(bestIdx + 2));
      }
      out.push(...tokens);
    }
    return out;
  }

  decode(ids) {
    let totalLen = 0;
    for (const id of ids) totalLen += this.vocab.get(id).length;
    const bytes = new Uint8Array(totalLen);
    let offset = 0;
    for (const id of ids) {
      const b = this.vocab.get(id);
      bytes.set(b, offset);
      offset += b.length;
    }
    return bytesToText(bytes);
  }

  toJSON() {
    return { merges: this.merges, vocabSize: this.vocabSize };
  }

  static fromJSON(data) {
    const tok = new ByteBPETokenizer();
    for (const [a, b, newId] of data.merges) {
      tok.merges.push([a, b, newId]);
      const merged = new Uint8Array(tok.vocab.get(a).length + tok.vocab.get(b).length);
      merged.set(tok.vocab.get(a), 0);
      merged.set(tok.vocab.get(b), tok.vocab.get(a).length);
      tok.vocab.set(newId, merged);
    }
    return tok;
  }
}

if (typeof module !== "undefined") module.exports = { ByteBPETokenizer, textToBytes, bytesToText };
