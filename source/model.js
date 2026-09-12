const { Linear, Embedding, LayerNorm, ReLUMLP, Dropout, CausalSelfAttention, softmaxCrossEntropy, addMat } = require("./nn.js");

class Block {
  constructor(dim, nHead, dropout = 0.1) {
    this.ln1 = new LayerNorm(dim);
    this.attn = new CausalSelfAttention(dim, nHead);
    this.drop1 = new Dropout(dropout);
    this.ln2 = new LayerNorm(dim);
    this.mlp = new ReLUMLP(dim);
    this.drop2 = new Dropout(dropout);
  }
  forward(x) {
    const a = this.drop1.forward(this.attn.forward(this.ln1.forward(x)));
    const x1 = addMat(x, a);
    const m = this.drop2.forward(this.mlp.forward(this.ln2.forward(x1)));
    return addMat(x1, m);
  }
  backward(dout) {
    const dx1 = addMat(dout, this.ln2.backward(this.mlp.backward(this.drop2.backward(dout))));
    const dx = addMat(dx1, this.ln1.backward(this.attn.backward(this.drop1.backward(dx1))));
    return dx;
  }
  params() {
    return [...this.ln1.params(), ...this.attn.params(), ...this.ln2.params(), ...this.mlp.params()];
  }
  setTraining(flag) {
    this.drop1.training = flag;
    this.drop2.training = flag;
  }
}

class GPT {
  constructor(vocabSize, blockSize, nLayer, nHead, nEmbd, dropout = 0.1) {
    this.vocabSize = vocabSize;
    this.blockSize = blockSize;
    this.tokEmb = new Embedding(vocabSize, nEmbd);
    this.posEmb = new Embedding(blockSize, nEmbd);
    this.embDrop = new Dropout(dropout);
    this.blocks = [];
    for (let i = 0; i < nLayer; i++) this.blocks.push(new Block(nEmbd, nHead, dropout));
    this.lnF = new LayerNorm(nEmbd);
    this.head = new Linear(nEmbd, vocabSize);
  }
  forward(idxArr) {
    const T = idxArr.length;
    const posArr = Array.from({ length: T }, (_, i) => i);
    const tok = this.tokEmb.forward(idxArr);
    const pos = this.posEmb.forward(posArr);
    let x = this.embDrop.forward(addMat(tok, pos));
    for (const blk of this.blocks) x = blk.forward(x);
    x = this.lnF.forward(x);
    return this.head.forward(x);
  }
  backward(dlogits) {
    let dx = this.head.backward(dlogits);
    dx = this.lnF.backward(dx);
    for (let i = this.blocks.length - 1; i >= 0; i--) dx = this.blocks[i].backward(dx);
    dx = this.embDrop.backward(dx);
    this.tokEmb.backward(dx);
    this.posEmb.backward(dx);
  }
  loss(idxArr, targets) {
    const logits = this.forward(idxArr);
    const { loss, backward } = softmaxCrossEntropy(logits, targets);
    this.backward(backward());
    return loss;
  }
  params() {
    let p = [...this.tokEmb.params(), ...this.posEmb.params()];
    for (const b of this.blocks) p = p.concat(b.params());
    p = p.concat(this.lnF.params(), this.head.params());
    return p;
  }
  numParameters() {
    let n = 0;
    for (const [, arr] of this.params()) {
      if (Array.isArray(arr[0])) for (const row of arr) n += row.length;
      else n += arr.length;
    }
    return n;
  }
  setTraining(flag) {
    this.embDrop.training = flag;
    for (const b of this.blocks) b.setTraining(flag);
  }
  generate(idxArr, maxNew, temperature = 0.8, topK = 20) {
    let idx = idxArr.slice();
    for (let step = 0; step < maxNew; step++) {
      const cond = idx.slice(Math.max(0, idx.length - this.blockSize));
      const logits = this.forward(cond);
      const lastLogits = logits[logits.length - 1].slice();
      for (let i = 0; i < lastLogits.length; i++) lastLogits[i] /= temperature;
      if (topK && topK < lastLogits.length) {
        const sorted = lastLogits.slice().sort((a, b) => b - a);
        const thresh = sorted[topK - 1];
        for (let i = 0; i < lastLogits.length; i++) if (lastLogits[i] < thresh) lastLogits[i] = -Infinity;
      }
      let maxv = -Infinity;
      for (const v of lastLogits) if (v > maxv) maxv = v;
      let sumExp = 0;
      const probs = new Array(lastLogits.length);
      for (let i = 0; i < lastLogits.length; i++) {
        const e = Math.exp(lastLogits[i] - maxv);
        probs[i] = e;
        sumExp += e;
      }
      for (let i = 0; i < probs.length; i++) probs[i] /= sumExp;
      let r = Math.random(),
        acc = 0,
        next = probs.length - 1;
      for (let i = 0; i < probs.length; i++) {
        acc += probs[i];
        if (r <= acc) {
          next = i;
          break;
        }
      }
      idx.push(next);
    }
    return idx;
  }
}

if (typeof module !== "undefined") module.exports = { Block, GPT };
