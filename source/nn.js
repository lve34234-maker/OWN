// 신경망 레이어 - 순수 JS (NumPy 없이). 배치 없이 "시퀀스 하나(T x C 행렬)"
// 단위로 처리한다. 배치가 필요하면 바깥 학습 루프에서 여러 번 돌려서 그래디언트를
// 누적하는 방식으로 처리한다 (이 파일 안에서는 항상 2차원 배열 T x C를 다룬다).

function zerosMat(rows, cols) {
  const m = new Array(rows);
  for (let i = 0; i < rows; i++) m[i] = new Array(cols).fill(0);
  return m;
}
function zerosVec(n) {
  return new Array(n).fill(0);
}
function randMat(rows, cols, scale) {
  const m = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols);
    for (let j = 0; j < cols; j++) row[j] = (Math.random() * 2 - 1) * scale;
    m[i] = row;
  }
  return m;
}
function randnMat(rows, cols, scale) {
  const m = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols);
    for (let j = 0; j < cols; j++) {
      // Box-Muller
      const u1 = Math.random() || 1e-9;
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      row[j] = z * scale;
    }
    m[i] = row;
  }
  return m;
}
function addMat(A, B) {
  return A.map((row, t) => row.map((v, i) => v + B[t][i]));
}
function isMatrix(p) {
  return Array.isArray(p[0]);
}
function zerosLike(p) {
  return isMatrix(p) ? p.map((row) => row.map(() => 0)) : p.map(() => 0);
}

class Linear {
  constructor(inDim, outDim) {
    this.inDim = inDim;
    this.outDim = outDim;
    const scale = 1 / Math.sqrt(inDim);
    this.W = randMat(inDim, outDim, scale);
    this.b = zerosVec(outDim);
    this.dW = zerosMat(inDim, outDim);
    this.db = zerosVec(outDim);
  }
  forward(x) {
    this.x = x;
    const T = x.length;
    const y = new Array(T);
    for (let t = 0; t < T; t++) {
      const row = new Array(this.outDim);
      for (let j = 0; j < this.outDim; j++) {
        let s = this.b[j];
        for (let i = 0; i < this.inDim; i++) s += x[t][i] * this.W[i][j];
        row[j] = s;
      }
      y[t] = row;
    }
    return y;
  }
  backward(dy) {
    const T = dy.length;
    this.dW = zerosMat(this.inDim, this.outDim);
    this.db = zerosVec(this.outDim);
    const dx = new Array(T);
    for (let t = 0; t < T; t++) dx[t] = new Array(this.inDim).fill(0);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < this.outDim; j++) {
        const dyv = dy[t][j];
        this.db[j] += dyv;
        for (let i = 0; i < this.inDim; i++) {
          this.dW[i][j] += this.x[t][i] * dyv;
          dx[t][i] += dyv * this.W[i][j];
        }
      }
    }
    return dx;
  }
  params() {
    return [
      ["W", this.W, this.dW],
      ["b", this.b, this.db],
    ];
  }
}

class Embedding {
  constructor(numEmb, dim) {
    this.table = randnMat(numEmb, dim, 0.02);
    this.dtable = zerosMat(numEmb, dim);
    this.numEmb = numEmb;
    this.dim = dim;
  }
  forward(idxArr) {
    this.idx = idxArr;
    return idxArr.map((i) => this.table[i].slice());
  }
  backward(dy) {
    this.dtable = zerosMat(this.numEmb, this.dim);
    for (let t = 0; t < dy.length; t++) {
      const row = this.dtable[this.idx[t]];
      for (let c = 0; c < row.length; c++) row[c] += dy[t][c];
    }
  }
  params() {
    return [["table", this.table, this.dtable]];
  }
}

class LayerNorm {
  constructor(dim, eps = 1e-5) {
    this.dim = dim;
    this.eps = eps;
    this.gamma = new Array(dim).fill(1);
    this.beta = new Array(dim).fill(0);
    this.dgamma = new Array(dim).fill(0);
    this.dbeta = new Array(dim).fill(0);
  }
  forward(x) {
    const T = x.length,
      D = this.dim;
    this.xhat = new Array(T);
    this.ivar = new Array(T);
    const y = new Array(T);
    for (let t = 0; t < T; t++) {
      let mean = 0;
      for (let i = 0; i < D; i++) mean += x[t][i];
      mean /= D;
      let vari = 0;
      for (let i = 0; i < D; i++) {
        const d = x[t][i] - mean;
        vari += d * d;
      }
      vari /= D;
      const iv = 1 / Math.sqrt(vari + this.eps);
      this.ivar[t] = iv;
      const xhatRow = new Array(D);
      const yRow = new Array(D);
      for (let i = 0; i < D; i++) {
        const xh = (x[t][i] - mean) * iv;
        xhatRow[i] = xh;
        yRow[i] = xh * this.gamma[i] + this.beta[i];
      }
      this.xhat[t] = xhatRow;
      y[t] = yRow;
    }
    return y;
  }
  backward(dy) {
    const T = dy.length,
      D = this.dim;
    this.dgamma = new Array(D).fill(0);
    this.dbeta = new Array(D).fill(0);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < D; i++) {
        this.dgamma[i] += dy[t][i] * this.xhat[t][i];
        this.dbeta[i] += dy[t][i];
      }
    }
    const dx = new Array(T);
    for (let t = 0; t < T; t++) {
      const dxhat = new Array(D);
      for (let i = 0; i < D; i++) dxhat[i] = dy[t][i] * this.gamma[i];
      let meanDxhat = 0;
      for (let i = 0; i < D; i++) meanDxhat += dxhat[i];
      meanDxhat /= D;
      let meanDxhatXhat = 0;
      for (let i = 0; i < D; i++) meanDxhatXhat += dxhat[i] * this.xhat[t][i];
      meanDxhatXhat /= D;
      const row = new Array(D);
      for (let i = 0; i < D; i++) {
        row[i] = this.ivar[t] * (dxhat[i] - meanDxhat - this.xhat[t][i] * meanDxhatXhat);
      }
      dx[t] = row;
    }
    return dx;
  }
  params() {
    return [
      ["gamma", this.gamma, this.dgamma],
      ["beta", this.beta, this.dbeta],
    ];
  }
}

class ReLUMLP {
  constructor(dim, mult = 4) {
    this.fc1 = new Linear(dim, dim * mult);
    this.fc2 = new Linear(dim * mult, dim);
  }
  forward(x) {
    const h = this.fc1.forward(x);
    this.mask = h.map((row) => row.map((v) => (v > 0 ? 1 : 0)));
    const hRelu = h.map((row, t) => row.map((v, i) => (v > 0 ? v : 0)));
    return this.fc2.forward(hRelu);
  }
  backward(dout) {
    const dh = this.fc2.backward(dout);
    const dhRelu = dh.map((row, t) => row.map((v, i) => v * this.mask[t][i]));
    return this.fc1.backward(dhRelu);
  }
  params() {
    return [...this.fc1.params(), ...this.fc2.params()];
  }
}

class Dropout {
  constructor(p = 0.1) {
    this.p = p;
    this.training = true;
  }
  forward(x) {
    if (!this.training || this.p <= 0) {
      this.mask = null;
      return x;
    }
    const keep = 1 - this.p;
    this.mask = x.map((row) => row.map(() => (Math.random() > this.p ? 1 / keep : 0)));
    return x.map((row, t) => row.map((v, i) => v * this.mask[t][i]));
  }
  backward(dy) {
    if (!this.mask) return dy;
    return dy.map((row, t) => row.map((v, i) => v * this.mask[t][i]));
  }
  params() {
    return [];
  }
}

class CausalSelfAttention {
  constructor(dim, nHead) {
    this.nHead = nHead;
    this.headDim = dim / nHead;
    this.dim = dim;
    this.Wq = new Linear(dim, dim);
    this.Wk = new Linear(dim, dim);
    this.Wv = new Linear(dim, dim);
    this.Wo = new Linear(dim, dim);
  }
  forward(x) {
    const T = x.length,
      H = this.nHead,
      hd = this.headDim;
    const q = this.Wq.forward(x);
    const k = this.Wk.forward(x);
    const v = this.Wv.forward(x);
    this.q = q;
    this.k = k;
    this.v = v;
    this.T = T;

    const attnPerHead = [];
    const outHeads = [];
    const invSqrt = 1 / Math.sqrt(hd);

    for (let h = 0; h < H; h++) {
      const attn = new Array(T);
      for (let t = 0; t < T; t++) {
        const scores = new Array(t + 1);
        let maxv = -Infinity;
        for (let t2 = 0; t2 <= t; t2++) {
          let s = 0;
          for (let c = 0; c < hd; c++) s += q[t][h * hd + c] * k[t2][h * hd + c];
          s *= invSqrt;
          scores[t2] = s;
          if (s > maxv) maxv = s;
        }
        let sumExp = 0;
        const row = new Array(t + 1);
        for (let t2 = 0; t2 <= t; t2++) {
          const e = Math.exp(scores[t2] - maxv);
          row[t2] = e;
          sumExp += e;
        }
        for (let t2 = 0; t2 <= t; t2++) row[t2] /= sumExp;
        attn[t] = row; // length t+1 (t2 = 0..t)
      }
      attnPerHead.push(attn);

      const outH = new Array(T);
      for (let t = 0; t < T; t++) {
        const rowOut = new Array(hd).fill(0);
        for (let t2 = 0; t2 <= t; t2++) {
          const a = attn[t][t2];
          for (let c = 0; c < hd; c++) rowOut[c] += a * v[t2][h * hd + c];
        }
        outH[t] = rowOut;
      }
      outHeads.push(outH);
    }

    const merged = new Array(T);
    for (let t = 0; t < T; t++) {
      const row = new Array(this.dim);
      for (let h = 0; h < H; h++) {
        for (let c = 0; c < hd; c++) row[h * hd + c] = outHeads[h][t][c];
      }
      merged[t] = row;
    }
    this.attnPerHead = attnPerHead;
    return this.Wo.forward(merged);
  }
  backward(dout) {
    const T = this.T,
      H = this.nHead,
      hd = this.headDim;
    const dmerged = this.Wo.backward(dout);

    const dq = new Array(T),
      dk = new Array(T),
      dv = new Array(T);
    for (let t = 0; t < T; t++) {
      dq[t] = new Array(this.dim).fill(0);
      dk[t] = new Array(this.dim).fill(0);
      dv[t] = new Array(this.dim).fill(0);
    }
    const invSqrt = 1 / Math.sqrt(hd);

    for (let h = 0; h < H; h++) {
      const attn = this.attnPerHead[h];
      const dOutH = new Array(T);
      for (let t = 0; t < T; t++) {
        const row = new Array(hd);
        for (let c = 0; c < hd; c++) row[c] = dmerged[t][h * hd + c];
        dOutH[t] = row;
      }

      const dAttn = new Array(T);
      for (let t = 0; t < T; t++) {
        dAttn[t] = new Array(t + 1);
        for (let t2 = 0; t2 <= t; t2++) {
          let s = 0;
          for (let c = 0; c < hd; c++) s += dOutH[t][c] * this.v[t2][h * hd + c];
          dAttn[t][t2] = s;
        }
      }
      for (let t = 0; t < T; t++) {
        for (let t2 = 0; t2 <= t; t2++) {
          const a = attn[t][t2];
          for (let c = 0; c < hd; c++) dv[t2][h * hd + c] += a * dOutH[t][c];
        }
      }

      const dScores = new Array(T);
      for (let t = 0; t < T; t++) {
        let dot = 0;
        for (let t2 = 0; t2 <= t; t2++) dot += dAttn[t][t2] * attn[t][t2];
        dScores[t] = new Array(t + 1);
        for (let t2 = 0; t2 <= t; t2++) {
          dScores[t][t2] = attn[t][t2] * (dAttn[t][t2] - dot);
        }
      }
      for (let t = 0; t < T; t++) {
        for (let t2 = 0; t2 <= t; t2++) {
          const g = dScores[t][t2] * invSqrt;
          for (let c = 0; c < hd; c++) {
            dq[t][h * hd + c] += g * this.k[t2][h * hd + c];
            dk[t2][h * hd + c] += g * this.q[t][h * hd + c];
          }
        }
      }
    }

    const dxQ = this.Wq.backward(dq);
    const dxK = this.Wk.backward(dk);
    const dxV = this.Wv.backward(dv);
    const dx = new Array(T);
    for (let t = 0; t < T; t++) {
      dx[t] = new Array(dxQ[t].length);
      for (let i = 0; i < dx[t].length; i++) dx[t][i] = dxQ[t][i] + dxK[t][i] + dxV[t][i];
    }
    return dx;
  }
  params() {
    return [...this.Wq.params(), ...this.Wk.params(), ...this.Wv.params(), ...this.Wo.params()];
  }
}

function softmaxCrossEntropy(logits, targets) {
  const T = logits.length,
    V = logits[0].length;
  const probs = new Array(T);
  let loss = 0;
  for (let t = 0; t < T; t++) {
    let maxv = -Infinity;
    for (let v = 0; v < V; v++) if (logits[t][v] > maxv) maxv = logits[t][v];
    let sumExp = 0;
    const row = new Array(V);
    for (let v = 0; v < V; v++) {
      const e = Math.exp(logits[t][v] - maxv);
      row[v] = e;
      sumExp += e;
    }
    for (let v = 0; v < V; v++) row[v] /= sumExp;
    probs[t] = row;
    loss += -Math.log(row[targets[t]] + 1e-12);
  }
  loss /= T;
  function backward() {
    const dlogits = new Array(T);
    for (let t = 0; t < T; t++) {
      const row = probs[t].slice();
      row[targets[t]] -= 1;
      for (let v = 0; v < V; v++) row[v] /= T;
      dlogits[t] = row;
    }
    return dlogits;
  }
  return { loss, backward };
}

if (typeof module !== "undefined") {
  module.exports = {
    Linear,
    Embedding,
    LayerNorm,
    ReLUMLP,
    Dropout,
    CausalSelfAttention,
    softmaxCrossEntropy,
    addMat,
    isMatrix,
    zerosLike,
    zerosMat,
    zerosVec,
    randMat,
    randnMat,
  };
}
