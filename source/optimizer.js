const { isMatrix, zerosLike } = require("./nn.js");

class Adam {
  constructor({ lr = 3e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.0 } = {}) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.weightDecay = weightDecay;
    this.m = new Map();
    this.v = new Map();
    this.t = 0;
  }
  step(paramsList) {
    this.t += 1;
    for (const [, p, g] of paramsList) {
      if (!this.m.has(p)) {
        this.m.set(p, zerosLike(p));
        this.v.set(p, zerosLike(p));
      }
      const m = this.m.get(p),
        v = this.v.get(p);
      const wd = this.weightDecay;
      const applyWD = isMatrix(p) && wd > 0;
      const b1t = Math.pow(this.beta1, this.t);
      const b2t = Math.pow(this.beta2, this.t);
      if (isMatrix(p)) {
        for (let i = 0; i < p.length; i++) {
          for (let j = 0; j < p[i].length; j++) {
            m[i][j] = this.beta1 * m[i][j] + (1 - this.beta1) * g[i][j];
            v[i][j] = this.beta2 * v[i][j] + (1 - this.beta2) * g[i][j] * g[i][j];
            const mhat = m[i][j] / (1 - b1t);
            const vhat = v[i][j] / (1 - b2t);
            if (applyWD) p[i][j] -= this.lr * wd * p[i][j];
            p[i][j] -= (this.lr * mhat) / (Math.sqrt(vhat) + this.eps);
          }
        }
      } else {
        for (let i = 0; i < p.length; i++) {
          m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i];
          v[i] = this.beta2 * v[i] + (1 - this.beta2) * g[i] * g[i];
          const mhat = m[i] / (1 - b1t);
          const vhat = v[i] / (1 - b2t);
          p[i] -= (this.lr * mhat) / (Math.sqrt(vhat) + this.eps);
        }
      }
    }
  }
}

if (typeof module !== "undefined") module.exports = { Adam };
