// 배치 없이 시퀀스 하나씩 처리하는 model.js/nn.js 위에서, 여러 시퀀스의 그래디언트를
// 누적(평균)한 뒤 옵티마이저를 한 번만 적용하는 미니배치 학습 스텝.

function cloneGrad(g) {
  return Array.isArray(g[0]) ? g.map((row) => row.slice()) : g.slice();
}
function addGradInPlace(acc, g) {
  if (Array.isArray(acc[0])) {
    for (let i = 0; i < acc.length; i++) for (let j = 0; j < acc[i].length; j++) acc[i][j] += g[i][j];
  } else {
    for (let i = 0; i < acc.length; i++) acc[i] += g[i];
  }
}
function scaleGradInPlace(acc, s) {
  if (Array.isArray(acc[0])) {
    for (const row of acc) for (let j = 0; j < row.length; j++) row[j] *= s;
  } else {
    for (let i = 0; i < acc.length; i++) acc[i] *= s;
  }
}

/**
 * batchSize개의 시퀀스에 대해 forward+backward를 각각 실행하고 그래디언트를
 * 누적/평균한 뒤 optimizer.step()을 한 번만 호출한다. sampleFn()은 [x, y]
 * (입력 토큰 배열, 정답 토큰 배열)를 반환해야 한다.
 */
function trainStep(model, batchSize, sampleFn, optimizer) {
  let paramRefs = null;
  let accum = null;
  let totalLoss = 0;

  for (let b = 0; b < batchSize; b++) {
    const [x, y] = sampleFn();
    totalLoss += model.loss(x, y);
    const params = model.params();
    if (!accum) {
      paramRefs = params.map(([name, p]) => [name, p]);
      accum = params.map(([, , g]) => cloneGrad(g));
    } else {
      params.forEach(([, , g], idx) => addGradInPlace(accum[idx], g));
    }
  }

  for (const acc of accum) scaleGradInPlace(acc, 1 / batchSize);
  optimizer.step(paramRefs.map(([name, p], idx) => [name, p, accum[idx]]));
  return totalLoss / batchSize;
}

if (typeof module !== "undefined") module.exports = { trainStep };
