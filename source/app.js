// ===== 브라우저 전용 글루 코드 =====
// tokenizer.js / nn.js / model.js / optimizer.js / train.js 가 이미 로드되어
// ByteBPETokenizer, GPT, Adam, trainStep 이 전역에 존재한다고 가정한다.

const CONFIG = {
  vocabSize: 800,
  blockSize: 64,
  nLayer: 2,
  nHead: 2,
  nEmbd: 64,
  dropout: 0.1,
  weightDecay: 0.01,
  lr: 3e-4,
};

const LS_KEYS = {
  tokenizer: "myai_tokenizer_v1",
  model: "myai_model_v1",
  corpusExtra: "myai_corpus_extra_v1",
  ingestedHashes: "myai_ingested_hashes_v1",
  githubUrl: "myai_github_url_v1",
  meta: "myai_meta_v1",
};

// ---- 간단한 문자열 해시 (중복 편입 방지용, 암호학적 용도 아님) ----
function simpleHash(str) {
  let h1 = 0xdeadbeef,
    h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (h1 >>> 0).toString(16) + (h2 >>> 0).toString(16);
}

function safeGet(key) {
  try {
    const v = localStorage.getItem(key);
    return v ? JSON.parse(v) : null;
  } catch (e) {
    return null;
  }
}
function safeSet(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (e) {
    console.error("localStorage 저장 실패:", e);
    return false;
  }
}

// 저장 용량을 줄이기 위해 소수점을 6자리로 반올림
function roundArr(a) {
  if (Array.isArray(a[0])) return a.map((row) => row.map((v) => Math.round(v * 1e6) / 1e6));
  return a.map((v) => Math.round(v * 1e6) / 1e6);
}

function saveModel(model) {
  const values = model.params().map(([, p]) => roundArr(p));
  return safeSet(LS_KEYS.model, { config: CONFIG, values });
}
function loadModelInto(model, saved) {
  const params = model.params();
  params.forEach(([, p], idx) => {
    const src = saved.values[idx];
    if (Array.isArray(p[0])) {
      for (let i = 0; i < p.length; i++) for (let j = 0; j < p[i].length; j++) p[i][j] = src[i][j];
    } else {
      for (let i = 0; i < p.length; i++) p[i] = src[i];
    }
  });
}

// ===== 앱 상태 =====
let tokenizer = null;
let model = null;
let optimizer = null;
let corpusExtra = "";
let ingestedHashes = [];
let stopRequested = false;
let autoTimer = null;

const $ = (id) => document.getElementById(id);
function log(msg) {
  const el = $("log");
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}
function status(msg) {
  $("status").textContent = msg;
}

function fullCorpus() {
  return DEFAULT_CORPUS + (corpusExtra ? "\n" + corpusExtra : "");
}

function sampleFnFor(ids, blockSize) {
  return function () {
    const start = Math.floor(Math.random() * (ids.length - blockSize - 1));
    return [ids.slice(start, start + blockSize), ids.slice(start + 1, start + 1 + blockSize)];
  };
}

async function runTraining(steps, batchSize) {
  const ids = tokenizer.encode(fullCorpus());
  if (ids.length < CONFIG.blockSize + 2) {
    log("코퍼스가 너무 작아서 학습할 수 없습니다.");
    return;
  }
  const sampleFn = sampleFnFor(ids, CONFIG.blockSize);
  model.setTraining(true);
  stopRequested = false;
  const t0 = performance.now();
  for (let it = 1; it <= steps; it++) {
    if (stopRequested) {
      log(`중지됨 (${it}/${steps})`);
      break;
    }
    const loss = trainStep(model, batchSize, sampleFn, optimizer);
    if (it % 10 === 0 || it === 1 || it === steps) {
      const sec = ((performance.now() - t0) / 1000).toFixed(1);
      log(`step ${it}/${steps} | loss ${loss.toFixed(4)} | ${sec}s 경과`);
      status(`학습 중... ${it}/${steps}`);
    }
    // 브라우저가 멈추지 않도록 매 스텝마다 이벤트 루프에 양보
    await new Promise((r) => setTimeout(r, 0));
  }
  saveModel(model);
  const meta = safeGet(LS_KEYS.meta) || { totalSteps: 0 };
  meta.totalSteps = (meta.totalSteps || 0) + steps;
  meta.lastTrainedAt = new Date().toISOString();
  safeSet(LS_KEYS.meta, meta);
  status(`대기 중 (누적 스텝: ${meta.totalSteps})`);
}

async function initApp() {
  status("초기화 중...");

  const savedTok = safeGet(LS_KEYS.tokenizer);
  if (savedTok) {
    tokenizer = ByteBPETokenizer.fromJSON(savedTok);
    log("저장된 토크나이저를 불러왔습니다 (vocab=" + tokenizer.vocabSize + ")");
  } else {
    log("토크나이저를 처음부터 학습합니다...");
    await new Promise((r) => setTimeout(r, 0));
    tokenizer = new ByteBPETokenizer();
    tokenizer.train(DEFAULT_CORPUS, CONFIG.vocabSize);
    safeSet(LS_KEYS.tokenizer, tokenizer.toJSON());
    log("토크나이저 학습 완료 (vocab=" + tokenizer.vocabSize + ")");
  }

  corpusExtra = localStorage.getItem(LS_KEYS.corpusExtra) || "";
  ingestedHashes = safeGet(LS_KEYS.ingestedHashes) || [];
  const savedUrl = localStorage.getItem(LS_KEYS.githubUrl);
  if (savedUrl) $("githubUrl").value = savedUrl;

  model = new GPT(CONFIG.vocabSize, CONFIG.blockSize, CONFIG.nLayer, CONFIG.nHead, CONFIG.nEmbd, CONFIG.dropout);
  optimizer = new Adam({ lr: CONFIG.lr, weightDecay: CONFIG.weightDecay });

  const savedModel = safeGet(LS_KEYS.model);
  if (savedModel) {
    loadModelInto(model, savedModel);
    log("저장된 모델 가중치를 불러왔습니다 (파라미터 " + model.numParameters().toLocaleString() + "개)");
  } else {
    log("새 모델을 초기화했습니다 (파라미터 " + model.numParameters().toLocaleString() + "개, 아직 미학습)");
  }

  const meta = safeGet(LS_KEYS.meta);
  status(meta ? `대기 중 (누적 스텝: ${meta.totalSteps || 0})` : "대기 중 (학습 전)");
  $("corpusInfo").textContent = `기본 코퍼스 ${DEFAULT_CORPUS.length.toLocaleString()}자 + 추가된 데이터 ${corpusExtra.length.toLocaleString()}자`;
}

// ===== UI 이벤트 =====
function wireUI() {
  $("trainBtn").addEventListener("click", async () => {
    const steps = parseInt($("steps").value, 10) || 500;
    const batchSize = parseInt($("batchSize").value, 10) || 4;
    $("trainBtn").disabled = true;
    await runTraining(steps, batchSize);
    $("trainBtn").disabled = false;
  });

  $("stopBtn").addEventListener("click", () => {
    stopRequested = true;
  });

  $("genBtn").addEventListener("click", () => {
    const prompt = $("prompt").value || "";
    if (!tokenizer || !model) return;
    model.setTraining(false);
    const ids = tokenizer.encode(prompt);
    const out = model.generate(ids, 60, 0.8, 20);
    $("output").textContent = tokenizer.decode(out);
  });

  $("fetchBtn").addEventListener("click", async () => {
    const url = $("githubUrl").value.trim();
    if (!url) {
      log("깃허브 raw URL을 먼저 입력하세요.");
      return;
    }
    localStorage.setItem(LS_KEYS.githubUrl, url);
    await fetchAndLearn(url);
  });

  $("autoCheck").addEventListener("change", (e) => {
    if (e.target.checked) {
      const minutes = parseFloat($("autoMinutes").value) || 5;
      log(`자동 확인 켬 (${minutes}분마다)`);
      autoTimer = setInterval(() => {
        const url = $("githubUrl").value.trim();
        if (url) fetchAndLearn(url);
      }, minutes * 60 * 1000);
    } else {
      if (autoTimer) clearInterval(autoTimer);
      log("자동 확인 끔");
    }
  });

  $("resetBtn").addEventListener("click", () => {
    if (!confirm("저장된 모델/코퍼스/토크나이저를 전부 지우고 처음부터 다시 시작합니다. 계속할까요?")) return;
    for (const k of Object.values(LS_KEYS)) localStorage.removeItem(k);
    location.reload();
  });
}

async function fetchAndLearn(url) {
  status("깃허브에서 가져오는 중...");
  let text;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    text = await res.text();
  } catch (e) {
    log("가져오기 실패: " + e.message + " (raw.githubusercontent.com 링크인지, 저장소가 공개인지 확인하세요)");
    status("대기 중");
    return;
  }

  const hash = simpleHash(text);
  if (ingestedHashes.includes(hash)) {
    log("이미 반영된 내용입니다 (변경 없음). 학습을 건너뜁니다.");
    status("대기 중");
    return;
  }

  corpusExtra += "\n# ---study 파일 추가분 (" + new Date().toISOString() + ")---\n" + text;
  ingestedHashes.push(hash);
  localStorage.setItem(LS_KEYS.corpusExtra, corpusExtra);
  safeSet(LS_KEYS.ingestedHashes, ingestedHashes);
  $("corpusInfo").textContent = `기본 코퍼스 ${DEFAULT_CORPUS.length.toLocaleString()}자 + 추가된 데이터 ${corpusExtra.length.toLocaleString()}자`;

  log(`새 내용 ${text.length}자를 코퍼스에 반영. 추가 학습을 시작합니다...`);
  const incrementalSteps = parseInt($("autoSteps").value, 10) || 200;
  await runTraining(incrementalSteps, parseInt($("batchSize").value, 10) || 4);
  log("깃허브 반영 학습 완료.");
}

let appStarted = false;
async function startAppAfterLogin() {
  if (appStarted) return;
  appStarted = true;
  wireUI();
  await initApp();
}
window.startAppAfterLogin = startAppAfterLogin;

