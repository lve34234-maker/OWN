"""
source/ 안의 JS 조각들 + 코퍼스 데이터를 하나로 합쳐서 ../index.html 을 만든다.
JS 파일을 수정했으면 이 스크립트를 다시 실행해서 index.html을 재생성해야 반영된다.

사용법: python3 assemble.py
"""
import re
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "gpu-version", "data")  # 이 패키지 안에 이미 들어있는 코퍼스
OUT_PATH = os.path.join(HERE, "..", "index.html")


def read(name):
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
        return f.read()


def escape_script_close(code):
    # HTML 파서는 문자열 안이든 밖이든 "</script"를 보면 그 자리에서 태그를 닫아버린다.
    # 코퍼스 예제 안에 실제 <script>...</script> 텍스트가 들어있어서 이스케이프가 필요하다.
    return re.sub(r"<(/script)", r"<\\\1", code, flags=re.IGNORECASE)


def strip_node_only_lines(code):
    lines = code.splitlines()
    return "\n".join(l for l in lines if not re.match(r'^\s*const \{.*\} = require\(', l))


def load_corpus():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not paths:
        raise SystemExit(f"코퍼스를 찾을 수 없습니다: {DATA_DIR}")
    texts = [open(p, encoding="utf-8").read() for p in paths]
    return "\n".join(texts)


import json

corpus = load_corpus()
corpus_json_str = escape_script_close(json.dumps(corpus, ensure_ascii=False))

tokenizer_js = escape_script_close(strip_node_only_lines(read("tokenizer.js")))
nn_js = escape_script_close(strip_node_only_lines(read("nn.js")))
model_js = escape_script_close(strip_node_only_lines(read("model.js")))
optimizer_js = escape_script_close(strip_node_only_lines(read("optimizer.js")))
train_js = escape_script_close(strip_node_only_lines(read("train.js")))
app_js = escape_script_close(read("app.js"))
auth_js = escape_script_close(read("auth.js"))

TEMPLATE_PATH = os.path.join(HERE, "template.html")
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    template = f.read()

html = (
    template
    .replace("__DEFAULT_CORPUS__", corpus_json_str)
    .replace("__TOKENIZER_JS__", tokenizer_js)
    .replace("__NN_JS__", nn_js)
    .replace("__MODEL_JS__", model_js)
    .replace("__OPTIMIZER_JS__", optimizer_js)
    .replace("__TRAIN_JS__", train_js)
    .replace("__APP_JS__", app_js)
    .replace("__AUTH_JS__", auth_js)
)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"생성 완료: {OUT_PATH} ({len(html.encode('utf-8')):,} bytes)")
