# my-own-ai-web

`my-own-ai`(파이썬/NumPy 버전)와 완전히 같은 걸 만드는 **순수 JS 버전**. `index.html`
파일 하나만 있으면 되고, 깃허브 Pages에 그대로 올리면 바로 돌아간다. Ollama도, 서버도,
API 키도 필요 없다 — 토크나이저부터 트랜스포머까지 전부 브라우저 안에서 처음부터
학습된다.

## 먼저 정직하게: 여기서도 "이해"는 안 된다

HTML로 바꾸고 깃허브에 올린다고 로드맵의 데이터/컴퓨팅 요구량이 바뀌지 않는다.
지금 이 버전도 `my-own-ai`와 똑같이 **암기 수준**이다. 자세한 내용과 실제 결과는
`my-own-ai/README.md`, 장기 계획은 `my-own-ai/ROADMAP.md`를 봐라. 진짜 이해 단계로
가려면 `gpu-version/`(아래 설명)으로 GPU를 써서 데이터를 훨씬 많이 학습시켜야 한다.

## 빠른 시작 (로그인 없이 테스트)

`index.html`을 그냥 더블클릭해서 브라우저로 열면 바로 된다 — `auth.js`의
`firebaseConfig`를 아직 안 채웠으면 로그인 화면 없이 바로 앱이 뜬다 (개발/테스트용).

## 실제로 쓸 때: 순서

### 1. Firebase 로그인 설정 (선택 사항이지만 요청한 기능)

1. [console.firebase.google.com](https://console.firebase.google.com) 접속, 구글 계정으로 로그인
2. "프로젝트 추가" → 이름 아무거나 → 프로젝트 생성
3. 왼쪽 메뉴 "빌드 → Authentication" → "시작하기" → "Google" 로그인 방법 사용 설정
4. 왼쪽 위 톱니바퀴(프로젝트 설정) → 아래로 스크롤 → "웹 앱 추가"(</> 아이콘)
5. 나오는 `firebaseConfig` 객체를 통째로 복사
6. `index.html`을 텍스트 에디터로 열어서 `const firebaseConfig = {{...}}` 부분을
   방금 복사한 값으로 교체
7. Firebase 콘솔의 Authentication → Settings → "승인된 도메인"에 네 깃허브
   Pages 주소(예: `사용자명.github.io`)를 추가해야 로그인이 실제로 동작한다

### 2. 깃허브 Pages로 올리기

1. 깃허브에 새 저장소 생성 (예: `my-own-ai-web`)
2. `index.html`을 그 저장소에 업로드 (커밋)
3. 저장소 Settings → Pages → Source에서 브랜치를 `main`, 폴더를 `/ (root)`로 설정 → Save
4. 몇 분 뒤 `https://사용자명.github.io/my-own-ai-web/` 에서 접속 가능

### 3. study 파일로 자동 학습시키기

1. 아무 공개 저장소에 `study.txt` (또는 원하는 이름) 파일을 만들고, 학습시키고
   싶은 텍스트/코드를 넣어라
2. 그 파일 페이지에서 "Raw" 버튼을 눌러 나오는 URL을 복사
   (`https://raw.githubusercontent.com/사용자명/저장소/main/study.txt` 형태)
3. `index.html` 앱의 "깃허브 study 파일 자동 학습" 섹션에 그 URL을 붙여넣고
   "지금 가져와서 학습" 클릭
4. "자동으로 주기 확인"을 켜두면 페이지가 열려있는 동안 주기적으로 `study.txt`가
   바뀌었는지 확인해서 바뀐 부분만 추가로 학습한다

**중요**: 이건 정적 페이지라서 브라우저 탭을 닫으면 자동 확인도 멈춘다. "실시간"은
"페이지가 열려있는 동안"이라는 뜻이지, 서버가 24시간 도는 게 아니다.

## 데이터는 어디에 저장되나

전부 브라우저의 `localStorage`에 저장된다 (그 컴퓨터의 그 브라우저에만 남음,
깃허브 서버에 올라가는 게 아님):
- 학습된 모델 가중치
- 토크나이저
- study 파일에서 가져온 누적 텍스트
- 이미 반영한 내용의 해시(중복 방지용)

기기를 바꾸면 처음부터 다시 학습해야 한다. "위험 구역"의 초기화 버튼을 누르면
전부 지우고 새로 시작한다.

## gpu-version/ — 진짜 이해 단계로 가는 방법

`gpu-version/train_gpu.py`는 같은 아키텍처를 **PyTorch**로 포팅한 것. NumPy/JS
버전과 수학적으로 동일하지만, backward를 손으로 안 짜도 되고(autograd), GPU가
있으면 수백 배 빠르다. 로드맵의 2단계(GPU 필요 시점)로 넘어갈 때 쓰는 파일이다.

**Google Colab에서 무료로 돌리는 법:**
1. [colab.research.google.com](https://colab.research.google.com) 새 노트북
2. 런타임 → 런타임 유형 변경 → T4 GPU 선택
3. 왼쪽 폴더 아이콘 → `train_gpu.py`, `tokenizer.py`, `data/` 폴더를 통째로 업로드
4. 셀에 입력: `!python train_gpu.py`

지금 `gpu-version/data/`에는 `my-own-ai`와 똑같은 14만 자짜리 예제 코퍼스가 들어있다.
**진짜 이해 단계로 가려면 이 데이터를 훨씬 늘려야 한다** — `ROADMAP.md`에서 말한
1GB+ 기준, 그리고 Karpathy의 nanochat 벤치마크($300/12시간 = GPT-2급)를 참고해라.
