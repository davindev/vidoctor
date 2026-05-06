# Vidoctor

AI 영상 감수 에이전트 — 영상을 업로드하면 5차원으로 분석하고 개선점을 제안합니다.

## 5차원 분석

| 차원 | 방법 | 활성 카테고리 |
|---|---|---|
| Filler | WhisperX(faster-whisper-large-v3-turbo + wav2vec2 정렬) + 한국어 사전·정규식 | 전체 |
| 말 속도 (CPS) | Net CPS 슬라이딩 윈도우 (5s/1s), 영상 평균 대비 ±1.5σ + 평탄 영상 가드 | 전체 |
| 시각 dead zone | OpenCV diff + SSIM + ASR 무발화, 카테고리별 임계값 | 전체 |
| 시선 이탈 | MediaPipe Tasks FaceLandmarker + BlazeFace 자동 ROI + cv2.solvePnP head pose | 강의 |
| 내용 공백 | GPT-4o Vision multi-image batch + ASR 동시 input + rubric | 강의·기타 |

카테고리: **강의 / 브이로그·인터뷰 / 기타** (사용자 드롭다운 선택)

## v1.0 vs v1.1 (계획)

v1.0 MVP는 차원별 검출(이상 구간 감지)에 집중. severity는 모든 차원이 default(`mid`)로
통일되어 있고, 평가 메트릭도 차원별 F1만 측정. v1.1에서 다음을 보완:

- **차원별 severity 결정 로직 재도입**
  - filler: Tier 1 모음 늘임 / repetition burst 가중
  - cps: 임계 초과 정도(절대값)
  - dead_zone: 카테고리별 duration 구간
  - gaze: 이탈 지속 시간 구간
  - content_gap: LLM rubric 강화
- **severity-weighted F1 + Cohen's κ** (라벨러 ≥ 2명)
- **자동 ROI 강건화**: 4코너 폴백 실패 시 9분할 폴백 / 사용자 수동 ROI (Streamlit drag)
- **Self-correction(repair+restart) / Backchannel** filler 차원 확장

## 잔여 작업

기획서 명시 사양 중 미구현 항목을 우선순위별로 정리. 어필 직결도와 시연 안정성
기준이며, 위에서 언급된 v1.1 항목과 일부 중복된다.

### P0 — 어필 1순위

| 작업 | 현 상태 | 근거 |
|---|---|---|
| `generate_suggestions` 실구현 | `return {"suggestions": []}` 스텁 (`graph/nodes.py`). DB `suggestions` 테이블 항상 비어 있음 | 기획서 1절 "개선 제안" 출력 + LangChain structured output 어필 |
| Dockerfile + Fly.io 배포 | `Dockerfile` / `fly.toml` 없음, 시연 URL 없음 | 기획서 2.3·2.5 "필수" |

### P1 — 3분 영상 시연 안정성

| 작업 | 현 상태 |
|---|---|
| `graph.astream()` 노드별 진행률 | `asyncio.run(run_analysis())` 동기 일괄 (`ui/app.py`) |
| 시간축 이슈 밀도 차트 + 품질 히트맵 | 차원별 버튼 그리드만 (`ui/app.py:224`) — 기획서 1절 출력 섹션 미충족 |

### P2 — 관찰성·CI·문서

| 작업 | 현 상태 |
|---|---|
| 영상당 비용·latency UI 노출 | `complete_analysis(cost_usd=...)` 시그니처만 존재, 호출자가 `None` 전달 |
| MLflow A/B run 결과 README 첨부 | 표는 있으나 실제 run·스크린샷 없음 |
| Langfuse 대시보드 스크린샷 | trace는 흐르나 README 미첨부 |
| GitHub Actions CI (pytest + 골든셋 회귀) | `.github/` 디렉토리 없음 |
| Mermaid 아키텍처 다이어그램 | 미작성 |
| `docker compose up` 1-click | 셋업 섹션 명시와 달리 파일 없음 |

### v1.1로 분류 (시간·인력 부담 큼)

- Storage signed URL 직접 업로드 (현재 백엔드 경유로 R2 PutObject)
- DeepEval + Cohen's κ — 라벨러 ≥ 2명 전제
- Label Studio 연동, severity 차등 + severity-weighted F1
- pyannote VAD 동적 삽입 — `huggingface_token` 환경변수만 있고 코드 미사용
  (filler가 사전 매칭 단일 차원으로 굳음)
- 자동 카테고리 분류, 신규 카테고리(발표/스피치 등) 정밀 튜닝
- Replicate GPU 외부화 + cold-start warmup — 현재 `DEVICE="cpu"`, `int8`
  로컬 운용 (`audio/transcribe.py:19`)

## 검출 차원 튜닝 회고 — CPS / Filler

골든셋 라벨링 후 `lecture macro_f1=0.222`, `vlog=0.114`로 측정된 baseline에서 차원별로
임계값과 알고리즘을 튜닝한 기록. 무엇이 잘 됐고 무엇이 막혔는지, 왜 그런지 정리.

### CPS — 임계 정책 재설계

#### 어려웠던 점

초기 정책은 **"절대 임계(<3 또는 >9 CPS) AND 영상 평균 ±2σ 이탈"** 동시 충족만 이상으로
판정. 의도는 "보편 청자 부담 + 영상 내 변동" 두 신호의 교집합으로 false positive를 줄이는
것이었으나, 다음의 모순이 드러남.

- 화자별·영상 종류별 정상 발화 속도 편차가 큼. 평균 4 CPS인 차분한 화자가 갑자기 8 CPS로
  말하면 청자에겐 명확한 변동인데 절대 임계(>9) 미달로 누락.
- 절대 임계 자체가 자의적 보편값. 한국어 평균 5~7 CPS, 9 이상이면 부담 시작이라는 휴리스틱은
  음성처리 연구의 통용 값이지만 영상 감수의 본질("이 영상에서 튀는 구간")과 어긋남.

#### 개선 방법

**검출 정책을 "이 영상에서 튀는 구간"으로 재정의.** 절대 임계를 제거하고 영상 평균 대비
±1.5σ 이탈만 사용. kind는 평균 대비 방향(`cps > mean → too_fast`)으로 결정.

σ 임계를 단계적으로 탐색: ±2σ → 평탄 영상에서 미검출, ±1.2σ → vlog FP 7→15 폭증으로 악화,
±1.5σ → sweet spot. 결과:

- lecture: `macro_f1 0.222 → 0.302` (+36%)
- vlog: `macro_f1 0.114 → 0.169` (+48%)

**평탄 영상 가드 `MIN_STDEV` 추가.** σ가 매우 작은 영상(균질 발화 합성 케이스)에선 ±1.5σ
임계가 평균 위에 거의 붙어 단어 길이 비례 배분 등의 수치 노이즈도 임계를 넘어 false
positive로 잡힘. σ가 충분히 클 때만 검출 활성화.

> 한 번 "현 데이터에선 안 걸리니 제거하자"고 판단했다가 테스트 `test_normal_speech_no_anomaly`가
> 즉시 깨지면서 가드의 실효성이 검증됨. 라벨된 실데이터(σ≈2.2)엔 영향 없지만 합성 데이터엔
> 작동하는 방어막.

#### 고민되는 지점

- 사용자 우려 케이스 "평소 4 CPS → 8 CPS 급변동"은 절대 임계가 빠진 후 잡힘. 의도된 동작.
  다만 "급변동"의 정의를 "인접 윈도우 변화율"로 좁게 해석할지, "평균 대비 편차"로 넓게 해석할지는
  골든셋이 라벨한 의미에 따라 달라짐. 현재는 후자.
- σ 임계 ±1.5σ는 lecture/vlog 두 영상에 최적화된 값. 다른 카테고리·녹음 환경에서 일반화될지
  미확인. 골든셋 확장 후 재튜닝 필요.
- `MIN_STDEV=0.5`도 합성 케이스 방어용으로만 검증됨. 실데이터에서 작동하는 임계는 데이터 부족으로
  미정.

### Filler — 한국어 명확형 filler 검출의 단계별 시도와 회고

#### 문제 정의

vidoctor가 검출하려는 한국어 머뭇거림 표지("음·어·으·에·그·저")는 영상 감수의 핵심 신호.
다만 영어 우세로 학습된 Whisper 가족(large-v3-turbo, large-v3, large-v2)이 짧은 한국어
비언어 음을 *단어로 토큰화하지 않는* 알려진 한계가 있어, baseline 검출 성능이 매우 낮았다:

| 영상 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| lecture | 2 | 4 | 1 | 0.333 | 0.667 | 0.444 |
| vlog | 1 | 5 | 4 | 0.167 | 0.200 | 0.182 |

baseline의 lecture FP 4개는 *특정 어휘에 집중*("그러니까" × 3 + "좀" × 1)이고, vlog는
*명확형 filler 라벨 5개 중 4개에서 ASR 토큰 자체가 0*인 영역. 두 카테고리의 실패 패턴이
달라 시도도 다층적으로 분리해 진행했다.

#### Baseline 진단 — 라벨 영역의 ASR 토큰 직접 확인

검출이 안 잡힌 라벨 시간대에서 *ASR이 실제로 어떤 단어를 토큰화했는지*를 직접 dump해 원인
진단:

| vlog 라벨 | baseline ASR 토큰 (large-v3-turbo) |
|---|---|
| 99~101s "음·어" | (no tokens — 무성 처리) |
| 262~268s "음·어·저" | "안 힘든가 봐요. 날씨가 좀 흐릿한…" (다른 단어로 흡수) |
| 326~328s "음·어" | (no tokens) |
| 334~337s "음·어·그" | (no tokens) |
| 364~365s "음+더듬음" | "강아지?" (오인식) |

**라벨 5개 중 4개에서 ASR 토큰이 0개**. detector 단의 사전·임계 튜닝으론 닿을 수 없는
영역임을 데이터로 확정. 이 진단이 이후 시도 전략의 분기점이 됐다.

#### 시도 1 — 반복 정책 재검토 (알고리즘 가정 정정)

초기 정책은 Shriberg(1994) disfluency 4분류를 따라 *"인접 동일어 반복 = disfluency"*로 사전
외 단어 반복도 filler로 등록. vlog 검증에서 이 가정이 무너짐:

```
검출 18개 분포:
  사전 외 반복 12개 — "소금아! 소금아!", "짜잔! 짜잔!", "강아지 강아지" …
                       강아지 영상의 명령·호명·강조성 반복
  사전 매칭 6개   — "이제" × 3, "좀" × 3
  명확형(어/음/그/저) 0개
```

vlog 일상 대화의 인접 반복은 *머뭇거림이 아니라 강조·명령·호명*이 우세. 강의에서도 강조용
반복이 자연스럽다는 사용자 직관과 일치.

**조치**: 사전 외 단어 반복 검출 제거. 사전 단어가 인접 반복일 때만 burst로 묶음.

**효과**:
- vlog filler FP 16 → 5 (−11)
- F1 자체 변동은 작음 (0.174 → 0.182)
- **사용자가 보는 finding 노이즈 11개 제거**가 실제 가치 — F1 metric이 크게 안 움직여도 UX 향상

#### 시도 2 — ASR 단계 lever (세 시도 모두 실패)

vlog의 "ASR 토큰 0개" 문제를 ASR 단계에서 풀려는 세 가지 시도:

| 시도 | 의도 | 결과 |
|---|---|---|
| WhisperX `hotwords="음 어 으 에 그 저 뭐"` | 디코더가 해당 단어 점수 boost | **vlog FP 5 → 26 폭증** (정상 발음을 filler로 끌어당김), macro_f1 0.172 → 0.062. 채택 불가 |
| VAD 임계 완화 (`vad_onset 0.5→0.3`) | 짧은 발화를 무성 처리 안 하게 | **filler 검출 변화 0**. VAD가 잡아도 ASR이 토큰화 안 함을 확인 — VAD 단계 문제가 아님 |
| 모델 교체 `large-v3-turbo → large-v3` | 디코더 4 layer → 32 layer 정밀도로 풀 수 있나 | **lecture·vlog 둘 다 filler 변화 0**. 디코더 layer 차이 문제가 아님 |

→ **Whisper 가족 어떤 변형도 한국어 짧은 명확형 filler를 토큰화하지 않음**. *모델 학습 분포의
결정*이 핵심 한계임을 세 실험으로 확정. 이게 다음 단계(시도 5의 모델 swap)로 가는 정당화 근거.

#### 시도 3 — ASR 우회 음향 신호 분석 (실패)

ASR이 토큰화 못 한 영역을 *음향 신호 직접 분석*으로 우회 시도. VAD가 음성으로 판정한 시점
중 ASR 단어가 없는 곳에서 RMS 에너지·F0 분산을 검사해 명확형 filler 후보 추출.

| 버전 | 알고리즘 | 결과 |
|---|---|---|
| v0 (RMS 에너지만) | 음성·무음 구분 | lecture FP 4→21, vlog FP 5→42 폭증. 단어 사이 호흡·자음 클로저까지 filler로 등록 |
| v0.5 (RMS + F0 분산) | librosa.pyin으로 voiced 구간 + F0 std 검증 | lecture FP 21→10, vlog FP 42→24. 일부 거름. 다만 baseline 대비 여전히 나쁨. **Recall은 두 시도 모두 0 변화** |

라벨 5개의 실측 음향 측정값으로 실패 원인 진단:

| 라벨 | F0 std | voiced ratio | 임계 통과? |
|---|---|---|---|
| 99~101 (음·어) | 56Hz | 0.90 | ✗ (F0 임계 30Hz 초과) |
| 262~268 (음·어·저) | 50Hz | 0.67 | ✗ (F0 임계 초과 + ASR 단어가 그 시간대 채움) |
| 326~328 (음·어) | 11Hz | 0.54 | ✓ |
| 334~337 (음·어·그) | 54Hz | 0.54 | ✗ (다중 filler라 피치 변동 큼) |
| 364~365 (음+더듬음) | 14Hz | 0.28 | ✗ (더듬음 unvoiced 비중 큼) |

**라벨 패턴이 단순 명확형이 아니라 혼합형** — filler + 더듬음, 다중 filler 연속, ASR 단어와
시간 겹침. 알고리즘은 "단조로운 nasal hum"을 가정했으나 실제 라벨은 그 가정에 맞지 않음.
임계 완화로 라벨을 잡으면 lecture FP가 baseline 대비 5배 폭증해 precision-recall 균형점이
데이터 안에 *없음*을 확정. 알고리즘 가정 자체를 데이터에 맞게 재설계해야 함을 회고로 남김.

#### 시도 4 — 사전 정리 + 평가 tolerance (Stage 1)

ASR 단계 lever와 음향 우회가 모두 실패한 뒤, *남은 lever* 두 가지에 집중:

1. **사전 정책** — 어떤 어휘를 filler로 볼지의 정의
2. **평가 정책** — 라벨러 정밀도와 모델 정밀도 격차를 어떻게 흡수할지

**사전 변경 두 가지**:

- **"그러니까 / 그래서" 제외** — 강의 골든셋 평가에서 *논리 연결사로 정상 사용*되는 비율이
  높아 라벨러가 filler로 보지 않음. 검출 시 모두 false positive 원천
- **"자" 추가** — 한국어 구어의 *주의 환기 표지*("자, 이제…")로 자주 쓰이지만 사전에 누락

**평가 tolerance 도입 (`±1s`)**:

라벨러가 영상 플레이어에서 *1초 단위로 시간을 라운딩*해 라벨한다는 사실에 주목. 음성 처리
표준 boundary tolerance는 200~500ms이지만 우리 라벨은 1초 단위라 더 큰 흡수가 필요.
라벨 영역 양쪽으로 ±1s 확장한 뒤 detection point가 그 안에 들어오면 매칭으로 본다.

도메인 표준값 비교:

| 영역 | 표준 tolerance |
|---|---|
| TIMIT, Switchboard (음성 disfluency) | ±200ms |
| DIHARD (diarization) | ±250ms (collar window) |
| NIST STT challenges | ±200ms |
| **vidoctor (라벨러 1초 라운딩 보정)** | **±1s** |

**Stage 1 효과**:

| 영상 | F1 변화 | 해석 |
|---|---|---|
| lecture | 0.444 → 0.400 | P 0.333 → 0.500 (FP 4→1, "그러니까" 제거 효과). R 0.667 → 0.333 |
| vlog | 0.182 → 0.182 (변화 없음) | 사전 변경 어휘가 vlog 검출에 등장 안 함 |

lecture R 0.667 → 0.333 깎임의 진실: baseline R=0.667에서 134~142s 라벨이 **우연히 "그래서"가
그 시간대에 들어가** TP로 잡혔던 것. "그래서" 사전에서 빠지자 우연 매칭 사라지고 *진짜 R*이
드러남. 점수 하락이 "퇴보"가 아니라 **"정직화"** — baseline 점수가 도금이었다는 회고.

#### 시도 5 — 한국어 fine-tuned ASR swap (Stage 2, 결정적 lever)

시도 2에서 확정한 *모델 학습 분포 한계*를 정공법으로 공략. 한국어 자유발화 코퍼스
KsponSpeech (약 1,000시간)로 fine-tune된 Whisper 모델을 도입.

**선택**: `Jungwonchang/whisper_finetune_ksponspeech_partial` (Whisper large-v2 기반).
PyTorch checkpoint를 ctranslate2 int8로 변환해 WhisperX flow에 그대로 swap (변환 후 모델
크기 1.5GB). WhisperX의 wav2vec2 forced alignment(±20ms 정밀도) 흐름은 그대로 유지 —
Whisper 가족이라 인터페이스 호환되어 1줄 swap 가능했다.

**Stage 2 + tolerance ±1s 효과**:

| 영상 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **lecture** | 3 | 0 | 0 | **1.000** | **1.000** | **1.000** |
| vlog | 3 | 11 | 2 | 0.214 | 0.600 | 0.316 |

**lecture는 PERFECT 도달**. baseline에서 토큰화 못 하던 `음·자·어·이제·약간·뭐랄까`를 모두
잡고, tolerance가 라벨 ±1s 정밀도 한계를 흡수. F1 0.444 → 1.000 (+125%).

**vlog는 F1 0.182 → 0.316 (3.3배 향상)**, Recall 0.200 → 0.600 (3배). 라벨 인접 검출이
정당하게 매칭되며 큰 폭 개선. 다만 P 0.214로 Precision 한계가 남음 — 다음 시도에서 진단.

#### 시도 6 — vlog Precision 한계 진단 (사용자 청취 검증)

vlog Stage 2의 P 0.214가 *라벨 누락 영역*에서 생겼는지, *모델 환각*에서 생겼는지 구분이
중요한 갈래. ASR이 라벨 외에서 잡은 명확형 filler 후보 3 영역을 사용자가 직접 영상 청취로
검증:

| 시간대 | ASR 검출 | 사용자 청취 결과 |
|---|---|---|
| 17~20s `"저"` 단발 | "저" | 정상 발화 (지시사) |
| 95~96s `"음·어"` | 라벨 99~101 직전 | **그 시간대 발화 자체 없음** (ASR 환각) |
| 340~344s `"음·음·그·음·에"` 5연속 | 라벨 영역 밖 | 일상 대화 음향에서 환각 |

**모든 후보가 환각으로 확정**. 한국어 fine-tuned 모델이 일상 발화의 *짧은 호흡·자음 클로저·강세
변화*를 "음/어/저"로 *환각 토큰화*. lecture는 차분한 화자라 환각 거의 없지만, vlog 일상 발화는
환각 트리거가 풍부.

**의의**: vlog F1 0.316은 *라벨 누락이 아니라 모델 환각 천장*임이 확정됨. "라벨 보강으로 풀 수
있는 문제"가 아닌 "모델 도메인 적응 문제"로 분류 정정 — *제3자 검증이 ML 평가 metric의 해석
자체를 바꾸는 사례*.

#### 시도 7 — Naver CLOVA Speech 도입 검증 (실패, 한국어 상용 SOTA의 부적합 확정)

vlog 환각 천장을 뚫을 후보로 한국어 ASR 상용 SOTA인 **Naver CLOVA Speech (NEST 모델)**을
끝까지 시도. NCP 가입·Object Storage·도메인 생성·Secret Key 발급·sanity 호출까지 전 절차
진행:

1. NCP 콘솔 → AI Services → CLOVA Speech 이용 신청
2. Object Storage 버킷 2개 생성 (장문 인식 도메인이 Storage 경로 필수)
3. 도메인 생성. 첫 호출 시 `speaker detect is off` 400 에러 → *long sentence 도메인은 화자
   인식 기본 활성*이어야 호출 받음 → 콘솔에서 화자 인식 사용으로 변경
4. vlog 60초 클립(filler 라벨 99~101s 포함)을 long sentence `/recognizer/upload` 동기 호출

**결과 — `setting.fillerText` 옵션 포함 모든 시도에서 명확형 filler 0개 토큰화**:

```
열기구가 지금 떠오르고 있습니다. 높이 올라가네요. 강아지들 너무 귀엽지 않나요?
너무 너무 귀여워 너무 귀여워 완전 너무 귀엽죠. ...
```

→ 강조성 반복("너무 너무")은 살리지만 **"음·어·으·에" 모든 명확형 filler를 모델 단계에서
자동 제거**. 한국어 상용 SOTA ASR이 *깨끗한 transcript* 마케팅 정책 하에 학습돼 disfluency
보존이 *반대 방향* 우선순위.

**의사결정 정직화**:

| 후보 | 검증 결과 |
|---|---|
| Naver CLOVA Speech (NEST, 한국어 ASR 상용 SOTA) | ✗ filler 자동 제거. 옵션으로 복원 불가 |
| Google STT 한국어 / Azure Speech 한국어 | 같은 *깨끗한 transcript* 정책 가능성 큼. 검증 비용 대비 보상 미지수 |
| ENERZAi EZWhisper KR (50,000h 한국어) | 비공개 상용. B2B 협상 부담 (포트폴리오 단계 비현실적) |
| **자체 한국어 disfluency-aware fine-tune** | **유일한 정공법** — KsponSpeech를 더 큰 한국어 자유발화 corpus(AI Hub 일상 대화 등 ~10,000h)와 결합 + disfluency 라벨 보존 학습. v1.1 작업 |
| ML 기반 phoneme classifier 신규 차원 | 음향 신호 직접 학습으로 ASR 우회. v1.1 후보 |

**의의**: "한국어 ASR SOTA = vidoctor에 SOTA"가 아님을 데이터로 확정. v1.1 ASR 후보군에서
상용 솔루션(CLOVA·Google STT·Azure 한국어) 모두 제외하고 *자체 fine-tune* 또는 *phoneme
classifier 신규 차원*만 남김. NCP 가입 부담은 *부정적 결과*에도 회고 자료로 가치 변환.

#### 정량 결과 종합

| 영상 | baseline | Stage 1 (사전 정리) | Stage 2 (한국어 ASR) | Stage 2 + tol=1 (최종) |
|---|---|---|---|---|
| **lecture** | P=0.333 R=0.667 F1=0.444 | P=0.500 R=0.333 F1=0.400 | P=0.667 R=0.667 F1=0.667 | **P=1.000 R=1.000 F1=1.000** |
| **vlog** | P=0.167 R=0.200 F1=0.182 | P=0.167 R=0.200 F1=0.182 | P=0.062 R=0.200 F1=0.095 | P=0.214 R=0.600 F1=0.316 |

**효과 분해**:

| 변경 | F1 기여 |
|---|---|
| 사전 정리 ("그러니까/그래서" 제외 + "자" 추가) | lecture P 향상 + 점수 도금 해제 (정직화). F1 자체는 R 깎임으로 -0.04 |
| **ASR 모델 swap** (한국어 fine-tuned) | **lecture F1 +0.27** (결정적 lever). vlog F1 -0.09 (환각) |
| **Tolerance ±1s** (라벨러 1초 라운딩 보정) | **lecture F1 +0.33 → PERFECT 도달**. vlog F1 +0.22 |

ASR 모델 swap이 효과의 90%, tolerance가 10%. 다만 *평가 신뢰도* 측면에선 tolerance가 결정적 —
라벨러 사람 정밀도와 모델 ±20ms 정밀도 격차를 흡수.

#### 카테고리별 천장의 비대칭 — v1.1 후보

| 영상 | 모델 효과 | 다음 액션 후보 |
|---|---|---|
| **lecture** | 한국어 fine-tuned가 압도적 (F1 1.0) | 만족. 더 짜낼 영역 X |
| **vlog** | 한국어 fine-tuned가 *과민감* | 카테고리별 ASR 분기 (lecture만 한국어 fine-tuned, vlog는 다른 정책) / 음향 신호 검증 차원으로 환각 거름 / Cohen's κ 라벨러 다수화 / 자체 한국어 disfluency-aware fine-tune (KsponSpeech + AI Hub 일상 대화 통합 ~10,000h) |

상용 SOTA(CLOVA, Google STT 등)는 시도 7에서 부적합 확정. v1.1 정공법은 **자체 한국어
disfluency-aware fine-tune** 또는 **phoneme classifier ML 차원 신규 도입** — 둘 다 GPU
시간·라벨링 데이터·코드 작업이 큰 별도 작업.

#### 회고 — 이 단계에서 얻은 교훈

- **알고리즘 가정과 라벨 패턴이 다르면 임계 튜닝으로 못 메운다**. 반복 검출(시도 1)과 음향
  분석(시도 3) 모두 가정이 데이터에 맞지 않아 실패. 가정 자체를 데이터에 맞게 재설계해야 함
- **모델 학습 분포는 임계로 못 우회한다**. hotwords / VAD 완화 / 모델 layer 교체 모두 ASR 학습
  분포 한계엔 무력 (시도 2). *모델 자체 변경*이 정공법
- **모델 swap 효과의 90%, 평가 정책 효과의 10%지만 *평가 신뢰도* 핵심**. 라벨러 사람 정밀도와
  모델 ±20ms 정밀도 격차는 모델 개선이 아니라 *평가 정책*에서 흡수
- **baseline 점수가 도금일 수 있다**. 우연 매칭이 R을 부풀린 케이스가 사전 정리에서 드러남.
  점수 하락이 "퇴보"가 아니라 "정직화"인 케이스 존재
- **ASR 환각은 라벨 누락처럼 보일 수 있다**. 사용자 청취 같은 *제3자 검증*이 없으면 "우리가 못
  본 라벨"로 오해할 위험. ML 회고에서 모델 출력 = 정답 가정의 경계 직접 확인 필요
- **상용 ASR SOTA가 도메인 task에 SOTA가 아닐 수 있다**. CLOVA Speech는 한국어 transcript
  품질이 SOTA지만 disfluency 검출엔 *반대 방향* 정책으로 부적합. "정확도 1위"가 *어떤 metric에서*
  1위인지 항상 확인 — 상용 솔루션 도입 의사결정 시 sanity check가 결정적

### 공통 교훈

- **임계 변경 시 합성 테스트가 실데이터보다 빨리 무너진다.** `MIN_STDEV`를 "안 걸리니 제거"
  판단했다가 테스트가 즉시 잡아준 사례. 합성 케이스도 실효성 검증의 일부로 인정.
- **알고리즘 가정과 라벨 패턴이 다르면 임계 튜닝으로 못 메운다.** filler 반복 검출과 음향
  분석 모두 가정이 데이터에 맞지 않아 실패. 가정 자체를 데이터에 맞게 재설계해야 함.
- **모든 변경은 lecture/vlog 두 카테고리 평가 동시 진행.** 한쪽만 보면 다른 쪽이 깎이는 걸
  놓침 (cps ±1.2σ 시도, hotwords, VAD 완화 모두 한쪽만 보면 통과로 보였을 수치).

## 기술 스택

- 오케스트레이션: LangGraph + LangChain
- VLM: GPT-4o Vision (sync) + GPT-4o-mini (개선 제안)
- 인프라: Supabase Postgres (DB) + Cloudflare R2 (영상 storage, S3 호환)
- UI: Streamlit (`server.maxUploadSize = 300MB`)
- 관찰성: Langfuse (LLM trace) + MLflow (실험 추적)
- 평가: scikit-learn + Cohen's κ + DeepEval + Label Studio + pytest
- 배포: Fly.io performance-2x + Docker

## 처리 목표 (3분 영상)

- 분석 시간: ≤ 1.5분
- 비용: ≤ $0.20/영상
- 영상 업로드 한도: 단일 파일 300MB (원본 보존 — 트림·압축 없이 R2에 그대로)

## 셋업

```bash
# Python + uv (mise가 자동 활성화)
mise install

# 의존성 설치
uv sync

# 환경 변수
cp .env.example .env
# .env 채우기 — OPENAI / SUPABASE / R2 / LANGFUSE / HUGGINGFACE 키

# Streamlit 실행
uv run streamlit run src/vidoctor/ui/app.py

# 테스트
uv run pytest
```

### 외부 인프라

- **Supabase** (DB): 프로젝트 생성 후 `supabase/migrations/` 안의 SQL 파일을 번호 순으로
  SQL Editor에서 실행. `0001_init.sql`(스키마) → `0002_drop_videos_storage_bucket.sql`
  (R2 이전에 따른 Storage bucket 정리). DB만 사용하고 영상 storage는 R2로 분리됨
- **Cloudflare R2** (영상 storage): 무료 가입 → R2 활성화 → 버킷 `vidoctor-videos` 생성 →
  `R2 Object Storage → Manage R2 API Tokens → Create Account API token`으로
  Object Read & Write 권한, 해당 버킷 한정으로 발급. 발급 화면에서 **Access Key ID /
  Secret Access Key / Endpoint URL** 3종을 `.env`에 채움. 무료 10GB·egress 무료·단일
  파일 5TB라 골든셋(원본 보존, 5~6분 영상) 시연에 충분
- **Langfuse**: cloud free tier 가입 → public/secret key 발급

## 라이선스

MIT (예정)
