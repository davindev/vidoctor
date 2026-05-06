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

### Filler — ASR 한계와 음향 분석 실패

#### 어려웠던 점 1: 반복 검출의 가정 붕괴

초기엔 Shriberg(1994) disfluency 분류를 따라 **"인접 동일어 반복 = disfluency"** 가정으로
사전 외 단어 반복도 filler로 등록. vlog 검증에서 이 가정이 무너짐.

```
검출된 18개 이벤트 중:
  반복 12개 — "소금아! 소금아!", "짜잔! 짜잔!", "강아지 강아지" …
                강아지 영상의 명령·호명·강조성 반복
  사전 매칭 6개 — "이제" × 3, "좀" × 3
  명확형 (어/음/그/저) 0개
```

vlog의 인접 반복은 머뭇거림이 아니라 **의도된 강조/명령**. 강의에서도 강조용 반복이 자연스럽다는
사용자 직관과 일치.

#### 개선 방법 1

**사전 외 단어 반복 검출 제거.** 사전 단어가 인접 반복되는 경우만 묶어서 단일 이벤트로 등록,
사전에 없는 단어는 반복되어도 무시.

- vlog filler FP 16 → 5 (−11)
- vlog F1 0.174 → 0.182 (소폭 상승), macro_f1 0.169 → 0.172
- lecture는 변동 없음 (반복 검출이 매칭에 기여 안 했음)

F1·macro_f1 상승 폭은 작지만 **사용자가 보는 "쓸데없는 finding" 11개 제거**가 실제 가치.

#### 어려웠던 점 2: ASR이 한국어 명확형 filler를 못 잡음

vlog 라벨 5개는 모두 "음·어·그·저 filler"인데 baseline에서 F1=0.182로 정체. 라벨 시간대에서
WhisperX(large-v3-turbo) transcribe 결과를 직접 확인:

| 라벨 | ASR transcribe |
|---|---|
| 99~106s "음·어" | 99~104s **무성**, 104부터 "있습니다." |
| 262~268s "음·어·저" | "안 힘든가 봐요. 날씨가 좀 흐릿한…" (다른 단어로 흡수) |
| 328~329s "음·어" | 라벨 1초 안에 단어 0개 (무성 처리) |
| 334~337s "음·어·그" | 라벨 3초 안에 단어 0개 (무성 처리) |
| 364~366s "음+더듬음" | "강아지?" (오인식) |

**전체 transcript 342 단어 중 명확형 사전 단어("어/음/그/저") 등장 0회.** WhisperX가 한국어
짧은 비언어 발음을 단어로 토큰화하지 않음. detector 단의 사전·임계 튜닝으로 해결 불가.

#### 개선 방법 2 (실패 기록)

ASR 단계의 시도와 결과:

1. **WhisperX `hotwords="음 어 으 에 그 저 뭐"`** — 디코더가 해당 단어 점수를 boost하는 옵션.
   - 결과: vlog FP 5 → **26 폭증** (정상 발음을 filler로 끌어당김), macro_f1 0.172 → 0.062. 채택 불가.

2. **VAD 임계 완화 `vad_onset 0.5→0.3, vad_offset 0.363→0.2`** — 짧은 발화를 무성 처리하지
   않게.
   - 결과: filler 검출 변화 0. VAD가 잡아도 ASR이 토큰화 안 함을 확인. **VAD 단계 문제가 아님**.

3. **모델 교체 `large-v3-turbo → large-v3`** — 디코더 4 layer → 32 layer.
   - 결과: lecture 무변, vlog cps 약간 악화, **filler 변화 0**. 모델 디코더 정밀도가 아닌
     **학습 분포의 결정**임을 확정.

→ **Whisper 가족 어떤 변형도 한국어 짧은 명확형 filler를 토큰화하지 않음**. 모델 자체의 한계.

#### 개선 방법 3 (실패 기록): 음향 신호 분석

ASR 우회 시도. **VAD ∩ ASR 단어 없는 시점 + 음향 검증**으로 명확형 filler 후보 추출.

- v0 (RMS 에너지만): lecture FP 4 → 21, vlog FP 5 → 42 폭증. 단어 사이 호흡·자음 클로저까지
  filler로 등록.
- v0.5 (RMS + F0 분산 검증, librosa.pyin): lecture FP 21 → 10, vlog FP 42 → 24. 일부 노이즈
  거름. 다만 baseline 대비 여전히 나쁨. **recall은 두 시도 모두 0 변화** — 음향이 추가로 잡은
  라벨이 0개.

라벨 5개의 실제 음향 측정값으로 진단:

| # | 라벨 | F0 std | voiced ratio | 통과 여부 |
|---|---|---|---|---|
| 1 | 99~101 (음·어) | 56Hz | 0.90 | ✗ (F0 임계 30Hz 초과) |
| 2 | 262~268 (음·어·저) | 50Hz | 0.67 | ✗ (F0 임계 초과 + ASR 단어가 그 시간대 채움) |
| 3 | 326~328 (음·어) | 11Hz | 0.54 | ✓ |
| 4 | 334~337 (음·어·그) | 54Hz | 0.54 | ✗ (다중 filler라 피치 변동 큼) |
| 5 | 364~365 (음+더듬음) | 14Hz | 0.28 | ✗ (더듬음 unvoiced 비중 큼) |

**라벨 패턴이 단순 명확형이 아니라 혼합형** (filler + 말 더듬음, 여러 filler 연속, ASR 단어와
시간 겹침). 알고리즘은 "단조로운 nasal hum"을 가정했으나 실제 라벨은 그 가정에 맞지 않음.

#### 고민되는 지점

- **임계 완화로 라벨을 잡으면 lecture FP 폭증.** F0 임계 30→60Hz, voiced ratio 0.5→0.3로
  완화하면 일부 라벨 통과 가능하지만 lecture FP가 이미 10개에서 더 늘어 baseline(4)을 더 멀어짐.
  precision-recall 트레이드 균형점이 데이터 안에 없음.
- **다양한 라벨 패턴을 한 알고리즘으로 커버하기 어려움.** filler+더듬음, 다중 filler 연속,
  ASR 단어와 겹침은 각각 다른 음향 특성을 보임. 각 패턴을 따로 검출하려면 phoneme classifier
  수준의 ML 모델이 필요. MVP 범위 초과.
- **라벨 자체의 검증 한계.** 사용자가 들어서 "음·어"라 라벨링해도 ASR 모델 입장에선 음향이
  애매할 수 있음. 라벨러 ≥ 2명 + Cohen's κ 측정으로 라벨 일관성을 먼저 확인하는 게 다음 시도의
  전제.
- **외부 한국어 ASR API 또는 한국어 fine-tuned Whisper 모델**은 시도 안 함. WhisperX 호환성
  검증과 비용·코드 변경이 큰 작업이라 현재 MVP에선 보류.

#### 결론

**Filler 명확형 검출은 v1.0의 알려진 한계.** ASR이 토큰화 못 하는 짧은 한국어 비언어 발음은
detector 단에서 우회 어려움. 다음 후보:

1. 한국어 특화 ASR로 교체 (Naver Clova / Google STT 한국어 / HuggingFace 한국어 fine-tuned
   Whisper)
2. ML 기반 phoneme classifier 차원 신규 도입 (반나절+ 작업)
3. 라벨러 다수화로 라벨 자체 정밀도 검증

이 셋 중 어느 것도 단순 임계 튜닝이 아니라 **별도 큰 작업**이라 MVP 이후로 미룸.

### Filler — Stage 2 한국어 fine-tuned ASR로 lecture 천장 돌파

위 결론에서 v1.1 후속으로 미뤘던 "한국어 특화 ASR 교체"를 실제 진행한 기록.

#### 변경 두 축

1. **사전 정리 (Stage 1)**: `"그러니까/그래서"`는 강의의 논리 연결사로 정상 사용되어 라벨러가
   filler로 보지 않음 → 사전에서 제외. `"자"`는 한국어 구어의 주의 환기 표지("자, 이제…")로
   누락돼 있어 추가.
2. **ASR 모델 swap (Stage 2)**: `large-v3-turbo`(영어 우세) → `Jungwonchang/whisper_finetune_ksponspeech_partial`
   (KsponSpeech 한국어 자유발화 코퍼스로 large-v2를 fine-tune한 모델)을 ctranslate2로
   변환해 WhisperX flow에 swap. 환경변수 `VIDOCTOR_WHISPER_MODEL`로 ASR backend 분기 가능.
3. **평가 tolerance 도입**: `match_points_in_intervals`에 `tolerance` 인자 추가, filler 평가는
   `FILLER_TOLERANCE_SEC=1.0` 적용. 라벨러가 영상 플레이어 1초 단위로 라운딩한 정밀도와
   ASR ±20ms 정밀도 격차를 평가 단계에서 흡수.

#### 정량 결과 (MLflow `vidoctor-filler` experiment, 6 + 4 runs)

| 영상 | baseline | stage 1 (사전 정리) | stage 2 (한국어 ASR) | stage 2 + tol=1 |
|---|---|---|---|---|
| **lecture** | P=0.333 R=0.667 F1=0.444 | P=0.500 R=0.333 F1=0.400 | P=0.667 R=0.667 F1=0.667 | **P=1.000 R=1.000 F1=1.000** |
| **vlog** | P=0.167 R=0.200 F1=0.182 | P=0.167 R=0.200 F1=0.182 | P=0.062 R=0.200 F1=0.095 | P=0.214 R=0.600 F1=0.316 |

- **Lecture는 PERFECT 도달**. 한국어 fine-tuned 모델이 baseline에서 토큰화 못 하던
  `음·자·어·이제·약간·뭐랄까`를 모두 잡고, tolerance가 라벨 ±1s 정밀도 한계를 흡수.
- **vlog F1 0.095 → 0.316 (3.3배)**. Recall 0.200 → 0.600. 라벨 인접 검출이 정당하게 매칭.
- **Stage 1 baseline 깎임은 점수 도금 해제**. baseline R=0.667이 실은 `"그래서"`가 라벨
  시간대에 우연히 들어가 TP로 잡힌 것 — Stage 1에서 그 우연 매칭 사라지며 진짜 R=0.333 노출.

#### Stage 1 자체의 한계 — 데이터 기반 정당화

Stage 1만으론 vlog 천장 못 뚫음을 *라벨 영역의 ASR 토큰을 직접 dump*해 확인:

| vlog 라벨 | baseline ASR 토큰 (large-v3-turbo) |
|---|---|
| 99~101s "음·어" | (no tokens — 무성 처리) |
| 262~268s "음·어·저" | "안 힘든가 봐요. 날씨가 좀 흐릿한…" (다른 단어 흡수) |
| 326~328s "음·어" | (no tokens) |
| 334~337s "음·어·그" | (no tokens) |
| 364~365s "음+더듬음" | "강아지?" (오인식) |

→ **라벨 5개 중 4개에서 ASR 토큰 자체가 0**. 사전 튜닝으론 닿을 수 없는 영역이 데이터로 확정.

#### Stage 2의 vlog 환각 — 사용자 청취 검증

Stage 2의 vlog P가 0.167 → 0.214로만 오른 이유: **한국어 fine-tuned 모델이 일상 발화의
짧은 호흡·자음 클로저·강세 변화를 "음/어/저"로 *환각 토큰화***. 실 영상 청취로 3 후보 영역
모두 진짜 filler 없음을 확인:

- 17~20s `"저"` 단발 → 정상 발화
- 95~96s `"음·어"` → 그 시간대 발화 자체 없음 (ASR 환각)
- 340~344s `"음·음·그·음·에"` 5연속 → 일상 대화 음향에서 환각

이 사용자 청취 검증으로 **vlog F1 0.316은 라벨 누락이 아니라 모델 환각 천장**임이 확정됨.
"라벨 보강으로 풀 수 있는 문제"가 아닌 "모델 도메인 적응 문제"로 분류 정정.

#### 카테고리별 모델 양분의 시사점 (v1.1 후보)

| | 모델 효과 | 다음 액션 후보 |
|---|---|---|
| **lecture** | 한국어 fine-tuned가 압도적 (F1 1.0) | 만족. 더 짜낼 영역 X |
| **vlog** | 한국어 fine-tuned가 *과민감* | 카테고리별 ASR 분기 (vlog는 baseline 또는 다른 fine-tuned) / 음향 신호 검증 차원으로 환각 거름 / Cohen's κ 라벨러 다수화 |

v1.0 시점에선 **lecture 정량 PERFECT + vlog 환각 천장 정직 회고**가 현 결과. v1.1에서는
카테고리별 ASR 모델 분기 또는 phoneme classifier 차원 도입으로 vlog 환각 거르기.

#### 공통 교훈 (이 단계에서 추가)

- **모델 swap 효과의 90%, tolerance 효과의 10%지만 평가 신뢰도 핵심**. 라벨러 사람 정밀도와
  모델 ±20ms 정밀도 격차는 *모델 개선*이 아니라 *평가 정책*에서 흡수.
- **baseline 점수가 도금일 수 있다**. 우연 매칭이 R을 부풀린 케이스가 사전 정리에서 드러남.
  점수 하락이 "퇴보"가 아니라 "정직화"인 케이스 존재.
- **ASR 환각은 라벨 누락처럼 보일 수 있다**. 사용자 청취 같은 *제3자 검증*이 없으면 "우리가
  못 본 라벨"로 오해할 위험. ML 회고에서 모델 출력 = 정답 가정의 경계 직접 확인 필요.

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
