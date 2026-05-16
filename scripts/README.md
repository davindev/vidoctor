# 평가·스모크 스크립트

`src/vidoctor/`가 *프로덕션 파이프라인*이라면 `scripts/`는 그 위의 *평가·진단 인프라*다. 모든 임계값 튜닝 의사결정의 정량적 근거가 여기서 나온다.

## 파일 지도

| 파일 | 목적 | 비싼 호출 | 캐시 |
|---|---|---|---|
| `filler_eval.py` | filler 차원 P/R/F1 | WhisperX ASR | transcript |
| `cps_eval.py` | cps 차원 + F0 결합 | ASR + librosa pYIN | transcript + F0 npz |
| `dead_zone_eval.py` | VAD 무발화 + flow gate | Silero VAD + Farneback flow | flow_max npz |
| `gaze_eval.py` | head pose yaw/pitch | MediaPipe FaceLandmarker | gaze_pose npz |
| `content_gap_eval.py` | GPT-4o Vision multi-image | LLM 1회 | transcript |
| `smoke_run.py` | 5차원 통합 스모크 | 전체 graph | 모두 |
| `eval_all.py` | 5차원 × 2영상 일괄 실행 | (각 스크립트 위임) | - |
| `mlflow_ui.sh` | MLflow UI launcher | - | - |

## 공통 CLI 인자

`build_eval_parser()`(`src/vidoctor/eval/_script_lib.py`)가 5개 차원 평가의 공통 인자를 제공:

```
positional:  video_path   (mp4)
             labels_csv   (golden CSV)
required:    --run-name   (MLflow run name, 예: baseline_lecture, stage1_lecture)
optional:    --no-cache   (transcript/feature 캐시 무시)
             --no-mlflow  (MLflow 로그 생략, 디버그용)
```

차원별 추가 옵션:
- `cps_eval.py`: `--no-pitch`
- `dead_zone_eval.py`: positional `category` + `--min-duration` + `--flow-threshold`
- `gaze_eval.py`: `--yaw-threshold` / `--pitch-threshold` / `--min-duration` / `--merge-gap` / `--no-baseline`
- `content_gap_eval.py`: `--category` / `--model`

## 튜닝 사이클

```
1. 코드 수정 (예: cps.py의 σ 임계 ±1.5 → ±1.3)
        │
        ▼
2. 차원별 평가:
   uv run python scripts/cps_eval.py data/golden/lecture.mp4 \
       data/golden/lecture_labels.csv --run-name stage16_tighter
        │
        ▼
3. 산출물 자동 생성:
   - mlruns/<exp>/<uuid>/            (MLflow run)
   - data/golden/eval_dumps/cps/lecture_stage16_tighter.json
        │
        ▼
4. 비교:
   - MLflow UI (bash scripts/mlflow_ui.sh → http://127.0.0.1:5001)
     · run 표 정렬·필터, parallel coordinate plot
   - JSON dump 본문 diff
     · 어떤 이벤트가 새로 잡혔/사라졌는지
        │
        ▼
5. 좋아지면 commit, 아니면 코드 되돌리기
```

전체 차원 한 번에 돌리려면:
```bash
uv run python scripts/eval_all.py --tag stage16
```

## 캐시 무효화

| 캐시 | 위치 | 무효화 방법 |
|---|---|---|
| transcript | `data/golden/inputs/transcript_*.json` | `--no-cache` |
| F0 (cps) | `data/golden/inputs/f0_*.npz` | `--no-cache` |
| flow (dead_zone) | `data/golden/inputs/flow_max_*.npz` | 파일 삭제 |
| gaze pose | `data/golden/inputs/gaze_pose_*.npz` | 파일 삭제 |

## 평가 산출물

| 종류 | 위치 | 용도 |
|---|---|---|
| MLflow run | `mlruns/<exp>/<uuid>/` | metric + param 비교용 요약 (UI) |
| JSON dump | `data/golden/eval_dumps/<dim>/` | detected/labels 본문 — 사후 디버깅 |
| feature 캐시 | `data/golden/inputs/` | 반복 평가 가속 |

## 한 줄 정리

> *"임계값을 어떻게 정했나"의 답은 모두 `data/golden/eval_dumps/`와 MLflow의 run 비교에 있다. 회고는 그 위에서 작성한다.*
