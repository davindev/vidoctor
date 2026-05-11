"use client";

import {
  CATEGORY_DIMENSIONS,
  DIMENSION_LABEL,
  type Category,
  type Dimension,
} from "@/lib/api";
import type { AnalyzingPhase } from "@/lib/sse";

type NodeState = "waiting" | "active" | "done" | "skipped";

interface Props {
  category: Category;
  phase: AnalyzingPhase;
  completed: Set<string>;
  errorMessage: string | null;
}

const ALL_DIMS: Dimension[] = [
  "filler",
  "cps",
  "dead_zone",
  "gaze",
  "content_gap",
];

// branch 좌표 — viewBox 360 기준 N+1 분할로 derive해 노드 추가/제거 시 한 곳만 갱신.
const VIEW_HEIGHT = 360;
const BRANCH_RATIOS = ALL_DIMS.map((_, i) => (i + 1) / (ALL_DIMS.length + 1));
const BRANCH_TOPS = BRANCH_RATIOS.map((r) => `${(r * 100).toFixed(2)}%`);
const WIRE_BRANCH_Y = BRANCH_RATIOS.map((r) => r * VIEW_HEIGHT);

function deriveStates(
  category: Category,
  phase: AnalyzingPhase,
  completed: Set<string>,
): {
  upload: NodeState;
  transcribe: NodeState;
  branches: NodeState[];
  suggest: NodeState;
} {
  const uploadDone = phase === "running";
  const transcribeDone = completed.has("transcribe");
  const activeDims = new Set(CATEGORY_DIMENSIONS[category]);
  const branches: NodeState[] = ALL_DIMS.map((dim) => {
    if (!activeDims.has(dim)) return "skipped";
    if (completed.has(`detect_${dim}`)) return "done";
    return transcribeDone ? "active" : "waiting";
  });
  const allDetectorsDone = CATEGORY_DIMENSIONS[category].every((d) =>
    completed.has(`detect_${d}`),
  );
  const suggestionsDone = completed.has("generate_suggestions");

  return {
    upload: uploadDone ? "done" : "active",
    transcribe: transcribeDone ? "done" : uploadDone ? "active" : "waiting",
    branches,
    suggest: suggestionsDone
      ? "done"
      : allDetectorsDone
        ? "active"
        : "waiting",
  };
}

function statusText(
  state: NodeState,
  doneLabel = "완료",
  activeLabel = "분석중",
): string {
  if (state === "done") return doneLabel;
  if (state === "active") return activeLabel;
  if (state === "skipped") return "건너뜀";
  return "대기중";
}

// 5단계 모델: 업로드 → 음성 전사 → 5차원 검출 → 개선 제안 → 완료. UI 라벨은 "N/4 진행중"
// 으로 표시(allDone 분기 별도)이라 step <= 4까지 매핑. 마지막 단계는 suggest=done이지만
// allDone 가드가 그 케이스를 가져가므로 여기 step=4는 "suggest 진행 중"을 의미.
function progressStep(s: ReturnType<typeof deriveStates>): number {
  const branchesDone = s.branches.every(
    (b) => b === "done" || b === "skipped",
  );
  if (branchesDone && s.transcribe === "done") return 4;
  if (s.transcribe === "done") return 3;
  if (s.upload === "done") return 2;
  return 1;
}

export function Pipeline({
  category,
  phase,
  completed,
  errorMessage,
}: Props) {
  const states = deriveStates(category, phase, completed);
  const step = progressStep(states);
  // URL 흐름의 다운로드 → R2 업로드를 첫 노드 한 곳에서 표시. 완료 시점부터는 "업로드 완료" 고정.
  const uploadLabel =
    phase === "downloading" ? "유튜브 다운로드" : "영상 업로드";
  const uploadActiveText =
    phase === "downloading" ? "다운로드중" : "업로드중";
  const allDone =
    states.upload === "done" &&
    states.transcribe === "done" &&
    states.branches.every((s) => s === "done" || s === "skipped") &&
    states.suggest === "done";

  return (
    <section className="rounded-[14px] border border-line bg-surface px-8 pt-7 pb-6">
      <div className="mb-[22px] flex items-center justify-between">
        <div className="text-[18px] font-semibold tracking-[-0.015em]">
          실시간 처리 파이프라인
        </div>
        <div className="text-xs text-ink-4">
          {allDone ? "완료" : `4단계 중 ${step}단계 진행중`}
        </div>
      </div>

      <div className="relative h-[360px] w-full">
        <svg
          viewBox="0 0 1000 360"
          preserveAspectRatio="none"
          className="pointer-events-none absolute inset-0 h-full w-full"
        >
          {/* upload → transcribe */}
          <Wire
            d="M 130 180 L 290 180"
            done={states.transcribe === "done" || states.transcribe === "active"}
            active={states.transcribe === "active"}
          />
          {/* transcribe → branches */}
          {WIRE_BRANCH_Y.map((y, i) => (
            <Wire
              key={`tr-br-${i}`}
              d={`M 410 180 C 480 180 500 ${y} 600 ${y}`}
              done={
                states.branches[i] === "done" ||
                states.branches[i] === "active"
              }
              active={states.branches[i] === "active"}
            />
          ))}
          {/* branches → suggest */}
          {WIRE_BRANCH_Y.map((y, i) => (
            <Wire
              key={`br-su-${i}`}
              d={
                y === 180
                  ? "M 720 180 L 880 180"
                  : `M 720 ${y} C 800 ${y} 820 180 880 180`
              }
              done={
                states.branches[i] === "done" &&
                (states.suggest === "active" || states.suggest === "done")
              }
              active={
                states.branches[i] === "done" && states.suggest === "active"
              }
            />
          ))}
        </svg>

        {/* Nodes */}
        <PipelineNode
          left="6.5%"
          top="50%"
          label={uploadLabel}
          state={states.upload}
          doneLabel="업로드 완료"
          activeLabel={uploadActiveText}
        />
        <PipelineNode
          left="35%"
          top="50%"
          label="음성 전사"
          state={states.transcribe}
        />
        {ALL_DIMS.map((dim, i) => (
          <PipelineNode
            key={dim}
            left="66%"
            top={BRANCH_TOPS[i]}
            label={`${DIMENSION_LABEL[dim]} 검출`}
            state={states.branches[i]}
            branch
          />
        ))}
        <PipelineNode
          left="93.5%"
          top="50%"
          label="개선 제안"
          state={states.suggest}
        />
      </div>

      {errorMessage && (
        <div className="mt-5 flex items-center gap-2 rounded-lg border border-[#EFCBB9] bg-[#FBEAE3] px-3 py-2.5 text-[13px] text-danger">
          분석 실패: {errorMessage}
        </div>
      )}
    </section>
  );
}

function Wire({
  d,
  done,
  active,
}: {
  d: string;
  done: boolean;
  active: boolean;
}) {
  return (
    <path
      d={d}
      fill="none"
      stroke={done || active ? "#D97757" : "#DBD0BF"}
      strokeWidth={1.5}
      strokeDasharray={active ? "5 6" : undefined}
      style={
        active
          ? { animation: "vidWireFlow 1.2s linear infinite" }
          : undefined
      }
    />
  );
}

function PipelineNode({
  left,
  top,
  label,
  state,
  doneLabel = "완료",
  activeLabel = "분석중",
  branch = false,
}: {
  left: string;
  top: string;
  label: string;
  state: NodeState;
  doneLabel?: string;
  activeLabel?: string;
  branch?: boolean;
}) {
  const tone =
    state === "active"
      ? "border-accent bg-surface shadow-[0_0_0_4px_var(--color-accent-tint),0_1px_0_rgba(0,0,0,0.02)]"
      : state === "done"
        ? "border-accent-soft bg-accent-tint"
        : "border-line-2 bg-surface";

  const labelColor =
    state === "waiting" || state === "skipped" ? "text-ink-3" : "text-ink";
  const statusColor =
    state === "active"
      ? "text-accent"
      : state === "done"
        ? "text-[#5A7A4A]"
        : "text-ink-4";

  return (
    <div
      className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-xl border p-3 text-center transition-[border-color,background,box-shadow,transform] duration-[250ms] ease-out ${tone} ${
        branch ? "min-w-[110px] py-2.5 px-3" : "min-w-[130px]"
      }`}
      style={{ left, top }}
    >
      <div className={`text-[12.5px] font-medium tracking-[-0.005em] ${labelColor}`}>
        {label}
      </div>
      <div className={`mt-1 h-3 text-[10.5px] ${statusColor} flex items-center justify-center gap-1.5`}>
        {state === "active" && <Spinner />}
        {state === "done" && <CheckBadge />}
        {state === "waiting" && <WaitDot />}
        <span>{statusText(state, doneLabel, activeLabel)}</span>
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <span
      className="inline-block h-[9px] w-[9px] rounded-full border-[1.5px] border-accent border-r-transparent"
      style={{ animation: "vidSpin 0.9s linear infinite" }}
    />
  );
}

function CheckBadge() {
  return (
    <span className="inline-grid h-[11px] w-[11px] place-items-center rounded-full bg-[#5A7A4A] text-[8px] leading-none text-white">
      ✓
    </span>
  );
}

function WaitDot() {
  return <span className="inline-block h-[5px] w-[5px] rounded-full bg-[#C9BEB1]" />;
}
