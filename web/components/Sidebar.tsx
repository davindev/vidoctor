"use client";

import type { AnalysisListItem } from "@/lib/api";
import { CATEGORY_LABEL } from "@/lib/api";
import { basename, fmtRelative } from "@/lib/format";

interface Props {
  items: AnalysisListItem[];
  selectedId: string | null;
  disabled: boolean;
  onSelect: (id: string) => void;
  onNewAnalysis: () => void;
}

export function Sidebar({
  items,
  selectedId,
  disabled,
  onSelect,
  onNewAnalysis,
}: Props) {
  return (
    <aside className="sticky top-0 flex h-screen w-[300px] flex-col border-r border-line bg-surface">
      <div className="flex items-center gap-2.5 border-b border-line px-[22px] py-[18px] pt-[22px]">
        <div className="grid h-[30px] w-[30px] place-items-center rounded-full bg-accent font-serif text-base font-medium text-white">
          V
        </div>
        <div className="font-serif text-xl font-medium tracking-[-0.02em]">
          vidoctor
        </div>
      </div>

      <button
        type="button"
        onClick={onNewAnalysis}
        disabled={disabled}
        className="mx-4 mt-[18px] mb-1.5 flex items-center justify-center gap-2 rounded-full border border-line-2 bg-surface px-3.5 py-2.5 text-[13px] font-medium text-ink transition-[background,border-color,color] duration-150 hover:border-accent-soft hover:bg-accent-tint active:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-50"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-accent">
          <path
            d="M7 3V11M3 7H11"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
        새 영상 분석
      </button>

      <div className="px-[22px] pt-[26px] pb-3">
        <div className="text-[11px] font-medium uppercase tracking-[0.16em] text-ink-4">
          이전 기록
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-0.5 overflow-y-auto px-3 pb-3">
        {items.length === 0 ? (
          <div className="mx-4 mt-3.5 mb-2 px-3 py-5 text-center">
            <div className="text-[13px] font-medium tracking-[-0.005em] text-ink-2">
              아직 분석 기록이 없어요
            </div>
          </div>
        ) : (
          items.map((item) => (
            <HistoryButton
              key={item.id}
              item={item}
              active={item.id === selectedId}
              disabled={disabled}
              onClick={() => onSelect(item.id)}
            />
          ))
        )}
      </div>
    </aside>
  );
}

function HistoryButton({
  item,
  active,
  disabled,
  onClick,
}: {
  item: AnalysisListItem;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  const filename = basename(item.storage_path) || "이름 없음";
  const catLabel = item.category ? CATEGORY_LABEL[item.category] : "—";
  const when = fmtRelative(item.started_at);

  const baseClass =
    "w-full text-left flex items-center gap-2.5 px-3 py-2.5 rounded-md border transition-[background,border-color] duration-[120ms] ease-out";
  const stateClass = active
    ? "bg-accent-tint border-accent-soft"
    : "bg-transparent border-transparent hover:bg-[#FBF8F2]";
  const disabledClass = disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer";

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`${baseClass} ${stateClass} ${disabledClass}`}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="truncate text-[13px] font-medium text-ink">
          {filename}
        </span>
        <div className="flex items-center gap-2 text-[11.5px] text-ink-4">
          <span>{when}</span>
          {item.error && (
            <>
              <span className="h-[3px] w-[3px] rounded-full bg-line-2" />
              <span className="text-danger">실패</span>
            </>
          )}
        </div>
      </div>
      <span
        className={`flex-shrink-0 rounded-full px-2 py-0.5 text-[10.5px] font-medium tracking-[-0.005em] ${
          active
            ? "bg-accent text-white"
            : "bg-surface-tint text-ink-3"
        }`}
      >
        {catLabel}
      </span>
    </button>
  );
}
