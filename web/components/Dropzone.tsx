"use client";

import { useRef, useState } from "react";
import { fileExt, fmtBytes } from "@/lib/format";

const MAX_BYTES = 300 * 1024 * 1024;
const ALLOWED_EXT = ["mp4", "mov", "mpeg4"];

interface Props {
  file: File | null;
  disabled: boolean;
  onChange: (file: File | null) => void;
}

function validate(f: File): string | null {
  const ext = (f.name.split(".").pop() ?? "").toLowerCase();
  if (!ALLOWED_EXT.includes(ext)) {
    return "지원하지 않는 확장자입니다. mp4, mov, mpeg4 만 가능합니다.";
  }
  if (f.size > MAX_BYTES) {
    return `파일 크기가 300MB를 초과했습니다. (현재 ${fmtBytes(f.size)})`;
  }
  return null;
}

export function Dropzone({ file, disabled, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (f: File | null) => {
    if (!f) {
      onChange(null);
      setError(null);
      return;
    }
    const err = validate(f);
    if (err) {
      setError(err);
      onChange(null);
      return;
    }
    setError(null);
    onChange(f);
  };

  // 이미 파일이 있으면 drop·click 모두 무시 — 사용자가 명시적으로 "제거" 버튼을 눌러야
  // 새 파일을 올릴 수 있다. 실수로 다른 영상을 드래그해 기존 선택을 잃는 사고 방지.
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (file || disabled) return;
    setDragging(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    if (file || disabled) return;
    handleFile(e.dataTransfer.files?.[0] ?? null);
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.mov,.mpeg4,video/mp4,video/quicktime,video/mpeg"
        hidden
        disabled={disabled}
        onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
      />

      {file ? (
        <FileCard file={file} disabled={disabled} onRemove={() => handleFile(null)} />
      ) : (
        <div
          onClick={() => !disabled && inputRef.current?.click()}
          onDragEnter={onDragOver}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={`relative cursor-pointer rounded-[10px] border border-dashed bg-[#F6F3F1] px-6 py-10 text-center transition-[background,border-color] duration-[120ms] ${
            dragging
              ? "border-accent bg-accent-tint"
              : "border-line-2 hover:border-accent hover:bg-accent-tint"
          } ${disabled ? "pointer-events-none opacity-50" : ""}`}
        >
          <div className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-lg border border-line bg-surface text-ink-2">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 16V4M12 4L7 9M12 4L17 9"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M4 17V19C4 19.5523 4.44772 20 5 20H19C19.5523 20 20 19.5523 20 19V17"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <div className="mb-1 text-sm font-semibold">
            파일을 끌어다 놓거나 <span className="font-semibold text-ink">파일 선택</span>
          </div>
          <div className="text-[12.5px] leading-[1.7] text-ink-4">
            최대 300MB · 지원 확장자 mp4, mov, mpeg4
          </div>
        </div>
      )}

      {error && (
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-[#EFCBB9] bg-[#FBEAE3] px-3 py-2.5 text-[13px] text-danger">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.4" />
            <path
              d="M7 4V7.5M7 9.5V10"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
            />
          </svg>
          <span>{error}</span>
        </div>
      )}
    </>
  );
}

function FileCard({
  file,
  disabled,
  onRemove,
}: {
  file: File;
  disabled: boolean;
  onRemove: () => void;
}) {
  return (
    <div className="flex items-center gap-3.5 rounded-[10px] border border-accent-soft bg-accent-tint px-5 py-4">
      <div className="grid h-11 w-11 flex-shrink-0 place-items-center rounded-lg bg-accent text-[10px] font-semibold text-white">
        .{fileExt(file.name)}
      </div>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-semibold">{file.name}</div>
        <div className="mt-0.5 flex gap-2.5 text-xs text-ink-3">
          <span>{fmtBytes(file.size)}</span>
          <span>·</span>
          <span>{file.type || "video"}</span>
        </div>
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        className="rounded-full border border-line-2 px-3 py-1.5 text-xs text-ink-3 transition-[border-color,color] duration-[120ms] hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
      >
        제거
      </button>
    </div>
  );
}
