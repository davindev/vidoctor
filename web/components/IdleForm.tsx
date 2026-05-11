"use client";

import { useState } from "react";
import { CATEGORY_CHOICE_LABEL, type CategoryChoice } from "@/lib/api";
import type { AnalyzeSource } from "@/lib/sse";
import { Dropzone } from "./Dropzone";
import { ErrorBanner } from "./ErrorBanner";

interface Props {
  disabled: boolean;
  lastError: string | null;
  onSubmit: (source: AnalyzeSource, category: CategoryChoice) => void;
}

type InputMode = "file" | "url";

// youtu.be / youtube.com / m.youtube.com — 백엔드 _HOST_PATTERN과 동일 규칙. 다운로드 전
// 사용자 입력을 빠르게 거르기 위한 UX 가드.
const YT_URL_RE = /^https?:\/\/(www\.|m\.)?(youtube\.com|youtu\.be)\//i;

export function IdleForm({ disabled, lastError, onSubmit }: Props) {
  const [category, setCategory] = useState<CategoryChoice>("auto");
  const [mode, setMode] = useState<InputMode>("file");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");

  const trimmedUrl = url.trim();
  const urlValid = YT_URL_RE.test(trimmedUrl);
  const ready =
    !disabled && (mode === "file" ? file !== null : urlValid);

  const handleSubmit = () => {
    if (!ready) return;
    if (mode === "file" && file) {
      onSubmit({ kind: "file", file }, category);
    } else if (mode === "url" && urlValid) {
      onSubmit({ kind: "url", url: trimmedUrl }, category);
    }
  };

  return (
    <section className="vid-page-enter mx-auto w-full max-w-[880px] px-16 pt-14 pb-20">
      <h1 className="font-serif text-[46px] font-semibold leading-[1.05] tracking-[-0.028em] text-ink">
        <span className="text-accent">분석</span> 시작하기
      </h1>
      <p className="mt-4 mb-11 text-base leading-[1.65] text-ink-3">
        영상 카테고리를 선택하고 파일 또는 유튜브 URL을 입력하면 자동으로 분석이 시작됩니다.
        <br />
        결과는 좌측 <b className="font-semibold text-ink">이전 기록</b>에 저장됩니다.
      </p>

      {lastError && (
        <ErrorBanner
          title="이전 분석이 실패했습니다."
          message={lastError}
          className="mb-6 px-4 py-3"
        />
      )}

      <div className="overflow-hidden rounded-xl border border-line bg-surface">
        {/* Field 01 — Category */}
        <div className="border-b border-line px-6 py-5">
          <div className="mb-3.5 flex items-center gap-2.5">
            <span className="text-[11px] font-medium tracking-[0.04em] text-accent">
              01
            </span>
            <span className="text-[14.5px] font-semibold">카테고리</span>
            <span className="ml-auto text-xs text-ink-4">required</span>
          </div>
          <div className="relative">
            <select
              className="w-full appearance-none cursor-pointer rounded-md border border-line-2 bg-surface px-3.5 py-3 pr-10 text-sm font-medium text-ink transition-[border-color,box-shadow] duration-[120ms] hover:border-ink-3 focus:border-accent focus:outline-none focus:ring-[3px] focus:ring-accent-tint disabled:cursor-not-allowed disabled:opacity-50"
              value={category}
              disabled={disabled}
              onChange={(e) => setCategory(e.target.value as CategoryChoice)}
            >
              {Object.entries(CATEGORY_CHOICE_LABEL).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
            <svg
              className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 text-ink-3"
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
            >
              <path
                d="M3 5.5L7 9.5L11 5.5"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        </div>

        {/* Field 02 — Source (file or YouTube URL) */}
        <div className="border-b border-line px-6 py-5">
          <div className="mb-3.5 flex items-center gap-2.5">
            <span className="text-[11px] font-medium tracking-[0.04em] text-accent">
              02
            </span>
            <span className="text-[14.5px] font-semibold">영상 입력</span>
            <span className="ml-auto text-xs text-ink-4">
              {mode === "file" ? "max 300MB" : "최대 10분"}
            </span>
          </div>

          <ModeTabs mode={mode} disabled={disabled} onChange={setMode} />

          <div className="mt-4">
            {mode === "file" ? (
              <Dropzone file={file} disabled={disabled} onChange={setFile} />
            ) : (
              <UrlInput
                value={url}
                disabled={disabled}
                invalid={url.length > 0 && !urlValid}
                onChange={setUrl}
              />
            )}
          </div>
        </div>

        {/* Submit */}
        <div className="flex items-center gap-3 px-6 py-5">
          <button
            type="button"
            disabled={!ready}
            onClick={handleSubmit}
            className={`inline-flex items-center gap-2 rounded-full border px-5 py-2.5 text-sm font-medium transition-[background,border-color,transform] duration-150 ease-out active:translate-y-[1px] ${
              ready
                ? "border-accent bg-accent text-white hover:border-accent-strong hover:bg-accent-strong"
                : "border-line-2 bg-transparent text-ink-4 cursor-not-allowed"
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path
                d="M3 7H11M11 7L7.5 3.5M11 7L7.5 10.5"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            분석 시작
          </button>
          <span className="ml-auto text-xs text-ink-4">
            {ready ? (
              <>
                <span className="text-accent">●</span> 준비 완료 — 분석을 시작할 수
                있습니다.
              </>
            ) : mode === "file" ? (
              "파일을 업로드하면 활성화됩니다."
            ) : (
              "유튜브 URL을 입력하면 활성화됩니다."
            )}
          </span>
        </div>
      </div>
    </section>
  );
}

function ModeTabs({
  mode,
  disabled,
  onChange,
}: {
  mode: InputMode;
  disabled: boolean;
  onChange: (m: InputMode) => void;
}) {
  const tabs: { key: InputMode; label: string }[] = [
    { key: "file", label: "파일 업로드" },
    { key: "url", label: "유튜브 URL" },
  ];
  return (
    <div
      role="tablist"
      className="inline-flex rounded-md border border-line-2 bg-[#F6F3F1] p-0.5"
    >
      {tabs.map((t) => {
        const active = mode === t.key;
        return (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={active}
            disabled={disabled}
            onClick={() => onChange(t.key)}
            className={`rounded-[5px] px-3.5 py-1.5 text-[13px] font-medium transition-[background,color] duration-[120ms] disabled:cursor-not-allowed disabled:opacity-50 ${
              active
                ? "bg-surface text-ink shadow-[0_1px_2px_rgba(0,0,0,0.06)]"
                : "text-ink-3 hover:text-ink"
            }`}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

function UrlInput({
  value,
  disabled,
  invalid,
  onChange,
}: {
  value: string;
  disabled: boolean;
  invalid: boolean;
  onChange: (v: string) => void;
}) {
  return (
    <>
      <input
        type="url"
        value={value}
        disabled={disabled}
        placeholder="https://www.youtube.com/watch?v=..."
        onChange={(e) => onChange(e.target.value)}
        className={`w-full rounded-md border bg-surface px-3.5 py-3 text-sm text-ink transition-[border-color,box-shadow] duration-[120ms] placeholder:text-ink-4 focus:outline-none focus:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50 ${
          invalid
            ? "border-danger-soft focus:border-danger focus:ring-danger-tint"
            : "border-line-2 hover:border-ink-3 focus:border-accent focus:ring-accent-tint"
        }`}
      />
      {invalid && (
        <ErrorBanner
          message="youtube.com 또는 youtu.be URL만 지원합니다."
          className="mt-3"
        />
      )}
      <p className="mt-2.5 text-[12.5px] leading-[1.7] text-ink-4">
        최대 10분 · youtube.com / youtu.be 링크
      </p>
    </>
  );
}
