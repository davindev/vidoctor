"use client";

import { useState } from "react";
import type { Category } from "@/lib/api";
import { CATEGORY_LABEL } from "@/lib/api";
import { Dropzone } from "./Dropzone";

interface Props {
  disabled: boolean;
  onSubmit: (file: File, category: Category) => void;
}

export function IdleForm({ disabled, onSubmit }: Props) {
  const [category, setCategory] = useState<Category>("lecture");
  const [file, setFile] = useState<File | null>(null);

  const ready = file !== null && !disabled;

  return (
    <section className="vid-page-enter mx-auto w-full max-w-[880px] px-16 pt-14 pb-20">
      <h1 className="font-serif text-[46px] font-semibold leading-[1.05] tracking-[-0.028em] text-ink">
        <span className="text-accent">분석</span> 시작하기
      </h1>
      <p className="mt-4 mb-11 max-w-[56ch] text-base leading-[1.65] text-ink-3">
        영상 카테고리를 선택하고 파일을 업로드하면 자동으로 분석을 시작합니다.
        <br />
        업로드 즉시 처리되며, 결과는 좌측{" "}
        <b className="font-semibold text-ink">이전 기록</b>에 저장됩니다.
      </p>

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
              onChange={(e) => setCategory(e.target.value as Category)}
            >
              {Object.entries(CATEGORY_LABEL).map(([k, v]) => (
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

        {/* Field 02 — Upload */}
        <div className="border-b border-line px-6 py-5">
          <div className="mb-3.5 flex items-center gap-2.5">
            <span className="text-[11px] font-medium tracking-[0.04em] text-accent">
              02
            </span>
            <span className="text-[14.5px] font-semibold">영상 업로드</span>
            <span className="ml-auto text-xs text-ink-4">max 300MB</span>
          </div>
          <Dropzone file={file} disabled={disabled} onChange={setFile} />
        </div>

        {/* Submit */}
        <div className="flex items-center gap-3 px-6 py-5">
          <button
            type="button"
            disabled={!ready}
            onClick={() => file && onSubmit(file, category)}
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
            ) : (
              "파일을 업로드하면 활성화됩니다."
            )}
          </span>
        </div>
      </div>
    </section>
  );
}
