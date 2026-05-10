import { CATEGORY_LABEL, type Category } from "@/lib/api";
import { fmtHMS } from "@/lib/format";

interface Props {
  /** "결과" / "진행 중" 등 H1 우측 텍스트. 액센트 단어 ("분석")는 컴포넌트가 항상 앞에 둔다. */
  trailing: string;
  filename: string | null;
  category: Category | null;
  /** 결과 화면에서만 영상 길이를 함께 표시. analyzing 중엔 모름이라 omit. */
  durationSec?: number | null;
}

export function ResultHeader({ trailing, filename, category, durationSec }: Props) {
  return (
    <header className="mb-9">
      <h1 className="font-serif text-[40px] font-semibold leading-[1.08] tracking-[-0.025em]">
        <span className="text-accent">분석</span> {trailing}
      </h1>
      <div className="mt-2.5 flex flex-wrap items-center gap-2.5 text-sm text-ink-3">
        <span>{filename ?? "—"}</span>
        {category && (
          <span className="rounded-full bg-surface-tint px-2.5 py-[3px] text-[11px] font-medium text-ink-2">
            {CATEGORY_LABEL[category]}
          </span>
        )}
        {durationSec ? (
          <>
            <span className="text-line-2">·</span>
            <span>{fmtHMS(durationSec)}</span>
          </>
        ) : null}
      </div>
    </header>
  );
}

export function ResultPage({ children }: { children: React.ReactNode }) {
  return (
    <section className="vid-page-enter mx-auto w-full max-w-[1040px] px-16 pt-14 pb-20">
      {children}
    </section>
  );
}
