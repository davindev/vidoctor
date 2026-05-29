/** 분석 상세·사이드바 초기 로드에서 공유하는 로딩 스피너. */
export function Spinner({
  size = 18,
  className,
}: {
  size?: number;
  className?: string;
}) {
  return (
    <span
      role="status"
      aria-label="불러오는 중"
      className={`inline-block rounded-full border-accent border-r-transparent ${className ?? ""}`}
      style={{
        width: size,
        height: size,
        borderWidth: Math.max(1.5, size / 9),
        animation: "vidSpin 0.9s linear infinite",
      }}
    />
  );
}
