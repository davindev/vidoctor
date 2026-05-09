-- v1.0 스코프에서 severity / priority 차등을 스펙아웃.
-- 적용: Supabase Dashboard → SQL Editor에 붙여넣고 실행
-- (CLI 사용 시: supabase db push)

-- ============================================================
-- findings.severity 제거
-- ============================================================
-- 모든 차원이 default "mid" 단일 값으로 고정 운영돼 분기에 의미가 없었고, 임계 결정
-- 근거(라벨링·평가)도 v1.0에서 도입되지 않았다. 신호 자체가 0차원 → DB 컬럼 자체를
-- 제거해 스키마 단순화. 향후 차등이 필요해지면 그때 도메인별 컬럼·테이블로 재도입.

alter table public.findings drop column severity;

-- ============================================================
-- suggestions.priority 제거
-- ============================================================
-- LLM이 priority를 항상 0~3 좁은 범위에서만 출력해 정렬 정보로서 가치가 미미했고,
-- UI도 라벨만 표시할 뿐 정렬 외 활용이 없었다. v1.0은 LLM 출력 순서를 그대로 보존.

alter table public.suggestions drop column priority;
