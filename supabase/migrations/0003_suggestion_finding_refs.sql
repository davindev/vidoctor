-- suggestions 테이블의 finding 근거 컬럼을 dimension:idx 형식의 text[] 로 교체.
-- 적용: Supabase Dashboard → SQL Editor에 붙여넣고 실행
-- (CLI 사용 시: supabase db push)

-- ============================================================
-- suggestions.finding_ids → finding_refs 교체
-- ============================================================
-- LLM 출력은 'filler:0'·'cps:2' 같은 ref 식별자다. 기존 finding_ids uuid[]는
-- finding row의 db id를 참조하려는 의도였지만, save_findings → save_suggestions
-- 흐름에 inserted finding id 매핑 단계가 없어 항상 빈 array로만 저장됐다. v1.0은
-- ref 문자열을 그대로 사용해 UI에서 차원·인덱스로 finding을 역참조한다.

alter table public.suggestions drop column finding_ids;
alter table public.suggestions
    add column finding_refs text[] not null default '{}'::text[];
