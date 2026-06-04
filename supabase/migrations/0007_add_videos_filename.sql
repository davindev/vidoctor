-- videos.filename: 원본 파일명(표시용).
-- 적용: Supabase Dashboard → SQL Editor에 붙여넣고 실행
-- (CLI 사용 시: supabase db push)

-- ============================================================
-- R2 키와 표시용 파일명 분리
-- ============================================================
-- storage_path를 uuid 기반 안전한 키(videos/{uuid}.mp4)로 바꾸면서, 사용자에게 보여줄
-- 원본 파일명은 이 컬럼에 따로 보관한다. 파일명의 특수문자(따옴표·슬래시 등)가 R2 키로
-- 새는 문제를 차단. 기존 row는 filename=null이며, 코드는 storage_path basename으로 폴백.

alter table public.videos
    add column filename text;
