import streamlit as st

from vidoctor import __version__

st.set_page_config(page_title="Vidoctor", layout="wide")
st.title("Vidoctor")
st.caption(f"AI 영상 감수 에이전트 · v{__version__}")
st.info("환경 셋업 완료. 5차원 분석 파이프라인은 추후 들어옵니다.")
