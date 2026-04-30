import asyncio

import streamlit as st

from vidoctor import __version__
from vidoctor.graph import build_graph

CATEGORY_LABEL = {
    "lecture": "강의",
    "vlog": "브이로그·인터뷰",
    "other": "기타",
}

st.set_page_config(page_title="Vidoctor", layout="wide")
st.title("Vidoctor")
st.caption(f"AI 영상 감수 에이전트 · v{__version__}")

with st.sidebar:
    category = st.selectbox(
        "카테고리",
        options=list(CATEGORY_LABEL.keys()),
        format_func=lambda x: CATEGORY_LABEL[x],
    )
    st.file_uploader("영상 파일 (placeholder, 아직 업로드 미연결)")

st.markdown("**5차원 분석 파이프라인 (skeleton dry run)**")

if st.button("dry run 시작"):
    graph = build_graph()

    async def run() -> list[dict]:
        steps: list[dict] = []
        async for chunk in graph.astream(
            {"video_path": "(placeholder)", "category": category},
        ):
            steps.append(chunk)
        return steps

    with st.status("실행 중...", expanded=True) as status:
        steps = asyncio.run(run())
        for chunk in steps:
            for node, output in chunk.items():
                st.write(f"`{node}` → {output}")
        status.update(label="완료", state="complete")
