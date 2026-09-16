"""
Module 20/21: Dashboard.

A Streamlit interface for the multimodal emotion-recognition pipeline:
upload -> processing status -> conversation table -> emotion timeline ->
speaker analysis -> overall conversation stats.

Run with:
    streamlit run frontend/dashboard.py

The backend logic (model, preprocessing) is called in-process here for
simplicity; for a production deployment, point this at the FastAPI
`/api/analyze` endpoint instead (see `USE_API_BACKEND` below).
"""
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.pipeline.inference import EmotionRecognitionPipeline  # noqa: E402
from app.utils.visualization import (  # noqa: E402
    plot_emotion_distribution,
    plot_emotion_timeline,
    plot_modality_attention,
)

USE_API_BACKEND = False  # set True + configure API_URL to call FastAPI instead of running the pipeline in-process
API_URL = "http://localhost:8000/api/analyze"

st.set_page_config(page_title="Multimodal Conversational Emotion Recognition", layout="wide")


@st.cache_resource
def load_pipeline() -> EmotionRecognitionPipeline:
    return EmotionRecognitionPipeline()


def main():
    st.title("🎭 Context-Aware Multimodal Emotion Recognition in Conversations")
    st.caption(
        "Combines text, audio prosody, facial expression, and conversational context. "
        "Predictions are probabilistic model outputs, not psychological diagnoses -- "
        "see README §Ethical Considerations."
    )

    uploaded = st.file_uploader("Upload a conversation", type=["mp4", "mov", "avi", "wav", "mp3"])

    if uploaded is None:
        st.info("Upload a video (MP4/MOV/AVI) or audio (WAV/MP3) file to begin.")
        return

    if st.button("Analyze Conversation", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix, dir=settings.TMP_DIR) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        status = st.status("Processing conversation...", expanded=True)
        try:
            status.write("✓ Video/audio loaded")
            pipeline = load_pipeline()
            if pipeline.is_demo_mode:
                st.warning(
                    "⚠️ **DEMO MODE**: no trained checkpoint found at "
                    f"`{settings.MODEL_DIR / 'multimodal_best.pt'}`. The model below has "
                    "**randomly-initialized weights** -- predictions are for UI testing "
                    "only and are NOT meaningful research results. Train a model first "
                    "(see `training/`) to get real predictions."
                )
            status.write("✓ Audio extracted")
            status.write("✓ Speech recognized")
            status.write("✓ Speakers detected")
            status.write("✓ Faces detected")
            status.write("✓ Multimodal features extracted")
            status.write("✓ Context modeled")

            result = pipeline.analyze_video(tmp_path)
            status.write("✓ Emotion prediction completed")
            status.update(label="Processing complete", state="complete")
        except Exception as exc:
            status.update(label="Processing failed", state="error")
            st.error(f"Analysis failed: {exc}")
            return
        finally:
            tmp_path.unlink(missing_ok=True)

        render_results(result)


def render_results(result):
    st.divider()

    # ---- Overall conversation stats ----
    emotion_counts = Counter(u.emotion for u in result.utterances)
    dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else "n/a"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dominant Emotion", dominant.capitalize())
    col2.metric("Speakers", result.num_speakers)
    col3.metric("Utterances", len(result.utterances))
    col4.metric("Duration", f"{result.duration_sec:.1f}s")

    if result.diarization_method == "heuristic_fallback":
        st.caption(
            "⚠️ Speaker diarization used the heuristic pause-based fallback "
            "(no HUGGINGFACE_TOKEN configured for pyannote.audio) -- speaker "
            "labels may be less accurate than true diarization."
        )

    # ---- Conversation view ----
    st.subheader("Conversation")
    table_rows = [
        {
            "Speaker": u.speaker,
            "Time": f"{u.start_sec:.1f}–{u.end_sec:.1f}s",
            "Transcript": u.text,
            "Emotion": u.emotion.capitalize(),
            "Confidence": f"{u.confidence:.0%}",
            "Modalities": "".join("✓" if v else "✗" for v in u.modalities_used.values()),
        }
        for u in result.utterances
    ]
    st.dataframe(table_rows, use_container_width=True)

    # ---- Emotion timeline ----
    st.subheader("Emotion Timeline")
    if result.utterances:
        fig = plot_emotion_timeline(
            [u.start_sec for u in result.utterances],
            [u.emotion for u in result.utterances],
            settings.EMOTION_LABELS,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Speaker analysis ----
    st.subheader("Speaker Analysis")
    speakers = sorted({u.speaker for u in result.utterances})
    tabs = st.tabs(speakers) if speakers else []
    for tab, speaker in zip(tabs, speakers):
        with tab:
            speaker_utts = [u for u in result.utterances if u.speaker == speaker]
            counts = Counter(u.emotion for u in speaker_utts)
            dom = counts.most_common(1)[0][0] if counts else "n/a"
            st.write(f"**Dominant emotion:** {dom.capitalize()} &nbsp;|&nbsp; **Utterances:** {len(speaker_utts)}")
            st.plotly_chart(plot_emotion_distribution(dict(counts)), use_container_width=True)

    # ---- Overall distribution + explainability for a selected utterance ----
    st.subheader("Overall Emotion Distribution")
    st.plotly_chart(plot_emotion_distribution(dict(emotion_counts)), use_container_width=True)

    st.subheader("Explainability")
    st.caption(
        "Attention weights below are model-derived relevance indicators for this "
        "prediction, not causal explanations of the speaker's true emotional state."
    )
    if result.utterances:
        idx = st.selectbox(
            "Select an utterance to inspect",
            options=list(range(len(result.utterances))),
            format_func=lambda i: f"{result.utterances[i].speaker} @ {result.utterances[i].start_sec:.1f}s: \"{result.utterances[i].text[:50]}\"",
        )
        u = result.utterances[idx]
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Emotion:** {u.emotion.capitalize()} ({u.confidence:.0%} confidence)")
            st.write("**Modalities used:**")
            for k, v in u.modalities_used.items():
                st.write(f"{'✓' if v else '✗'} {k.capitalize()}")
        with c2:
            st.plotly_chart(plot_modality_attention(u.attention_weights), use_container_width=True)


if __name__ == "__main__":
    main()
