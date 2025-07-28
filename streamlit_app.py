import streamlit as st
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from model import Model
from i3d_feature_extractor import load_and_preprocess_video, I3DFeatureExtractor
from video_utils import extract_sliding_clips, annotate_anomalies_on_video

st.set_page_config(page_title="RTFM Video Anomaly Detection", layout="wide")
st.title("🚦 RTFM Anomaly Detection on Video")

uploaded_file = st.file_uploader("🎬 Upload a test video (.mp4)", type=["mp4"])
threshold = st.slider("🎚️ Anomaly Threshold", 0.0, 1.0, value=0.5, step=0.01)

if uploaded_file is not None:
    with st.spinner("Processing video..."):

        # Save uploaded file
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load frames and extract clips
        frames = load_and_preprocess_video(video_path)
        clips, frame_indices = extract_sliding_clips(frames, clip_len=32, stride=16)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = I3DFeatureExtractor(model_path='rgb_imagenet.pt', device=device)
        rtfm_model = Model(n_features=1024, batch_size=1).to(device)
        rtfm_model.load_state_dict(torch.load("ckpt/rtfmfinal.pkl", map_location=device))
        rtfm_model.eval()

        frame_scores = np.zeros(len(frames))
        frame_counts = np.zeros(len(frames))

        with torch.no_grad():
            for idx, clip in enumerate(clips):
                clip_tensor = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, 3, 32, 224, 224)
                features = extractor(clip_tensor)  # (32, 1024)
                inputs = features.unsqueeze(0)     # (1, 32, 1024)
                _, _, _, _, _, _, logits, _, _, _ = rtfm_model(inputs)  # (1, 32, 1)

                logits = logits.squeeze()
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
                logits_np = logits.cpu().numpy()
                start, end = frame_indices[idx]
                # DEBUG: Log raw logits
                print(f"🧪 Clip {idx}:")
                print(f"Logits shape: {logits_np.shape}")
                print(f"Logits sample: {logits_np[:10]}")
                print(f"Mapped frames: {frame_indices[idx]}")
                print("-" * 50)

                valid_len = min(len(logits_np), end - start)
                for i in range(valid_len):
                    frame_idx = start + i
                    if frame_idx < len(frames):
                        frame_scores[frame_idx] += logits_np[i]
                        frame_counts[frame_idx] += 1

        # Normalize overlapping frames
        frame_scores = frame_scores / np.clip(frame_counts, a_min=1, a_max=None)
        import pandas as pd

        # DEBUG: Display score table
        frame_df = pd.DataFrame({
            'Frame Index': np.arange(len(frame_scores)),
            'Anomaly Score': frame_scores
        })
        st.subheader("🧾 Debug: Frame Score Table")
        st.dataframe(frame_df.head(100))  # Show first 100 rows


        # Detect anomaly intervals
        anomaly_intervals = []
        start = None
        for i, score in enumerate(frame_scores):
            if score >= threshold:
                if start is None:
                    start = i
            else:
                if start is not None:
                    anomaly_intervals.append((start, i))
                    start = None
        if start is not None:
            anomaly_intervals.append((start, len(frame_scores)))

        # Annotate and save output video
        output_video = "annotated_output.mp4"
        annotate_anomalies_on_video(video_path, anomaly_intervals, output_video)

        st.success("✅ Inference complete!")

        # Show annotated video
        st.subheader("📺 Annotated Video")
        st.video(output_video)

        # Plot anomaly scores
        st.subheader("📈 Frame-level Anomaly Scores")
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(frame_scores)), frame_scores, color='crimson')
        ax.axhline(y=threshold, color='blue', linestyle='--', label=f'Threshold = {threshold}')
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Anomaly Score")
        ax.set_title("Anomaly Score per Frame")
        ax.legend()
        st.pyplot(fig)

        # DEBUG: Score histogram
        st.subheader("📊 Score Distribution Histogram")
        fig2, ax2 = plt.subplots()
        ax2.hist(frame_scores, bins=20, color='purple', edgecolor='black')
        ax2.set_xlabel("Anomaly Score")
        ax2.set_ylabel("Frame Count")
        ax2.set_title("Histogram of Frame-Level Anomaly Scores")
        st.pyplot(fig2)

