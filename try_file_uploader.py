import streamlit as st

uploaded_files = st.file_uploader('Choose a video', type='mp4')

if uploaded_files is not None:
    video_bytes = uploaded_files.read()
    st.video(video_bytes)
