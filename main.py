# Streamlit app

import streamlit as st

st.title("The Bard 📖")
st.markdown("### _Bring your books to life_")
st.write("")

pdf_file = st.file_uploader("Upload Book PDF", type=['pdf'])

if pdf_file:
    st.write("File uploaded successfully!")
    st.write("")
    st.write("##### Select the chapter you want to visualize")
    chapter = st.selectbox("Chapters", ["Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4", "Chapter 5"])

    st.spinner("Visualizing your book...")
