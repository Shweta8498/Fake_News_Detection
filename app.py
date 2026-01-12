import streamlit as st
import joblib
import gdown
import os

# ===== Google Drive File IDs =====
MODEL_ID = "1o-WzhU7iHw_Y3YI496Sx3O2dO7QXsGC1"
VECTORIZER_ID = "1VpGxXPfzXSn-3YA5i7cZKe4633MDnwft"

# ===== Download model files if not present =====
if not os.path.exists("lr_model.jb"):
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_ID}",
        "lr_model.jb",
        quiet=False
    )

if not os.path.exists("vectorizer.jb"):
    gdown.download(
        f"https://drive.google.com/uc?id={VECTORIZER_ID}",
        "vectorizer.jb",
        quiet=False
    )

# ===== Load model and vectorizer =====
model = joblib.load("lr_model.jb")
vectorizer = joblib.load("vectorizer.jb")

# ===== Streamlit UI =====
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

inputn = st.text_area("News Article:", "")

if st.button("Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Real!")
        else:
            st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to Analyze.")
