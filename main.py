import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os

# ── Load models once ──────────────────────────────────────
@st.cache_resource
def load_all():
    caption_model     = load_model("models/model.keras")
    feature_extractor = load_model("models/feature_extractor.keras")
    tokenizer         = pickle.load(open("models/tokenizer.pkl", "rb"))
    return caption_model, feature_extractor, tokenizer

caption_model, feature_extractor, tokenizer = load_all()

MAX_LENGTH = 35  # ✅ change this if your max_length is different

# ── Caption generator ─────────────────────────────────────
def generate_caption(image, max_length=MAX_LENGTH, img_size=224):
    img = image.resize((img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img_features = feature_extractor.predict(img, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        sequence   = tokenizer.texts_to_sequences([in_text])[0]
        sequence   = pad_sequences([sequence], maxlen=max_length)
        yhat       = caption_model.predict([img_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word       = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption

# ── Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="Image Captioning", page_icon="🖼️", layout="centered")

st.title("🖼️ Image Captioning")
st.write("Using CNN (DenseNet) + LSTM")
st.divider()

# ── Option 1: Upload image ────────────────────────────────
st.subheader("Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open("uploaded_image.jpg").convert("RGB")
    st.image(image, use_column_width=True)

    with st.spinner("Generating caption..."):
        caption = generate_caption(image)

    st.success(f"📝 **Caption:** {caption}")

st.divider()

# ── Option 2: Test with input_imgs folder ─────────────────
st.subheader("Or Test with Sample Images")

input_folder = "input_imgs"
image_files  = [f for f in os.listdir(input_folder) 
                if f.endswith((".jpg", ".jpeg", ".png"))]

if image_files:
    selected = st.selectbox("Choose a sample image", image_files)

    if st.button("Generate Caption"):
        image_path = os.path.join(input_folder, selected)
        image      = Image.open(image_path).convert("RGB")
        st.image(image, use_column_width=True)

        with st.spinner("Generating caption..."):
            caption = generate_caption(image)

        st.success(f"📝 **Caption:** {caption}")
else:
    st.info("No images found in input_imgs folder. Paste some .jpg images there!")