import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from youtubesearchpython import VideosSearch

# Load FLAN-T5
st.set_page_config(page_title="🔧 AI Vehicle Repair Assistant", layout="centered")
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# App UI

st.title("🔧 AI Vehicle Repair Assistant")
st.caption("Get brief repair instructions and a helpful YouTube video.")

# Inputs
vehicle = st.selectbox("Choose Vehicle Type:", ["Car", "Bike"])
issue = st.text_area("Describe the issue (e.g. engine won't start, AC not cooling):", height=100)

# Button
if st.button("🔍 Get Repair Instructions"):
    if issue.strip() == "":
        st.warning("Please enter a vehicle issue.")
    else:
        # Prompt for FLAN-T5
        prompt = f"Give a short repair tip for a {vehicle.lower()} with this issue: {issue}."
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=100)
        instruction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Show AI output
        st.markdown("### 🛠 Repair Tip:")
        st.success(instruction)

        # Search YouTube
        query = f"{vehicle} {issue} repair"
        videos_search = VideosSearch(query, limit=1)
        results = videos_search.result()["result"]
        if results:
            video_url = results[0]["link"]
            st.markdown("### 📺 Watch Tutorial:")
            st.video(video_url)
        else:
            st.info("No matching video found.")