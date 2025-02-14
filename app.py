import streamlit as st
import spacy
import matplotlib.pyplot as plt
import pandas as pd

    npl = spacy.download("en_core_web_sm")

# Try loading the SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please install it by running: python -m spacy download en_core_web_sm")
    st.stop()

# Streamlit App Title
st.title("📝 NLP Contract Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload a Contract Document (TXT)", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("📄 Contract Preview")
    st.text_area("Contract Content", text, height=200)

    # Process text with NLP
    doc = nlp(text)

    # Extract Named Entities
    entities = {"ORG": [], "DATE": [], "MONEY": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    # Display Extracted Entities
    st.subheader("🔍 Extracted Key Information")
    for key, values in entities.items():
        if values:
            st.write(f"**{key}:** {', '.join(set(values))}")

    # Create a DataFrame for Visualization
    entity_counts = {key: len(values) for key, values in entities.items() if values}
    if entity_counts:
        df = pd.DataFrame(list(entity_counts.items()), columns=["Entity Type", "Count"])
        st.subheader("📊 Entity Frequency Chart")
        st.bar_chart(df.set_index("Entity Type"))

st.sidebar.markdown("💡 **Tip:** Upload a contract to see the NLP analysis in action!")
