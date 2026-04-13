import streamlit as st
import pandas as pd
import random # Placeholder for actual model inference

# Page configuration
st.set_page_config(page_title="AI Content Moderation System", layout="wide")

st.title("🛡️ AI Content Moderation System")
st.markdown("""
This system uses **NLP and Machine Learning** to detect harmful content in real-time. 
It classifies text into multiple categories and assigns a **Severity Score**.
""")

# Sidebar info based on project doc
st.sidebar.header("System Information")
st.sidebar.info("""
**Domain:** Trust & Safety Systems [cite: 7]
**Dataset:** Jigsaw Toxic Comment [cite: 25]
**Tech:** Python, Scikit-learn, NLP [cite: 17, 36]
""")

# Layout: Input Area
st.subheader("Analyze User-Generated Content")
user_input = st.text_area("Enter text to moderate (comment, post, or message):", 
                          placeholder="Type something here...")

if st.button("Analyze Content"):
    if user_input:
        # SIMULATION OF MODEL OUTPUT [cite: 26, 27]
        # In a real app, you would call your Flask/Django API or model.predict() here
        labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
        scores = {label: random.uniform(0, 1) for label in labels}
        overall_severity = max(scores.values()) * 100

        # Display Results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Overall Severity Score", f"{overall_severity:.2f}%")
            if overall_severity > 70:
                st.error("⚠️ High Risk Content Detected")
            elif overall_severity > 30:
                st.warning("🟡 Moderate Risk")
            else:
                st.success("✅ Safe Content")

        with col2:
            st.write("### Multi-Label Classification Breakdown ")
            df = pd.DataFrame(list(scores.items()), columns=['Category', 'Probability'])
            st.bar_chart(df.set_index('Category'))

        # Detailed breakdown table
        st.write("### Detection Details")
        st.table(df)
    else:
        st.error("Please enter some text first.")

st.divider()
st.caption("Developed as an AI-based system for automatic toxic content detection. [cite: 19]")