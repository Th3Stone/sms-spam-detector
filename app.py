import streamlit as st
import joblib

@st.cache_resource

def load_assets():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('spam_vectorizer.pkl')
    return model, vectorizer
    
model, vectorizer = load_assets()

st.title("SMS Spam Detector")
st.subheader("Is that text message safe or a scam?")

message = st.text_area("Enter your message:", height= 150)

if st.button("Check"):
    if message:
        vec_text = vectorizer.transform([message])
        prediction = model.predict(vec_text)
        
        probs = model.predict_proba(vec_text)[0]

        if prediction[0] == 1:
            st.error("Spam Detected")
            st.write(f"Confidence: {probs[1]*100:.2f}%")
        else:
            st.success("The message looks safe")
            st.write(f"Confidence: {probs[0]*100:.2f}%")
            st.balloons()
    else:
        st.warning("Please type a message first!")

st.sidebar.title("About")
st.sidebar.info("This app uses a Naive Bayes classifier trained on 5,000+ real text messages to detect spam with 99% accuracy.")
st.sidebar.info("Ceated by Jose Piedrahita")
st.sidebar.info("More Projects: github.com/Th3Stone")
