import streamlit as st
import replicate
import os
from textblob import TextBlob

# Streamlit App Configuration
st.set_page_config(page_title="Math Anxiety Helper", layout="wide")
st.title("Math Anxiety Helper 📘")

# Sidebar Functionality
def initialize_sidebar():
    st.sidebar.header("Settings")

    # API Key for Replicate
    replicate_token = st.secrets.get("REPLICATE_API_TOKEN")
    if replicate_token:
        os.environ["REPLICATE_API_TOKEN"] = replicate_token

    # Reset Chat History
    if st.sidebar.button("Clear Chat"):
        reset_chat_history()

    # Display Chat Download Option
    st.sidebar.download_button(
        label="Download Chat History",
        data=format_chat_history(),
        file_name="math_chat_history.txt",
        mime="text/plain",
        key="download_chat"
    )

    # Add Style to Download Button
    st.markdown("""
        <style>
        .stDownloadButton > button {
            background-color: #28a745;
            color: white;
        }
        .stDownloadButton > button:hover {
            background-color: #218838;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Chat History Initialization
def reset_chat_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm here to help you solve math problems, especially fractions. How are you feeling today?"}
    ]
    st.session_state["stage"] = 1

def initialize_session_state():
    if "messages" not in st.session_state:
        reset_chat_history()

# Format Chat for Download
def format_chat_history():
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"]])

# Sentiment Analysis Function
def assess_emotion(user_text):
    sentiment = TextBlob(user_text).sentiment
    keywords_negative = ["nervous", "sad", "anxious", "worried", "scared"]
    keywords_positive = ["happy", "excited", "great", "confident", "motivated"]

    if any(kw in user_text.lower() for kw in keywords_negative):
        return "negative"
    if any(kw in user_text.lower() for kw in keywords_positive):
        return "positive"

    if sentiment.polarity > 0.3:
        return "positive"
    elif sentiment.polarity < -0.3:
        return "negative"
    return "neutral"

# Construct Response Using Replicate API
def fetch_assistant_response(user_input):
    conversation_context = "You are a helpful, kind and supportive assistant specialized in solving fraction-related math problems. Be supportive and motivational and help students to Break down math problems into simple, clear steps that are easy to understand. Communicate in an empathetic, friendly, and conversational tone\n\n"
    for msg in st.session_state["messages"]:
        conversation_context += f"{msg['role'].capitalize()}: {msg['content']}\n"

    conversation_context += f"User: {user_input}\nAssistant:"

    try:
        response_gen = replicate.run(
            "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea",
            input={
                "prompt": conversation_context,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 128,
                "repetition_penalty": 1.2,
            }
        )
        return "".join(list(response_gen)).strip()
    except Exception as e:
        st.error(f"Error fetching response: {e}")
        return "I'm sorry, something went wrong. Can you please try again?"

# Display Chat
def display_chat():
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Handle User Input
def handle_user_input(user_input):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    response = ""
    if st.session_state["stage"] == 1:
        sentiment = assess_emotion(user_input)
        if sentiment == "negative":
            response = (
                "I'm sorry to hear that you're feeling this way. Don't worry, I'm here to help you every step of the way. "
                "Let's tackle this math problem together! What problem are you working on?"
            )
        elif sentiment == "positive":
            response = "That's a wonderful attitude! Let's dive right in. What fraction problem would you like to solve today?"
        else:
            response = "Alright, let's get started. What math problem can I help you with?"
        st.session_state["stage"] = 2
    elif st.session_state["stage"] == 2:
        response = (
            "Great! Now, what do you think is the first step to solving this problem? "
            "If you're not sure, don't worry—I can guide you."
        )
        st.session_state["stage"] = 3
    elif st.session_state["stage"] == 3:
        response = fetch_assistant_response(user_input)

    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Main Execution
initialize_session_state()
initialize_sidebar()
display_chat()

if user_query := st.chat_input("Type your question here..."):
    handle_user_input(user_query)