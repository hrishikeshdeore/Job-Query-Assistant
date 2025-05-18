import streamlit as st
import os
import sys
from dotenv import load_dotenv
from chatbot import JobChatBot
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import queue
import requests
import base64
import time
import av
import threading
import io
import wave

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="JobSevak - Lokal Job Assistant",
    page_icon="💼",
    layout="centered"
)

def transcribe_audio_whisper(audio_bytes, hf_api_token):
    """
    Transcribe audio using Hugging Face Whisper API (openai/whisper-large-v3).
    """
    if not hf_api_token:
        return "[Error: Missing API token for voice transcription]"
        
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {"Authorization": f"Bearer {hf_api_token}"}
    
    try:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"task": "transcribe", "language": "en"}
        
        # Try with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, files=files, data=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return result.get("text", "")
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    st.error(f"Transcription API error: {response.status_code} - {response.text}")
                    return f"[Voice transcription failed: {response.text}]"
            except Exception as e:
                st.error(f"Error during voice transcription: {str(e)}")
                return f"[Voice transcription error: {str(e)}]"
        
        return "[Voice transcription failed after multiple attempts]"
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return f"[Audio processing error: {str(e)}]"

# Function to initialize session state
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "chatbot" not in st.session_state:
        hf_api_token = os.getenv("HF_API_TOKEN")
        if not hf_api_token:
            st.error("CRITICAL: Hugging Face API Token (HF_API_TOKEN) not found in your .env file. The chatbot will not be able to connect to the language model. Please create a .env file with your token.")
            st.session_state.chatbot = JobChatBot(hf_api_token=None)
        else:
            st.session_state.chatbot = JobChatBot(hf_api_token=hf_api_token)
    
    if "audio_frames" not in st.session_state:
        st.session_state.audio_frames = []
        
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False

# Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .stApp { background-color: #f5f7f9; }
    .chat-message { padding: 1rem; border-radius: 0.8rem; margin-bottom: 1rem; display: flex; flex-direction: row; align-items: flex-start; gap: 0.8rem; }
    .chat-message .message { flex-grow: 1; color: #333333; }
    .chat-message.user { background-color: #e6f3ff; border: 1px solid #cce5ff; }
    .chat-message.user .message { color: #2c3e50; text-align: right; }
    .chat-message.bot { background-color: #ffffff; border: 1px solid #e6e6e6; }
    .chat-message.bot .message { color: #333333; text-align: left; }
    .chat-message .avatar { width: 35px; height: 35px; border-radius: 50%; object-fit: cover; flex-shrink: 0; }
    /* For user messages, avatar should be on the right */
    .chat-message.user .avatar { order: 1; }
    .header-container { display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; padding: 1rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-logo { font-size: 2rem; }
    .header-title { font-size: 1.8rem; font-weight: bold; color: #2e75b6; margin: 0; flex-grow: 1; }
    
    /* Improve text input styling and cursor visibility */
    .stTextInput > div > div > input {
        background-color: white !important; 
        color: #333333 !important;
        caret-color: #2e75b6 !important; /* Bright blue cursor */
        border: 1px solid #cccccc !important;
        padding: 0.5rem !important;
    }
    
    /* Make sure the text input has focus styles */
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 2px rgba(46, 117, 182, 0.5) !important;
        border-color: #2e75b6 !important;
    }
    
    /* General text color */
    body, .stMarkdown, .stText { color: #333333 !important; }
    
    /* Make sure buttons have good contrast */
    .stButton > button {
        background-color: #2e75b6 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        background-color: #1c5794 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to render chat messages
def render_chat_messages():
    for message in st.session_state.chat_history:
        avatar_url_user = "https://ui-avatars.com/api/?background=random&name=User"
        avatar_url_bot = "https://ui-avatars.com/api/?background=2E75B6&color=fff&name=JS"
        
        if message["role"] == "user":
            st.markdown(f'''
            <div class="chat-message user">
                <div class="message">{message["content"]}</div>
                <img class="avatar" src="{avatar_url_user}" alt="User avatar">
            </div>
            ''', unsafe_allow_html=True)
        else: # Bot message
            st.markdown(f'''
            <div class="chat-message bot">
                <img class="avatar" src="{avatar_url_bot}" alt="JobSevak avatar">
                <div class="message">{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)

# Audio processing functions
def process_audio_frame(frame):
    """Process incoming audio frames during recording"""
    if st.session_state.is_recording:
        sound = frame.to_ndarray().copy()
        st.session_state.audio_frames.append(sound)
    return frame

def create_wav_from_frames(frames):
    """Convert audio frames to WAV file bytes"""
    if not frames:
        return None
        
    # Concatenate all audio chunks
    audio_data = np.concatenate(frames, axis=0)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(48000)  # Sample rate
        wav_file.writeframes(audio_data.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.getvalue()

# Main function
def main():
    initialize_session_state()
    apply_custom_css()
    
    st.markdown("""
    <div class="header-container">
        <div class="header-logo">💼</div>
        <h1 class="header-title">JobSevak - Lokal Job Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to **JobSevak**! I can help you find local job opportunities and answer questions about Lokal's job platform. Ask me anything!
    """)
    
    render_chat_messages()
    
    # --- Voice Input Section ---
    # st.markdown("**Use voice input:**")
    
    # # WebRTC setup for audio recording
    # webrtc_ctx = webrtc_streamer(
    #     key="voice-input",
    #     mode=WebRtcMode.SENDONLY,
    #     rtc_configuration=RTCConfiguration(
    #         {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    #     ),
    #     media_stream_constraints={"video": False, "audio": True},
    #     audio_frame_callback=process_audio_frame,
    #     async_processing=True,
    # )
    
    # # Recording status indicator
    # if st.session_state.is_recording:
    #     st.warning("⚫ Recording in progress...")
    
    # Control buttons
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     if st.button("Start Recording"):
    #         st.session_state.is_recording = True
    #         st.session_state.audio_frames = []
    #         st.success("Recording started!")
    #         st.rerun()
            
    # with col2:
    #     if st.button("Stop Recording"):
    #         st.session_state.is_recording = False
    #         st.info("Recording stopped.")
    #         st.rerun()
            
    # with col3:
    #     if st.button("Transcribe"):
    #         if st.session_state.audio_frames:
    #             with st.spinner("Transcribing your voice..."):
    #                 wav_bytes = create_wav_from_frames(st.session_state.audio_frames)
    #                 if wav_bytes:
    #                     hf_api_token = os.getenv("HF_API_TOKEN")
    #                     transcript = transcribe_audio_whisper(wav_bytes, hf_api_token)
                        
    #                     if transcript and not transcript.startswith("[Error") and not transcript.startswith("[Voice transcription failed"):
    #                         st.success(f"Transcribed: {transcript}")
                            
    #                         # Process the transcribed input
    #                         st.session_state.chat_history.append({"role": "user", "content": transcript})
    #                         if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot.hf_api_token:
    #                             with st.spinner("JobSevak is thinking..."):
    #                                 response = st.session_state.chatbot.chat(transcript)
    #                             st.session_state.chat_history.append({"role": "assistant", "content": response})
            #                     # Clear audio frames after successful processing
            #                     st.session_state.audio_frames = []
            #                     st.rerun()
            #             else:
            #                 st.error(f"Sorry, could not transcribe your audio: {transcript}")
            #         else:
            #             st.error("Failed to create audio file from recording.")
            # else:
            #     st.warning("No audio recorded. Please record some audio first.")
    
    # --- Text Input Section ---
    with st.form(key="chat_form", clear_on_submit=True):
        text_input = st.text_input("Type your message here:", key="user_input", placeholder="Ask about jobs in Warangal, how to post a job, etc.")
        submit_button = st.form_submit_button("Send")
        if submit_button and text_input:
            st.session_state.chat_history.append({"role": "user", "content": text_input})
            if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot.hf_api_token:
                with st.spinner("JobSevak is thinking..."):
                    response = st.session_state.chatbot.chat(text_input)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "I am currently unable to process your request. The Hugging Face API token (HF_API_TOKEN) is missing. Please ensure it is set correctly in your .env file and restart the application."})

if __name__ == "__main__":
    main() 