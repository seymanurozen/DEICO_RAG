import streamlit as st
import numpy as np

import streamlit as st

# Function to create user login
def login(username, password):
    # a simple user verification for demo purposes
    return username == "user" and password == "password"

# Initialize session state for user login and conversation history
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# Create login form
if not st.session_state['logged_in']:
    st.title("Echo Chatbot - Login")
    
    # User inputs
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid credentials. Please try again.")

# Main Chatbot Application
if st.session_state['logged_in']:
    st.title("Echo Chatbot")

    # Display conversation history
    st.subheader("Conversation History:")
    for user_message in st.session_state['conversation_history']:
        st.text(f"You: {user_message}")
        st.text(f"Bot: {user_message}")  # Echo back the user input

    # Input for user message
    user_input = st.text_input("You:", "")
    
    # Process user input
    if user_input:
        st.session_state['conversation_history'].append(user_input)
        st.text(f"Bot: {user_input}")  # Echo back the user input

    # Logout button
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['conversation_history'] = []  # Clear history on logout
        st.success("You have logged out.")

