from ragify import Ragify
import streamlit as st


@st.cache_resource
def load_rag_chain(llm_name):
    return Ragify(llm_name=llm_name)


# Function to create user login
def login(username, password):
    # a simple user verification for demo purposes
    return username == "user" and password == "password"


# Initialize session state for user login and conversation history
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = "Guest"
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Create login form
if not st.session_state['logged_in']:
    st.title("Ragify - Login")

    # User inputs
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Logged in successfully!")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Welcome {st.session_state.username}"})
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

# Main Chatbot Application
if st.session_state['logged_in']:
    st.title("Ragify")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("How can I help you?"):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Responding..."):
                response = Ragify.generate_response(question=question)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})