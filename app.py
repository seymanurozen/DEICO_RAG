import streamlit as st
from ragify import Ragify
import yaml


USER_DETAILS = {
    "user": {
        "password": "password",
        "chat_history": []
    },
    "admin": {
        "password": "password",
        "chat_history": []
    }
}


@st.cache_resource
def load_rag_chain(llm_name, embedding_name, chunk_size):
    return Ragify(
        pdf_paths=[
            r"./documents/METU_Regulation.pdf",
            r"./documents/ISStudentGuide_2023-2024_v1.5.pdf"
        ],
        llm_name=llm_name,
        embedding_name=embedding_name,
        chunk_size=chunk_size
    )


# Function to create user login
def login(username, password):
    # a simple user verification for demo purposes
    if username in list(USER_DETAILS.keys()):
        if password == USER_DETAILS[username]["password"]:
            return True
    return False


# Initialize session state for user login and conversation history
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = "Guest"
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'chats' not in st.session_state:
    st.session_state['chats'] = {}  # Dictionary to hold multiple chat histories
if 'current_chat' not in st.session_state:
    st.session_state['current_chat'] = "Default Chat"

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Create login form
if not st.session_state['logged_in']:
    st.title("Ragify - Login")

    # Create a form for username and password inputs
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # Use form submit button for Enter key support
        submit_button = st.form_submit_button("Login")

    # Handle login logic
    if submit_button:
        if login(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Logged in successfully!")
            st.rerun()  # Reload the app
        else:
            st.error("Invalid credentials. Please try again.")

# Main Chatbot Application
if st.session_state['logged_in']:
    if st.session_state["username"] == "admin":
        selected_model = st.sidebar.selectbox("Please select an LLM model", config["model_list"])
        selected_embedder = st.sidebar.selectbox("Please select an embedder", config["embedder_list"])
        selected_chunk_size = st.sidebar.number_input("Please select a chunk size", value=1000, step=100)
    else:
        selected_model = "llama3.2:latest"
        selected_embedder = "nomic-embed-text"
        selected_chunk_size = 1000


    st.sidebar.write("---")  # Divider for better organization
    st.session_state['current_chat'] = st.sidebar.selectbox("Previous chats list", list(st.session_state['chats'].keys()), index=len(list(st.session_state['chats'].keys()))-1)

    # Section to create a new chat

    new_chat_name = st.sidebar.text_input("Enter a name for the new chat")
    if st.sidebar.button("Create Chat"):
        if new_chat_name and new_chat_name not in st.session_state['chats']:
            st.session_state['chats'][new_chat_name] = []  # Initialize new chat history
            st.session_state['current_chat'] = new_chat_name
            st.rerun()  # Reload to reflect the new chat
        elif new_chat_name:
            st.sidebar.error("Chat name must be unique and non-empty.")

    col1, col2, col3 = st.columns(3)
    with col2:
        # st.image(r"./images/ragify_logo.jpg")
        st.header("Ragify")
    # Check if a current chat is selected and load its history
    st.session_state['chat_history'] = st.session_state['chats'].get(st.session_state['current_chat'], [])

    # Display chat messages from the selected chat
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    ragify_pipeline = load_rag_chain(llm_name=selected_model, embedding_name=selected_embedder, chunk_size=selected_chunk_size)

    # Handle new user input
    if question := st.chat_input("How can I help you?"):
        # Display user input
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state['chat_history'].append({"role": "user", "content": question})

        # Simulate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Responding..."):
                response, time_collapsed = ragify_pipeline.generate_response(question=question, chat_history=st.session_state['chat_history'])
            st.markdown(response)
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

        # Update the chat history in the selected chat
        st.session_state['chats'][st.session_state['current_chat']] = st.session_state['chat_history']
