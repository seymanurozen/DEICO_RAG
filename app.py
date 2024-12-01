import streamlit as st
from ragify import Ragify

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
    return username == "user" and password == "password"
    
#ragify_pipeline = load_rag_chain(llm_name="llama3.2:1b")

# Initialize session state for selected pipeline
if 'selected_pipeline' not in st.session_state:
    st.session_state['selected_pipeline'] = "Pipeline 1"  # Default pipeline



ragify_pipeline = load_rag_chain(llm_name="llama3.2:latest", embedding_name="nomic-embed-text", chunk_size=1000)

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

    # Sidebar for chat selection and creation
    st.sidebar.title("Chats")
    chat_names = list(st.session_state['chats'].keys())

    # Show existing chats one by one
    for chat_name in chat_names:
        if st.sidebar.button(chat_name):
            st.session_state['current_chat'] = chat_name

    # Section to create a new chat
    st.sidebar.write("---")  # Divider for better organization
    st.sidebar.subheader("Create a New Chat")

    new_chat_name = st.sidebar.text_input("Enter a name for the new chat", key="new_chat_name")
    if st.sidebar.button("Create Chat", key="create_chat"):
        if new_chat_name and new_chat_name not in st.session_state['chats']:
            st.session_state['chats'][new_chat_name] = []  # Initialize new chat history
            st.session_state['current_chat'] = new_chat_name
            st.rerun()  # Reload to reflect the new chat
        elif new_chat_name:
            st.sidebar.error("Chat name must be unique and non-empty.")

    current_chat_history = st.session_state['chats'].get(st.session_state['current_chat'], [])

    # Dropdown menu

    # Create a layout with two columns
    col1, col2, col3 = st.columns([4, 1, 1])  # Adjust the proportions to make the second column smaller

    # Add the dropdown to the right corner
    with col2:
        options = ["Model 1", "Model 2", "Model 3"]
        selected_model = st.selectbox("", options, label_visibility="collapsed")  # Hide the label for a cleaner look 

    # Add the dropdown to the right corner
    with col3:
        options = ["Embedder 1", "Embedder 2", "Embedder 3"]
        selected_embedder = st.selectbox("", options, label_visibility="collapsed")  # Hide the label for a cleaner look

    with col1:
        st.title("Ragify")
        # Display the selected option
        
        st.write(f"You selected: {selected_model} {selected_embedder}")
        st.subheader(f"Chat: {st.session_state['current_chat']}")

    # Check if a current chat is selected and load its history
    current_chat_history = st.session_state['chats'].get(st.session_state['current_chat'], [])
    st.session_state['chat_history'] = current_chat_history

    # Display chat messages from the selected chat
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if question := st.chat_input("How can I help you?"):
        # Display user input
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state['chat_history'].append({"role": "user", "content": question})

        # Simulate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Responding..."):
                #response = ragify_pipeline.generate_response(question=question)
                response = f"Simulated response to: {question}"
            st.markdown(response)
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

        # Update the chat history in the selected chat
        st.session_state['chats'][st.session_state['current_chat']] = st.session_state['chat_history']
