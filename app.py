import streamlit as st
from ragify import Ragify
import yaml
import json
import base64

# Set Streamlit page config (title and page icon)
st.set_page_config(page_title="Ragify", page_icon=r"images/logo_light.png")

# Hide Streamlit's default hamburger menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def get_base64_image(image_path: str) -> str:
    """
    Reads an image file and returns its contents in base64-encoded string format.

    Args:
        image_path (str): Local path to the image file.

    Returns:
        str: Base64-encoded string of the image content.
    """
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode()
    return encoded_image


# Load the application config (model list, embedder list, etc.) from a YAML file
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Load the user database from a JSON file (contains user credentials and chat histories)
with open('user_database.json', 'r') as file:
    USER_DETAILS = json.load(file)


@st.cache_resource
def load_rag_chain(llm_name: str, embedding_name: str, chunk_size: int) -> Ragify:
    """
    Constructs a Ragify pipeline object with the given LLM, embedder,
    and chunk size using PDFs located in ./documents/.

    Args:
        llm_name (str): The name of the Large Language Model to use.
        embedding_name (str): The name of the embedding model.
        chunk_size (int): The size of the text chunks for indexing.

    Returns:
        Ragify: An instance of Ragify with the loaded pipeline.
    """
    return Ragify(
        pdf_paths=[
            r"./documents/METU_Regulation.pdf",
            r"./documents/ISStudentGuide_2023-2024_v1.5.pdf"
        ],
        llm_name=llm_name,
        embedding_name=embedding_name,
        chunk_size=chunk_size
    )


def login(username: str, password: str) -> bool:
    """
    Verifies user credentials against the USER_DETAILS dictionary.

    Args:
        username (str): The username entered by the user.
        password (str): The password entered by the user.

    Returns:
        bool: True if the credentials are valid, False otherwise.
    """
    if username in list(USER_DETAILS.keys()):
        if password == USER_DETAILS[username]["password"]:
            return True
    return False


def register(username: str, password: str) -> bool:
    """
    Registers a new user if the username doesn't exist in the database.

    Args:
        username (str): New username to register.
        password (str): Password for the new user.

    Returns:
        bool: True if the registration is successful, False if the user already exists.
    """
    if username in list(USER_DETAILS.keys()):
        return False
    else:
        # Initialize user data structure in memory
        USER_DETAILS[username] = {
            "password": password,
            "chat_history": {"default": []}
        }
        return True


def logout() -> None:
    """
    Logs out the currently logged-in user by setting
    the 'logged_in' flag in session_state to False, then reruns the app.
    """
    st.session_state['logged_in'] = False
    specific_rerun()


def theme_change() -> None:
    """
    Toggles the theme (dark/light) by flipping the 'theme' key in session_state,
    then reruns the app to apply the changes.
    """
    st.session_state['theme'] = not st.session_state['theme']


def specific_rerun() -> None:
    """
    Saves the updated USER_DETAILS to the user_database.json file,
    then reruns the Streamlit app to reflect changes.
    """
    with open('user_database.json', 'w') as file:
        json.dump(USER_DETAILS, file, indent=4)
    st.rerun()


# Initialize session states for user login and theme toggling
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'theme' not in st.session_state:
    st.session_state['theme'] = True

# Create a toggle button for Light / Dark mode
col1, col2 = st.columns([4, 1])
with col2:
    st.toggle(
        label="Dark Mode" if st.session_state['theme'] else "Light Mode",
        value=st.session_state['theme'],
        on_change=theme_change
    )

# Paths to the light and dark mode images for background
light_mode_image_path = "images/background_light.jpg"
dark_mode_image_path = "images/background_dark.jpg"

# Convert images to base64 strings
light_mode_image = get_base64_image(light_mode_image_path)
dark_mode_image = get_base64_image(dark_mode_image_path)

# Determine which background image to use based on the theme
background_image = dark_mode_image if st.session_state['theme'] else light_mode_image

# Determine which logo to use based on the theme
logo_image = (
    r"images/logo_dark.png"
    if st.session_state['theme'] else r"images/logo_light.png"
)

text_color = "white" if st.session_state['theme'] else "black"
text_color_inverse = "black" if st.session_state['theme'] else "white"

# Header text, styled according to the theme
header = f"""
        <div style="text-align: justify; color:{text_color};">
            Welcome to <strong>Ragify</strong>, your personalized assistant designed to simplify the complexities of 
            METU's <em>"Rules and Regulations Governing Graduate Studies."</em> 
            Ragify helps graduate students navigate enrollment procedures, course requirements, thesis guidelines, and more with ease and confidence.
        </div>
        <hr style="border: 1px solid {text_color}; background-color: {text_color};">
        """

# Footer text, styled according to the theme
footer = f"""
        <div style="text-align: justify; font-size: 0.9em; margin-top: 50px; padding-top: 10px; color: {text_color};">
        This chatbot was developed by the <strong>Ragify team</strong>, including Barış Coşkun, Laya Moridsedaghat, Şeymanur Özen, and Şeyma Şimşek, on behalf of the <strong>Graduate School of Informatics</strong>. It was first released in <strong>December 2024</strong> and last updated in <strong>December 2024</strong>. For further inquiries, please contact <strong>Res. Assist. Şeyma Şimşek</strong> at 
        <a href="mailto:sseyma@metu.edu.tr" style="text-decoration: none; color: blue;">
        sseyma@metu.edu.tr
        </a>.
        </div>
        """

# CSS layout changes for dark mode or light mode
css_layout = f"""
        <style>
            body {{
                background-color: {text_color_inverse} !important;
            }}
            label {{
                color: {text_color} !important;
            }}
            input {{
                color: {text_color} !important;
                background-color: {text_color_inverse} !important;
                border: 1px solid {text_color} !important;
            }}
        </style>
        """

# CSS for the chat message container and input box
message_layout = f"""
        <style>
        div[data-testid="stChatMessageContent"] {{
            background-color: {text_color_inverse};
            color: {text_color};
        }}

        div[data-testid="stChatMessage"] {{
            background-color: {text_color_inverse};
        }}

        div[data-testid="stChatInput"] {{
            background-color: {text_color_inverse};
        }}

        </style>
    """

# Set the background image to the chosen theme
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{background_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Apply the message layout styling
st.markdown(message_layout, unsafe_allow_html=True)

# Center the main logo
col1, col2, col3 = st.columns(3)
with col2:
    st.image(logo_image)

# ---------------------
#  LOGIN/REGISTRATION
# ---------------------
if not st.session_state['logged_in']:
    # Header and page styling
    st.markdown(header, unsafe_allow_html=True)
    st.markdown(css_layout, unsafe_allow_html=True)

    # User input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login and Registration buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        login_button = st.button("Login", icon=":material/login:")
    with col2:
        register_button = st.button("Register", icon=":material/app_registration:")

    # Handle login logic
    if login_button:
        if login(username, password):
            # If credentials match, mark user as logged in
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Logged in successfully!")
            specific_rerun()  # Reload the app
        else:
            st.error("Invalid credentials. Please try again.")

    # Handle registration logic
    if register_button:
        if register(username, password):
            # If user successfully registered, mark user as logged in
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Registered successfully!")
            specific_rerun()
        else:
            st.error("User already exists. Please try again.")

# ---------------------
#  MAIN CHAT INTERFACE
# ---------------------
if st.session_state['logged_in']:
    # Create a sidebar with user greeting and logout button
    st.sidebar.title(f"Welcome {st.session_state['username']}")
    logout_button = st.sidebar.button("Log out", use_container_width=True, icon=":material/logout:")
    if logout_button:
        logout()

    # If admin user, allow model and embedder selection from config; else, fixed defaults
    if st.session_state["username"] == "admin":
        selected_model = st.sidebar.selectbox("Please select an LLM model", config["model_list"])
        selected_embedder = st.sidebar.selectbox("Please select an embedder", config["embedder_list"])
        selected_chunk_size = st.sidebar.number_input("Please select a chunk size", value=1000, step=100)
        st.sidebar.write("---")  # Divider for better organization
    else:
        # Defaults for non-admin users
        selected_model = "llama3.2:latest"
        selected_embedder = "nomic-embed-text"
        selected_chunk_size = 1000

    # Allow the user to switch between chat sessions
    current_chat_name = st.sidebar.selectbox(
        "Previous chats list",
        list(USER_DETAILS[st.session_state['username']]["chat_history"].keys()),
        index=len(list(USER_DETAILS[st.session_state['username']]["chat_history"].keys())) - 1
    )

    # Create a new chat session
    new_chat_name = st.sidebar.text_input("Enter a name for the new chat")
    if st.sidebar.button("Create Chat", icon=":material/add_circle:"):
        if new_chat_name and new_chat_name not in USER_DETAILS[st.session_state['username']]["chat_history"].keys():
            # Initialize the new chat in the user's history
            USER_DETAILS[st.session_state['username']]["chat_history"][new_chat_name] = []
            specific_rerun()
        elif new_chat_name:
            st.sidebar.error("Chat name must be unique and non-empty.")

    # Display the message history for the selected chat
    for message in USER_DETAILS[st.session_state['username']]["chat_history"][current_chat_name]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load the Ragify pipeline for LLM-based QA
    ragify_pipeline = load_rag_chain(
        llm_name=selected_model,
        embedding_name=selected_embedder,
        chunk_size=selected_chunk_size
    )

    # Chat input container
    with st.container():
        # Check if the user has submitted a question
        if question := st.chat_input("How can I help you?"):
            # Display user input in the conversation
            with st.chat_message("user"):
                st.markdown(question)

            # Generate response from the Ragify pipeline
            with st.chat_message("assistant"):
                with st.spinner("Responding..."):
                    response, time_collapsed = ragify_pipeline.generate_response(
                        question=question,
                        chat_history=USER_DETAILS[st.session_state['username']]["chat_history"][current_chat_name]
                    )
                st.markdown(response)

            # Save user input and the assistant's response to the chat history
            USER_DETAILS[st.session_state['username']]["chat_history"][current_chat_name].append(
                {"role": "user", "content": question}
            )
            USER_DETAILS[st.session_state['username']]["chat_history"][current_chat_name].append(
                {"role": "assistant", "content": response}
            )
            specific_rerun()

# Display a footer
st.markdown(footer,unsafe_allow_html=True)
