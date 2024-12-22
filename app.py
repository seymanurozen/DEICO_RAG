import streamlit as st
from ragify import Ragify
import yaml
import json
import base64


def get_base64_image(image_path):
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode()
    return encoded_image


with open("config.yml", "r") as file:
    config = yaml.safe_load(file)


with open('user_database.json', 'r') as file:
    USER_DETAILS = json.load(file)


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


def register(username, password):
    if username in list(USER_DETAILS.keys()):
        return False
    else:
        USER_DETAILS[username] = {
            "password": password,
            "chat_history":
                {"default": []}
        }
        return True


def logout():
    st.session_state['logged_in'] = False
    specific_rerun()


def specific_rerun():
    with open('user_database.json', 'w') as file:
        json.dump(USER_DETAILS, file, indent=4)
    st.rerun()


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# Initialize session state for user login and conversation history
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Create login form
if not st.session_state['logged_in']:

    # Paths to the light and dark mode images
    light_mode_image_path = "images/background_light.jpg"
    dark_mode_image_path = "images/background_dark.jpg"

    # Encode both images
    light_mode_image = get_base64_image(light_mode_image_path)
    dark_mode_image = get_base64_image(dark_mode_image_path)

    # Create a manual toggle for Light or Dark mode
    col1, col2, col3 = st.columns([5,1,1])
    with col3:         
        theme_choice = st.radio("Select Theme", [":rainbow[Light]", ":rainbow[Dark]"], index=0)

    # Choose the background image based on the selected theme
    background_image = dark_mode_image if theme_choice == ":rainbow[Dark]" else light_mode_image
    logo_image =  r"./images/inverted_logo_image.png" if theme_choice == ":rainbow[Dark]" else r"./images/ragify_logo2.png"
    header = """
            <div style="text-align: justify; color:white;">
                Welcome to <strong>Ragify</strong>, your personalized assistant designed to simplify the complexities of 
                METU's <em>"Rules and Regulations Governing Graduate Studies."</em> 
                Ragify helps graduate students navigate enrollment procedures, course requirements, thesis guidelines, and more with ease and confidence.
            </div>
            
            <hr style="border: 1px solid white; background-color: white;">
            """ if theme_choice == ":rainbow[Dark]" else """
            <div style="text-align: justify;">
                Welcome to <strong>Ragify</strong>, your personalized assistant designed to simplify the complexities of 
                METU's <em>"Rules and Regulations Governing Graduate Studies."</em> 
                Ragify helps graduate students navigate enrollment procedures, course requirements, thesis guidelines, and more with ease and confidence.
            </div>
            
            <hr>  <!-- Add a horizontal line -->
            """
    footer = """
            <div style="text-align: justify; font-size: 0.9em; margin-top: 50px; border-top: 1px solid #ddd; padding-top: 10px; color: white;">
            This chatbot was developed by the <strong>Ragify team</strong>, including Barış Coşkun, Laya Moridsedaghat, Şeymanur Özen, and Şeyma Şimşek, on behalf of the <strong>Graduate School of Informatics</strong>. It was first released in <strong>December 2024</strong> and last updated in <strong>December 2024</strong>. For further inquiries, please contact <strong>Res. Assist. Şeyma Şimşek</strong> at 
            <a href="mailto:sseyma@metu.edu.tr" style="text-decoration: none; color: lightblue;">
            sseyma@metu.edu.tr
            </a>.
            </div>
            """ if theme_choice == ":rainbow[Dark]" else """
            <div style="text-align: justify; font-size: 0.9em; margin-top: 50px; border-top: 1px solid #ddd; padding-top: 10px;">
            This chatbot was developed by the <strong>Ragify team</strong>, including Barış Coşkun, Laya Moridsedaghat, Şeymanur Özen, and Şeyma Şimşek, on behalf of the <strong>Graduate School of Informatics</strong>. It was first released in <strong>December 2024</strong> and last updated in <strong>December 2024</strong>. For further inquiries, please contact <strong>Res. Assist. Şeyma Şimşek</strong> at 
            <a href="mailto:sseyma@metu.edu.tr" style="text-decoration: none; color: blue;">
            sseyma@metu.edu.tr
            </a>.
            </div>
            """
    css_layout = """
            <style>
                /* Style for text input labels */
                label {
                    color: white !important;
                }

                /* Optional: Style for text input boxes */
                input {
                    color: white !important;
                    background-color: black !important;
                    border: 1px solid white !important;
                }
            </style>
            """ if theme_choice == ":rainbow[Dark]" else"""
            <style>
                /* Style for text input labels */
                label {
                    color: black !important;
                }

                /* Optional: Style for text input boxes */
                input {
                    color: black !important;
                    background-color: white !important;
                    border: 1px solid black !important;
                }
            </style>
            """

    # Inject CSS for the background image
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

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(logo_image, use_column_width=True)

    st.markdown(header, unsafe_allow_html=True)
    st.markdown(css_layout,unsafe_allow_html=True)

    # Text input fields with white labels
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        login_button = st.button("Login", icon=":material/login:")
    with col2:
        register_button = st.button("Register", icon=":material/app_registration:")

    # Handle login logic
    if login_button:
        if login(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Logged in successfully!")
            specific_rerun()  # Reload the app
        else:
            st.error("Invalid credentials. Please try again.")
    if register_button:
        if register(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Registered successfully!")
            specific_rerun()
        else:
            st.error("User already exists. Please try again.")

    # Footer or informational text at the bottom of the page
    st.markdown(footer,unsafe_allow_html=True)

# Main Chatbot Application
if st.session_state['logged_in']:
    st.sidebar.title(f"Welcome {st.session_state['username']}")
    logout_button = st.sidebar.button("Log out", use_container_width=True, icon=":material/logout:")
    if logout_button:
        logout()

    if st.session_state["username"] == "admin":
        selected_model = st.sidebar.selectbox("Please select an LLM model", config["model_list"])
        selected_embedder = st.sidebar.selectbox("Please select an embedder", config["embedder_list"])
        selected_chunk_size = st.sidebar.number_input("Please select a chunk size", value=1000, step=100)
        st.sidebar.write("---")  # Divider for better organization

    else:
        selected_model = "llama3.2:latest"
        selected_embedder = "nomic-embed-text"
        selected_chunk_size = 1000

    current_chat_name = st.sidebar.selectbox("Previous chats list", list(USER_DETAILS[st.session_state['username']]["chat_history"].keys()), index=len(list(USER_DETAILS[st.session_state['username']]["chat_history"].keys()))-1)

    # Section to create a new chat
    new_chat_name = st.sidebar.text_input("Enter a name for the new chat")
    if st.sidebar.button("Create Chat", icon=":material/add_circle:"):
        if new_chat_name and new_chat_name not in USER_DETAILS[st.session_state['username']]["chat_history"].keys():
            USER_DETAILS[st.session_state['username']]["chat_history"][new_chat_name] = []  # Initialize new chat history
            specific_rerun() # Reload to reflect the new chat
        elif new_chat_name:
            st.sidebar.error("Chat name must be unique and non-empty.")
    
    st.sidebar.markdown("""
        <div style="text-align: justify; font-size: 0.9em; margin-top: 50px; border-top: 1px solid #ddd; padding-top: 10px;">
            This chatbot was developed by the <strong>Ragify team</strong>, including Barış Coşkun, 
            Laya Moridsedaghat, Şeymanur Özen, and Şeyma Şimşek, on behalf of the 
            <strong>Graduate School of Informatics</strong>. It was first released in 
            <strong>December 2024</strong> and last updated in <strong>December 2024</strong>. 
            For further inquiries, please contact <strong>Res. Assist. Şeyma Şimşek</strong> at 
            <a href="mailto:sseyma@metu.edu.tr" style="text-decoration: none; color: blue;">
            sseyma@metu.edu.tr
            </a>.
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(r"./images/ragify_logo.jpg", use_column_width=True)
        #st.header("Ragify")

    # Display chat messages from the selected chat
    for message in USER_DETAILS[st.session_state['username']]["chat_history"][current_chat_name]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    ragify_pipeline = load_rag_chain(llm_name=selected_model, embedding_name=selected_embedder, chunk_size=selected_chunk_size)

    # Handle new user input
    if question := st.chat_input("How can I help you?"):
        # Display user input
        with st.chat_message("user"):
            st.markdown(question)

        # Simulate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Responding..."):
                response, time_collapsed = ragify_pipeline.generate_response(
                   question=question,
                   chat_history=USER_DETAILS[st.session_state['username']]["chat_history"][current_chat_name]
                )

            st.markdown(response)

        USER_DETAILS[st.session_state['username']]["chat_history"][current_chat_name].append(
            {"role": "user", "content": question})
        USER_DETAILS[st.session_state['username']]["chat_history"][current_chat_name].append(
            {"role": "assistant", "content": response})

        with open('user_database.json', 'w') as file:
            json.dump(USER_DETAILS, file, indent=4)