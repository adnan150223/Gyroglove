import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio
from dotenv import load_dotenv
import pyttsx3

# Load environment variables
load_dotenv()

# Access environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")



# Streamlit page configuration
st.set_page_config(page_title="GyroGlove AI Assistant", layout="wide")
st.title("🤖 GyroGlove AI Assistant for Hand Tremor")

# Memory to store conversation
memory = []  # List to store previous questions and answers

# Sidebar: Memory options and GyroGlove image
with st.sidebar:
    st.image("gyroglove.jpeg", width=300)  # Increased size of the GyroGlove image
    st.header("🔧 Features")  # Title for feature buttons

    # Clear Memory Button
    if st.button("🧹 Clear Memory"):
        memory.clear()
        st.write("Memory Cleared!")
    
    # Show Frequently Asked Questions
    st.markdown("<h3 style='color: #007BFF;'>Frequently Asked Questions:</h3>", unsafe_allow_html=True)
    
    # Enhanced FAQ presentation with clickable questions
    faq_style = """
    <style>
        .faq-button {
            display: block;
            background-color: #EFEFEF;
            color: #007BFF;
            padding: 10px;
            margin: 5px;
            border: none;
            width: 100%;
            text-align: left;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 5px;
        }
        .faq-button:hover {
            background-color: #D9D9D9;
        }
    </style>
    """
    st.markdown(faq_style, unsafe_allow_html=True)
    
    # Function for handling FAQ clicks and passing them as questions
    def handle_faq_click(question):
        return question

    # Display clickable FAQs
    if st.button("What is GyroGlove?", key="faq_gyroglove"):
        user_question = handle_faq_click("What is GyroGlove?")
    
    if st.button("How does GyroGlove work?", key="faq_work"):
        user_question = handle_faq_click("How does GyroGlove work?")
    
    if st.button("What is a tremor?", key="faq_tremor"):
        user_question = handle_faq_click("What is a tremor?")
    
    if st.button("How do tremors affect daily life?", key="faq_life"):
        user_question = handle_faq_click("How do tremors affect daily life?")
    
    if st.button("What are common medications for tremors?", key="faq_medications"):
        user_question = handle_faq_click("What are common medications for tremors?")
    
    # Feedback Button
    st.markdown("<h3 style='color: #007BFF;'>Feedback:</h3>", unsafe_allow_html=True)
    if st.button("📝 Provide Feedback"):
        feedback = st.text_area("Enter your feedback here:")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

    # Article and Video buttons
    st.markdown("<h3 style='color: #007BFF;'>Resources:</h3>", unsafe_allow_html=True)
    if st.button("📚 View Article on Hand Tremors"):
        st.markdown("[Read the Article](https://www.sciencedirect.com/topics/medicine-and-dentistry/hand-tremor)", unsafe_allow_html=True)

    if st.button("🎥 Watch Hand Tremor Video"):
        st.video("https://www.youtube.com/watch?v=2xihXzJNd8Y")

# Google API Key check
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set it in the .env file.")
else:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Step 1: Initialize Google Generative AI (Gemini) for answering questions (LLM)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Choose gemini-1.5-flash or another model if needed
    temperature=0.7,  # Control randomness of the response
    max_tokens=512,   # Set a limit for the number of tokens in the response
    max_retries=2,    # Number of retries in case of failure
    timeout=None,     # Timeout for the API call
)

# Step 2: Initialize Google Generative AI Embeddings (for embedding PDFs)
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"  # Embedding model for handling document embeddings
)

# Step 3: Initialize the Cassandra session using cassio
cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN,
    database_id=ASTRA_DB_ID
)

# Initialize Cassandra VectorStore for querying using the embeddings from Google Generative Embeddings
astra_vector_store = Cassandra(
    embedding=embed_model,  # Use the embeddings from the embedding model
    table_name="qa_mini_demo",  # The table where embeddings are stored
    session=None,  # Let cassio manage the session
    keyspace=None  # Replace with your keyspace name
)

# Wrap the vector store for querying purposes
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Function to retrieve relevant text from the PDF stored in Astra DB
def get_relevant_text_from_pdf(question: str):
    relevant_docs = astra_vector_store.similarity_search(question, k=3)
    if len(relevant_docs) == 0:
        st.warning("No relevant documents were retrieved from Astra DB.")
        return ""
    
    combined_context = " ".join([doc.page_content for doc in relevant_docs])
    return combined_context

# Function to check for questions about the creator
def check_for_creator_question(question: str):
    if "who is your creator" in question.lower() or "who is your owner" in question.lower():
        return (
            "Hello! I was created by **Muhammad Adnan**, a passionate Electrical Engineering "
            "student at **NUST** (7th semester). His **Final Year Project (FYP)** is to design a revolutionary "
            "'**GyroGlove**' aimed at reducing hand tremors. Inspired by this groundbreaking work, "
            "Adnan built me to assist with knowledge about the GyroGlove and hand tremors."
        )
    return None

# Create a prompt template for the LLM to generate answers based on the context and memory
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that remembers previous conversations. Use the following memory: {memory}. "
            "Here is the context: {context}. Answer the user's question.",
        ),
        ("human", "{question}"),  # The user's question will be inserted dynamically
    ]
)

# Combine LLM and prompt template for querying (Define the chain)
chain = prompt_template | llm

# Function to clean the response and remove metadata
def clean_response(response):
    return response.content.replace('\n', ' ').strip()

# Function to generate speech using pyttsx3 without saving the file
def generate_audio(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()

# Function to interact with the LLM, taking memory into account
def ask_question(question):
    memory.clear()  # Clear previous memory
    creator_response = check_for_creator_question(question)
    if creator_response:
        return creator_response
    context_from_pdf = get_relevant_text_from_pdf(question)
    if not context_from_pdf:
        return "No context retrieved from the PDF."
    memory_context = " ".join(memory)
    response = chain.invoke({
        "memory": memory_context,
        "context": context_from_pdf,
        "question": question
    })
    cleaned_answer = clean_response(response)
    memory.append(f"Q: {question}")
    memory.append(f"A: {cleaned_answer}")
    return cleaned_answer

# Main section for chat and AI interaction
st.subheader("Ask the AI about GyroGlove or Hand Tremors")

# Input for user question
user_question = st.text_input("Enter your question:")

# Button to ask the AI
if st.button("🧠 Ask the AI"):
    if user_question:
        answer = ask_question(user_question)
        st.write(f"**Answer:** {answer}")
        generate_audio(answer)
    else:
        st.warning("Please enter a question before asking the AI.")

# Footer with some info
st.markdown("---")
st.markdown("This AI-powered assistant provides insights on GyroGlove technology and its role in reducing hand tremors.")
