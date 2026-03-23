import os
import requests
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

# --- Configuration ---
DATA_DIR = "./data"
DB_DIR = "./chroma_db"
# If running in Docker compose this is http://ollama:11434, otherwise default to localhost
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

st.set_page_config(page_title="Agentic File Share & Web QA", page_icon="🕵️")
st.title("🕵️ Intelligent File Share & Web Agent")
st.markdown("Ask anything. The agent will decide whether to search your **private local files** or the **public internet**. *Local data is strictly guarded from internet queries!*")

# --- Setup Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---
def get_available_models():
    """Fetches the list of downloaded models from the local Ollama instance."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    except requests.exceptions.RequestException:
        return []

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        st.info("Building database from files for the first time... this may take a minute depending on data size.")
        os.makedirs(DATA_DIR, exist_ok=True)
        loader = DirectoryLoader(DATA_DIR, glob="**/*.*", use_multithreading=True)
        docs = loader.load()
        if not docs:
            st.warning(f"No documents found in `{DATA_DIR}`. Please add some files to the share.")
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
        st.success("Database built successfully!")
        return vectorstore

def get_agent(vectorstore, selected_model):
    # Initialize Ollama Local LLM
    llm = ChatOllama(model=selected_model, base_url=OLLAMA_BASE_URL, temperature=0)

    # --- DEFINE TOOLS ---
    
    # Tool 1: Local File Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    local_search_tool = create_retriever_tool(
        retriever,
        "search_local_files",
        "Searches and returns excerpts from the user's private local file share. Always use this first for any questions about private internal data, projects, or documents."
    )

    # Tool 2: Web Search
    web_search_tool = DuckDuckGoSearchRun(
        name="search_web",
        description="Searches the public internet for current events, external knowledge, or information not found in the local files."
    )

    tools = [local_search_tool, web_search_tool]

    # --- DEFINE PRIVACY SYSTEM PROMPT ---
    system_prompt = (
        "You are an expert, autonomous AI assistant. "
        "You have access to two tools: `search_local_files` for private user data, and `search_web` for the public internet. "
        "CRITICAL PRIVACY RULE: You must NEVER include any information, names, sensitive details, or context from the local files into the queries you send to `search_web`. "
        "If a user asks a question, first determine if it's about their private files or a general internet query. "
        "Use the appropriate tool. Answer concisely. If you use a tool, mention what you searched."
    )

    # Create the React Agent (Requires a tool-calling capable model like llama3.1 or llama3.2)
    return create_react_agent(llm, tools, state_modifier=system_prompt)

# --- Sidebar Model Selection ---
with st.sidebar:
    st.header("⚙️ Settings")
    available_models = get_available_models()
    
    if not available_models:
        st.error(f"Cannot connect to Ollama at `{OLLAMA_BASE_URL}` or no models are downloaded.")
        st.markdown("Ensure the container is running and pull a model: `docker exec -it ollama ollama run llama3.2`")
        selected_model = None
    else:
        st.info("Note: Tool calling requires an advanced model like `llama3.1` or `llama3.2`.")
        selected_model = st.selectbox("Select local LLM model:", available_models)

# --- App Logic ---
vectorstore = get_vectorstore()

if vectorstore and selected_model:
    agent_executor = get_agent(vectorstore, selected_model)
    
    # Render chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input and execution
    if agent_executor and (user_query := st.chat_input("Ask a question about your files or the internet...")):
        # Append and show User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # Prepare chat history for LangGraph (List of Tuples: (role, content))
        chat_history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "assistant"
            chat_history.append((role, msg["content"]))

        # Generate output
        with st.spinner(f"Agent `{selected_model}` is deciding how to answer..."):
            try:
                # Invoke the LangGraph agent
                response = agent_executor.invoke({"messages": chat_history})
                
                # The final answer is the last message in the response state
                final_answer = response["messages"][-1].content

                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.chat_message("assistant").write(final_answer)
            except Exception as e:
                st.error(f"Error executing agent. Note: ensure you are using a model that supports Tool Calling (like llama3.1 or llama3.2). Error details: {e}")
