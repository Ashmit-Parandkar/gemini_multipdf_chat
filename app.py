import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from streamlit.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)

logger = get_logger(__name__)

# read all pdf files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = splitter.split_text(text)
    return chunks  # list of strings


# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context just say, "answer is not available in the context"; don't provide a wrong answer.

    Context:
    {context}?

    Question: 
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        client=genai,
        temperature=0.3
    )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 

    # Retrieve documents WITH scores
    docs_with_scores = new_db.similarity_search_with_score(user_question, k=4)
    if not docs_with_scores:
        # No relevant docs
        return {"output_text": "No relevant context found."}, 0.0, 0.0

    # Separate out the docs from the scores
    docs = [item[0] for item in docs_with_scores]
    distances = [item[1] for item in docs_with_scores]
    
    # Convert FAISS distances into a "similarity" measure in [0..1] 
    similarities = [1 / (1 + d) for d in distances] if distances else [0]

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question},
                     return_only_outputs=True)

    # Compute two basic "scoring metrics" from the retrieved docs
    avg_similarity = sum(similarities) / len(similarities)
    max_similarity = max(similarities)

    return response, avg_similarity, max_similarity


def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon=""
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}
        ]

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Wait for new user input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # If the latest message is user, generate a new assistant response
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, avg_score, max_score = user_input(prompt)

                # Convert output_text to a single string in case it's a list
                raw_output = response.get("output_text", "")
                if isinstance(raw_output, list):
                    raw_output = "".join(raw_output)

                
                # Stream the text character by character
                placeholder = st.empty()
                full_response = ""
                for ch in raw_output:
                    full_response += ch
                    placeholder.markdown(full_response)

            # Once finished, store assistant's entire text
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Now display metrics RIGHT here in the assistant's message bubble
            # st.write("---")
            # st.write("**Similarity Metrics**")
            # Show them as text:
            # st.write(f"- **Average Similarity Score**: {avg_score:.3f}")
            # st.write(f"- **Max Similarity Score**: {max_score:.3f}")

            logger.info("**Similarity Metrics**")
            logger.info(f"- **Average Similarity Score**: {avg_score:.3f}")
            logger.info(f"- **Max Similarity Score**: {max_score:.3f}")
            # Alternatively, you can also show them as st.metric:
            # st.metric(label="Average Similarity Score", value=round(avg_score, 3))
            # st.metric(label="Max Similarity Score", value=round(max_score, 3))


if __name__ == "__main__":
    main()
