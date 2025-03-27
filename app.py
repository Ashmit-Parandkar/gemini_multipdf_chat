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
from dotenv import load_dotenv

# For BLEU & ROUGE scoring
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Make sure you have the relevant NLTK data installed:
# nltk.download('punkt')  # if you see errors about missing tokenizers

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)


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
        # If no docs are returned, just pass empty placeholders
        return {"output_text": "No relevant context found"}, 0, 0, 0, 0

    # Sort them ascending by distance (lowest distance = highest similarity)
    docs_with_scores.sort(key=lambda x: x[1])

    # Separate out the docs from the scores
    docs = [item[0] for item in docs_with_scores]
    distances = [item[1] for item in docs_with_scores]

    # Convert FAISS distances into a naive [0..1] similarity
    # e.g. similarity = 1 / (1 + distance)
    similarities = [1 / (1 + d) for d in distances]

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question},
                     return_only_outputs=True)

    # Compute two basic "scoring metrics" from the retrieved docs
    avg_similarity = sum(similarities) / len(similarities)
    max_similarity = max(similarities)

    # Let's also compute BLEU & ROUGE vs. the single highest-similarity doc
    # This is a naive approach to show overlap. 
    highest_sim_doc_text = docs_with_scores[0][0].page_content

    # Convert doc text + response text to token lists
    reference_tokens = nltk.word_tokenize(highest_sim_doc_text)
    candidate_tokens = nltk.word_tokenize(response["output_text"])

    # BLEU
    # We pass a list of references = [reference_tokens], 
    # plus candidate tokens. We use a small smoothing to avoid zero scores.
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, 
                               smoothing_function=smoothie)

    # ROUGE
    # We'll measure ROUGE-1 and ROUGE-L (F1).
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(highest_sim_doc_text, response["output_text"])
    rouge_1 = rouge_scores["rouge1"].fmeasure
    rouge_l = rouge_scores["rougeL"].fmeasure

    return response, avg_similarity, max_similarity, bleu_score, rouge_1


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

    # Check for user input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # If the latest message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, avg_score, max_score, bleu, rouge_1 = user_input(prompt)
                placeholder = st.empty()
                full_response = ""

                # response["output_text"] might be a string or a list of strings
                # If it's a list of strings, let's join them
                output_text = response["output_text"]
                if isinstance(output_text, list):
                    output_text = "".join(output_text)

                # Show the assistant's text in "real-time" style
                for ch in output_text:
                    full_response += ch
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

            # Display the four metrics after the answer
            st.write("---")
            st.metric(label="Average Similarity", value=round(avg_score, 3))
            st.metric(label="Max Similarity", value=round(max_score, 3))
            st.metric(label="BLEU Score", value=round(bleu, 3))
            st.metric(label="ROUGE-1 F1", value=round(rouge_1, 3))


if __name__ == "__main__":
    main()
