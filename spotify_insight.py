# Intialize GEMINI AI API KEY
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key and credentials path
API_KEY = os.getenv("GOOGLE_CREDENTIALS")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Configure the generative AI with the API key
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    raise ValueError("API_KEY is not set in the environment variables.")

# Set the GOOGLE_APPLICATION_CREDENTIALS if you are using a service account
if CREDENTIALS_PATH:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
else:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the environment variables.")

print(f"API_KEY: {API_KEY}")
print(f"CREDENTIALS_PATH: {CREDENTIALS_PATH}")

# Establish LLM Model
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-pro",
                               google_api_key=API_KEY,
                               temperature=0.6,
                               stream = True,
                               max_output_token=256
                               )
print(f"Model: {model}")

# Load the vector store from the specified directory
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

vectorstore_path = "C:\Mekari\chroma_spotify_2"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
vectorstore = Chroma(persist_directory=vectorstore_path)
print(len(vectorstore.get()['documents']))

# Prompt Engineering
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


# Streamlit Application
import streamlit as st
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.title("Spotify Reviews Insights")

# Question input

with st.form(key="question_form"):
    question = st.text_input("Enter your question about Spotify reviews:")
    submit_button = st.form_submit_button(label="Get Insight")

if submit_button:
    if question:
        # Retrieve the API key and credentials path
        API_KEY = os.getenv("GOOGLE_CREDENTIALS")
        CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # Configure the generative AI with the API key
        if API_KEY:
            genai.configure(api_key=API_KEY)
        else:
            raise ValueError("API_KEY is not set in the environment variables.")

        # Set the GOOGLE_APPLICATION_CREDENTIALS if you are using a service account
        if CREDENTIALS_PATH:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
        else:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the environment variables.")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)

        # Create retriever from the vector store
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 100, "fetch_k": 1000})
        
        prompt_template = """
        You are a friendly Q&A Chatbot capable of extracting meaningful information from the reviews data of the music streaming application: Spotify.
        You should provide brief and insightful responses to a variety of management questions.
        Since you're analyzing for the company's executives, give a summary of no more than a short paragraph.

        Context:\n {context}\n
        Question:\n {question}\n

        Answer (Maximum 50 Words):
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Load the QA chain
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # Retrieve documents
        docs = retriever.invoke(question)
        print(docs)
        if not docs:
            st.error("No documents were retrieved for the given question.")
        else:
            # Generate response
            response = chain.invoke({"input_documents": docs, "question": question}, return_only_outputs=True)

            # Typing animation for response
            response_placeholder = st.empty()
            full_response = response.get("output_text", "")
            typing_speed = 0.05  # Adjust the speed as needed

            response_text = ""
            for char in full_response:
                response_text += char
                response_placeholder.markdown(
                    f"<div style='border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px; background-color: #034a2e;'>{response_text}</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(typing_speed)

            # # Evaluate response quality
            # def evaluate_response_quality(docs, generated_response):
            #     # Get vector representation of the context (input documents)
            #     context_vectors = embeddings.embed_documents([doc.page_content for doc in docs])

            #     # Check if context_vectors are not empty
            #     if not context_vectors or len(context_vectors) == 0:
            #         raise ValueError("Context vectors are empty.")

            #     # Get vector representation of the generated response
            #     response_vector = embeddings.embed_query(generated_response)

            #     # Check if response_vector is not empty
            #     if response_vector is None or len(response_vector) == 0:
            #         raise ValueError("Failed to generate embedding for the response.")

            #     # Convert response vector to a NumPy array
            #     response_vector = np.array(response_vector)

            #     # Reshape if necessary
            #     if len(response_vector.shape) == 1:
            #         response_vector = response_vector.reshape(1, -1)

            #     # Calculate cosine similarity
            #     similarities = cosine_similarity(response_vector, context_vectors)
            #     avg_similarity = np.mean(similarities)

            #     return avg_similarity

            # try:
            #     quality_score = evaluate_response_quality(docs, full_response)
            #     st.write(f"**Response Quality Score:** {quality_score:.2f}")
            # except ValueError as e:
            #     st.error(f"Error evaluating response quality: {str(e)}")
    else:
        st.error("Please enter a question.")