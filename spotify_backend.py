import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_CREDENTIALS")

# Configure Google Generative AI
genai.configure(api_key=API_KEY)

# Load your dataset
dataset_path = "C:/Mekari/SPOTIFY_REVIEWS.csv" 
df = pd.read_csv(dataset_path)

# Data cleansing
df = df.filter(['review_text', 'review_rating', 'review_likes', 'review_timestamp'])
new_reviews = df[(df['review_timestamp'] >= '2020-01-01') & (df['review_timestamp'] <= '2023-12-31')]
insightful_reviews = new_reviews[new_reviews['review_likes'] >= 50]

# Set up documents and embeddings
loader = DataFrameLoader(insightful_reviews, page_content_column="review_text")
documents = loader.load()
split_texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
reviews = split_texts.split_documents(documents)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
print(len(reviews))

# Create the Chroma vector store with the embedding function
# vectorstore = Chroma(embedding_function=embeddings.embed_documents)

max_batch_size = 1500  # Define maximum batch size

# Process documents in smaller batches
for i in range(0, len(reviews), max_batch_size):
    batch_reviews = reviews[i:i + max_batch_size]  # Create a batch of documents
    vectorstore = Chroma.from_documents(documents=batch_reviews, embedding=embeddings, persist_directory="C:/Mekari/chroma_spotify_2")
    print(f"Processed batch {i // max_batch_size + 1}/{(len(reviews) // max_batch_size) + 1}")

print(batch_reviews)
print(len(vectorstore.get()['documents']))
# vectorstore.persist("C:/Mekari/chroma_vectorstore_spotify")
# print("Vectorstore persisted.")