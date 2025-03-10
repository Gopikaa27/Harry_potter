import os
import fitz
import re
import nltk
import gradio as gr
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the Hugging Face API token from the environment variable
api_token = os.getenv("HF_API_TOKEN")

# Load pre-trained models with authentication
llama_model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=api_token)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, use_auth_token=api_token)

# Function to extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return text

# Function to clean text
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', '', text)
    return text.strip()

# Function to chunk text
def chunk_text(text, max_length=500):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Load and preprocess documents
pdf_text = extract_text_from_pdf("harrypotter.pdf")
summary_text = extract_text_from_pdf("harrypotter_summary.pdf")

cleaned_text = clean_text(pdf_text)
cleaned_summary = clean_text(summary_text)

chunks_text = chunk_text(cleaned_text)
chunks_summary = chunk_text(cleaned_summary)

# Encode embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorized_text = np.array([embed_model.encode(chunk) for chunk in chunks_text], dtype="float32")
vectorized_summary = np.array([embed_model.encode(chunk) for chunk in chunks_summary], dtype="float32")

# Create FAISS index
index_text = faiss.IndexFlatL2(vectorized_text.shape[1])
index_text.add(vectorized_text)

index_summary = faiss.IndexFlatL2(vectorized_summary.shape[1])
index_summary.add(vectorized_summary)

# Retrieval function
def retrieve_relevant_passage(query, top_k=3):
    query_embedding = embed_model.encode(query).reshape(1, -1)
    distances, indices = index_text.search(query_embedding, top_k)
    results = [chunks_text[i] for i in indices[0]]
    return results

# Answer generation
def generate_answer(query):
    relevant_passages = retrieve_relevant_passage(query)
    context = " ".join(relevant_passages)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = llama_model.generate(input_ids, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Gradio UI
def chatbot_interface(user_query):
    return generate_answer(user_query)

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(placeholder="Ask about Harry Potter..."),
    outputs="text",
    title="Harry Potter Chatbot",
    description="Ask me anything about Harry Potter!"
)

if __name__ == "__main__":
    iface.launch()
    
    