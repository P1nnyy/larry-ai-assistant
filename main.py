# main.py

"""
A beginner-friendly Python script to load and run the Alpaca 7B model offline
using llama.cpp through the llama-cpp-python library.
"""

# Step 1: Import the library
import textwrap
# llama_cpp is a Python wrapper for llama.cpp, which lets Python talk to your GGUF model.
from llama_cpp import Llama

# Step 2: Tell Python where your model file is located
MODEL_PATH = "/Users/pranavgupta/Desktop/Larry/language model/alpaca-7b.Q4_K_M.gguf"

# Step 3: Load the model
# n_ctx = number of tokens the model can "remember" in one go (context window)
# n_threads = number of CPU threads to use
print("Loading model... This may take a few seconds.")
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=4)

print("Model loaded successfully!")

history = []

import pickle  # For loading UFC embeddings
# Embedding setup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load UFC embeddings and texts from UFC_DATA folder ---
# The embeddings are stored as a pickled numpy array.
# The texts are stored as one line per text in a txt file.
UFC_EMBEDDINGS_PATH = "/Users/pranavgupta/Desktop/larry/UFC_DATA/ufc_embeddings.pkl"
UFC_TEXTS_PATH = "/Users/pranavgupta/Desktop/larry/UFC_DATA/ufc_texts.txt"

faiss_index = None
stored_texts = []
authoritative_facts = set()  # To store authoritative facts and corrections

try:
    # Load UFC embeddings
    with open(UFC_EMBEDDINGS_PATH, "rb") as f:
        ufc_embeddings = pickle.load(f)
    # Load UFC texts
    with open(UFC_TEXTS_PATH, "r", encoding="utf-8") as f:
        stored_texts = [line.strip() for line in f.readlines()]
    # Build FAISS index from embeddings
    dimension = ufc_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(ufc_embeddings))
    print("Loaded UFC embeddings and texts. FAISS index built with UFC knowledge.")
except Exception as e:
    print(f"Failed to load UFC data, starting with empty FAISS index: {e}")
    faiss_index = None
    stored_texts = []

conditioning_prompts = [
    "You are a helpful assistant named Larry.",
    "Be concise and factual, unless the user asks for a detailed story or explanation.",
    "If you are unsure, say so briefly.",
    "Always fact-check your responses against the provided UFC info and authoritative facts.",
    "If the user provides a correction or authoritative fact, update your knowledge accordingly."
]

base_prompt = "\n".join(conditioning_prompts) + "\n"

FEWSHOT = ""

# Initial greeting from Larry before entering the chat loop (randomized)
import random

intro_greetings = [
    "Hello! I'm Larry, your helpful assistant.",
    "Hi there! Larry at your service.",
    "Greetings! Larry here to assist you.",
    "Hey! I'm Larry, ready to help.",
    "Hello! Larry is here, how can I assist?"
]

greet = random.choice(intro_greetings)
print("Larry:", greet)
history.append({"role": "assistant", "content": greet})


def add_to_faiss(text, is_authoritative=False):
    global faiss_index, stored_texts, authoritative_facts
    # Check if text is a correction or authoritative fact
    # We treat authoritative facts differently by storing separately
    if is_authoritative:
        print("[Debug] Adding authoritative fact to knowledge base.")
        authoritative_facts.add(text)
    else:
        print("[Debug] Adding user or AI text to FAISS index.")
        embedding = embedder.encode([text])
        if faiss_index is None:
            dimension = embedding.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embedding))
        stored_texts.append(text)


def retrieve_context(query, top_k=5):
    global faiss_index, stored_texts, authoritative_facts
    print("[Debug] Retrieving context for query:", query)
    results = []
    if faiss_index is not None and len(stored_texts) > 0:
        query_embedding = embedder.encode([query])
        D, I = faiss_index.search(np.array(query_embedding), top_k)
        for idx in I[0]:
            if idx < len(stored_texts):
                results.append(stored_texts[idx])
    else:
        print("[Debug] FAISS index empty or no stored texts.")
    # Add authoritative facts to context always
    if authoritative_facts:
        print("[Debug] Adding authoritative facts to context.")
        results.extend(list(authoritative_facts))
    return results


# Step 4: Create a simple chat loop
# This will keep asking the user for input and give the AI's reply until you type 'exit'.
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        # Save FAISS index and stored texts before exiting
        try:
            if faiss_index is not None:
                faiss.write_index(faiss_index, "larry.index")
            with open("larry_texts.txt", "w", encoding="utf-8") as f:
                for text in stored_texts:
                    f.write(text + "\n")
            with open("larry_authoritative_facts.txt", "w", encoding="utf-8") as f:
                for fact in authoritative_facts:
                    f.write(fact + "\n")
            print("Saved FAISS index, stored texts, and authoritative facts. Goodbye!")
        except Exception as e:
            print(f"Error saving FAISS index or texts: {e}")
        break

    # Detect if user input is a correction or authoritative fact
    # Simple heuristic: if user input starts with "Correction:" or "Fact:" or "Correction -"
    is_authoritative = False
    lower_input = user_input.lower()
    if lower_input.startswith("correction:") or lower_input.startswith("fact:") or lower_input.startswith("correction -"):
        is_authoritative = True
        print("[Debug] User input detected as authoritative fact or correction.")

    # Add user input to history and FAISS or authoritative facts
    history.append({"role": "user", "content": user_input})
    add_to_faiss(user_input, is_authoritative=is_authoritative)

    context = retrieve_context(user_input)

    if len(history) > 20:
        old_messages = history[:-8]  # keep last 8
        # Extract key points, not just raw text
        key_points = []
        for msg in old_messages:
            content = msg['content']
            # take only first 100 chars of each message to keep it concise
            snippet = (content[:100] + "...") if len(content) > 100 else content
            key_points.append(f"{msg['role']}: {snippet}")
        # Join them and wrap nicely
        summary_text = " | ".join(key_points[:10])
        summary = "Summary so far (key points):\n" + textwrap.fill(summary_text, width=80)
        history = history[-8:]
    else:
        summary = ""

    # Build prompt with multi-shot style: include last 3-4 exchanges explicitly
    last_few = history[-8:]  # last 4 exchanges = 8 messages (user+assistant)
    prompt = base_prompt
    if summary:
        prompt += f"\n{summary}\n"

    if context:
        prompt += "\nRelevant UFC info and authoritative facts:\n"
        for c in context:
            prompt += f"- {c}\n"

    for msg in history[-8:]:
        if msg["role"] == "user":
            prompt += f"\nPranav: {msg['content']}"
        else:
            prompt += f"\nLarry: {msg['content']}"

    prompt += "\nLarry:"

    print("[Debug] Sending prompt to model...")
    response = llm(
        prompt,
        max_tokens=1000,
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\nPranav:", "\nLarry:"]
    )

    ai_response = response["choices"][0]["text"].strip()
    print("[Debug] Model response received.")
    history.append({"role": "assistant", "content": ai_response})
    print("AI:", ai_response)
    add_to_faiss(ai_response)