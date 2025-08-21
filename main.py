# main.py

"""
A beginner-friendly Python script to load and run the Alpaca 7B model offline
using llama.cpp through the llama-cpp-python library.
"""

# Step 1: Import the library
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

# Embedding setup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = None
stored_texts = []

# Try loading previously saved FAISS index and stored texts
if os.path.exists("larry.index") and os.path.exists("larry_texts.txt"):
    try:
        faiss_index = faiss.read_index("larry.index")
        with open("larry_texts.txt", "r", encoding="utf-8") as f:
            stored_texts = [line.strip() for line in f.readlines()]
        print("Loaded existing FAISS index and stored texts.")
    except Exception as e:
        print(f"Failed to load FAISS index or texts: {e}")

conditioning_prompts = [
    "You are a helpful assistant named Larry.",
    "Be concise and factual, unless the user asks for a detailed story or explanation.",
    "If you are unsure, say so briefly."
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


def add_to_faiss(text):
    global faiss_index, stored_texts
    embedding = embedder.encode([text])
    if faiss_index is None:
        dimension = embedding.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embedding))
    stored_texts.append(text)

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
            print("Saved FAISS index and stored texts. Goodbye!")
        except Exception as e:
            print(f"Error saving FAISS index or texts: {e}")
        break

    # Add user input to history and FAISS
    history.append({"role": "user", "content": user_input})
    add_to_faiss(user_input)

    if len(history) > 20:
        old_messages = history[:-8]  # keep last 8
        summary = "Summary so far: " + " | ".join(
            [f"{msg['role']}: {msg['content']}" for msg in old_messages[:10]]
        )
        history = history[-8:]
    else:
        summary = ""

    # Build prompt with multi-shot style: include last 3-4 exchanges explicitly
    last_few = history[-8:]  # last 4 exchanges = 8 messages (user+assistant)
    prompt = base_prompt
    if summary:
        prompt += f"\n{summary}\n"

    for msg in history[-8:]:
        if msg["role"] == "user":
            prompt += f"\nPranav: {msg['content']}"
        else:
            prompt += f"\nLarry: {msg['content']}"

    prompt += "\nLarry:"

    response = llm(
        prompt,
        max_tokens=1000,
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\nPranav:", "\nLarry:"]
    )

    ai_response = response["choices"][0]["text"].strip()
    history.append({"role": "assistant", "content": ai_response})
    print("AI:", ai_response)
    add_to_faiss(ai_response)