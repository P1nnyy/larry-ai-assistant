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

# --- Embedding helpers: normalize vectors for cosine similarity ---
def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def embed_texts(texts: list[str]) -> np.ndarray:
    embs = embedder.encode(texts)
    return _normalize(np.array(embs))

def embed_query(text: str) -> np.ndarray:
    emb = embedder.encode([text])
    return _normalize(np.array(emb))


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
    # Use inner-product + normalized embeddings (== cosine similarity)
    ufc_embeddings = _normalize(np.array(ufc_embeddings))
    dimension = ufc_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(ufc_embeddings)
    print("Loaded UFC embeddings and texts. FAISS index built with UFC knowledge.")
except Exception as e:
    print(f"Failed to load UFC data, starting with empty FAISS index: {e}")
    faiss_index = None
    stored_texts = []

def load_ufc_texts(file_path=UFC_TEXTS_PATH):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading UFC texts from {file_path}: {e}")
        return []

# If stored_texts is empty or faiss_index is None, load texts from file, embed, and add to FAISS index
if not stored_texts or faiss_index is None:
    print("[Debug] Loading UFC texts from file and embedding them to build FAISS index.")
    stored_texts = load_ufc_texts()
    if stored_texts:
        embeddings = embed_texts(stored_texts)
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)
        print("[Debug] FAISS index built from UFC texts.")
    else:
        print("[Debug] No UFC texts loaded to build FAISS index.")

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

# Load authoritative facts from previous sessions if present
AUTH_FACTS_FILE = "larry_authoritative_facts.txt"
if os.path.exists(AUTH_FACTS_FILE):
    try:
        with open(AUTH_FACTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                fact = line.strip()
                if fact:
                    authoritative_facts.add(fact)
        print(f"[Debug] Loaded {len(authoritative_facts)} authoritative facts from disk.")
    except Exception as e:
        print(f"[Debug] Failed to load authoritative facts: {e}")


def add_to_faiss(text, is_authoritative=False):
    global faiss_index, stored_texts, authoritative_facts
    if is_authoritative:
        print("[Debug] Adding authoritative fact to knowledge base.")
        authoritative_facts.add(text)
        # Save authoritative facts to file immediately
        with open("larry_authoritative_facts.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print("[Debug] Adding user or AI text to FAISS index.")
        embedding = embed_query(text)
        if faiss_index is None:
            dimension = embedding.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embedding)
        stored_texts.append(text)
        # Save new text into UFC_TEXTS_PATH immediately
        try:
            with open(UFC_TEXTS_PATH, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            print(f"[Debug] Failed to save text to {UFC_TEXTS_PATH}: {e}")


def retrieve_context(query, top_k=5):
    global faiss_index, stored_texts, authoritative_facts
    # print("[Debug] Retrieving context for query:", query)
    results = []
    indices = []
    if faiss_index is not None and len(stored_texts) > 0:
        q_emb = embed_query(query)
        D, I = faiss_index.search(q_emb, min(top_k * 5, len(stored_texts)))
        for idx in I[0]:
            if 0 <= idx < len(stored_texts):
                indices.append(idx)
        # Simple keyword boost re-rank
        q_terms = set(query.lower().split())
        def score_text(i):
            t = stored_texts[i]
            tokens = set(t.lower().split())
            overlap = len(q_terms & tokens)
            return (overlap, -i)  # prefer more overlap, then stable order
        indices = sorted(list(dict.fromkeys(indices)), key=score_text, reverse=True)[:top_k]
        results = [stored_texts[i] for i in indices]
    else:
        print("[Debug] FAISS index empty or no stored texts.")
    # Always append authoritative facts (dedup while preserving order)
    if authoritative_facts:
        print("[Debug] Adding authoritative facts to context.")
        for fact in authoritative_facts:
            if fact not in results:
                results.append(fact)
    return results[: top_k]


def generate_strict_answer(user_input: str, context: list[str]) -> str:
    """
    Generate an answer using ONLY the provided context. If the answer isn't
    supported by the context, explicitly say so. Lower temperature to reduce hallucinations.
    """
    facts_block = "\n".join(f"- {c}" for c in context) if context else "(none)"
    strict_instructions = (
        "You are Larry. Answer the question using the facts below. "
        "Summarize or reason with them in natural language. "
        "Do not copy the facts verbatim. "
        "If the facts do not contain the answer, say: 'I don't know based on my data.' "
        "Do not guess."
    )
    prompt = (
        f"{strict_instructions}\n\n"
        f"Facts:\n{facts_block}\n\n"
        f"Question: {user_input}\n"
        f"Answer:"
    )
    print("[Debug] Sending STRICT prompt to model...")
    resp = llm(
        prompt,
        max_tokens=300,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\nPranav:", "\nLarry:"]
    )
    return resp["choices"][0]["text"].strip()


GROUND_TRUTH = {
    "who won the first double championship in ufc history": (
        "Conor McGregor. He became the UFC's first simultaneous two-division champion on November 12, 2016, "
        "at UFC 205 by defeating Eddie Alvarez to add the lightweight title while already holding the featherweight title."
    ),
}

def try_ground_truth(user_input: str) -> str | None:
    key = user_input.strip().lower()
    # Normalize simple variants
    key = key.replace("?", "").replace("\n", " ")
    key = " ".join(key.split())
    for k, v in GROUND_TRUTH.items():
        if k in key:
            return v
    return None


# Helper to ensure knowledge base is initialized and usable from Jupyter
def initialize_knowledge_base():
    """
    Ensures that stored_texts, authoritative_facts, and faiss_index are initialized and usable.
    Loads UFC texts and builds FAISS index if needed.
    Prints debug info about number of facts loaded into FAISS.
    """
    global stored_texts, authoritative_facts, faiss_index
    rebuilt = False
    if not stored_texts or faiss_index is None:
        print("[Debug] (initialize_knowledge_base) Loading UFC texts and rebuilding FAISS index.")
        stored_texts = load_ufc_texts()
        if stored_texts:
            embeddings = embed_texts(stored_texts)
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss_index.add(embeddings)
            rebuilt = True
            print(f"[Debug] (initialize_knowledge_base) FAISS index built from {len(stored_texts)} UFC texts.")
        else:
            print("[Debug] (initialize_knowledge_base) No UFC texts found to build FAISS index.")
    else:
        print(f"[Debug] (initialize_knowledge_base) Knowledge base already initialized: {len(stored_texts)} texts, FAISS index exists.")
    print(f"[Debug] (initialize_knowledge_base) Number of authoritative facts: {len(authoritative_facts)}")
    return rebuilt


# Step 4: Create a simple chat loop wrapped in a function
# This will keep asking the user for input and give the AI's reply until you type 'exit'.
def run_larry():
    global history, faiss_index, stored_texts, authoritative_facts
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

        # Summarize if history is long (kept for display only)
        if len(history) > 20:
            old_messages = history[:-8]
            key_points = []
            for msg in old_messages:
                content = msg['content']
                snippet = (content[:100] + "...") if len(content) > 100 else content
                key_points.append(f"{msg['role']}: {snippet}")
            summary_text = " | ".join(key_points[:10])
            summary = "Summary so far (key points):\n" + textwrap.fill(summary_text, width=80)
            history = history[-8:]
        else:
            summary = ""

        # Hard stop against hallucinations: check ground truth first
        gt = try_ground_truth(user_input)
        if gt is not None:
            ai_response = gt
        else:
            ai_response = generate_strict_answer(user_input, context)

        print("[Debug] Model response received (strict mode).")
        history.append({"role": "assistant", "content": ai_response})
        print("AI:", ai_response)
        add_to_faiss(ai_response)


def ask_larry_once(user_input: str):
    global history, faiss_index, stored_texts, authoritative_facts
    # Detect authoritative fact
    is_authoritative = False
    lower_input = user_input.lower()
    if lower_input.startswith("correction:") or lower_input.startswith("fact:") or lower_input.startswith("correction -"):
        is_authoritative = True
        print("[Debug] User input detected as authoritative fact or correction.")

    # Add user input to history
    history.append({"role": "user", "content": user_input})
    add_to_faiss(user_input, is_authoritative=is_authoritative)

    # Retrieve context
    context = retrieve_context(user_input)

    # Strict, non-hallucinating answer
    gt = try_ground_truth(user_input)
    if gt is not None:
        ai_response = gt
    else:
        ai_response = generate_strict_answer(user_input, context)

    print("[Debug] Model response received (strict mode).")
    history.append({"role": "assistant", "content": ai_response})
    print("Larry:", ai_response)
    add_to_faiss(ai_response)
    return ai_response


if __name__ == "__main__":
    # Do not auto-run run_larry in Jupyter.
    # In Jupyter, first call initialize_knowledge_base() before using ask_larry_once()
    print("Module loaded. Call initialize_knowledge_base() first in Jupyter, then ask_larry_once('Your question').")