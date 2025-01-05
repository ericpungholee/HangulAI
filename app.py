from flask import Flask, request, jsonify
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Load retriever and index
retriever = SentenceTransformer("./retriever_model")
index = faiss.read_index("retriever_index.faiss")
with open("answers.json", "r", encoding="utf-8") as f:
    answers = json.load(f)

# Load generator
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
generator = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token  # Suppress padding warnings

def retrieve(question):
    """Retrieve the most relevant answer."""
    query_embedding = retriever.encode([question], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), k=1)
    return answers[indices[0][0]]

def generate_response(prompt):
    """Generate a response using the generator."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = generator.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],  # Ensure proper handling
        max_new_tokens=100,  # Increase token generation limit
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/rag", methods=["POST"])
def rag():
    """Handle RAG requests."""
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    try:
        # Retrieve and generate response
        retrieved_answer = retrieve(question)
        response = generate_response(f"Question: {question}\nContext: {retrieved_answer}\nAnswer:")
        return jsonify({"retrieved_answer": retrieved_answer, "response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
