import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load retriever model
retriever = SentenceTransformer("./retriever_model")

# Load dataset
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Encode answers
answers = [item["answer"] for item in data]
answer_embeddings = retriever.encode(answers, convert_to_tensor=False)

# Build Faiss index
index = faiss.IndexFlatL2(answer_embeddings.shape[1])
index.add(np.array(answer_embeddings))

# Save index and answers
faiss.write_index(index, "retriever_index.faiss")
with open("answers.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False)
