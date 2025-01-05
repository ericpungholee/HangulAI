# HangulAI: Retrieval-Augmented Generation (RAG) for Korean

HangulAI is a **Retrieval-Augmented Generation (RAG)** system designed to process and respond to Korean language queries effectively. The system combines a **retriever** and a **generator** to provide context-aware and high-quality answers to user queries. 

This project leverages **SentenceTransformers**, **Huggingface Transformers**, and **Faiss** for building an efficient and scalable pipeline.

---

## 🚀 Features

- **Retrieval-Augmented Generation**: Combines retrieval and generative capabilities for accurate and contextual answers.
- **Domain-Specific Adaptability**: Easily extendable for different datasets and Korean-specific domains.
- **Flask API**: Provides a simple RESTful interface for integrating the system with external applications.
- **Korean Language Support**: Optimized for Korean datasets and linguistic nuances.

---

## 🛠️ Technologies Used

- **Python**: Core language for implementation.
- **Flask**: Web framework for API development.
- **SentenceTransformers**: For retriever fine-tuning and embedding generation.
- **Huggingface Transformers**: For the generator model.
- **Faiss**: For efficient vector search and retrieval.
- **Korean Dataset**: Domain-specific data for training and testing.

---

## 📂 Project Structure

```plaintext
HangulAI/
├── app.py                 # Flask API for the RAG pipeline
├── build_index.py         # Script to build the retriever index
├── train_retriever.py     # Script to fine-tune the retriever model
├── generator.py           # Script for generator setup
├── test_request.py        # Script to test the Flask API
├── data.json              # Sample dataset (questions and answers)
├── answers.json           # Processed answers for retrieval
├── retriever_model/       # Fine-tuned retriever model (excluded from Git)
├── retriever_index.faiss  # Faiss index for retrieval (excluded from Git)
├── checkpoints/           # Training checkpoints (optional, excluded from Git)
├── .gitignore             # Git ignore rules
