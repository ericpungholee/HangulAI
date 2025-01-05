from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

# Load dataset
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare training data
train_examples = [
    InputExample(texts=[item["question"], item["answer"]]) for item in data
]

# Load pre-trained model
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

# DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# Loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, output_path="./retriever_model")
