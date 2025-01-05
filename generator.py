from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
generator = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = generator.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
