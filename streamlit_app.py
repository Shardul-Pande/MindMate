import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Function to load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(base_model_path, adapter_model_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, adapter_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return model, tokenizer

# Paths to your base model and adapter model
base_model_path = 'meta-llama/Llama-2-7b-chat-hf'
adapter_model_path = '/content/drive/MyDrive/mentalHealth'

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(base_model_path, adapter_model_path)

# Function to generate response
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids)
    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return predicted_text

# Streamlit app layout
st.title("Mental Health Chatbot")
st.write("This chatbot is designed to help you with mental health-related queries. Please enter your text below.")

# Text input from user
user_input = st.text_area("Enter your text here:")

# Generate response when button is clicked
if st.button('Generate Response'):
    if user_input:
        response = generate_response(user_input)
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter some text to get a response.")
