import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import PeftModel
import torch

# Function to initialize the app
@st.cache_resource
def init_app(hf_token):
    # Login or setup using hf_token
    login(hf_token)
    st.info("Login successful")
    
    # Determine device (CUDA or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.info(f"Loading model on device: {device}")

    # Paths to your base model and adapter model
    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    adapter_model = 'Mental-Health-Chatbot'  

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(model, adapter_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    st.info("Model and tokenizer loaded correctly")
    return model, tokenizer, device

# Streamlit app layout
st.title("Mental Health Chatbot")
st.write("This chatbot is designed to help you with mental health-related queries. Please enter your text below.")

# Accessing Hugging Face token from secrets
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Unable to retrieve Hugging Face token. Please check your secrets configuration.")
    st.stop()

# Initialize the app
model, tokenizer, device = init_app(hf_token)

# Function to generate response
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids)
    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return predicted_text

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
