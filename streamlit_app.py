import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from peft import PeftModel
import torch
import asyncio

# Function to initialize the app
@st.cache(allow_output_mutation=True)
def init_app(hf_token):
    st.info("Initializing app...")
    
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
    st.info("Loading model and tokenizer...")
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
except Exception as e:
    st.error(f"Unable to retrieve Hugging Face token: {e}")
    st.stop()

# Initialize the app
model, tokenizer, device = init_app(hf_token)

# Function to preprocess text
@st.cache(allow_output_mutation=True)
def preprocess(texts):
    st.info("Preprocessing text...")
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

# Function to postprocess text
@st.cache(allow_output_mutation=True)
def postprocess(outputs):
    st.info("Postprocessing text...")
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Function to generate responses
@st.cache(allow_output_mutation=True)
async def generate_responses(prompts):
    st.info("Generating responses...")
    inputs = await asyncio.gather(*[preprocess(prompt) for prompt in prompts])
    with torch.no_grad():
        outputs = await asyncio.gather(*[model.generate(**input, max_length=512, num_return_sequences=1) for input in inputs])
    return [postprocess(output) for output in outputs]

# BatchProcessor class to handle batch processing
class BatchProcessor:
    def __init__(self, batch_size, interval):
        st.info("Initializing BatchProcessor...")
        self.batch_size = batch_size
        self.interval = interval
        self.queue = asyncio.Queue()
        self.results = {}

    async def add_task(self, prompt, task_id):
        st.info(f"Adding task {task_id} to queue...")
        await self.queue.put((prompt, task_id))

    async def process_batch(self):
        st.info("Starting batch processing...")
        while True:
            if self.queue.qsize() >= self.batch_size:
                st.info("Processing batch...")
                prompts, task_ids = [], []
                for _ in range(self.batch_size):
                    prompt, task_id = await self.queue.get()
                    prompts.append(prompt)
                    task_ids.append(task_id)

                responses = await generate_responses(prompts)
                for task_id, response in zip(task_ids, responses):
                    self.results[task_id] = response
            else:
                st.info("Waiting for batch to fill...")
                await asyncio.sleep(self.interval)

    def get_result(self, task_id):
        st.info(f"Retrieving result for task {task_id}...")
        while task_id not in self.results:
            asyncio.sleep(0.1)
        return self.results.pop(task_id)

# Instantiate and start the batch processor
batch_processor = BatchProcessor(batch_size=4, interval=1)
asyncio.create_task(batch_processor.process_batch())

# Function to generate response
def generate_response(input_text):
    st.info("Generating single response...")
    task_id = len(batch_processor.results) + 1
    asyncio.create_task(batch_processor.add_task(input_text, task_id))
    st.info("Waiting for response...")
    return batch_processor.get_result(task_id)

# Text input from user
user_input = st.text_area("Enter your text here:")

# Generate response when button is clicked
if st.button('Generate Response'):
    if user_input:
        st.info("Button clicked, generating response...")
        response = generate_response(user_input)
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter some text to get a response.")
