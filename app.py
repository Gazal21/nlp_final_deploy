import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer

model, tokenizer = load_model()

# Title of the Streamlit app
st.title("GPT-2 Chatbot")

# Input text from user
user_input = st.text_input("You: ", "Hello, how are you?")

if st.button("Generate Response"):
    # Tokenize and encode the input text
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # Generate a response
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Display the response
    st.text_area("GPT-2:", value=response, height=200, max_chars=None, key=None)

