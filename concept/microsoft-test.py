import torch
from transformers import pipeline

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1

# Define the messages
messages = "Who are you?"

# Initialize the pipeline with the specified model and load it onto the GPU if available
pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True, device=device)

# Generate the response
response = pipe(messages, max_length=500)

# Print the response
print(response)