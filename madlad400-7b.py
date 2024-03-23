from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = 'jbochi/madlad400-7b-mt-bt'

# Ensure CUDA is available; if not, this will raise an error.
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. Make sure you're running on a GPU-enabled setup.")

device = torch.device("cuda")  # This specifies that a CUDA device (GPU) should be used.
print(device)
print(torch.cuda.is_available())
# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_name)# Moves the model to the GPU
tokenizer = T5Tokenizer.from_pretrained(model_name)

print("model moved to T5ForConditionalGeneration")
text = """<2en> Litwo! Ojczyzno moja! ty jesteś jak zdrowie:
Ile cię trzeba cenić, ten tylko się dowie,
Kto cię stracił. Dziś piękność twą w całej ozdobie
Widzę i opisuję, bo tęsknię po tobie. 
Panno święta, co Jasnej bronisz Częstochowy
I w Ostrej świecisz Bramie! Ty, co gród zamkowy
Nowogródzki ochraniasz z jego wiernym ludem!
Jak mnie dziecko do zdrowia powróciłaś cudem
(Gdy od płaczącej matki, pod Twoją opiekę
"""
#text = """<2en> Lithuania! My homeland! you are like health: O quanto você deve ser valorizado, só ele vai descobrir, """
# Tokenize the input text and move the tokens to the same device as the model
input_ids = tokenizer(text, return_tensors="pt").input_ids

# Generate output
outputs = model.generate(input_ids=input_ids, max_length=8000)

# Decode the output tokens to a string
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translation)
