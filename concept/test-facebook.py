import torch
from transformers import pipeline

# Check if a GPU is available
device = 0 if torch.cuda.is_available() else -1

from transformers import pipeline

pipe = pipeline("translation_en_to_fr", model="facebook/nllb-200-distilled-600M")
print(pipe("Let's go to france and see the eiffel tower"))