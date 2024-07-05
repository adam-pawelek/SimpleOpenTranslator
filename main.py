from fastapi import FastAPI
import uvicorn
from utils.simple_language_translator import translate_text_one_language

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/translate-text")
async def translate(text: str, to_language: str):
    return translate_text_one_language(text, to_language)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
