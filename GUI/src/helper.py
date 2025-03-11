from fastapi import FastAPI, File, Form
import base64
from io import BytesIO
from PIL import Image
import torch

app = FastAPI()

def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))  
    return torch.tensor(image) 

@app.post("/vqa/")
async def process_vqa(image_base64: str = Form(...), question: str = Form(...)):
    image_data = base64.b64decode(image_base64)
    processed_image = preprocess_image(image_data)
    
    # Call your VQA model here
    answer = "This is a sample VQA response."
    
    return {"answer": answer}


