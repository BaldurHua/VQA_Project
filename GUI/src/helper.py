import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import torchvision.transforms as transforms

from utils.model import SANModel, VocabDict
from utils.utils import encode_question, run_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n Registered Routes in FastAPI:")
    for route in app.routes:
        print(f" {route.path} [{', '.join(route.methods)}]")
    yield 
    print("\n FastAPI is shutting down...")

app = FastAPI(
    title="FastAPI Test API",
    description="This API is used for testing endpoints",
    version="1.0",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

print("\n FastAPI is running.")

@app.get("/test", tags=["Debug"])
async def test_api():
    print("/test endpoint was accessed!")
    return {"message": "FastAPI is receiving requests!"}

# @app.post("/vqa/")
# async def process_vqa(request: Request):
#     raw_body = await request.body()
#     print("\n Raw Request Body (received by FastAPI):", raw_body.decode("utf-8"))

#     return {"answer": "This is a sample VQA response."}

@app.post("/vqa/")
async def vqa_inference(question: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Encode 
        question_tensor = encode_question(question).unsqueeze(0).to(device)

        # Inference
        predictions = run_inference(img_tensor, question_tensor)

        # output
        result = [
            {"answer": ans, "confidence": round(prob, 4)}
            for ans, prob in predictions
        ]

        return JSONResponse(content={"predictions": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})








