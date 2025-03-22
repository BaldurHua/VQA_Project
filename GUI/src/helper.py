from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

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

@app.post("/vqa/")
async def process_vqa(request: Request):
    raw_body = await request.body()
    print("\n Raw Request Body (received by FastAPI):", raw_body.decode("utf-8"))

    return {"answer": "This is a sample VQA response."}









