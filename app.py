from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from inference import VQAPipeline
import uvicorn
import os
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = VQAPipeline(
    model_path="vqa_custom_model.pth",
    vocab_path="vocab.pkl",
    answer_map_path="answer_mapping.pkl"
)

@app.get("/")
async def root():
    return {"message": "VQA API is running! Use POST /predict"}

@app.post("/predict")
async def predict(
    question: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
        temp_path = "/tmp/temp_image.jpg"
        image_pil.save(temp_path)
        answer = pipeline.predict(temp_path, question)
        return {
            "question": question,
            "answer": answer,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
