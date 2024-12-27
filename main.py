from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

# FastAPI 앱 생성
app = FastAPI()

# 모델 및 프로세서 로드
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

@app.post("/emotion_predict")
async def emotion_predict(file: UploadFile = File(...)):
    try:
        # 업로드된 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 이미지 전처리
        inputs = processor(images=image, return_tensors="pt")

        # 모델 추론
        with torch.no_grad():
            outputs = model(**inputs)

        # 결과 처리
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits).item()
        label = model.config.id2label[predicted_class_idx]

        # 결과 반환
        return JSONResponse(content={"predicted_class": label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

