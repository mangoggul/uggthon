from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from pydantic import BaseModel
import io
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import pymysql
from dotenv import load_dotenv
import os


# FastAPI 앱 생성
# .env 파일 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (특정 도메인만 허용하려면 여기에 추가)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 모델 및 프로세서 로드
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DB 연결 정보
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT")),
    "charset": os.getenv("DB_CHARSET")
}
class DiaryRequest(BaseModel):
    id: int

def insert_letter_into_db(diary_id: int, letter_content: str):
    """letter 테이블에 diary_id와 content 삽입"""
    connection = None
    try:
        # MySQL 데이터베이스 연결
        connection = pymysql.connect(**DB_CONFIG)

        # 커서 생성 및 데이터 삽입
        with connection.cursor() as cursor:
            query = "INSERT INTO letter (diary_id, content) VALUES (%s, %s)"
            cursor.execute(query, (diary_id, letter_content))
            connection.commit()  # 변경 사항 커밋

    except Exception as e:
        print(f"DB 연결 또는 데이터 삽입 실패: {e}")
        raise HTTPException(status_code=500, detail="편지 데이터를 DB에 저장하는 데 실패했습니다.")

    finally:
        # 연결 닫기
        if connection:
            connection.close()


def fetch_diary_content_by_id(diary_id: int):
    """특정 ID로 diary content 조회"""
    connection = None
    try:
        # MySQL 데이터베이스 연결
        connection = pymysql.connect(**DB_CONFIG)

        # 커서 생성 및 데이터 가져오기
        with connection.cursor() as cursor:
            query = "SELECT content FROM diary WHERE id = %s"  # 특정 ID의 content 가져오기
            cursor.execute(query, (diary_id,))
            row = cursor.fetchone()

            if row:
                return row[0]
            else:
                return None

    except Exception as e:
        print(f"DB 연결 또는 데이터 가져오기 실패: {e}")
        return None

    finally:
        # 연결 닫기
        if connection:
            connection.close()


def generate_response_from_diary(content: str) -> str:
    """LangChain을 이용해 diary content로부터 답장 생성"""
    # GPT 모델 초기화
    chat = ChatOpenAI(temperature=0.7, model="gpt-4o")

    # 메모리와 프롬프트 템플릿 설정
    memory = ConversationBufferMemory()
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=(
            "다음은 사용자가 작성한 개인 일기입니다. "
            "이 일기를 바탕으로 AI 비서로서 조언, 격려 또는 성찰을 담은 "
            "따뜻하고 사려 깊은 답장 편지를 작성하세요. "
            "편지는 4문장 정도로 작성해주세요.\n\n"
            "끝 문장은 해요체로 작성해주세요"
            "{history}\n사용자의 일기: {input}\nAI의 편지:"
        )
    )

    # ConversationChain 생성
    conversation = ConversationChain(
        llm=chat,
        prompt=prompt_template,
        memory=memory,
        verbose=True,
    )

    # LangChain을 통해 응답 생성
    response = conversation.predict(input=content)
    return response


@app.post("/generate-letter/")
async def generate_letter(request: DiaryRequest):
    """일기 ID를 받아 편지를 생성"""
    # DB에서 diary content 조회
    content = fetch_diary_content_by_id(request.id)

    if not content:
        raise HTTPException(status_code=404, detail="해당 ID의 일기를 찾을 수 없습니다.")

    # LangChain을 이용해 편지 생성
    letter = generate_response_from_diary(content)

    # 생성된 편지를 DB에 저장
    try:
        insert_letter_into_db(request.id, letter)
    except HTTPException as e:
        return JSONResponse(content={"error": str(e.detail)}, status_code=500)

    return {"id": request.id, "diary_content": content, "letter": letter}



def insert_emotion_into_db(user_id: int, emotion: str):
    """diary 테이블에 user_id와 emotion 삽입"""
    connection = None
    try:
        # MySQL 데이터베이스 연결
        connection = pymysql.connect(**DB_CONFIG)

        # 커서 생성 및 데이터 삽입
        with connection.cursor() as cursor:
            query = "INSERT INTO diary (user_id, emotion) VALUES (%s, %s)"
            cursor.execute(query, (user_id, emotion))
            connection.commit()  # 변경 사항 커밋

    except Exception as e:
        print(f"DB 연결 또는 데이터 삽입 실패: {e}")
        raise HTTPException(status_code=500, detail="감정 데이터를 DB에 저장하는 데 실패했습니다.")

    finally:
        # 연결 닫기
        if connection:
            connection.close()


@app.post("/emotion_predict")
async def emotion_predict(
    user_id: int = Query(..., description="사용자 ID"),  # 쿼리 파라미터로 user_id 받기
    file: UploadFile = File(...)
):
    """
    사용자의 user_id와 업로드된 이미지를 받아 감정 예측 결과를 반환하고 DB에 저장.
    """
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

        # 감정 데이터를 DB에 저장
        try:
            insert_emotion_into_db(user_id=user_id, emotion=label)
        except Exception as db_error:
            print(f"DB 저장 실패: {db_error}")
            return JSONResponse(content={"error": "DB 저장 중 문제가 발생했습니다."}, status_code=500)

        # 결과 반환
        return JSONResponse(content={"user_id": user_id, "predicted_class": label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
