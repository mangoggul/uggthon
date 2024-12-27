from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
import pymysql


def fetch_diary_contents():
    # MySQL DB 연결 정보
    host = "uggdb.c9ocs4kg22yt.ap-northeast-2.rds.amazonaws.com"
    user = "admin"
    password = "gbvsfy7z!"
    database = "uggDB"  # 사용할 데이터베이스 이름
    port = 3306

    try:
        # MySQL 데이터베이스 연결
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            charset='utf8mb4'
        )

        print("DB 연결 성공")

        # 커서 생성 및 데이터 가져오기
        with connection.cursor() as cursor:
            query = "SELECT content FROM diary"  # diary 테이블에서 content 열 선택
            cursor.execute(query)
            rows = cursor.fetchall()

            # 결과를 리스트로 변환
            contents = [row[0] for row in rows]
            return contents

    except Exception as e:
        print(f"DB 연결 또는 데이터 가져오기 실패: {e}")
        return []

    finally:
        # 연결 닫기
        if connection:
            connection.close()

def main():
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

    print("AI와의 대화를 시작합니다. DB에서 일기를 읽고 처리합니다.")

    # DB에서 diary 테이블의 content 가져오기
    questions = fetch_diary_contents()

    for user_input in questions:
        print(f"사용자: {user_input}")
        # LangChain을 통해 응답 생성
        response = conversation.predict(input=user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()