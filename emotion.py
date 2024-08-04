import cv2
from deepface import DeepFace
import threading
import time

# 전역 변수 설정
emotion = None
emotion_lock = threading.Lock()
exit_flag = threading.Event()

# 설문 질문과 답변 옵션
questions = [
    "1. 당신의 투자 목표는 무엇인가요?",
    "2. 시장 변동성에 대한 당신의 반응은?",
    "3. 투자 기간은 얼마나 길게 계획하고 있나요?",
    "4. 손실을 감내할 수 있는 정도는?",
    "5. 투자 경험이 얼마나 되나요?"
]

options = [
    ["a) 자산 보전", "b) 꾸준한 수익 추구", "c) 중간 정도의 성장과 수익", "d) 높은 성장 추구", "e) 매우 공격적인 투자"],
    ["a) 불안해서 매도한다", "b) 걱정되지만 유지한다", "c) 변동성에 개의치 않는다", "d) 변동성을 이용해 추가 매수한다", "e) 변동성을 적극 활용한다"],
    ["a) 1년 이하", "b) 1-3년", "c) 3-5년", "d) 5-10년", "e) 10년 이상"],
    ["a) 전혀 감내할 수 없다", "b) 약간의 손실은 감수할 수 있다", "c) 보통 정도의 손실은 감내할 수 있다", "d) 상당한 손실도 감내할 수 있다", "e) 큰 손실도 감내할 수 있다"],
    ["a) 전혀 경험이 없다", "b) 약간의 경험이 있다", "c) 보통 정도의 경험이 있다", "d) 많은 경험이 있다", "e) 매우 많은 경험이 있다"]
]

# 답변별 가중치
answer_weights = {'a': -2, 'b': -1, 'c': 0, 'd': 1, 'e': 2}

# 감정별 가중치
emotion_weights = {'happy': 1, 'neutral': 0, 'sad': -1, 'angry': -2, 'fear': -3, 'surprise': 0, 'disgust': -2}

# 얼굴 감지 및 감정 분석 함수 (Thread 1)
def detect_emotion():
    global emotion
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True and not exit_flag.is_set():
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽는 데 실패했습니다.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            with emotion_lock:
                emotion = result[0]['dominant_emotion']

        # 현재 emotion 변수를 업데이트한 후 다른 스레드에서 이를 사용
        # cv2.imshow는 다른 스레드에서 실행하지 않음.

    cap.release()

# 설문 응답 받기 함수 (Thread 2)
def ask_questions():
    global emotion
    total_score = 0

    for idx, question in enumerate(questions):
        print(question)
        for option in options[idx]:
            print(option)
        
        select = input("선택지를 입력하세요 (a, b, c, d, e) 또는 q 를 눌러 종료하세요: ").lower()
        if select == 'q':
            exit_flag.set()  # 감정 감지 루프를 종료하기 위해 플래그 설정
            break

        while select not in ['a', 'b', 'c', 'd', 'e']:
            select = input("올바른 선택지를 입력하세요 (a, b, c, d, e): ").lower()

        with emotion_lock:
            current_emotion = emotion or 'neutral'
        
        # 감정 가중치와 답변 가중치 합산
        emotion_score = emotion_weights.get(current_emotion, 0)
        answer_score = answer_weights.get(select, 0)
        combined_score = emotion_score + answer_score
        total_score += combined_score

        print(f"현재 감정: {current_emotion}, 감정 점수: {emotion_score}, 답변 점수: {answer_score}, 총 점수: {total_score}")

    classify_investment_type(total_score)

# 투자 유형 분류 함수
def classify_investment_type(total_score):
    if total_score <= -10:
        print("고객의 투자 유형: 안정형")
    elif -10 < total_score <= -5:
        print("고객의 투자 유형: 안정추구형")
    elif -5 < total_score <= 0:
        print("고객의 투자 유형: 위험중립형")
    elif 0 < total_score <= 5:
        print("고객의 투자 유형: 적극투자형")
    else:
        print("고객의 투자 유형: 공격투자형")

# Main 실행
if __name__ == "__main__":
    # 감정 감지 스레드 시작
    emotion_thread = threading.Thread(target=detect_emotion)
    emotion_thread.start()

    # 설문 응답 받기 시작 (Main Thread)
    ask_questions()

    # 감정 감지 스레드 종료 대기
    emotion_thread.join()
