import cv2
import mediapipe as mp
import time


class handDetector():
    static_image_mode = False,
    max_num_hands = 2,
    model_complexity = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
    def __init__(self, mode=False, maxHands=2,complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.holistic
        self.hands = self.mpHands.Holistic()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if draw:
            self.mpDraw.draw_landmarks(img,self.results.right_hand_landmarks , self.mpHands.HAND_CONNECTIONS)
            self.mpDraw.draw_landmarks(img,self.results.left_hand_landmarks , self.mpHands.HAND_CONNECTIONS)
            for hands in self.results.left_hand_landmarks:
                for id, lm in enumerate(hands.landmark):
                    print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        #if self.results.multi_hand_landmarks:
            #myHand = self.results.multi_hand_landmarks[handNo]
        #     for id, lm in enumerate(myHand.landmark):
        #         # print(id, lm)
        #         h, w, c = img.shape
        #         cx, cy = int(lm.x * w), int(lm.y * h)
        #         # print(id, cx, cy)
        #         lmList.append([id, cx, cy])
        #         if draw:
        #             cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        # return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        # lmlist=detector.findPosition(img)
        # if(len(lmlist)!=0):
        #     print(lmlist[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
