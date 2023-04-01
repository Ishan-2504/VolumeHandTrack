import cv2
import mediapipe as mp
import time
 
cap = cv2.VideoCapture(0) #opening camera
 
mpHands = mp.solutions.hands #initialising an object
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #for making markings
 
pTime = 0 #previous time
cTime = 0 #current time
 
while True: #for video
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversion into RGB img
    results = hands.process(imgRGB)
 
    if results.multi_hand_landmarks: #if detected successfully
        for handLms in results.multi_hand_landmarks: #multiple hands can be there
            for id, lm in enumerate(handLms.landmark): #for getting components of each hand
                # lm is landmark
                h, w, c = img.shape #height width channels
                cx, cy = int(lm.x * w), int(lm.y * h) #for pixels
                print(id, cx, cy)
                # if id == 4: #points out which specific portion of hand
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)#circles
 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime) #frames per sec
    pTime = cTime
 
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, #for designing display
                (255, 0, 255), 3)
 
    cv2.imshow("Image", img)
    cv2.waitKey(1)
