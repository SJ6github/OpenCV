import cv2
import mediapipe as mp

#SINGLE HAND AT A TIME------------------------------------------------------------------------------------------------------------

#Initializing mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#Creating a hand detector/tracking object
hands = mp_hands.Hands(
    static_image_mode=False, #works on live video
    max_num_hands=1,#will look for only 1 hand
    min_detection_confidence=0.7#will detect a hand only if its atleast 70% confident
    )

capture = cv2.VideoCapture(0)
tip_ids = [4,8,12,16,20] #ids of tips of fingers: [thumb,index,middle,ring,pinky]

while True:
    isTrue, frame = capture.read()
    frame = cv2.flip(frame, 1)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#since mediapipe requires rgb
    results=hands.process(rgb_img) #sending the frame to mediapipe 

    #counting the fingers
    fingers_up=0
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx in range(len(results.multi_hand_landmarks)):
            hand_landmarks = results.multi_hand_landmarks[idx]
            hand_info = results.multi_handedness[idx]#looping through all 21 landmarks
            lm_list=[]
            h, w, _ = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
               cx, cy = int(lm.x*w), int(lm.y*h) #getting their pixel posns from their relative posns
               lm_list.append((id, cx, cy))

            # Thumb
            thumb_tip_x = lm_list[4][1]
            thumb_base_x = lm_list[2][1]
            if abs(thumb_tip_x - thumb_base_x) > 30:
                fingers_up += 1

            # Other Fingers
            for id in [8, 12, 16, 20]:
                if lm_list[id][2] < lm_list[id - 2][2]:
                    fingers_up += 1

            #drawing hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    #showing results
    cv2.putText(frame, f'Fingers Up: {fingers_up}', (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Finger Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()










































