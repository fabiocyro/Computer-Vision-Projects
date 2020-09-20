#Smile Recognition Algorithm

#Importing necessary libraries
import cv2

#Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#Defining a function that will do the detections
def detect_smile(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) #Tweak these values
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame

#Applying the detection function into the webcam frames
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect_smile(gray, frame)
    cv2.imshow("Video", canvas)
    if cv2.waitKey(1) & 0xFFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()