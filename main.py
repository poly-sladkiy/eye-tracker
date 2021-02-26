import cv2

cap = cv2.VideoCapture(0)
eye_detect = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:
    ret, frame = cap.read()

    if ret:
        rects = eye_detect.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

        if len(rects) == 2:
            for rect in rects:
                x, y, w, h = rect
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                frame = cv2.line(frame, (x, y + h // 2), (x + w, y + h // 2),
                                 (133, 133, 133))  # horizontal-line

                frame = cv2.line(frame, (x + w // 2, y), (x + w // 2, y + h),
                                 (133, 133, 133))  # vertical-line

        cv2.imshow('frame', frame)

        # work until pressed ESC
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
