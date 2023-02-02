import cv2

videoCaptureObject = cv2.VideoCapture(0)
result = True
i = 1

while result:
    ret, frame = videoCaptureObject.read()
    cv2.imwrite("calib/data/%.jpg".format(i), frame)
    i += 1

    result = ret

    cv2.imshow("Recording", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

videoCaptureObject.release()
cv2.destroyAllWindows()
