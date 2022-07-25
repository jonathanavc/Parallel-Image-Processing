import cv2

capture = cv2.VideoCapture('videoplayback (1).mp4')
cont = 0
path = ""

while (capture.isOpened()):
    ret, frame = capture.read()
    if (ret == True):
        cv2.imwrite(path + 'IMG_%04d.jpg' % cont, frame)    
        cont += 1
        if (cv2.waitKey(1) == ord('s')):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()