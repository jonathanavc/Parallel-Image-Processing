import cv2
import os
import glob

video = 'video_2_LI'

for filename in glob.glob('./' + video + '/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    break


out = cv2.VideoWriter(video + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 27.0, size)

for filename in sorted(glob.glob('./' + video + '/*.jpg'), key=lambda x: int(((os.path.splitext(os.path.basename(x))[0]).split('_'))[1])):
    out.write(cv2.imread(filename))

out.release()