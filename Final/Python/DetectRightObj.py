import cv2 as cv
from collections import defaultdict
cvNet = cv.dnn.readNetFromTensorflow('/Users/Tinku/Desktop/RoboticsShit/Skystone-Vision/Final/image_tensor/PB_file/model.pb', '/Users/Tinku/Desktop/RoboticsShit/Skystone-Vision/Final/image_tensor/Pbtxt/model.pbtxt')

img = cv.imread('/Users/Tinku/Desktop/TestStuff/work.jpeg')
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()
numOfRect = 0
objects_dict = {}
for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    objects = detection
    if score > 0.75:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        objects_dict[score]=[left,top,right,bottom]
        numOfRect += 1

        print(score)
#print(objects_dict)
l=list(objects_dict.keys())
l.sort(reverse=True)
print(l)
for i in l[:6] :
    print(i,objects_dict[i])
    cv.rectangle(img, (int(objects_dict[i][0]), int(objects_dict[i][1])), (int(objects_dict[i][2]), int(objects_dict[i][3])), (23, 230, 210), thickness=5)



cv.imshow('img', img)
print(numOfRect)
print(objects)
cv.waitKey()
