import cv2 as cv

cvNet = cv.dnn.readNetFromTensorflow('/Users/Tinku/Desktop/RoboticsShit/Skystone-Vision/Final/image_tensor/PB_file/model.pb', '/Users/Tinku/Desktop/RoboticsShit/Skystone-Vision/Final/image_tensor/Pbtxt/model.pbtxt')

img = cv.imread('/Users/Tinku/Desktop/TestStuff/work.jpeg')
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()
numOfRect = 0
for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > .85:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=5)
        numOfRect += 1
        print(score)

cv.imshow('img', img)
print(numOfRect)
cv.waitKey()
