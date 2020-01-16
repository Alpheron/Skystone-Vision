import cv2 as cv
from collections import defaultdict
cvNet = cv.dnn.readNetFromTensorflow('model.pb', 'model.pbtxt')

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    #img = cv.imread('/Users/Tinku/Desktop/TestStuff/pls.jpeg')
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    numOfSkystone = 0
    objects_dict = {}
    StoneOrder = []
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.85:
            objectClass = detection[1]
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            objects_dict[score]=[left,top,right,bottom, objectClass]
            
        
    l=list(objects_dict.keys())
    l.sort(reverse=True)
    for i in l[:6] :
        #print(i,objects_dict[i])
        if objects_dict[i][4] == 3.0:
            detectedObject = "Skystone"
            numOfSkystone +=1
        elif objects_dict[i][4] == 4.0:
            detectedObject = "Stone"
        elif objects_dict[i][4] == 2.0:
            detectedObject = "Red Foundation"
        elif objects_dict[i][4] == 1.0:
            detectedObject = "Blue Foundation"
        cv.rectangle(img, (int(objects_dict[i][0]), int(objects_dict[i][1])), (int(objects_dict[i][2]), int(objects_dict[i][3])), (23, 230, 210), thickness=5)
        cv.putText(img, detectedObject, (int(objects_dict[i][0]), int(objects_dict[i][1])), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 7, cv.LINE_AA)
        #print(i, objects_dict)
        StoneOrder.append(detectedObject)
        if numOfSkystone==2:
            
            #print(StoneOrder)
            SkystonePosition = [i for i, value in enumerate(StoneOrder) if value == "Skystone"]
            print('The Skystone is located in positions: ' + str(int(SkystonePosition[0]) + 1) + ' and ' + str(int(SkystonePosition[1]) + 1))

    cv.imshow('img', img)
    cv.waitKey(1)
    cv.destroyAllWindows()