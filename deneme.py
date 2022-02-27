from re import L
import cv2
import argparse
import imutils
import numpy as np

 
#img = cv2.imread('/home/pyarena/python/OpenCV/objectDetection/image1.jpg')
 
cap = cv2.VideoCapture(0)
cap.set(3, 640)#
cap.set(4, 480)#
 
classNames = []
 
classFile = 'coco.names'

with open(classFile,'rt') as f:
    classNames=[line.rstrip() for line in f] 
 
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'
 
 
 
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

insan=0
boş_masa=0
org = (20, 460)
org_masa=(20, 400)
org_masa_2=(20, 430)


while True:  #
    success, img = cap.read()  #
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds,bbox)
    person=0 
    masa=0
    boş_masa=0
    if len(classIds) != 0:   #
        person=0 
        table=0
    
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId == 1 or classId == 67:
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId-1].upper(), (box[0]+20, box[1]+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
                if classId == 1:
                    insan_kordi= bbox
                    person+=1
                    print(f'ahada insan{insan_kordi}')
                if classId == 67:
                    masa_kordi= bbox
                    masa+=1
                    print(f'ahada masaaaa{masa_kordi}')
                    

            


        
        cv2.putText(img, f'Insan sayisi : {person}', org, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)
        cv2.putText(img, f'Masa Sayisi : {masa}', org_masa, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)
        cv2.putText(img, f'Bos Masa Sayisi : {boş_masa}', org_masa_2, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

            

    
        
    # if classNames[classId-1] == "diner table":
    #     cv2.putText(img, f'{classNames[classId-1]} : {table}', org, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

    #     person+=1
    # else:
    #     cv2.putText(img, f'{classNames[classId-1]} : {table}', org, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

    #     person-=1
            


  
         
               
                   
 
 
 
    cv2.imshow('output', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.waitKey(1)