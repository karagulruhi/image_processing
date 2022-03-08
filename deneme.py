from math import dist
from pickle import TRUE
from re import L, T
from turtle import distance
import cv2



 
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
bos_masa=0
org = (20, 460)
org_masa=(20, 400)
org_masa_2=(20, 430)
insan_bulundugu_kordi=[]

table_cordi=[]

while True:  #
    success, img = cap.read()  #
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    # print(classIds,bbox)
    person=0 
    masa=0
    bos_masa=0

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
                    if not table_cordi: 
                        insan_bulundugu_kordi.append(insan_kordi[0][0])#left
                        insan_bulundugu_kordi.append(insan_kordi[0][1])#bottom
                        insan_bulundugu_kordi.append(insan_kordi[0][2])#right
                        insan_bulundugu_kordi.append(insan_kordi[0][3])#top
                    

            
                
                
                
                if classId == 67:
                    masa_kordi= bbox
                    masa+=1
                    if not table_cordi:  
                        table_cordi.append(masa_kordi[0][0])#x
                        table_cordi.append(masa_kordi[0][1])#y
                        table_cordi.append(masa_kordi[0][2])#w
                        table_cordi.append(masa_kordi[0][3])#h
                    

                      
                try:
                    # print(table_cordi[0])
                    # print(insan_kordi[0][0])
                    bos_masa_mesafesi=abs(max(abs(table_cordi[0]-insan_kordi[0][0])-(table_cordi[2]+insan_kordi[0][2])/2,abs(table_cordi[1]-insan_kordi[0][1])-(table_cordi[3]+insan_kordi[0][3])/2))
                    # dx = abs(table_cordi[0] - insan_kordi[0][0]) - (table_cordi[2] + insan_kordi[0][2])
                    print(bos_masa_mesafesi)
                    if bos_masa_mesafesi < 180 and masa!=0: 
                        bos_masa = 1
                    else:
                        bos_masa = 0
                    # dy = abs(table_cordi[1] - insan_kordi[0][1] )- (table_cordi[3] + insan_kordi[0][3])
                    # print(dx+dy)
     
                except:             # rectangles intersect
                
                    continue
                
                   
                    # Length(Max((0, 0), Abs(Center - otherCenter) - (Extent + otherExtent)))    
                    # print(dist((table_cordi[0], table_cordi[3]), (insan_bulundugu_kordi[2], insan_bulundugu_kordi[1])))

                    # print(dist((table_cordi[0], table_cordi[1]), (insan_bulundugu_kordi[2], insan_bulundugu_kordi[3])))

                    # print(dist((table_cordi[2], table_cordi[1]), (insan_bulundugu_kordi[0], insan_bulundugu_kordi[3])))

                    # print(dist((table_cordi[2], table_cordi[3]), (insan_bulundugu_kordi[0], insan_bulundugu_kordi[1])))

                    # print(table_cordi[0] - insan_bulundugu_kordi[2])

                    # print(insan_bulundugu_kordi[0] - table_cordi[2])

                    # print(table_cordi[1] - insan_bulundugu_kordi[3])

                #     # print(insan_bulundugu_kordi[1] - table_cordi[3])


            
                
                    #     print(i)
                    
                    # table_cordi[0][1][0]
                    
                # print(f'ahada masaaaa{table_cordi}')
        
        
        # print(table_cordi)    


        

        
        cv2.putText(img, f'Insan sayisi : {person}', org, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)
        cv2.putText(img, f'Masa Sayisi : {masa}', org_masa, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)
        cv2.putText(img, f'Bos Masa Sayisi : {bos_masa}', org_masa_2, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 5)

            
 
    cv2.imshow('output', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.waitKey(1)