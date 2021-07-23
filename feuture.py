import cv2
import numpy as np
from detect2 import detection as detection

cap =cv2.VideoCapture('t\\v6.wmv')




first= detection.process_img(cap.read()[1])
pro=detection(first)

point=np.float32([[175,242]])
while True:
    ret,sec = cap.read()
    
    
    if ret==False:
        break
    
    sec=detection.process_img(sec)
    
    copy=sec.copy()
    copy=cv2.cvtColor(copy,cv2.COLOR_GRAY2BGR)

    img3=pro.dnd(sec)

    pps=pro.mapp(point)
    pps=pps.reshape((pps.shape[1],pps.shape[2]))

    cv2.circle(copy,(pps[0,0],pps[0,1]),7,(0,255,255),-1)
    cv2.imshow('sss',copy)
    cv2.imshow('sss2',img3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print(pro.all_outs)
cap.release()

cv2.destroyAllWindows()


