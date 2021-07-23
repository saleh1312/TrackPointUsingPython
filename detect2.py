import cv2
import numpy as np

class detection:
    #detect first frame and his feautures
    def __init__(self,im1):
        self.im1=im1
        
        self.detector=cv2.FastFeatureDetector_create(
            threshold=10,nonmaxSuppression=True,type=2
            )
        self.computer = cv2.xfeatures2d.SIFT_create()
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict()   # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
            
        self.kp1 = self.detector.detect(
                self.im1,None)
        self.des1 = self.computer.compute(self.im1,self.kp1)
            
        self.all_outs=0
        self.hc=np.eye(3)


    def dnd(self,im2):
        kp2 = self.detector.detect(
                im2,None)
        des2 = self.computer.compute(im2,kp2)

        matches = self.matcher.knnMatch(self.des1[1],des2[1],k=2)
       
 
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(self.im1,self.kp1,im2,kp2
                                  ,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        points = np.zeros((len(good), 4), dtype=np.float32)  

        for i, m in enumerate(good):
           
            points[i, :2] = self.kp1[m[0].queryIdx].pt    #gives index of the descriptor in the list of query descriptors
            points[i, 2:] = kp2[m[0].trainIdx].pt    #gives index of the descriptor in the list of train descriptors
            
     
        

        l1=points[:,:2].astype(np.float32)
        l2=points[:,2:].astype(np.float32)
        #cv2.LMEDS
        #cv2.RANSAC
        #threshold up = outliers down
        #ransac to detect outs and remove them then use LMEDS 
        # to detect more accurate matrix
        h,mask=cv2.findHomography(l1, l2, cv2.RANSAC,30.0)
        h2=self.remove_outs(l1,l2,h,mask)
       
        self.hc=np.dot(self.hc,h2)



        self.des1=des2
        self.kp1=kp2
        self.im1=im2
        return img3
    
    
    def remove_outs(self,l1,l2,h,mask):
        mask=np.array(mask)
        mask=mask.reshape((mask.shape[0]))
        self.all_outs+=len(np.where(mask==0)[0])
        inpox=[]
        inpoy=[]
        for x in range(l1.shape[0]):
            if mask[x]==1:
                inpox.append([l1[x,0],l1[x,1]])
                inpoy.append([l2[x,0],l2[x,1]])
        inpox=np.array(inpox)
        inpoy=np.array(inpoy)
 
        h2,mask2=cv2.findHomography(inpox, inpoy, cv2.LMEDS)
        return h2
        


    @staticmethod
    def process_img(img):
        c=img.copy()
        c=cv2.resize(c,(512,512))
        c=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)

        return c
    
    def mapp(self,points):
        #sometimes it fails to detect the matrix
        
        ds=cv2.perspectiveTransform(points[None,:,:],self.hc)
 
        return ds    

