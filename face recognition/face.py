import cv2
import sys ,numpy,os

haar_file="C:\\Users\\berza\\OneDrive\\Documenten\\cv\\face recognition\\haarcascade_frontalface_default.xml"
dataset='dataset'
subdata='Berzan'
count=1

path=os.path.join(dataset,subdata)

if not os.path.isdir(path):
    os.mkdir(path)

(width,height)=(130,100)
face_cascade=cv2.CascadeClassifier(haar_file)

webcam=cv2.VideoCapture(0)

while count <30:
    (_,im)=webcam.read()
    grey=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)#face detection works better in a grey scale image    
    faces=face_cascade.detectMultiScale(grey,1.3,4)#1.3 is the scaling factor 4 are the minimum neighbors for accurecy control
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)

        face=grey[y:y+h,x:x+w]#this line is to only extract the face of the frame
        face_resize=cv2.resize(face,(width,height))
        cv2.imwrite('%s/%s.png' % (path,count),face_resize)
        count+=1
        cv2.imshow("face",im)
        key=cv2.waitKey(0)
        if key==27:
            break
        cv2.destroyAllWindows()
