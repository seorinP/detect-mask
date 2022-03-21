import cv2
import os
import time

dirname = '/home/jovyan/Face-Mask-Detection/recFrame'
url = 'rtsp://1.214.48.119:5540/ch0'
videoCap = cv2.VideoCapture(url)
videoCap.set(cv2.CAP_PROP_POS_MSEC, 1)
videoCap.set(cv2.CAP_PROP_BUFFERSIZE, 38)
count = 0

while(videoCap.isOpened()):

    ret, frame = videoCap.read()
    
    if not ret:
        break
     
    # image show
    # cv2.imshow('stream', frame)
    # 프레임 저장
    name = "rec_frame"+str(count)+".jpg"  
    cv2.imwrite(os.path.join(dirname, name), frame)      
    count += 1
    print(count)
        
    # q키를 누르면 종료
    if cv2.waitKey(10)&0xFF == ord('q'):
        print('종료')
        break

videoCap.release()
print("done!")
#cv2.destroyAllWindows()

