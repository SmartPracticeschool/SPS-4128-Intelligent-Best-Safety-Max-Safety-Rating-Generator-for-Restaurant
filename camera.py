import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIA3HETSUJQT7763AXO",
                        aws_secret_access_key="+x1/IWZoYqCb5ROSUQxDIxffGwZY0pVRiL54NUYd",
                     aws_session_token="FwoGZXIvYXdzEDAaDDT652D4KmLlpniuvSLDAVWBdHtUsfwVzYDTp6e4wUPahb4xIck3G0jaro9PSa6S/HIPA+Dv1AGqIbI42jkWsXYHmrVbFlxbYQckKrtxX1pMfHlRxhPuqbK5i0W9lvwGXCg/brmpQ2XYOkrYGs19Kiked7tCU7tjADg+d+r62eTPBr5bBl7OegNniK9P+9JWc9SNvDV4DPxiJDRsgDjv/THQgnLJbzOppr1PstO/5yYuVg9WXwvlBKrOZx2ZFvMRvqEcIARQpabT0br4LcZT7LwaTCj/09z6BTItnzWYyflWNS0pnbmm3AKaqa5fvNP/X5+6qzmTsQTrR2/Hs2svV53fo1jKj837",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:771256132193:project/mask-detection/version/mask-detection.2020-09-06T23.56.52/1599416814058',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://ovj7dfacxa.execute-api.us-east-1.amazonaws.com/countmask/countmasks?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
