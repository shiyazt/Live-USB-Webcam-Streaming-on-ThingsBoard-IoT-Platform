"""
A Simple program to detect the human faces using Haarcascade Classifier

Date : 21 March 2019
Author : Shiyaz T
"""


import cv2
haar_cascade_face = cv2.CascadeClassifier('/home/shiyaztech/Documents/Applications/OpenCV/opencv/data/haarcascades/haarcascade_frontalface_default.xml')


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.frameWidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5);
        # Let us print the no. of faces found
        #print('Faces found: ', len(faces_rects))

        for (x, y, w, h) in faces_rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(faces_rects) > 0:
            
            cv2.putText(img = image, text = 'Faces found:' + str(len(faces_rects)), org=(50,50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 0, 255))
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
