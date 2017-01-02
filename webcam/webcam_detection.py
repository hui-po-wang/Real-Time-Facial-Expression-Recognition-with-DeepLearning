import argparse
import sys, os
sys.path.append("../")

import cv2
import numpy as np

import face_detection_utilities as fdu

import model.myVGG as vgg

windowsName = 'Preview Screen'

parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
parser.add_argument('-testImage', help=('Given the path of testing image, the program will predict the result of the image.'
"This function is used to test if the model works well."))

args = parser.parse_args()
FACE_SHAPE = (48, 48)

model = vgg.VGG_16('my_model_weights_83.h5')
#model = vgg.VGG_16()

emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

def refreshFrame(frame, faceCoordinates):
    if faceCoordinates is not None:
        fdu.drawFace(frame, faceCoordinates)
    cv2.imshow(windowsName, frame)


def showScreenAndDectect(capture):
    while (True):
        flag, frame = capture.read()
        faceCoordinates = fdu.getFaceCoordinates(frame)
        refreshFrame(frame, faceCoordinates)
        
        if faceCoordinates is not None:
            face_img = fdu.preprocess(frame, faceCoordinates, face_shape=FACE_SHAPE)
            #cv2.imshow(windowsName, face_img)

            input_img = np.expand_dims(face_img, axis=0)
            input_img = np.expand_dims(input_img, axis=0)

            result = model.predict(input_img)[0]
            index = np.argmax(result)
            print (emo[index], 'prob:', max(result))
            # print(face_img.shape)
            # emotion = class_label[result_index]
            # print(emotion)

def getCameraStreaming():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming ")
        sys.exit(1)
    else:
        print("Successed to capture video streaming")
        
    return capture

def main():
    '''
    Arguments to be set:
        showCam : determine if show the camera preview screen.
    '''
    print("Enter main() function")
    
    if args.testImage is not None:
        img = cv2.imread(args.testImage)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, FACE_SHAPE)
        print(class_label[result[0]])
        sys.exit(0)

    showCam = 1

    capture = getCameraStreaming()

    if showCam:
        cv2.startWindowThread()
        cv2.namedWindow(windowsName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(windowsName, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    
    showScreenAndDectect(capture)

if __name__ == '__main__':
    main()
