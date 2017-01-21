import argparse
import sys, os
sys.path.append("../")

import cv2
import numpy as np

import face_detection_utilities as fdu

import model.myVGG as vgg
import speech_recognition as sr

windowsName = 'Preview Screen'

parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')

args = parser.parse_args()
FACE_SHAPE = (48, 48)

model = vgg.VGG_16('my_model_weights_83.h5')
#model = vgg.VGG_16()

emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

def speechRecognition():
    # obtain audio from the microphone 
    print("Press 'y' to start~")
    inputdata = input()
    if inputdata == 'y':
        inputdata = 0
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
        # recognize speech using Google Speech Recognition
        try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
            recSuccess = 1
            recContent = r.recognize_google(audio)
            print("Speech Recognition thinks you said " + recContent)#,language="cmn-Hant-TW")
            return recContent
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

def refreshFrame(frame, faceCoordinates):
    if faceCoordinates is not None:
        fdu.drawFace(frame, faceCoordinates)
    cv2.imshow(windowsName, frame)


def showScreenAndDectect(capture):
    print("Face dececting...")
    cnt = 5;
    while cnt:
        flag, frame = capture.read()
        faceCoordinates = fdu.getFaceCoordinates(frame)
        refreshFrame(frame, faceCoordinates)
        
        if faceCoordinates is not None:
            cnt -= 1
            face_img = fdu.preprocess(frame, faceCoordinates, face_shape=FACE_SHAPE)
            #cv2.imshow(windowsName, face_img)

            input_img = np.expand_dims(face_img, axis=0)
            input_img = np.expand_dims(input_img, axis=0)

            result = model.predict(input_img)[0]
            if cnt == 4:
                tot_result = result
            else:
                tot_result += result
            index = np.argmax(result)
            print ('Frame',5-cnt,':', emo[index], 'prob:', max(result))
            #index = np.argmax(result)
            #print (emo[index], 'prob:', max(result))
            # print(face_img.shape)
            # emotion = class_label[result_index]
            # print(emotion)
    index = np.argmax(tot_result)
    print ('Final decision:',emo[index], 'prob:', max(tot_result))
    return emo[index]

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
    
    capture = getCameraStreaming()

    cv2.startWindowThread()
    cv2.namedWindow(windowsName, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(windowsName, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    
    while True:
        recContent = speechRecognition()
        if recContent is not None:
            emotion = showScreenAndDectect(capture)
            if emotion == "Angry":
                emoji = " >:O"
            elif emotion == "Fear":
                emoji = " :-S"
            elif emotion == "Happy":
                emoji = " :-D"
            elif emotion == "Sad":
                emoji = " :'("
            elif emotion == "Surprise":
                emoji = " :-O"
            else:
                emoji = " "
            print("Output result: " + recContent + emoji)

if __name__ == '__main__':
    main()
