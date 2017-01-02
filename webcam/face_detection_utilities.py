import cv2
import numpy as np

CASCADE_PATH = "haarcascade_frontalface_default.xml"

RESIZE_SCALE = 3
REC_COLOR = (0, 255, 0)

def getFaceCoordinates(image):
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    rects = cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(48, 48)
        )

    # For now, we only deal with the case that we detect one face.
    if(len(rects) != 1) :
        return None
    
    face = rects[0]
    bounding_box = [face[0], face[1], face[0] + face[2], face[1] + face[3]]

    # return map((lambda x: x), bounding_box)
    return bounding_box

def drawFace(img, faceCoordinates):
    cv2.rectangle(np.asarray(img), (faceCoordinates[0], faceCoordinates[1]), \
    (faceCoordinates[2], faceCoordinates[3]), REC_COLOR, thickness=2)

def crop_face(img, faceCoordinates):
    '''
    extend_len_x =  (256 - (faceCoordinates[3] - faceCoordinates[1]))/2
    extend_len_y =  (256 - (faceCoordinates[0] - faceCoordinates[2]))/2
    img_size = img.shape
    if (faceCoordinates[1] - extend_len_x) >= 0 :
        faceCoordinates[1] -= extend_len_x
    if (faceCoordinates[3] + extend_len_x) < img_size[0]:
        faceCoordinates[3] += extend_len_x
    if (faceCoordinates[0] - extend_len_y) >= 0 :
        faceCoordinates[0] -= extend_len_y
    if (faceCoordinates[2] + extend_len_y) < img_size[1]:
        faceCoordinates[2] += extend_len_y
    '''
    return img[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

def preprocess(img, faceCoordinates, face_shape=(48, 48)):
    '''
        This function will crop user's face from the original frame
    '''
    face = crop_face(img, faceCoordinates)
    #face = img
    face_scaled = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)
    
    return face_gray
