import sys, os
import cv2
import numpy as np
def preprocessing(img, size=(48, 48)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, size).astype(np.float32)
    #img = img.transpose((2, 0, 1))
    # img = np.expand_dims(img, axis=0)

    return img
    
def extract_features(path):
    X, y = [], []
    label = 0
    for dirnames in os.listdir(path):
        # print(dirnames)
        sub_path = os.path.join(path, dirnames)
        # print(sub_path)
        for filename in os.listdir(sub_path):
            # print (filename)
            file_path = os.path.join(sub_path, filename)
            img = cv2.imread(file_path)
            img = preprocessing(img)
            X.append(img)

            class_label = [0, 0, 0, 0, 0, 0, 0]
            class_label[label] = 1
            y.append(class_label)
        label += 1
    
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

if __name__ == "__main__" :
    X, y = extract_features(sys.argv[1])
    print(X, y)
    print(type(X), type(X[0]))
    print(X[212])
    print(len(X))
