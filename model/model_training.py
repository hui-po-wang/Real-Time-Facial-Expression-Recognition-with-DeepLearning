import argparse

import cv2
import numpy as np

from keras.callbacks import LambdaCallback, EarlyStopping

import feature_utility as fu
import myVGG

parser = argparse.ArgumentParser(description=("Model training process."))
# parser.add_argument('data_path', help=("The path of training data set"))
parser.add_argument('--test', help=("Input a single image to check if the model works well."))

args = parser.parse_args()

def main():
    model = myVGG.VGG_16()

    if args.test is not None:
        print ("Test mode")
        img = cv2.imread(args.test)
        img = fu.preprocessing(img)
        img = np.expand_dims(img, axis=0)
        y = np.expand_dims(np.asarray([0]), axis=0)
        batch_size = 1
        model.fit(img, y, nb_epoch=400, \
                batch_size=batch_size, \
                validation_split=0.2, \
                shuffle=True, verbose=0)
        return

    #input_path = args.data_path
    #print("training data path : " + input_path)
    #X_train, y_train = fu.extract_features(input_path)
    X_fname = '../data/X_train_train.npy'
    y_fname = '../data/y_train_train.npy'
    X_train = np.load(X_fname)
    y_train = np.load(y_fname)
    print(X_train.shape)
    print(y_train.shape) 
   
    print("Training started")

    callbacks = []
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))
    epoch_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print("epoch:", epoch))
    callbacks.append(earlystop_callback)
    callbacks.append(batch_print_callback)
    callbacks.append(epoch_print_callback)

    batch_size = 512
    model.fit(X_train, y_train, nb_epoch=400, \
            batch_size=batch_size, \
            validation_split=0.2, \
            shuffle=True, verbose=0, \
            callbacks=callbacks)

    model.save_weights('my_model_weights.h5')
    scores = model.evaluate(X_train, y_train, verbose=0)
    print ("Train loss : %.3f" % scores[0])
    print ("Train accuracy : %.3f" % scores[1])
    print ("Training finished")

if __name__ == "__main__":
    main()
