# Real-Time-Facial-Expression-Recognition-with-DeepLearning
A real-time facial expression recognition system through webcam streaming and CNN

## Environment
I provide my work environment for references.

### Hadware
CPU : i5-6500  
GPU : nvidia GTX 960 2G  
RAM : 8G  

### Software
OS  : Ubuntu 16.04  
Keras 1.2.0  
scikit-learn 0.18.1  
opencv 3.1.0  

## Installation
I strongly recommend you to use [`Anaconda`](https://www.continuum.io/downloads), which is a package manager and provides python virtual environment.  
After you install Anaconda, you can create a virtual environment with python 3.4.
```
conda create -n env-name python=3.4
```
you can also check if your env. has been created by,
```
conda info --envs
```
You should activate your virtual environment in different way corresponding to your operating system.
For example, In Ubuntu, you can activate your virtual environment by,
```
source activate env-name
```
The following instructions will lead you to install dependencies, and I suggest you to fllow the order.
#### Install scikit-learn
```
conda install scikit-learn
```
#### Install OpenCV
Note that the version `Anaconda` provided may not be the latest one.
```
conda install opencv
```
#### Install Keras
Keras is a high-level wrapper of Theano and Tensorflow, it provides friendly APIs to manipulate several kinds of deep learning models.
```
pip install --upgrade keras
```
#### Install pandas and h5py
`pandas` can help you to preprocess data if you want train your own deep learning model.
```
conda install pandas
```
`h5py` is used to save weights of pre-trained models.
```
conda install h5py
```
## Usage
After installing dependencies, you can move to `webcam` directory and simply type this command,
```
python webcam_detection.py
```
and the system will start detecting user's emotions and print results to the console.
