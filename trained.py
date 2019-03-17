
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
import cv2
from scipy.misc import imresize
imgDimension=100

json_file = open('model (1).json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
img1=r"C:\Users\PIYUSH\Desktop\Images\TCImages\TC001_39.png"
img2=r"C:\Users\PIYUSH\Desktop\Images\TCImages\TS001_11.png"
img1 = cv2.imread(img1)

img1=imresize(img1,[imgDimension,imgDimension,3])
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1=img1.reshape(1,100,100,1)
score1 = loaded_model.predict_classes(img1, verbose=1)
#score2 = loaded_model.predict_classes(img2, verbose=1)
if score1==[0]:
    print("NOT AUTISTIC")
