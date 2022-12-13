import time
from os.path import exists
import tensorflow as tf
import tensorflow_hub as hub
changemodel_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
IMAGE_SHAPE = (224, 224)
changelayer = hub.KerasLayer(changemodel_url, input_shape=IMAGE_SHAPE+(3,))
changemodel = tf.keras.Sequential([changelayer])
from scipy.spatial import distance

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import ssl
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import yolov5.detect2 as d2

#for printing
class Flt(float):
    def __str__(self):
        return '{:.2}'.format(self)

class Percent(float):
    def __str__(self):
        return '{:.2%}'.format(self)

#Change Model
def vectorize(filename):
    img1 = image.load_img(filename)
    file = img1.convert('L').resize(IMAGE_SHAPE)  
    file = np.stack((file,)*3, axis=-1)                       
    file = np.array(file)/255.0                               
    embedding = changemodel.predict(file[np.newaxis, ...])
    embedding_np = np.array(embedding)
    change1 = embedding_np.flatten()
    return change1

def change(oldv,newv,threshold):
    metric = 'cosine'
    cosineDistance = distance.cdist([oldv], [newv], metric)[0]
    cosd = str(Flt(cosineDistance))
    outputstring = "From the Change Module\n"
    if (cosineDistance>threshold):
        outputstring += "Change Module detect big change in the variance of " + cosd+".\n"
    else:
        outputstring += "Change Module detect small change in the variance of " + cosd+".\n"
    return outputstring

#Mobile Net Model
ssl._create_default_https_context = ssl._create_unverified_context
mobile = tf.keras.applications.mobilenet.MobileNet()

#reading the train result
posword = {}
negword = {}
fpos = open("mobpos.txt","r")
posLines = fpos.readlines()
fneg = open("mobneg.txt","r")
negLines = fneg.readlines()
for pos in posLines:
    pos = pos[:-1]
    posword[pos] = 1
for neg in negLines:
    neg = neg[:-1]
    negword[neg] = 1

def mobilenet(filename):
    img = image.load_img(filename,target_size = (224,224))
    resized_img = image.img_to_array(img)
    final_image = np.expand_dims(resized_img,axis = 0)
    final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)
    predictions = mobile.predict(final_image)
    results = imagenet_utils.decode_predictions(predictions)
    posprobablity = 0
    negprobablity = 0
    for p in results[0]:
        if (p[2]>0.01):
            if p[1] in posword:
                posprobablity += p[2]
            if p[1] in negword:
                negprobablity += p[2]
    
    pospro = str(Percent(posprobablity))
    negpro = str(Percent(negprobablity))
    outputstring = "From the Mobilenet module\n"
    outputstring += "Mobilenet detect pos-wildfire object in the probability of " + pospro+".\n"
    outputstring += "Mobilenet detect neg-wildfire object in the probability of " + negpro+".\n"
    #print(outputstring)
    return outputstring

#SVM model
#SVM model trained in SVM_train.py
def svm_test(filename):
    model = load_model('wildfire_test1.h5')
    test_image = image.load_img(filename, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image=test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    outputstring = "From the SVM module\n" 
    pos = str(Flt(result[0]))

    if (result[0]>0):
        outputstring += "SVM Model detect it is positive in the result of " + pos+".\n"
    else:
        outputstring += "SVM Model detect it is negative in the result of " + pos+".\n"
    return outputstring

current = 0
filename = "input/" + str(current)+".jpg"
while (True):
    file_exists = exists(filename)
    if (file_exists):
        print(filename,"has been read!")
        
        
        if (current == 0):
            outputmobilenet = mobilenet(filename)
            outputsvm = svm_test(filename)
            opt = d2.parse_opt(current)
            d2.main(opt)
            print("====================")
            print(outputmobilenet)
            print(outputsvm)
            print("====================")
        else:
            filelast = "input/" + str(current-1)+".jpg"
            newv = vectorize(filename)
            oldv = vectorize(filelast)
            outputchange = change(oldv,newv,0.1)
            outputmobilenet = mobilenet(filename)
            outputsvm = svm_test(filename)
            opt = d2.parse_opt(current)
            d2.main(opt)
            print("====================")
            print(outputchange)
            print(outputmobilenet)
            print(outputsvm)
            print("====================")


        current = current + 1
        filename = "input/" + str(current)+".jpg"
        file_exists = exists(filename)
    else:
        print("Not yet")
        time.sleep(5)


