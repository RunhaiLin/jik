import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import ssl
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
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

class Percent(float):
    def __str__(self):
        return '{:.2%}'.format(self)

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


str0 = mobilenet("input/0.jpg")
str1 = mobilenet("input/1.jpg")
str2 = mobilenet("input/2.jpg")