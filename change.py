import tensorflow as tf
import tensorflow_hub as hub
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
IMAGE_SHAPE = (224, 224)
layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
model = tf.keras.Sequential([layer])

import numpy as np
from tensorflow.keras.preprocessing import image
from scipy.spatial import distance

class Flt(float):
    def __str__(self):
        return '{:.2}'.format(self)

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

oldv = vectorize("input/0.jpg")
newv = vectorize("input/2.jpg")
str0 = change(oldv,newv,0.1)
print(str0)

