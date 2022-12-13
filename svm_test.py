import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class Flt(float):
    def __str__(self):
        return '{:.2}'.format(self)

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

str0 = svm_test("input/1.jpg")
print(str0)
        
