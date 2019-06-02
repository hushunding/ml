import os
from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np
import pickle


def genSingleCharData(fontFile= None):
    text = chr(random.randint(ord('A'), ord('Z')))
    im = Image.new('L', (28, 28), (255))
    dr = ImageDraw.Draw(im)
    if fontFile == None:
        fontFile = "msyh.ttc"
    font = ImageFont.truetype(fontFile, 26)
    dr.text((4, -4), text, font=font, fill="#000000")
    # for i in range(50):
    #     x = random.random()*28
    #     y = random.random()*28
    #     dr.point((x, y), random.randint(0,255))

    im = im.rotate(random.randint(-30, 30), fillcolor=255)
    # im.show()
    return im.tobytes(), text


def genCharsData(trainSize, testSize):   
    for ext,size in zip(('training', 'test'), (trainSize, testSize)):
        with open(os.path.join('data', 'ABC_{0}_data_{1}'.format(ext, size)), 'wb') as df, \
             open(os.path.join('data', 'ABC_{0}_tag_{1}'.format(ext, size)), 'wb') as tf:
            for i in range(size):
                x, t = genSingleCharData()
                df.write(x)
                tf.write(t.encode())


def loadCharsData(trainSize, testSize):
    training_data = []
    test_data = []
    for ext,size,data in zip(('training', 'test'), (trainSize, testSize),(training_data, test_data)):
        with open(os.path.join('data', 'ABC_{0}_data_{1}'.format(ext, size)), 'rb') as df, \
             open(os.path.join('data', 'ABC_{0}_tag_{1}'.format(ext, size)), 'rb') as tf:
            for i in range(size):
                x = list(df.read(784))
                t = tf.read(1).decode()
                input = -np.reshape(x, (784, 1))/255.0+1
                if ext == 'training':
                    result = vectorized_result(t)
                else:
                    result = ord(t) - ord('A')
                data.append((input, result))
    return training_data, test_data
    

def vectorized_result(c):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((26, 1))
    e[ord(c)-ord('A')] = 1.0
    return e

if __name__ == "__main__":
    training_data_size, test_data_size = 60000,10000
    # genCharsData(training_data_size, test_data_size)
    training_data, test_data = loadCharsData(training_data_size, test_data_size)
    while True:
        i = int(input())
        d,t = training_data[i]
        td,tt = test_data[i]
        print(chr(np.argmax(t)+ord('A')), chr(tt+ord('A')))
        Image.fromarray(np.reshape(d*255, (28, 28))).show()
        Image.fromarray(np.reshape(td*255, (28, 28))).show()
    pass
    