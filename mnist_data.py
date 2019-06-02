import mnist_loader
from PIL import Image, ImageFont, ImageDraw
import numpy as np

training_data, test_data = mnist_loader.load_data_wrapper2('./data')
while True:
    i = int(input())
    d,t = training_data[i]
    td,tt = test_data[i]
    print(chr(np.argmax(t)+ord('A')), chr(tt+ord('A')))
    Image.fromarray(np.reshape(d*255, (28, 28))).show()
    Image.fromarray(np.reshape(td*255, (28, 28))).show()
