from PIL import Image,ImageFilter
import numpy as np


def handlsingpic(path):
    img = Image.open(path)
    img = img.convert("L")
    img = img.crop((2, 2, img.size[0], 30))
    for i in range(4):
        img = img.filter(ImageFilter.SMOOTH)
    for i in range(4):
        img = img.filter(ImageFilter.SHARPEN)
    bs = np.copy(np.asarray(img))
    for x in np.nditer(bs, op_flags=['readwrite']):
        x[...] = 255-(255 if x > 128 else x)
    # Image.fromarray(bs).show()
    return divPic(bs)

def divPic(bs):
    divs = [1 if s > 100 else 0 for s in np.sum(bs, axis=0)]
    dx = []
    newdiv = False
    for d in range(len(divs)):
        dsum = np.sum(divs[d:d+2])
        if dsum == 0:
            if newdiv : dx.append(d+2)
            newdiv = False
        else:
            newdiv = True
    bss = np.split(bs,dx, axis=1)
    bss = [expandpic(b) for b in bss if b.shape[1] > 3]
    bss = [Image.fromarray(b).show() for b in bss]
    

def expandpic(b):
    left = int ((28 - b.shape[1])/2)
    right = 28 -left - b.shape[1]
    if left > 0 :
        left = np.zeros(((28,left)), dtype= b.dtype)
        b = np.concatenate(left, b)
    if right > 0:
        b = np.concatenate(b, np.zeros((28,right)))
    return b 
    
#     img.show()
#     x, y = img.size
#     for i in range(0, 4):
#         s = i*x/4.0
#         im = img.crop((s, 0, s+x/4.0, y)).resize((28,28))
#         im.show()


if __name__ == "__main__":
    handlsingpic('./pic/cap3.png')
