from PIL import Image


def handlsingpic(path):
    img = Image.open(path)
    img = img.convert("L")
    img = img.crop((2, 2, img.size[0], 30))
    x, y = img.size
    for i in range(0, 4):
        s = i*x/4.0
        im = img.crop((s, 0, s+x/4.0, y)).resize((28,28))
        im.show()


if __name__ == "__main__":
    handlsingpic('./pic/cap3.png')
