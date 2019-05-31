import worddata
import NNDL

training_data, test_data = worddata.loadCharsData(60000,10000)

net = NNDL.Network([784, 60 ,26])
net.SGD(training_data, 60,10,3.0, 'charOCR', test_data)