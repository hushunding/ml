import mnist_loader
import NNDL

training_data, test_data = mnist_loader.load_data_wrapper2('./data')

net = NNDL.Network([784, 30 ,10])
net.SGD(training_data, 30,10,3.0, 'mnist', test_data)