import mnist_loader
import NNDL
import matplotlib.pyplot as plt

training_data, test_data = mnist_loader.load_data_wrapper2('./data')

net = NNDL.Network([784, 30, 10])
training_cost, training_accuracy,evaluation_cost, evaluation_accuracy = net.SGD(training_data, 30, 10, 3.0, 'mnist',
        lmbda=6.0,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True
        )
    
plt.plot(training_accuracy)
plt.plot(evaluation_accuracy)
plt.show()