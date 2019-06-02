import worddata
import NNDL
from costfunc import QuadraticCost
import matplotlib.pyplot as plt

training_data, test_data = worddata.loadCharsData(60000,10000)

net = NNDL.Network([784, 60 ,26])
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