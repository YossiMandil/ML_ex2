import numpy as np
import math as math
from matplotlib import pyplot as plt

""" 
generate 100 examples for each class (each class conditional density is normal)
"""
def generate_examples(classess,num_of_examples = 100):
    dataset = []
    for c in classess:
        X = np.random.normal(2*c, 1, num_of_examples)
        for x in X:
            dataset.append((x, c))
    return dataset



"""
returns a n-dim array that in the index i there is the softmax for class i
"""

def calculate_softmax(x,W,b):
    v = W*x+b
    v_exp = np.array([math.e**vi for vi in v])
    sum_of_all_classes = np.sum(v_exp)
    return np.array([yi/sum_of_all_classes for yi in v_exp])


"""
calculates probability density function
"""
def calculate_PDF(x,mean,variance=1):
    #pdf
    return [(1/(math.sqrt(2*math.pi*variance)))*np.exp(-np.square(x-m)/(2*variance))for m in mean]


"""
train the examples with lr and run it num_of_iteration times
"""
def train(examples, lr=0.1, num_of_iterations=30, num_of_classes=3):
        W = np.zeros(num_of_classes)
        b = np.zeros(num_of_classes)
        for i in range(num_of_iterations):
            np.random.shuffle(examples)
            for (x, y) in examples:
                softmax = calculate_softmax(x, W, b)
                gradiemt_W = x * softmax
                gradiemt_b = softmax
                gradiemt_W[y - 1] -= x
                gradiemt_b[y - 1] -= 1
                W -= lr * gradiemt_W
                b -= lr * gradiemt_b
        return W, b


"""
Draws the graph of the function the was learned by the program 
and the probability density function 
"""
def draw_graphs(W,b,classes,mean=2,variance=1):
    X = np.arange(0.0, 11.0, 0.25)
    predictions = []
    true_value = []
    for x in X:
        v=calculate_softmax(x, W, b)
        predictions.append(v[0])
        fx=calculate_PDF(x,mean*classes,variance)
        true_value.append(fx[0]/np.sum(fx))
    plt.plot(X, predictions, 'r', label= "learned function")
    plt.plot(X, true_value, 'b', label= "prob function")
    plt.legend()
    plt.axis([0, 10, 0, 1])
    plt.show()




def main(classes):
    dataset = generate_examples(classes)
    W, b = train(dataset)
    draw_graphs(W, b, np.array(classes))




if __name__=="__main__":
    main([1, 2, 3])