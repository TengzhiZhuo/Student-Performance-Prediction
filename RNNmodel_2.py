import random
import numpy as np
import math
import matplotlib.pyplot as plt

def forward_propagation(layer_num, neuron_list, weights_list, instance):
    pred = []
    a_list = []
    if weights_list is None:
        weights_list = []
        for i in range(layer_num + 1):
            curweight = []
            if i == layer_num:
                weights = []
                pre_layer_neuron_num = neuron_list[i - 1] + 1
                for n in range(pre_layer_neuron_num):
                    weights.append(random.uniform(-1.0, 1.0))
                curweight.append(weights)
            else:
                for m in range(neuron_list[i]):
                    weights = []
                    pre_layer_neuron_num = 0
                    if i == 0:
                        pre_layer_neuron_num = len(instance[0]) + 1
                    else:
                        pre_layer_neuron_num = neuron_list[i-1] + 1
                    for n in range(pre_layer_neuron_num):
                        weights.append(random.uniform(-1.0, 1.0))
                    curweight.append(weights)
            weights_list.append(curweight)
    #print(weights_list)
    a1 = []
    a1.append([1])
    for i in instance[0]:
        a1.append([i])
    a = a1.copy()
    n = 1
    #print("a" + str(n) + ": " + str(a1))
    a_list.append(a)
    for layer in range(layer_num):
        n += 1
        z = np.dot(np.array(weights_list[layer]), np.array(a))
        #print("z" + str(n) + ": " + str(z))
        a = []
        a.append([1])
        for i in z:
            a.append([1 / (1 + math.exp(-i))])
        a_list.append(a)
        #print("a" + str(n) + ": " + str(a))
    n += 1
    z = np.dot(weights_list[-1], a)
    #print("z" + str(n) + ": " + str(z))
    a = []
    for i in z:
        a.append([1 / (1 + math.exp(-i))])
    pred = a
    a_list.append(a)
    #print("a" + str(n) + ": " + str(a))
    #print("Predicted output for instance: " + str(pred))
    #print("Expected output for instance: " + str(instance[1]))
    return pred, a_list, weights_list

def cost_function(layer_num, neuron_list, weights_list, ins, lambda_value):
    J = 0
    y = []
    y.append([ins[1]])
    output = forward_propagation(layer_num, neuron_list, weights_list, ins)
    weights_list = output[2]
    cur_J = []
    for j in range(len(output[0])):
        cur_J.append([-y[j][0] * math.log(output[0][j][0]) - (1-y[j][0]) * math.log(1-output[0][j][0])])
    sum_J = 0
    for j in cur_J:
        sum_J += j[0]
    J += sum_J
    #print("Cost, J , associated with instance: " + str(sum_J))
    S = 0
    for layer in weights_list:
        for row in layer:
            new_row = row[1:]
            for r in new_row:
                S += r*r
    S *= (lambda_value / (2))
    #print("Final (regularized) cost, J, based on the complete training set: " + str(J+S))
    return J+S

def backpropagation(layer_num, neuron_list, weights_list, instance, lambda_value):
    ini_J = cost_function(layer_num, neuron_list, weights_list, instance, lambda_value)
    D_list = []
    for i in range(layer_num + 1):
        D_list.append([])
    output, a_list, wl = forward_propagation(layer_num, neuron_list, weights_list, instance)
    weights_list = wl
    y = []
    delta_list = []
    y.append([instance[1]])
    delta = np.array(output) - np.array(y)
    delta_list.insert(0, delta)
    #print("delta: " + str(delta))
    for num in range(layer_num):
        curweight = weights_list[-num-1]
        tweight = np.transpose(np.array(curweight))
        delta = np.dot(tweight, delta)
        for i in range(len(delta)):
            delta[i][0] = delta[i][0] * a_list[-num-2][i][0] * (1-a_list[-num-2][i][0])
        delta = delta[1:]
        delta_list.insert(0, delta)
        #print("delta: " + str(delta))
    for num in range(layer_num + 1):
        cur = np.array(D_list[-num-1])
        if len(cur) == 0:
            D_list[-num - 1] = np.dot(delta_list[-num-1], np.transpose(a_list[-num-2]))
        else:
            D_list[-num-1] = cur + np.dot(delta_list[-num-1], np.transpose(a_list[-num-2]))
    #for i in range(len(delta_list)):
        #print("Gradients of Theta" + str(len(delta_list) - i) + " based on training instance: " + str(np.dot(delta_list[-i-1], np.transpose(a_list[-i-2]))))
    for layer in range(layer_num+1):
        P = []
        for w in weights_list[-layer-1]:
            lst = []
            for e in w:
                lst.append(lambda_value * e)
            lst[0] = 0
            P.append(lst)
        D_list[-layer-1] = 1 * (np.array(D_list[-layer-1]) + np.array(P))
    #for i in range(len(D_list)):
        #print("Final regularized gradients of Theta" + str(i+1) + ": " + str(D_list[i]))
    for layer in range(layer_num+1):
        weights_list[-layer-1] = np.array(weights_list[-layer-1]) - 2 * np.array(D_list[-layer-1])
    cur_J = cost_function(layer_num, neuron_list, weights_list, instance, lambda_value)
    if ini_J - cur_J < 0.001:
        return weights_list
    else:
        backpropagation(layer_num, neuron_list, weights_list, instance, lambda_value)

def run_rnn(trainingset, layer_num, neuron_list, lambda_value, weight_list = None):
    for instance in trainingset:
        weight_list = backpropagation(layer_num, neuron_list, weight_list, instance,lambda_value)
    return weight_list


# trainingset = [   [[codestate_array1], [codestate_label1]] ,    [[codestate_array2], [codestate_label2]] ...    ]
# testingset same as trainingset