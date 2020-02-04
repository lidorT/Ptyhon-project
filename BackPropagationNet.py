import numpy as np

def get_count_of_char_for_each_row(list_strings, find_char):
    return [row.count(find_char) for row in list_strings]

## gets pattern of 100X100 chars, and flattens it a list of the counts of '*' in each of the pattern rows.
def flatten_input(pattern):
    on_symbol = '*'
    return get_count_of_char_for_each_row(pattern, on_symbol)

def activation_func(x):
    """Activation Function for Each Perceptron"""
    a = -1
    return 1/(1+np.exp(-a*x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def calculate_output(input_neurons, weights, layer_neurons, final='no'):
    output = []
    
    if (final=='yes'):
        output = list(softmax([sum(input_neurons*weights[i]) for i in range(layer_neurons)]))
        output = output.index(max(output))
    else:
        for neuron in range(layer_neurons):
            output.append(activation_func(sum(input_neurons*weights[neuron])))
            # output.append(sum(input_neurons*weights[neuron]))
    return output

def get_initialized_weights(input_neurons, output_neurons):
    return np.random.rand(output_neurons, input_neurons)

def is_it_error(output,target):
    return not(output==target)

def sqr(x):
    squared = [n*n for n in x]
    return np.array(squared)

def adjust_weights(weights, output, target, layers):
    learning_rate = 0.05
    # calculate final output delta
    out_delta = np.array([(1 - sqr(weight)) * (target - output) for weight in weights[-1] ])        
     
    # calculate hidden_deltas for each layer
    # hidden_deltas = [(1 - sqr(layer_weights)) * out_delta.mean() * layer_weights for layer_weights in weights[:-1]]
    hidden_deltas = [sqr(1 - layer_weights) * out_delta.mean() * layer_weights for layer_weights in weights[:-1]]

    # change weights
    weights[-1] = list(np.array(weights[-1]) + learning_rate * out_delta * np.array(layers[-1]))
    for i in range(len(weights)-2,0,-1):
        weights[i] = list(np.array(weights[i])+(learning_rate * np.array(hidden_deltas[i]) * np.array(layers[i])))
        # weights[i] = list(np.array(weights[i-1])+learning_rate * np.array(hidden_deltas[i]))
    return weights

def train_network(patterns, num_groups):
    loop = 0
    max_loops = 3000
    num_input_neurons = 100
    num_hidden_layer_neurons =100
    num_output_neurons = 3
    num_hidden_layers = 1
    weights=[]
    if(num_hidden_layers==1):
        weights.append(get_initialized_weights(num_input_neurons,num_hidden_layer_neurons))
        weights.append(get_initialized_weights(num_hidden_layer_neurons,num_output_neurons))
    else:
        weights.append(get_initialized_weights(num_input_neurons,num_hidden_layer_neurons))
        weights.append(get_initialized_weights(num_hidden_layer_neurons,num_hidden_layer_neurons))
        weights.append(get_initialized_weights(num_hidden_layer_neurons,num_hidden_layer_neurons))
        weights.append(get_initialized_weights(num_hidden_layer_neurons,num_output_neurons))
    pre_weights = weights[-1]
    print('--------------------Initialized Weights-------------------------')
    print()
    print('----------------------------------------------------------------')
    print()
    print('-------------------- Training Network --------------------------')
    print()
    print('----------------------------------------------------------------')
    success=0
    max_success=0
    while (loop<max_loops and success<90):
        loop += 1
        print('----------------------------------------------------------------')
        print('------------------ Training Loop : %d  -------------------------' %loop)
        print('----------------------------------------------------------------')
        errors = 0

        for group in range(num_groups):
            for i,shape in enumerate(patterns[6*group:6*(group+1)]):
                layers=[]
                layers.append(flatten_input(shape))
                layers.append(calculate_output(layers[0], weights[0], num_hidden_layer_neurons))       ##input to hidden1
                # layers.append(calculate_output(layers[1], weights[1], num_hidden_layer_neurons))       ##hidden1 to hidden2
                # layers.append(calculate_output(layers[2], weights[2], num_hidden_layer_neurons))       ##hidden2 to hidden3
                output = calculate_output(layers[1], weights[1], num_output_neurons, final='yes')              ##hidden3 to output
                target = i%3
                # print(output, "     target: ", target)
                if(is_it_error(output, target)):
                    errors += 1
                    adjust_weights(weights,output, target,layers)
                    #print(weights[-1])
        success= round((1- errors/(6*num_groups))*100, ndigits=None)
        print('----------------------------------------------------------------')
        print('---------------------%d - Percent Success -------------------------------' %success)
        print('----------------------------------------------------------------')
        max_success = success if success>max_success else max_success
    if loop==max_loops:
        print('training network failed')
        print(pre_weights)
        return False , weights, max_success
    else :
        print('training network succeeded')
        return True , weights, max_success

def test_network(patterns, test_group, weights):
    num_input_neurons = 100
    num_hidden_layer_neurons = 100
    errors = 0
    for shape in patterns[6*test_group:6*(test_group+1)]:
     input_layer = flatten_input(shape)
     output = calculate_output(input_layer, weights, num_hidden_layer_neurons)       ##input to hidden
     ##output = calculate_output(output, weights, hidden_layer_neurons)     ##hidden1 to hidden2
     ##output = calculate_output(output, weights, hidden_layer_neurons)     ##hidden2 to hidden3
     output = calculate_output(output, weights, 3, final='yes')              ##hidden to output
     target = shape%3
     if(is_it_error(output, target)):
         errors += 1
    success= round((1- errors/6)*100,ndigits=None)
    print('----------------------------------------------------------------')
    print('---------------------%d percent accuracy------------------------' %success)
    print('----------------------------------------------------------------')

import DATA

TRAINED, WEIGHTS,  MAX_SUCCESS= train_network(DATA.SHAPES, num_groups=1)


print('---------------------%d--max success------------------------' %MAX_SUCCESS)
print(WEIGHTS[-1])
if TRAINED:
    test_network(DATA.SHAPES, test_group=2, weights=WEIGHTS)


