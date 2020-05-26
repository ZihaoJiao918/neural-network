# neural-networkManual calculation of Backpropagation

Consider a MLP that is described in word as follows:

input layer has 3 components (i1,i2,i3)
hidden layer 1 (processing the input layer) has 3 components (j1,j2,j3)
fully connected to the input layer
w_{1,{a,b}} indicates the weight from input a to hidden component b
W_1 is the matrix of all weights above; it is thus a 3x3 matrix
b_1 is the matrix of biases
relu is the nonlinearity used
j = relu(W_1*i + b_1), where relu is a vector function defined by the component-wise application of relu to the components of its vector argument, describes the computation of the components of this layer
hidden layer 2 (processing hidden layer 1) has 3 components (k1,k2,k3)
analogous definitions to the above, but with:
w_{2,{a,b}}
W_2
b_2
logistic nonlinearity
k = logistic(W_2*j + b_2)
output layer has 3 components (l1,l2,l3)
analogous definition to the above but with the subscripts changed, the argument changed from j to k, and the nonlinearity changed to softmax
cross entropy error is used for training


 Employ the back propagation algorithm to write the update rules for the weights and biases of this network.


Code the above network in Python. Use JAX to compute the update rule. Generate a random set of weights, biases, inputs and outputs to show that your update rule and the one computed via JAX are identical.
