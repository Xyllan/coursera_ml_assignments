function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
a1 = [ones(m, 1) X]; % m x (1 + input_layer_size)

% Encode y's into vectors.
y_enc = zeros(size(y,1),num_labels); % m x num_labels
y_enc(sub2ind(size(y_enc), 1:m, y')) = 1;

% Calculate the activations.
z2 = a1 * Theta1'; % m x hidden_layer_size
a2 = [ones(m, 1) sigmoid(z2)]; % m x (1 + hidden_layer_size)
z3 = a2 * Theta2'; % m x num_labels
a3 = sigmoid(z3); % m x num_labels
ht = a3; % For this network, the hypothesis is the activations of the third layer

% Calculate the cost of each hypothesis for each output, for each training data.
% The hypothesis matrix is of size (m x num_labels)
% The encoded y matrix is also of size (m x num_labels)
% Hence, we construct the cost matrix and sum over the whole matrix.
J_unreg = (1.0/m)*sum(sum(-y_enc.*log(ht) - (1-y_enc).*log(1-ht)));

% Regularize over all values of theta, excluding the +1 bias unit.
J = J_unreg + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));

% Compute the error terms
d3 = a3 - y_enc; % m x num_labels
d2 = (d3 * Theta2)(:,2:end) .* sigmoidGradient(z2); % m x hidden_layer_size

% Compute gradient terms
D2 = ((1./m) * (d3' * a2)); % num_labels x (1 + hidden_layer_size)
D1 = ((1./m) * (d2' * a1)); % hidden_layer_size x (1 + input_layer_size)

% Regularization
D2(:,2:end)+= Theta2(:,2:end) * (lambda / m); % num_labels x hidden_layer_size
D1(:,2:end)+= Theta1(:,2:end) * (lambda / m); % hidden_layer_size x input_layer_size

% Return the calculated gradients.
Theta1_grad = D1;
Theta2_grad = D2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
