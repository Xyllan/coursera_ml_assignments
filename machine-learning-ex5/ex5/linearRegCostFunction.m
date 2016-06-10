function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute the hypothesis
ht = X * theta;

% Compute cost and add regularization cost
J = sum((ht - y) .** 2)/(2*m) + (lambda/(2*m))*sum((theta .^ 2)(2:end));

% Compute the unregularized gradient
grad = (sum((ht - y) .* X, 1)*1/m)';

% Add regularization
grad(2:end) += (lambda/m)*theta(2:end);


% =========================================================================

grad = grad(:);

end
