function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% List of possible values for C and sigma.
vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% Variable to hold the least prediction error.
% The max possible error is 1.
err_least = 1.1;

% Check all possible pairs (perhaps building a MH algorithm would've been
% nicer).
for sigma_trial = vals
	for C_trial = vals
		model= svmTrain(X, y, C_trial, @(x1, x2) gaussianKernel(x1, x2, sigma_trial));
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		if err < err_least
			err_least = err;
			C = C_trial;
			sigma = sigma_trial;
		end
	end
end





% =========================================================================

end
