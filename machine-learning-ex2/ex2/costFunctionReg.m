function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

delta = 0;
reg = zeros(size(theta));

hypothesis = sigmoid(X * theta);

delta = sum((-1 * y).*log(hypothesis) -(1 - y) .* log(1 - hypothesis));


grad(1) = 1/m.* sum((hypothesis - y).* X(:, 1));
for j = 2:size(grad)  
    grad(j) = 1/m.* sum((hypothesis - y).* X(:, j)) + ((lambda/m).*theta(j));
    %reg(j) = lambda/(2 * m)* sum(theta(j).^2);
end
theta1 = zeros(size(theta));
for i = 2:size(theta)
    theta1(i) = theta(i);
J = (1/m).*(delta) + ((lambda/(2 * m))* sum(theta1.^2));


% =============================================================

end
