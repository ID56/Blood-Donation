function [J, grad] = costFunctionReg(theta, X, y, lambda)

%%Cost function with a regularization parameter, for running with higher order polynominals
%%In this version of my approach, I'm using features of the first degree, so lambda is set to
%%zero and actually performs no regularization

m = length(y); 
J = 0;
grad = zeros(size(theta));

t2=theta.^2;
n=size(theta);
J=-sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta)))/m + ...
lambda*sum(t2(2:n))/(2*m);

grad=(X'*(sigmoid(X*theta)-y))/m +(lambda/m)*theta;

grad(1)= sum((sigmoid(X*theta)-y).*X(:,1))/m;

end
