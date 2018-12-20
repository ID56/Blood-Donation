function p = predict(theta, X)
%Predicting the probability of test examples being 1, using our learned
%logistic regression parameters.

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

p=sigmoid(X*theta);

%%If you actually want to predict whether the label is zero or one, you
%%need to set a threshold parameter between 0 and 1. For example, assuming a 
%%threshold of 0.5, we could predict:
%%for i=1:m
%%if p(i)>=0.5
%%  p(i)=1;
%%else
%%  p(i)=0;
%%endif
%%endfor

end
