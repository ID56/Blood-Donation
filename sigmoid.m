function g = sigmoid(z)
%Basically computes the sigmoid that I use in logistic regression/nn models
g = zeros(size(z));

E=e.^(-z);       %Broke it down to simpler terms for understanding 
g=1./(1+E);      %Sigmoid always outputs values between 0 and 1

end
