clear ; close all; clc

%%  Logistic regression approach for DrivenData competition 'Blood Donation'
%%  So we'll initially get four X columns, and one y column, from our training set.

data = csvread('training.csv');
X = data(2:end,2:5); 
y = data(2:end, 6);

%% From our data, it is easy to see that the third column holds no real value.
%% It is merely the second column multiplied by a constant. Instead, we're calculating 
%% a new feature: Average donation per month, dividing the third column by the fourth.

X(:,3)=X(:,3)./X(:,4);

X=featureNormalize(X);      %% normalizing into values easier to work with.

X=[ones(size(X,1),1) X];    %% adding bias

initial_theta = zeros(size(X, 2), 1);

% In a previous attempt, I actually mapped polynominal features to X. If you increase
% features, increasing lambda will decrease the new variance, and provide a better model.

lambda = 0;

% Setting Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimizing using the built-in octave function fminunc
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);


data=csvread('test.csv');   %%Loading test data

%%Preprocessing the data with feature normalization, like before.

X = data(2:end,2:5); 

X(:,3)=X(:,3)./X(:,4);

X=featureNormalize(X);

X=[ones(size(X,1),1) X]; 

%%predicting the final output

p = predict(theta, X);

csvwrite('answer.csv',[data(2:end,1) p2]); 


