function X = featureNormalize(X)
  %Preprocessing data before use by standard normalizing them: (x-mean)/sigma
  m=size(X,1);
  n=size(X,2);
  
  mean=zeros(n,1);
  sigma=zeros(n,1);
  
  for i=1:n
    X(:,i)=(X(:,i)-(sum(X(:,i))/m))/std(X(:,i));
  endfor
endfunction
