function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Cvec = [ 0.3 ];
sigmavec = [0.1];

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
min_err=1000;
c_inx=0;
sig_inx=0;

for i=1,length(Cvec),
  for j=1,length(sigmavec),
    model= svmTrain(X, y, Cvec(i), @(x1, x2) gaussianKernel(x1, x2, sigmavec(j)));
    predictions = svmPredict(model, Xval);
    err=mean(double(predictions ~= yval));
    if err<=min_err,
      c_inx=i;
      sig_inx=j;
    endif
  endfor
endfor

C=Cvec(c_inx);
sigma=sigmavec(sig_inx);





% =========================================================================

end
