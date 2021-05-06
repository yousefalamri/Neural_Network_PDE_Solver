function w = InitializeWeights(num_ws,cnxns)
% 
% Initialize weights from the normal distribution
% with mean = 0 and variance = 2/connections
% with zero mean and variance 2/Ni

weights = randn(num_ws,'single') * sqrt(2/cnxns);
w = dlarray(weights);

end