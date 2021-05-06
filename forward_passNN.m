function dlU = forward_passNN(parameters,dlX,dlT)
dlXT = [dlX;dlT];
layers = numel(fieldnames(parameters));

% computes the weighted sum of input in the first layer
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
dlU = fullyconnect(dlXT,weights,bias);

% forward pass in the hidden layer
for thislayer=2:layers
    name = "fc" + thislayer;
    % pass the weighted sum of input through the activation function
    dlU = tanh(dlU);
    % store parameters
    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    % computes the weighted sum of input 
    dlU = fullyconnect(dlU, weights, bias);
end

end