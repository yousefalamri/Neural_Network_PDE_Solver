function Biases = InitializeBiases(num_bs)
% Initialize the bias terms as zero
B = zeros(num_bs,'single');
Biases = dlarray(B);
end