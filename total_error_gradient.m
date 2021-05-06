function [gradients,loss] = total_error_gradient(parameters,dlX,dlT,dlX0,dlT0,dlU0)

% Make predictions with the initial conditions.
U = forward_passNN(parameters,dlX,dlT);

% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,'all'),{dlX,dlT},'EnableHigherDerivatives',true);
Ux = gradientsU{1};
Ut = gradientsU{2};

% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);

% Calculate lossF. Enforce Burger's equation.
f = Ut + U.*Ux - (0.01./pi).*Uxx;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

% Calculate lossU. It enforces the initial and boundary conditions.
dlU0Pred = forward_passNN(parameters,dlX0,dlT0);
lossU = mse(dlU0Pred, dlU0);

% total loss function
loss = lossF + lossU;

% Calculate gradients with respect to the NN parameters.
gradients = dlgradient(loss,parameters);

end