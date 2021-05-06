clc,clear,close all

%% Generate Training Data
% generate 100 points for ICs and BCs
Nu = 100;

% space domain for ICs
x0IC = linspace(-1,1,Nu);

% ICs
t0IC = zeros(1,Nu);
u0IC = -sin(pi*x0IC);

% Space domain for BCs
x0BC1 = -1*ones(1,Nu);
x0BC2 = ones(1,Nu);

% time interval for BCs
t0BC1 = linspace(0,1,Nu);
t0BC2 = linspace(0,1,Nu);

% BCs
u0BC1 = zeros(1,Nu);
u0BC2 = zeros(1,Nu);

% Concatenate ICs and BCs
X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];

% Convert the initial and boundary conditions to dlarray. 
% specifiy input as 'CB' (channel, batch).
dlX0 = dlarray(X0,'CB');
dlT0 = dlarray(T0,'CB');
dlU0 = dlarray(U0);

% generate 10,000 collocation points  
Nf = 10000;
 
% draw numbers from quasirandom set
indep_vars = 2;
pointSet = sobolset(indep_vars);
points = net(pointSet,Nf);

% assign random values to the grid (x,t)
dataX = 2*points(:,1)-1;
dataT = points(:,2);

% store grid in a data structure
ds = arrayDatastore([dataX dataT]);

%% Initialize the structure of the Neural Network

% number of layers and neurons in the NN
layers = 9;
neurons = 20;

% layers = 4;
% neurons = 40;

% two inputs to the NN x and t
input_size = 2; 
% one output of the NN u(x,t)
output_size = 1;

% empty struct for the NN parameters
parameters = struct;

% Data structure for the first layer
% Each neuron is assigned two weights and one bias term 
parameters.fc1.Weights = InitializeWeights([neurons 2],input_size);
parameters.fc1.Bias = InitializeBiases([neurons 1]);

% Data structure for the hidden layers
for thislayer=2:layers-1
    name = "fc"+thislayer;
    incoming_conxns = neurons;
    % Each neuron is assigned two weights and one bias term 
    parameters.(name).Weights = InitializeWeights([neurons neurons],incoming_conxns);
    parameters.(name).Bias = InitializeBiases([neurons 1]);
end

% Initialize the parameters for the final fully connect operation. The final fully connect operation has one output channel.
% Data structure for the last layer
incoming_conxns = neurons;
% nine weights incoming to the last layer
parameters.("fc" + layers).Weights = InitializeWeights([output_size neurons],incoming_conxns);
% one bias term for one output
parameters.("fc" + layers).Bias = InitializeBiases([output_size output_size]);


%% Training settings
%Train the model for 3000 epochs with a mini-batch size of 1000.

% Initialize the number of the epochs: the number of times the whole
% training data is passed through the NN
epochs = 3000;
% size of the batch for the SGD 
batch_size = 1000;

%Specify ADAM optimization options.
initialLearnRate = 0.01;
decayRate = 0.005;

%% Train Network using stochastic gradient descent

% Arrange minibatches to be used for training
mbq = minibatchqueue(ds,'MiniBatchSize',batch_size,'MiniBatchFormat','BC');

% empty vector for optimization.
averageGrad = [];
averageSqGrad = [];

% Initialize the training progress plot.
figure(1)
an_error = animatedline('Color',[0 0 0],'linewidth',2);
ylim([0 inf])
xlabel("Iteration")
ylabel("Training Error")

% initialize iteration (note: iteration here is epoch*batchsize)
iteration = 0;
% iterate through the number of epochs and draw minibatches
for epoch = 1:epochs
    % draw a minibatch
    reset(mbq);
    % iterate through the elements of the minibatch
    while hasdata(mbq)
        iteration = iteration + 1;
        % Returns the next mini-batch
        dlXT = next(mbq);
        % separate new minibatch
        dlX = dlXT(1,:);
        dlT = dlXT(2,:);
        
        % Calculate the gradient and minimize the total error 
        [gradients,err] = dlfeval(dlaccelerate(@total_error_gradient),parameters,dlX,dlT,dlX0,dlT0,dlU0);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network learnable parameters (i.e. weights, biases,
        % and learned x and t)
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end
    % Plot training progress.
    err = double(gather(extractdata(err)));
    addpoints(an_error,iteration, err);

    title("Epochs: " + epoch + ", Final Total Error: " + err)    
    drawnow
    xlabel('Iteration')
    ylabel('Training Error')
    set(gca,'Fontsize',13)
end


%% plot prediction in 2D plots

% pick three time points
tgrid = [0.25 0.5 0.75];
% grid settings
plt_grid = 1000;
xgrid = linspace(-1,1,plt_grid);

figure(2)

for i=1:numel(tgrid)
    t = tgrid(i);
    TTest = t*ones(1,plt_grid);

    % get the final output from the NN.
    dlXTest = dlarray(xgrid,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred = forward_passNN(parameters,dlXTest,dlTTest);

    % Calcualte analytic solution of the Burger's equation.
    U_exact = get_analytic(xgrid,t);

    % Calculate error between the prediction and analytic solution.
    err = norm(extractdata(dlUPred) - U_exact) / norm(U_exact);

    % Plot predictions.
    subplot(1,3,i)
    plot(xgrid, U_exact, '-','LineWidth',4)
    ylim([-1.1, 1.1])

    % overlay the analytic solution.
    hold on
    plot(xgrid,extractdata(dlUPred),'--','LineWidth',4);
    xlabel('x')
    ylabel('u(x,t)')
    set(gca,'FontSize',18)
    hold off
    title("t = " + t + ", Error = " + gather(err));
end
legend('Exact','Predicted')


%% store prediction for the 3D plot
plt_grid = 1000;
tgrid = linspace(0,1,plt_grid);
xgrid = linspace(-1,1,plt_grid);

figure(3)
emptyMat = zeros(length(tgrid),plt_grid);
Pred.x = emptyMat;
Pred.u = emptyMat;
Pred.t = tgrid;

for i=1:numel(tgrid)
    t = tgrid(i);
    TTest = t.*ones(1,plt_grid);

    % get the final output from the NN.
    dlXTest = dlarray(xgrid,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred = forward_passNN(parameters,dlXTest,dlTTest);
   
    % create new space vector ovre [-1,1]
    Pred.x(i,:) = xgrid;
    % get predicted solution vector of the Burger's equation.
    Pred.u(i,:) = extractdata(dlUPred);

    % Calcualte analytic solution of the Burger's equation.
    U_exact = get_analytic(xgrid,t);
end

% create 3D plot 
t = repmat(Pred.t',1,plt_grid);
subplot(2,1,2)
surf(t,Pred.x,Pred.u)
xlabel('t')
ylabel('x')
zlabel('u(x,t)')
title('u(x,t)')
colorbar 
colormap jet
shading interp
view(2)
set(gca,'FontSize',13)
subplot(2,1,1)
surf(t,Pred.x,Pred.u)
xlabel('t')
ylabel('x')
zlabel('u(x,t)')
title('u(x,t)')
set(gca,'FontSize',13)
colorbar 
colormap jet
shading interp



