%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this

newHouse = [1 1650 3];
newHouseNorm = (newHouse(:,[2:3])-mu)./sigma;
newHouseNorm = [1 newHouseNorm];
price = newHouseNorm*theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this

newHouse = [1 1650 3];
price = newHouse*theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
         
%% ============= Part 4: Visualizing J(theta_0, theta_1, theta_2) =============
%fprintf('Visualizing J(theta_0, theta_1, theta_2) ...\n')
%
%% Grid over which we will calculate J
%theta0_vals = linspace(-10, 10, 100);
%theta1_vals = linspace(-1, 4, 100);
%theta2_vals = linspace(-10, 10, 100);
%
% initialize J_vals to a matrix of 0's
%J_vals = zeros(length(theta0_vals), length(theta1_vals), length(theta2_vals));
%
%% Fill out J_vals
%for i = 1:length(theta0_vals),
%    for j = 1:length(theta1_vals),
%      for h = 1:length(theta2_vals),
%	      t = [theta0_vals(i); theta1_vals(j); theta2_vals(h)];    
%	      J_vals(i,j,h) = computeCostMulti(X, y, t);
%      end
%    end
%end
%
%
%% Because of the way meshgrids work in the surf command, we need to 
%% transpose J_vals before calling surf, or else the axes will be flipped
%%J_vals = J_vals';
%%% Surface plot
%%figure;
%%surf(theta0_vals, theta1_vals, theta2_vals, J_vals)
%%xlabel('\theta_0'); ylabel('\theta_1'); zlabel('\theta_2');
%
%% Contour plot
%figure;
%% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
%contour(theta0_vals, theta1_vals, theta2_vals, J_vals, logspace(-2, 3, 20))
%xlabel('\theta_0'); ylabel('\theta_1'); zlabel('\theta_2');
%hold on;
%plot(theta(1), theta(2), theta(3), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


