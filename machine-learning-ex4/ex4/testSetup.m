% [J] = nnCostFunction(t, 2, 4, 4, Xm, ym, 0);
% [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)  
  
  X = reshape(sin(1:32), 16, 2) / 5;
  y = 1 + mod(1:16,4)';
  t1 = sin(reshape(1:2:24, 4, 3));
  t2 = cos(reshape(1:2:40, 4, 5));
  t  = [t1(:) ; t2(:)];
  nn_params = t;
  input_layer_size = 2;
  hidden_layer_size = 4;
  num_labels = 4;
  lambda = 0;
  
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
m = size(X, 1);

X = [ones(m,1) X];
y_matrix = eye(num_labels)(y,:);

z2 = X*Theta1';
a2 = [ones(m,1) sigmoid(z2)];
a3 = sigmoid(a2*Theta2');