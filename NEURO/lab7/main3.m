clear;
clc;

% Создание обучающего множества
phi = 0 : 0.025 : 2 * pi;
r = 2 * phi;

x = [r .* cos(phi); r .* sin(phi); phi];
x_seq = con2seq(x);

% Создание и конфигурация сети
net = feedforwardnet([10 2 10], 'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';

net = configure(net, x_seq, x_seq);
net = init(net);

net.trainParam.epochs = 1000;
net.trainParam.goal = 1.0e-5;

% Обучение сети
net = train(net, x_seq, x_seq);
view(net);

% Рассчет выхода сети
y_pred_seq = sim(net, x_seq);
result = cell2mat(y_pred_seq);

% Отображение обучающего множества и выхода сети
plot3(x(1, :), x(2, :), x(3, :), '-r', 'LineWidth', 2);
hold on;
grid on;
plot3(result(1, :), result(2, :), result(3, :), '-b', 'LineWidth', 2);
