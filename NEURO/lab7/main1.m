clear;
clc;

% Создание обучающего множества
t = 0 : 0.025 : 2 * pi;

a = 0.7;
b = 0.7;
alpha = -pi / 6;
x0 = 0;
y0 = -0.1;

x = ellipse(t, a, b, alpha, y0, x0);
x_seq = con2seq(x);

% Создание и конфигурация сети
net = feedforwardnet(1, 'trainlm');
net.layers{1}.transferFcn = 'purelin';

net = configure(net, x_seq, x_seq);
net = init(net);

net.trainParam.epochs = 100;
net.trainParam.goal = 1.0e-5;

% Обучение сети
net = train(net, x_seq, x_seq);

% Рассчет выхода сети
y_pred_seq = sim(net, x_seq);
result = cell2mat(y_pred_seq);

% Отображение обучающего множества и выхода сети
plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);
hold on;
grid on;
plot(result(1, :), result(2, :), '-b', 'LineWidth', 2);
