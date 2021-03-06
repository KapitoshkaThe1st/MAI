clear;
clc;

% Формирование множества точек
X = [0 1.5;
     0 1.5];
clusters = 8;
points = 10;
deviation = 0.1;
P = nngenc(X, clusters, points, deviation);

% Создание и конфигурация сети
net = competlayer(8);
net = configure(net, P);
net.divideFcn = '';

% Обучение сети
net.trainParam.epochs = 50;
net = train(net, P);

disp("First layer weights:");
disp(net.IW{1});

% Проверка качества
sample = 1.5*rand(2, 5);
result = vec2ind(sim(net, sample));

% Результат кластеризации тестового множества
disp("Result:");
disp(result);

figure;
hold on;
grid on;
scatter(P(1, :), P(2, :), 5, [0 1 0], 'filled');
scatter(net.IW{1}(:, 1), net.IW{1}(:, 2), 5, [0 0 1], 'filled');
scatter(sample(1, :), sample(2, :), 5, [1 0 0], 'filled');
