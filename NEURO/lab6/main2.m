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
net = newsom(X, [2 4]);
net = configure(net, X);
net.trainParam.epochs = 150;

% Обучение сети
net.divideFcn = '';
net = train(net, P);

figure;
plotsomhits(net,P)
figure;
plotsompos(net,P)

disp("First layer weights:");
disp(net.IW{1});

% Проверка качества разбиения
sample = 1.5 * rand(2, 5);
result = vec2ind(sim(net, sample));

disp("Result:");
disp(result);

figure;
hold on;
grid on;
plotsom(net.IW{1,1},net.layers{1}.distances);
scatter(P(1, :), P(2, :), 5, [0 1 0], 'filled');
scatter(net.IW{1}(:, 1), net.IW{1}(:, 2), 5, [0 0 1], 'filled');
scatter(sample(1, :), sample(2, :), 5, [1 0 0], 'filled');