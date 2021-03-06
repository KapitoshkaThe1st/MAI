clear;
clc;

% Формирование множества точек
N = 20;
T = -1.5 + 3*rand(2, N);

figure;
hold on;
grid on;
plot(T(1,:), T(2,:), '-V', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);

% Создание и конфигурация сети
net = newsom(T, N);
net = configure(net, T);
net.trainParam.epochs = 3000;
net = train(net, T);

% Отображение координат городов и центров кластеров
figure;
hold on;
grid on;
plotsom(net.IW{1,1}, net.layers{1}.distances);
plot(T(1,:), T(2,:), 'V', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
hold off;

