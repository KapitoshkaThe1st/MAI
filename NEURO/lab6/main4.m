clear;
clc;

% Задание обучающего множества
P = [0.6 0.2 1.2 0.9 0.2 -0.3 -1.1 -0.3 0.2 0.5 0.4 -1.3;
     -0.5 -1.2 1.1 -0.8 -1.5 -0.6 -1 -1.3 -0.1 0.5 -1.4 -0.6];
T = [-1 1 -1 -1 1 -1 -1 -1 1 -1 1 -1];

figure;
hold on;
grid on;
plotpv(P, max(T, 0));

Ti = T;
Ti(Ti == 1) = 2;
Ti(Ti == -1) = 1;

p_class1 = sum(Ti(:) == 1);
n_samples = size(Ti ,2);

Ti = ind2vec(Ti);

% Создание и конфигурация сети
net = newlvq(minmax(P), 12, [p_class1 / n_samples, 1 - p_class1 / n_samples], 0.1);
net = configure(net, P, Ti);
net.trainParam.epochs = 300;

% Обучение сети
net = train(net, P, Ti);
disp("Weights:");
disp("IW_1:");
disp(net.IW{1,1});
disp("IW_2:");
disp(net.LW{2,1});

% Проверка качества обучения
[X,Y] = meshgrid(-1.5 : 0.1 : 1.5, -1.5 : 0.1 : 1.5);
result = sim(net, [X(:)'; Y(:)']);
result = vec2ind(result) - 1;

figure;
plotpv([X(:)'; Y(:)'], result);
point = findobj(gca,'type','line');
set(point,'Color','g');
hold on;
plotpv(P, max(0, T));