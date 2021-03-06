clear;
clc;

% Создание обучающего множества
k1 = 0 : 0.025 : 1;
p_1 = sin(4 * pi * k1);
t_1 = -ones(size(p_1));

k2 = 2.38 : 0.025 : 4.1;
g = @(k)cos(cos(k) .* k .* k + 5*k);

p_2 = g(k2);
t_2 = ones(size(p_2));
R = {1; 3; 5};
P = [repmat(p_1, 1, R{1}), p_2, repmat(p_1, 1, R{2}), p_2, repmat(p_1, 1, R{3}), p_2];
T = [repmat(t_1, 1, R{1}), t_2, repmat(t_1, 1, R{2}), t_2, repmat(t_1, 1, R{3}), t_2];
Pseq = con2seq(P);
Tseq = con2seq(T);

% Создание и конфигурация сети
D = 4;

net = distdelaynet({0 : D, 0 : D}, 8, 'trainoss');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.divideFcn = '';

net = configure(net, Pseq, Tseq);

[Xs, Xi, Ai, Ts] = preparets(net, Pseq, Tseq); 

net.trainParam.epochs = 100;
net.trainParam.goal =  1.0e-5;

% Обучение сети
net = train(net, Xs, Ts, Xi, Ai);

% Рассчет выхода сети для обучающего множества
Y = sim(net, Xs, Xi, Ai);

% График обучающего множества и выход сети до порогового элемента
figure;
hold on;
grid on;
plot(cell2mat(Ts), '-b');
plot(cell2mat(Y), '-r');

Yc = sign(cell2mat(Y));

fprintf('Correctly recognized (train set): %d\\%d\n', nnz(Yc == T(D+1 : end)), length(T)-D-1);

% Формирование тестового множества
R = {1; 4; 5}; 
P = [repmat(p_1, 1, R{1}), p_2, repmat(p_1, 1, R{2}), p_2, repmat(p_1, 1, R{3}), p_2];
T = [repmat(t_1, 1, R{1}), t_2, repmat(t_1, 1, R{2}), t_2, repmat(t_1, 1, R{3}), t_2];

Pseq = con2seq(P);
Tseq = con2seq(T);

[Xs, Xi, Ai, Ts] = preparets(net, Pseq, Tseq);

% Рассчет выхода для тестового множетсва
Y = sim(net, Xs, Xi, Ai);

Yc = sign(cell2mat(Y));

fprintf('Correctly recognized (test set): %d\\%d\n', nnz(Yc == T(D+1 : end)), length(T)-D-1);

% График тестового множества и выход сети до порогового элемента
figure;
hold on;
grid on;
pLine = plot(cell2mat(Ts), 'b');
rLine = plot(cell2mat(Y), 'r');

legend([rLine, pLine], 'Predicted', 'Target');
