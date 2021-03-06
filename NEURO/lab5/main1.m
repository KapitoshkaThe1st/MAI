% Основной сигнал
k_1 = 0:0.025:1;
p_1 = sin(4*pi*k_1);

% Сигнал для распознавания
k_2 = 2.38:0.025:4.1;
p_2 = cos(cos(k_2).*k_2.*k_2 + 5*k_2);

% Целевой выход
t_2 = ones(size(p_2));

% Целевой выход основного сигнала
t_1 = -ones(size(p_1));

% Длительности основного сигнала
R = {1; 3; 5}; 

% Формирование входного множества
P = [repmat(p_1, 1, R{1}), p_2, repmat(p_1, 1, R{2}), p_2, repmat(p_1, 1, R{3}), p_2];
T = [repmat(t_1, 1, R{1}), t_2, repmat(t_1, 1, R{2}), t_2, repmat(t_1, 1, R{3}), t_2];
Pseq = con2seq(P);
Tseq = con2seq(T);

% Создание и конфигурация сети
net = layrecnet(1 : 2, 100, 'trainoss');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net = configure(net, Pseq, Tseq);

% Подготовка массивов ячеек
[p, Xi, Ai, t] = preparets(net, Pseq, Tseq);

% Задаем параметры обучения:
% число эпох обучения, предельное значение критерия обучения
net.trainParam.epochs = 100;
net.trainParam.goal = 1.0e-5;

% Обучение сети
net = train(net, p, t, Xi, Ai);
Y = sim(net, p, Xi, Ai);

% Отображение структуры сети
view(net);

figure;
hold on;

rLine = plot(cell2mat(Y), 'r');
pLine = plot(cell2mat(t), 'b');
legend([rLine,pLine],'Target', 'Predicted');

% Преобразование значений
tc = sign(cell2mat(Y));

fprintf('Correctly recognized (train set): %d\\%d\n',nnz(tc == T(3 : end)), length(T)-3);

% Формирование тестового множества
R = {1; 4; 5}; 
P = [repmat(p_1, 1, R{1}), p_2, repmat(p_1, 1, R{2}), p_2, repmat(p_1, 1, R{3}), p_2];
T = [repmat(t_1, 1, R{1}), t_2, repmat(t_1, 1, R{2}), t_2, repmat(t_1, 1, R{3}), t_2];

Pseq = con2seq(P);
Tseq = con2seq(T);

[p, Xi, Ai, t] = preparets(net, Pseq, Tseq);

% Рассчет выходя для тестового множетсва
Y = sim(net, p, Xi, Ai);

figure;
hold on;

rLine = plot(cell2mat(Y), 'r');
pLine = plot(cell2mat(t), 'b');
legend([rLine,pLine],'Target', 'Predicted');

% Преобразование значений
tc = sign(cell2mat(Y));

fprintf('Correctly recognized (test set): %d\\%d\n',nnz(tc == T(3 : end)), length(T)-3);