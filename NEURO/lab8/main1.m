clear;
clc;

% Считывание датасета из файла
fd = fopen('data.txt','r');
formatSpec = '%f %f';
sizeData = [1 Inf];
data = fscanf(fd, formatSpec, sizeData).';
fclose(fd);

% Сглаживание данных
x = smooth(data, 12);

% График исходные/сглаженные данные
figure;
hold on;
grid on;
plot(data, '-b');
plot(x, '-r');

% Формирование обучающего множества
D = 5;
ntrain = 500;
nval = 100;
ntest = 50;

trainInd = 1 : ntrain; % 1..500
valInd = ntrain + 1 : ntrain + nval; % 501..600
testInd = ntrain + nval + 1 : ntrain + nval + ntest; % 601..650

x_train = x(trainInd);
x_val = x(valInd);
x_test = x(testInd);

% Создание и конфигурация сети
hiddenSizes = 10;
net = timedelaynet(1:D, hiddenSizes, 'trainlm'); %1:10 delay, hid. l. size.
net.divideFcn = '';

x_train_seq = con2seq(x_train');
x_val_seq = con2seq(x_val');
x_test_seq = con2seq(x_test');

net = configure(net, x_train_seq, x_train_seq);
net = init(net);

net.trainParam.epochs = 2000;
net.trainParam.max_fail = 2000;
net.trainParam.goal = 1.0e-5;

% Обучение сети
[Xs, Xi, Ai, Ts] = preparets(net, x_train_seq, x_train_seq); 
net = train(net, Xs, Ts, Xi, Ai);

% Рассчет выхода сети для обучающего множества
Y = sim(net, Xs, Xi);

% График выхода сети на обучающем множестве 
figure;
hold on;
grid on;
plot(x_train, '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');

% График ошибки на обучающем множестве 
figure;
hold on;
grid on;
plot([cell2mat(Xi) cell2mat(Y)] - x_train', '-r');

% Рассчет выхода сети для контрольного множества
[Xs, Xi, Ai, Ts] = preparets(net, x_val_seq, x_val_seq);
Y = sim(net, Xs, Xi);

% График выхода сети на контрольном множестве 
figure;
hold on;
grid on;
plot(x_val, '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');
