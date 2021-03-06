clear;
clc;

% Создание обучающего множества
t0 = 0;
tn = 10;
dt = 0.01;

n = (tn - t0) / dt + 1;

f = @(k)sin(k.^2 - 2 * k + 3);
f1 = @(y, u)y ./ (1 + y.^2) + u.^3;

u = zeros(1, n);

u(1) = f(0);
x = zeros(1, n);
for i = 2 : n
    t = t0 + (i - 1) * dt;
    x(i) = f1(x(i - 1), u(i - 1));
    u(i) = f(t);
end

% График управляющего и целевого сигнала
figure
subplot(2,1,1)
plot(t0:dt:tn, u, '-b'),grid
ylabel('control')
subplot(2,1,2)
plot(t0:dt:tn, x, '-r'), grid
ylabel('state')
xlabel('t')

D = 3;
ntrain = 700;
nval = 200;
ntest = 97;

trainInd = 1 : ntrain;
valInd = ntrain + 1 : ntrain + nval;
testInd = ntrain + nval + 1 : ntrain + nval + ntest;

% Создание и конфигурация NARX-сети
net = narxnet(1 : D, 1, 8);
net.trainFcn = 'trainlm';

net.divideFcn = '';
net.trainParam.epochs = 20000;
net.trainParam.max_fail = 2000;
net.trainParam.goal = 1.0e-5;

u_train = u(trainInd);
u_val = u(valInd);
u_test = u(testInd);

x_train = x(trainInd);
x_val = x(valInd);
x_test = x(testInd);

[Xs, Xi, Ai, Ts] = preparets(net, con2seq(u_train), {}, con2seq(x_train));

% Обучение сети
net = train(net, Xs, Ts, Xi, Ai);

% Рассчет выхода сети для обучающего множества
Y = sim(net, Xs, Xi);

t = t0:dt:tn;
train_range = t(1 : ntrain);

% График управляющего сигнала, истинного значения функции, выхода сети и
% ошибки на обучающем отрезке
figure
subplot(3,1,1)
plot(train_range, u(1:ntrain), '-b'),grid
ylabel('control')
subplot(3,1,2)
plot(train_range, x(1:ntrain), '-b', train_range, [x(1:D) cell2mat(Y)], '-r'), grid
ylabel('state')
subplot(3,1,3)
plot(train_range(D+1:end), x(D+1:ntrain) - cell2mat(Y)), grid
ylabel('error')
xlabel('t')

[Xs, Xi, Ai, Ts] = preparets(net, con2seq([u_val(end-D+1:end) u_test]), {}, con2seq([x_val(end-D+1:end) x_test]));

% Рассчет выхода сети для тестового множества
Y = sim(net, Xs, Xi);

t = t0:dt:tn;
test_range = t(end-ntest-D+1:end);

% График управляющего сигнала, истинного значения функции, выхода сети и
% ошибки на тестовом отрезке
figure;
subplot(3,1,1)
plot(t(end-ntest+1:end), u(end-ntest+1:end), '-b'),grid
ylabel('control')
subplot(3,1,2)
plot(t(end-ntest+1:end), x_test, '-b', t(end-ntest+1:end), cell2mat(Y), '-r'), grid
ylabel('state')
subplot(3,1,3)
plot(t(end-ntest+1:end), x_test - cell2mat(Y)), grid
ylabel('error')
xlabel('t')