clc
clear

% 1:
P = [-2.8 -0.2 2.8 -2.1 0.3 -1;
    1.4 -3.5 -4 -2.7 -4.1 -4];
T = [0 1 1 0 1 0];

net = newp([-5 5; -5 5],[0,1]);

net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';

net = init(net);

IW = cell2mat(net.IW(1));
display(IW);

b = cell2mat(net.b(1));
display(b);

% Y = net(P);
% display(Y)
% net.trainParam.epochs = 20;
% net = train(net, P, T);
% Y = net(P);
% display(Y)

epochs = 20;
net = rosenblatt_rule(net, P, T, epochs);

plotpv(P,T), grid
plotpc(net.IW{1},net.b{1})

Y = net(P);
display(Y)

function net = rosenblatt_rule(net, P, T, epochs)
    IW_ = cell2mat(net.IW(1));
    b_ = cell2mat(net.b(1));
    for i = 1:epochs
        was_error = 0;
        for j = 1:(size(P, 2))
            p = P(:,j);
            t = T(:,j);
            a = hardlim(IW_ * p + b_);
            e = t - a;
            if mae(e)
                was_error = 1;
                IW_ = IW_ + e * p.';
                b_ = b_ + e;
            end
        end
        if was_error == 0
           break;
        end
    end
    net.IW(1) = mat2cell(IW_, size(IW_, 1));
    net.b(1) = mat2cell(b_, size(b_, 1));
end
