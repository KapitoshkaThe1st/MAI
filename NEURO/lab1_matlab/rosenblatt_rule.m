function net = rosenblatt_rule(net, P, T, epochs)
    IW_ = net.IW{1,1};
    b_ = net.b{1,1};
    fprintf(repmat('IW[%d]\t\t', 1, size(IW_,2)), 1:size(IW_,2));
    fprintf(repmat('b[%d]\t\t', 1, size(b_, 2)), 1:size(b_,2));
    fprintf("MAE\n");
    for i = 1:epochs
        was_error = 0;
        for j = 1:(size(P, 2))
            IW_ = net.IW{1,1};
            b_ = net.b{1,1};
            p = P(:,j);
            e = T(:,j) - net(p);
            if mae(e)
                was_error = 1;
                IW_ = IW_ + e * p.';
                b_ = b_ + e;
                net.IW{1,1} = IW_;
                net.b{1,1} = b_;
            end
        end
        if was_error == 0
           break;
        end
        
        fprintf(repmat('%f\t', size(IW_)), IW_);
        fprintf(repmat('%f\t', size(b_)), b_);
        fprintf("%f\n", mae(T - net(P)));
    end
end