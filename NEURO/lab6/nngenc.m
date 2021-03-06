function v = nngenc(x,c,n,d)



if nargin < 3, error(message('nnet:Args:NotEnough')), end
if nargin == 3, d = 1; end

[r,q] = size(x);
minv = min(x')';
maxv = max(x')';
v = rand(r,c) .* ((maxv-minv) * ones(1,c)) + (minv * ones(1,c));
t = c*n;
v = repmat(v,1,n) + randn(r,t)*d;