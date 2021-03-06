function [ res ] = ellipse( t, a, b, alpha, x0, y0)

x = a * cos(t); 
y = b * sin(t);
res = [x * cos(alpha) - y * sin(alpha) + x0;
     x * sin(alpha) + y * cos(alpha) + y0];
end

