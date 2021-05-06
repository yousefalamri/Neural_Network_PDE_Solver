function U = get_analytic(xgrid,t)
nu = 0.01/pi;
f = @(y) exp(-cos(pi*y)/(2*pi*nu));
g = @(y) exp(-(y.^2)/(4*nu*t));

U = zeros(size(xgrid));

for i = 1:length(xgrid)
    x = xgrid(i);
    % leave BCs at x=-1 and x=1
    if abs(x) ~= 1
        term = @(n) sin(pi*(x-n)) .* f(x-n) .* g(n);
        top = -integral(term,-inf,inf);
        term = @(n) f(x-n) .* g(n);
        U(i) = top / integral(term,-inf,inf);
    end
end

end