function y = CostDifference(x1,x2,type)
switch lower(type)
    case {'mse'}
        y = 2 * (x2 - x1);
    case {'entropy'}
        y = x1 .* (- 1 ./ x2);
end
end