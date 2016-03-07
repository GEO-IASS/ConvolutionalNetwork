function y = Difference(x,type)
switch lower(type)
    case {'relu'}
        y = x > 0;
    case {'tanh'}
        y = 1 - x .^ 2;
    case {'sigmoid'}
        y = x .* (1 - x);
    case {'linear'}
        y = ones(size(x));
end
end