function y = Activation(x,type)
switch lower(type)
    case {'relu'}
        y = max(x,0);
    case {'tanh'}
        y = tanh(x);
    case {'sigmoid'}
        y = 1 ./ (1 + exp(- x));
    case {'linear'}
        y = x;
end
end
