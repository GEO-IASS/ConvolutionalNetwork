function cost = CalculateCost(GroundTruth,OutputProbability,type)
switch lower(type)     
    case {'mse'}
        cost =  0.5 * sum((GroundTruth(:) - OutputProbability(:)) .^ 2);%
    case {'entropy'}
        cost = - mean(sum(GroundTruth .* (log(OutputProbability)),1),2);% + 0.0001
end
end
