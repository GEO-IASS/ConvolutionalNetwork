function outputData = LocalResponNormalize(LocalResponseNorm,InputData)
outputData = zeros(size(InputData));
k = LocalResponseNorm.k;
alpha = LocalResponseNorm.alpha;
n = LocalResponseNorm.n;
beta = LocalResponseNorm.beta;
for h = 1 : size(InputData,3)
    outputData(:,:,h,:) = InputData(:,:,h,:) ./ ...
        ((k + alpha * sum(InputData(:,:,max(1,h - floor(n / 2)) : min(size(InputData,3),h + floor(n / 2)),:) .^ 2,3)) .^ beta);
end
end