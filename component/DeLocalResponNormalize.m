function LRN = DeLocalResponNormalize(LocalResponseNorm,InputData)
LRN = zeros(size(InputData));
k = LocalResponseNorm.k;
alpha = LocalResponseNorm.alpha;
n = LocalResponseNorm.n;
beta = LocalResponseNorm.beta;
for h = 1 : size(LRN,3)
    sum_part = sum(InputData(:,:,max(1,h - floor(n / 2)) : min(size(LRN,3),h + floor(n / 2)),:) .^ 2,3);
    LRN(:,:,h,:) = (k + alpha * sum_part - 2 * beta * alpha * (InputData(:,:,h,:) .^ 2)) ./ ((k + alpha * sum_part) .^ (beta + 1));
end