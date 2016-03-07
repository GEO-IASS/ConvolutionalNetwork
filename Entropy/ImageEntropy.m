function y = ImageEntropy(FeatureImage)
y = zeros(1,size(FeatureImage,4));
for i = 1 : size(FeatureImage,4)
    b = FeatureImage(:,:,:,i);
    max_b = max(b(:));
    min_b = min(b(:));
    b = (b - min_b) ./ (max_b - min_b);
    y(1,i) = entropy(b);
end
end