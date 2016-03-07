function WeightGrad = CalculateWeightGradient(filter,sensitivity,InputData)
WeightGrad = zeros(size(filter.kernel));            
inputData = zeros((filter.MapSize(1,1) - 1) * filter.stride(1,1) + filter.size(1,1),(filter.MapSize(1,2) - 1) ...
                * filter.stride(1,2) + filter.size(1,2),size(InputData,3),size(InputData,4));
inputData(1 : size(InputData,1),1 : size(InputData,2),:,:) = InputData;
for l = 1 : size(sensitivity,4)
    for m = 1 : size(sensitivity,3)
        amplify = zeros(filter.stride);
        amplify(1,1) = 1;
        new_sensitivity = kron(sensitivity(:,:,m,l),amplify);
        new_sensitivity = new_sensitivity(1 : size(inputData,1) - size(filter.kernel,1) + 1, ...
            1 : size(inputData,2) - size(filter.kernel,2) + 1);
        new_sensitivity = rot90(rot90(new_sensitivity));
        for n = 1 : size(inputData,3)
            WeightGrad(:,:,n,m) = WeightGrad(:,:,n,m) + ...
                rot90(rot90(conv2(inputData(:,:,n,l),new_sensitivity,'valid')));
        end
    end
end
WeightGrad = WeightGrad ./ size(sensitivity,4);
end