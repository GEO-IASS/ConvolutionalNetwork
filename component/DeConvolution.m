function outputData = DeConvolution(filter,sensitivity)
filter_size = filter.size;
stride = filter.stride;
kernel_size = size(filter.kernel);
outputData = zeros(stride(1,1) * (size(sensitivity,1) - 1) + filter_size(1,1),stride(1,2) * (size(sensitivity,2) - 1) + ...
    filter_size(1,2),kernel_size(1,3),size(sensitivity,4));
for j = 1 : size(outputData,4)
    for n = 1 : size(filter.kernel,3)
        for m = 1 : size(filter.kernel,4)
            amplify = zeros(filter.stride);
            amplify(1,1) = 1;
            new_sensitivity = kron(sensitivity(:,:,m,j),amplify);
            new_sensitivity = new_sensitivity(1 : (size(sensitivity,1) - 1) * filter.stride(1,1) + 1, ...
                1 : (size(sensitivity,2) - 1) * filter.stride(1,2) + 1);
            outputData(:,:,n,j) = outputData(:,:,n,j) + conv2(new_sensitivity,rot90(rot90(filter.kernel(:,:,n,m))),'full');
        end
    end
end     
end