function OutputData = Convolution(filter,InputData)
% inputData = double(zeros((filter.MapSize(1,1) - 1) * filter.stride(1,1) + filter.size(1,1),(filter.MapSize(1,2) - 1) ...
%     * filter.stride(1,2) + filter.size(1,2),size(InputData,3),size(InputData,4)));
% inputData(1 : size(InputData,1),1 : size(InputData,2),:,:) = InputData;
% InputData = inputData;
OutputData = double(zeros((filter.MapSize(1,1) - 1) * filter.stride(1,1) + 1, ...
    (filter.MapSize(1,2) - 1) * filter.stride(1,2) + 1,size(filter.kernel,4),size(InputData,4)));
for j = 1 : size(filter.kernel,4)
    for m = 1 : size(filter.kernel,3)
        for n = 1 : size(InputData,4)
            OutputData(:,:,j,n) = OutputData(:,:,j,n) + conv2(InputData(:,:,m,n),filter.kernel(:,:,m,j),'valid');
        end
    end
    OutputData(:,:,j,:) = OutputData(:,:,j,:) + filter.bias(1,j);
end
OutputData = OutputData(1 : filter.stride(1,1) : end,1 : filter.stride(1,2) : end,:,:);
end
