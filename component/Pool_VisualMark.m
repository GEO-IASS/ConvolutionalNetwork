function OutputData = Pool_VisualMark(pool,InputData,mark)
MapSize = pool.MapSize;
% Pool_size = pool.size;
stride = pool.stride;
OutputData = zeros([MapSize,size(InputData,4)]);
for i = 1 : size(InputData,4)
    for j = 1 : MapSize(1,3)
        for k = 1 : MapSize(1,2)
            for l = 1 : MapSize(1,1)
                OutputData(l,k,j,i) = InputData((l - 1) * stride(1,1) + mark(1,1,l,k,j,i) + 1,(k - 1) * stride(1,2) + ...
                    mark(1,2,l,k,j,i) + 1,j,i);
            end
        end
    end
end
end