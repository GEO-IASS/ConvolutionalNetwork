function outputData = DePool(pool,sensitivity,mark)
stride = pool.stride;
pool_size = pool.size;
MapSize = pool.MapSize;
outputData = zeros(stride(1,1) * (MapSize(1,1) - 1) + pool_size(1,1),stride(1,2) * (MapSize(1,2) - 1) + pool_size(1,2),...
    size(sensitivity,3),size(sensitivity,4));
switch lower(pool.type)
    case {'max'}
        for j = 1 : size(mark,6)
            for k = 1 : size(mark,5)
                for m = 1 : size(mark,4)
                    for n = 1 : size(mark,3)
                        outputData((n - 1) * stride(1,1) + 1 : (n - 1) * stride(1,1) + pool_size(1,1),(m - 1) * stride(1,2) + 1 : (m - 1) * stride(1,2) + pool_size(1,2),k,j) = ...
                            outputData((n - 1) * stride(1,1) + 1 : (n - 1) * stride(1,1) + pool_size(1,1),(m - 1) * stride(1,2) + 1 : (m - 1) * stride(1,2) + pool_size(1,2),k,j) + ...
                            mark(:,:,n,m,k,j) * sensitivity(n,m,k,j);
                    end
                end
            end
        end
    case {'average'}
        for i = 1 : size(sensitivity,4)
            for j = 1 : size(sensitivity,3)
                for k = 1 : size(sensitivity,2)
                    for l = 1 : size(sensitivity,1)
                        outputData((l - 1) * stride(1,1) + 1: (l - 1) * stride(1,2) + pool_size(1,1),(k - 1) * stride(1,2) + 1 : (k - 1) * stride(1,2) ...
                            + pool_size(1,2),j,i) = outputData((l - 1) * stride(1,1) + 1: (l - 1) * stride(1,2) + pool_size(1,1),(k - 1) * stride(1,2) + 1 : (k - 1) * stride(1,2) ...
                            + pool_size(1,2),j,i) + sensitivity(l,k,j,i) ./ prod(pool_size);
                    end
                end
            end
        end
end
end