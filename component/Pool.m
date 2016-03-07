function [OutputData,mark] = Pool(pool,InputData)
% usage : [OutputData,mark] = Pool(pool,InputData)
%   you must specialize the pool.type(max/abs_max),pool.size, pool.stride,etc.
% 
%%
% InputData_2 = zeros((pool.MapSize(1,1) - 1) * pool.stride(1,1) + pool.size(1,1),(pool.MapSize(1,2) - 1) * pool.stride(1,2) + pool.size(1,2),pool.MapSize(1,3), ...
%     size(InputData,4));
% InputData_2(1 : size(InputData,1),1 : size(InputData,2),:,:) = InputData;
% InputData = InputData_2;
SampleNum = size(InputData,4);
mark = zeros([pool.size,pool.MapSize(1,1),pool.MapSize(1,2), ...
    pool.MapSize(1,3),SampleNum]);
OutputData = zeros([pool.MapSize,SampleNum]);
abs_OutputData = OutputData;
switch lower(pool.type)
    case {'max'}
        for m = 1 : pool.MapSize(1,2)
            for n = 1 : pool.MapSize(1,1)
                PoolData = InputData((n - 1) * pool.stride(1,1) + 1 : (n - 1) * pool.stride(1,1) + ...
                    pool.size(1,1),(m - 1) * pool.stride(1,2) + 1 : (m - 1) * pool.stride(1,2) + ...
                    pool.size(1,2),:,:);
                [max_x,loc_x] = max(PoolData,[],1);
                [OutputData(n,m,:,:),loc_y] = max(max_x,[],2);
                for j = 1 : SampleNum
                    for k = 1 : pool.MapSize(1,3)
                        mark(loc_x(1,loc_y(1,1,k,j),k,j),loc_y(1,1,k,j),n,m,k,j) = 1;   
                    end
                end
            end
        end
    case {'average'}
        for m = 1 : pool.MapSize(1,2)
            for n = 1 : pool.MapSize(1,1)
                PoolData = InputData((n - 1) * pool.stride(1,1) + 1 : (n - 1) * pool.stride(1,1) + ...
                    pool.size(1,1),(m - 1) * pool.stride(1,2) + 1 : (m - 1) * pool.stride(1,2) + ...
                    pool.size(1,2),:,:);
                OutputData(n,m,:,:) = sum(sum(PoolData,1),2) ./ prod(pool.size);
            end
        end
    case {'abs_max'}
        for m = 1 : pool.MapSize(1,2)
            for n = 1 : pool.MapSize(1,1)
                PoolData = InputData((n - 1) * pool.stride(1,1) + 1 : (n - 1) * pool.stride(1,1) + ...
                    pool.size(1,1),(m - 1) * pool.stride(1,2) + 1 : (m - 1) * pool.stride(1,2) + ...
                    pool.size(1,2),:,:);
                max_x = max(PoolData,[],1);
                OutputData(n,m,:,:) = max(max_x,[],2);
                
                abs_PoolData = abs(PoolData);
                [abs_max_x,abs_loc_x] = max(abs_PoolData,[],1);
                [abs_OutputData(n,m,:,:),abs_loc_y] = max(abs_max_x,[],2);
                
                for j = 1 : SampleNum
                    for k = 1 : pool.MapSize(1,3)
                        mark(abs_loc_x(1,abs_loc_y(1,1,k,j),k,j),abs_loc_y(1,1,k,j),n,m,k,j) = 1;   
                    end
                end
            end
        end
        OutputData = (2 * (OutputData == abs_OutputData) - 1) .* abs_OutputData;
end
end