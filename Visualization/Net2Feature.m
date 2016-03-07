function [feature,FeatureData,net] = Net2Feature(net,feature)
if isfield(feature,'layer')
    layer = feature.layer;
else
    layer = 2;
    feature.layer = layer;
end
if isfield(feature,'num')
    num = feature.num;
else
    num = 1;
    feature.num = num;
end

% assignment output arguement FeatureData
FeatureData = [];

switch lower(net.layers{layer}.type)
    case {'conv'}
        feature.layers{layer}.filter.MapSize = [1,1,1];
        feature.layers{layer}.filter.stride = net.layers{layer}.filter.stride;
        feature.layers{layer}.filter.size = net.layers{layer}.filter.size;
        feature.layers{layer}.filter.kernel = net.layers{layer}.filter.kernel(:,:,:,num);
        feature.layers{layer}.filter.bias = net.layers{layer}.filter.bias(1,num);
        feature.layers{layer}.activation = 'linear';
        feature.layers{layer}.type = net.layers{layer}.type;
    case {'full'}
        feature.layers{layer}.weight = net.layers{layer}.weight(num,:);
        feature.layers{layer}.bias = net.layers{layer}.bias(1,num);
        feature.layers{layer}.activation = 'linear';
        feature.layers{layer}.type = net.layers{layer}.type;
end

for i = layer - 1 : -1 : 1
    switch lower(net.layers{i}.type)
        case {'full'}
            feature.layers{i}.type = net.layers{i}.type;
            feature.layers{i}.weight = net.layers{i}.weight;
            feature.layers{i}.bias = net.layers{i}.bias;      
            feature.layers{i}.OutputMapSize = net.layers{i}.OutputMapSize;
            feature.layers{i}.activation = net.layers{i}.activation;
        case {'conv'}
            switch lower(net.layers{i + 1}.type)
                case {'conv'}
                    MapSize = feature.layers{i + 1}.filter.MapSize;
                    FilterSize = feature.layers{i + 1}.filter.size;
                    stride = feature.layers{i + 1}.filter.stride;
                    feature.layers{i}.OutputMapSize =[(MapSize(1,1) - 1) * stride(1,1) + FilterSize(1,1),(MapSize(1,2) - 1) * stride(1,2) + FilterSize(1,2), ...
                        net.layers{i}.OutputMapSize(1,3)];
                    feature.layers{i}.filter.MapSize = feature.layers{i}.OutputMapSize;
                case {'full','output'}
                    feature.layers{i}.OutputMapSize = net.layers{i}.OutputMapSize;
                    feature.layers{i}.filter.MapSize = net.layers{i}.filter.MapSize;
            end
            
            if isfield(net.layers{i},'LocalResponseNorm')
                    feature.layers{i}.LocalResponseNorm = net.layers{i}.LocalResponseNorm;
            end
            
            if isfield(net.layers{i},'pool')
                MapSize = feature.layers{i}.OutputMapSize;
                stride = net.layers{i}.pool.stride;
                PoolSize = net.layers{i}.pool.size;
                
                feature.layers{i}.pool.MapSize = MapSize;
                feature.layers{i}.pool.type = net.layers{i}.pool.type;
                feature.layers{i}.pool.stride = stride;
                feature.layers{i}.pool.size = PoolSize;
                feature.layers{i}.filter.MapSize = [(MapSize(1,1) - 1) * stride(1,1) + PoolSize(1,1),(MapSize(1,2) - 1) * stride(1,2) + PoolSize(1,2),MapSize(1,3)];
                
                switch lower(net.layers{i}.pool.type)
                    case {'max'}
                        if ~isfield(net.layers{i}.pool,'VisualMark')
                            net.layers{i}.pool.VisualMark = 'free';
                            feature.layers{i}.pool.VisualMark = 'free';
                        else
                            feature.layers{i}.pool.VisualMark = net.layers{i}.pool.VisualMark;
                            switch lower(feature.layers{i}.pool.VisualMark)
                                case {'fixed'}
                                    if ~isfield(net.layers{i}.pool,'mark')
                                        net.layers{i}.pool.mark = zeros([1,2,MapSize]);
                                    end
                                    FeatureData.layers{i}.pool.mark = net.layers{i}.pool.mark;
                                case {'random'}
%                                     if ~isfield(net.layers{i}.pool,'mark')
                                    net.layers{i}.pool.mark = randi(net.layers{i}.pool.size(1,1),[1,2,MapSize]) - 1;% pool implement
%                                     end
                                    FeatureData.layers{i}.pool.mark = net.layers{i}.pool.mark;
                            end
                        end
                    case {'average','ave'}
%                         if ~isfield(net.layers{i}.pool,'VisualMark')
%                             net.layers{i}.pool.VisualMark = 'free';
%                             feature.layers{i}.pool.VisualMark = 'free';
%                         else
%                             feature.layers{i}.pool.VisualMark = net.layers{i}.pool.VisualMark;
%                             switch lower(feature.layers{i}.pool.VisualMark)
%                                 case {'fixed'}
%                                     if ~isfield(net.layers{i}.pool,'mark')
%                                         net.layers{i}.pool.mark = zeros([1,2,MapSize]);
%                                     end
%                                     FeatureData.layers{i}.pool.mark = net.layers{i}.pool.mark;
%                                 case {'random'}
% %                                     if ~isfield(net.layers{i}.pool,'mark')
%                                     net.layers{i}.pool.mark = randi(net.layers{i}.pool.size(1,1),[1,2,MapSize]) - 1;% pool implement
% %                                     end
%                                     FeatureData.layers{i}.pool.mark = net.layers{i}.pool.mark;
%                             end
%                         end
                end
            end
            
            feature.layers{i}.filter.stride = net.layers{i}.filter.stride;
            feature.layers{i}.filter.size = net.layers{i}.filter.size;
            feature.layers{i}.filter.kernel = net.layers{i}.filter.kernel;
            feature.layers{i}.filter.bias = net.layers{i}.filter.bias;
            feature.layers{i}.type = net.layers{i}.type;
            feature.layers{i}.activation = net.layers{i}.activation;
        case {'input'}
            switch lower(net.layers{i + 1}.type)
                case {'conv'}
                    MapSize = feature.layers{i + 1}.filter.MapSize;
                    FilterSize = feature.layers{i + 1}.filter.size;
                    stride = feature.layers{i + 1}.filter.stride;
                    if net.layers{i}.OutputMapSize(1,1) <= (MapSize(1,1) - 1) * stride(1,1) + FilterSize(1,1)
                        feature.layers{i}.OutputMapSize = net.layers{i}.OutputMapSize;
                    else
                        feature.layers{i}.OutputMapSize = [(MapSize(1,1) - 1) * stride(1,1) + FilterSize(1,1),(MapSize(1,2) - 1) * stride(1,2) + FilterSize(1,2), ...
                            net.layers{i}.OutputMapSize(1,3)];    
                    end
                case {'full'}
                   feature.layers{i}.OutputMapSize =  net.layers{i}.OutputMapSize;
            end
            feature.layers{i}.type = net.layers{i}.type;       
    end
end
end