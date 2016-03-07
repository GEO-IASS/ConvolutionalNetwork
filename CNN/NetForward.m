function data = NetForward(net,data,option)
% usage: [data,CostHistory] = NetForward(net,data,CostHistory)
%   you must initialize the OutputData of first layer in data
%      and the GroundTruth belonged to the last layer in data
%         net          : essential
%         data         : essential
%         option       : optional,specialize the last layer you want to
%         forward to.
% author: Tang Jianbo,all rights reserved,2014-08-01
if nargin < 3
    option = [];
end

if ~isfield(option,'layer')
    option.layer = single(size(net.layers,2));
else
    option.layer = single(option.layer);
end

if ~isfield(option,'WeightDecay')
    option.WeightDecay = single(5e-4); 
else
    option.WeightDecay = single(option.WeightDecay); 
end

if ~isfield(option,'OutputDecay')
    option.OutputDecay = single(5e-4); 
else
    option.OutputDecay = single(option.OutputDecay); 
end

if ~isfield(option,'phase')
    option.phase = 'test';
end

if isfield(data,'layers')
    if ~isfield(data.layers{1},'OutputData')
        error('you must initialize the OutputData of the first layer in data');
    end
    if option.layer == size(net.layers,2) && ~isfield(data.layers{size(net.layers,2)},'GroundTruth')
        error('you must initialize the GroundTruth of the last layer in data')
    end
else
    error('you must have initialize the first layer');
end

data.layers{1}.WeightCost = single(0);
SampleNum = single(size(data.layers{1}.OutputData,4));
for i = 1 : option.layer
    switch lower(net.layers{i}.type)
        case {'conv'}
            data.layers{i}.filter.OutputData = Activation(Convolution_cpp(net.layers{i}.filter,data.layers{i - 1}.OutputData),net.layers{i}.activation);
            data.layers{i}.OutputData = data.layers{i}.filter.OutputData;
            data.layers{i}.WeightCost = data.layers{i - 1}.WeightCost + (option.WeightDecay / 2) * sum((net.layers{i}.filter.kernel(:)) .^ 2);% ...
               %  + (option.OutputDecay / 2) * mean((data.layers{i}.filter.OutputData(:)) .^ 2);
            
            if isfield(net.layers{i},'LocalResponseNorm')
                data.layers{i}.LocalResponseNorm.InputData = data.layers{i}.OutputData;
                data.layers{i}.OutputData = LocalResponNormalize_cpp(net.layers{i}.LocalResponseNorm,data.layers{i}.LocalResponseNorm.InputData);
            end
            
            if isfield(net.layers{i},{'pool'})
                switch lower(net.layers{i}.pool.type)
                    case {'max'}
                        if ~isfield(net.layers{i}.pool,'VisualMark')
                            net.layers{i}.pool.VisualMark = 'free';
                        end
                        switch(lower(net.layers{i}.pool.VisualMark))
                            case {'free'}
                                [data.layers{i}.OutputData,data.layers{i}.pool.mark] = Pool_cpp(net.layers{i}.pool,data.layers{i}.OutputData);
                            case {'fixed','random'}
                                data.layers{i}.OutputData = Pool_VisualMark_cpp(net.layers{i}.pool,data.layers{i}.OutputData,data.layers{i}.pool.mark);
                        end
%                         [data.layers{i}.OutputData,data.layers{i}.pool.mark] = Pool_cpp(net.layers{i}.pool,data.layers{i}.OutputData);
                    case {'average','ave'}
                        data.layers{i}.OutputData = Pool_cpp(net.layers{i}.pool,data.layers{i}.OutputData);
                end
            end
            if isfield(net.layers{i},'OutputAlignment')
                if strcmpi(net.layers{i}.OutputAlignment,'random')
                    OutputData = reshape(data.layers{i}.OutputData,[prod(net.layers{i}.OutputMapSize),SampleNum]);
                    data.layers{i}.OutputData = reshape(OutputData(net.layers{i}.ForwardAlignment,:),[net.layers{i}.OutputMapSize,SampleNum]);
                end
            end
                
        case {'full'}
             switch lower(net.layers{i - 1}.type)
                case {'conv','input'}
                    MapSize = single(size(data.layers{i - 1}.OutputData));
                    data.layers{i}.InputData = reshape(data.layers{i - 1}.OutputData,prod(MapSize(1 : 3)),SampleNum);
                case {'full'}
                    data.layers{i}.InputData = data.layers{i - 1}.OutputData;                   
             end
            data.layers{i}.OutputData = Activation(bsxfun(@plus,net.layers{i}.weight * data.layers{i}.InputData,net.layers{i}.bias'),net.layers{i}.activation);
            if isfield(net.layers{i},'DropoutRate') 
                switch lower(option.phase)
                    case {'train'}
                        data.layers{i}.DropoutMark = single(rand(size(data.layers{i}.OutputData)) >= net.layers{i}.DropoutRate);
                        data.layers{i}.OutputData = data.layers{i}.OutputData .* data.layers{i}.DropoutMark;
                    case {'test'}
                        data.layers{i}.OutputData = data.layers{i}.OutputData .* single((1 - net.layers{i}.DropoutRate));
                end
            end
            data.layers{i}.WeightCost = data.layers{i - 1}.WeightCost + (option.WeightDecay / 2) * sum((net.layers{i}.weight(:)) .^ 2) ;% ...
              % + (option.OutputDecay / 2) * mean((data.layers{i}.OutputData(:)) .^ 2);
        case {'output'}
            data.layers{i}.InputData = reshape(data.layers{i - 1}.OutputData,[prod(net.layers{i - 1}.OutputMapSize),SampleNum]);
            data.layers{i}.OutputData = data.layers{i}.InputData;
            data.layers{i}.WeightCost = data.layers{i - 1}.WeightCost;
            if isfield(net.layers{i},'classifier')
                data.layers{i}.OutputData = Classifier(data.layers{i}.OutputData,net.layers{i}.weight,net.layers{i}.classifier.type);
                data.OutputProbability = data.layers{i}.OutputData;
                data.layers{i}.WeightCost = data.layers{i}.WeightCost + (option.WeightDecay / 2) * sum((net.layers{i}.weight(:)) .^ 2);
            end
%             data.WeightCost = data.WeightCost + data.layers{i - 1}.WeightCost;
            assert(numel(data.layers{i}.GroundTruth) == numel(data.layers{i}.OutputData),'the size of GroundTruth is not in accordance with the one of final OutputData');
            data.layers{i}.GroundTruth = reshape(data.layers{i}.GroundTruth,size(data.layers{i}.OutputData));
            data.OutputCost = CalculateCost(data.layers{i}.GroundTruth,data.layers{i}.OutputData,net.layers{i}.CostFunction);
            data.cost = data.OutputCost + data.layers{i}.WeightCost;
    end
end
end
