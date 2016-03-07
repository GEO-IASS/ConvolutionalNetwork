function [BackPro,net] = NetBackPropagation(net,data,option)
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

if ~isfield(option,'LearningRate')
    option.LearningRate = single(0.05);
else
    option.LearningRate = single(option.LearningRate);
end

if ~isfield(option,'MomentumRate')
    option.MomentumRate = single(0.9);
else
    option.MomentumRate = single(option.MomentumRate);
end

if ~isfield(option,'OutputDecay')
    option.OutputDecay = single(5e-4); 
else
    option.OutputDecay = single(option.OutputDecay); 
end



SampleNum = single(size(data.layers{1}.OutputData,4));
for i = option.layer : -1 : 1
    switch lower(net.layers{i}.type)
        case {'output'}
            BackPro.layers{i}.sensitivity = CostDifference(data.layers{i}.GroundTruth,data.layers{i}.OutputData,net.layers{i}.CostFunction);
            if isfield(net.layers{i},'classifier')
                if strcmpi(net.layers{i}.classifier.type,'softmax') && strcmpi(net.layers{i}.CostFunction,'entropy')
                    BackPro.layers{i}.sensitivity = single(- (data.layers{i}.GroundTruth - data.layers{i}.OutputData));
                    BackPro.layers{i}.WeightGrad = 1/SampleNum * (BackPro.layers{i}.sensitivity * data.layers{i}.InputData') + option.WeightDecay * net.layers{i}.weight;
                end
                if ~isfield(BackPro.layers{i},'momentum')
                    BackPro.layers{i}.momentum.weight = - option.LearningRate * BackPro.layers{i}.WeightGrad;
                else
                    BackPro.layers{i}.momentum.weight = option.MomentumRate * BackPro.layers{i}.momentum.weight  - option.LearningRate * BackPro.layers{i}.WeightGrad;
                end
                BackPro.layers{i}.sensitivity = net.layers{i}.weight' * BackPro.layers{i}.sensitivity;
            end
            BackPro.layers{i}.sensitivity = reshape(BackPro.layers{i}.sensitivity,[net.layers{i - 1}.OutputMapSize,SampleNum]);  
        case {'full'}
            BackPro.layers{i}.sensitivity = BackPro.layers{i + 1}.sensitivity;
%             BackPro.layers{i}.sensitivity = BackPro.layers{i}.sensitivity + option.OutputDecay * data.layers{i}.OutputData ./ numel(data.layers{i}.OutputData);% 
            if isfield(net.layers{i},'DropoutRate')
                BackPro.layers{i}.sensitivity = BackPro.layers{i}.sensitivity .* data.layers{i}.DropoutMark;
            end
            BackPro.layers{i}.sensitivity = BackPro.layers{i}.sensitivity .* Difference(data.layers{i}.OutputData,net.layers{i}.activation);
            
            BackPro.layers{i}.WeightGrad = 1/SampleNum * (BackPro.layers{i}.sensitivity * data.layers{i}.InputData') + option.WeightDecay * net.layers{i}.weight;
            BackPro.layers{i}.BiasGrad = mean(BackPro.layers{i}.sensitivity,2);
            if ~isfield(BackPro.layers{i},'momentum')
                BackPro.layers{i}.momentum.weight = - option.LearningRate * BackPro.layers{i}.WeightGrad;
                BackPro.layers{i}.momentum.bias = - option.LearningRate * BackPro.layers{i}.BiasGrad';
            else
                BackPro.layers{i}.momentum.weight = option.MomentumRate * BackPro.layers{i}.momentum.weight  - option.LearningRate * BackPro.layers{i}.WeightGrad;
                BackPro.layers{i}.momentum.bias = option.MomentumRate * BackPro.layers{i}.momentum.bias  - option.LearningRate * BackPro.layers{i}.BiasGrad';
            end
            BackPro.layers{i}.sensitivity = net.layers{i}.weight' * BackPro.layers{i}.sensitivity;
            BackPro.layers{i}.sensitivity = reshape(BackPro.layers{i}.sensitivity,[net.layers{i - 1}.OutputMapSize,SampleNum]);  
        case {'conv'}
            BackPro.layers{i}.sensitivity = BackPro.layers{i + 1}.sensitivity(1 : net.layers{i}.OutputMapSize(1,1),1 : net.layers{i}.OutputMapSize(1,2),:,:);    
            
            if strcmpi(net.layers{i}.OutputAlignment,'random')
                sensitivity = reshape(BackPro.layers{i}.sensitivity,[prod(net.layers{i}.OutputMapSize),SampleNum]);
                BackPro.layers{i}.sensitivity = reshape(sensitivity(net.layers{i}.BackwardAlignment,:),[net.layers{i}.OutputMapSize,SampleNum]);
            end
            
            if isfield(net.layers{i},'pool')
                switch lower(net.layers{i}.pool.type)
                    case {'max'}
                        BackPro.layers{i}.sensitivity = DePool_cpp(net.layers{i}.pool,BackPro.layers{i}.sensitivity,data.layers{i}.pool.mark);   
                    case {'average'}
                        BackPro.layers{i}.sensitivity = DePool_cpp(net.layers{i}.pool,BackPro.layers{i}.sensitivity); 
                end     
                BackPro.layers{i}.sensitivity = BackPro.layers{i}.sensitivity(1 : net.layers{i}.filter.MapSize(1,1),1 : net.layers{i}.filter.MapSize(1,2),:,:);
            end           
            
            if isfield(net.layers{i},'LocalResponseNorm')
                BackPro.layers{i}.sensitivity = BackPro.layers{i}.sensitivity .* DeLocalResponNormalize_cpp(net.layers{i}.LocalResponseNorm,data.layers{i}.LocalResponseNorm.InputData);
            end
            
%             BackPro.layers{i}.sensitivity = BackPro.layers{i}.sensitivity + option.OutputDecay * data.layers{i}.filter.OutputData ./ numel(data.layers{i}.filter.OutputData);% 
            BackPro.layers{i}.sensitivity = BackPro.layers{i}.sensitivity .* Difference(data.layers{i}.filter.OutputData,net.layers{i}.activation);
            
            BackPro.layers{i}.WeightGrad = CalculateWeightGradient_cpp(net.layers{i}.filter,BackPro.layers{i}.sensitivity, ...
                data.layers{i - 1}.OutputData);
            BackPro.layers{i}.WeightGrad = BackPro.layers{i}.WeightGrad + option.WeightDecay * net.layers{i}.filter.kernel;
            
            biasgrad = mean(BackPro.layers{i}.sensitivity,4);
            biasgrad = squeeze(sum(sum(biasgrad,1),2));
            BackPro.layers{i}.BiasGrad = biasgrad';
            if ~isfield(BackPro.layers{i},'momentum')
                BackPro.layers{i}.momentum.weight = - option.LearningRate * BackPro.layers{i}.WeightGrad;
                BackPro.layers{i}.momentum.bias = - option.LearningRate * BackPro.layers{i}.BiasGrad;
            else
                BackPro.layers{i}.momentum.weight = option.MomentumRate * BackPro.layers{i}.momentum.weight  - option.LearningRate * BackPro.layers{i}.WeightGrad;
                BackPro.layers{i}.momentum.bias = option.MomentumRate * BackPro.layers{i}.momentum.bias  - option.LearningRate * BackPro.layers{i}.BiasGrad;
            end

            BackPro.layers{i}.sensitivity = DeConvolution_cpp(net.layers{i}.filter,BackPro.layers{i}.sensitivity);  
    end
end
end



