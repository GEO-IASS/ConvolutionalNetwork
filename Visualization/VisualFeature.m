function [kernel,CostHistory,MaxOutput,net] = VisualFeature(net,option)
if nargin <= 1
    option.layer = 2;
    option.display = 'off';
    option.count = 60;
    option.amplify = 1;
    option.penalty = 50;
    option.LearningRate = 0.02;
    option.VisualType = 'own';
end

%%
if ~isfield(option,'layer')
    option.layer = 2;
end

if ~isfield(option,'display')
    option.display = 'off';
end

if ~isfield(option,'count')
    option.count = 60;
end

if ~isfield(option,'amplify')
    option.amplify = 1;
end

if ~isfield(option,'penalty')
    option.penalty = 5;
end

if ~isfield(option,'LearningRate')
    option.LearningRate = 0.02;
end

if ~isfield(option,'VisualType')
    option.VisualType = 'own';
end

layer = option.layer;
count = option.count;
amplify = option.amplify;
penalty = option.penalty;
LearningRate = option.LearningRate;
feature.layer = layer;

if ~isfield(option,'num')
    switch lower(net.layers{layer}.type)
        case {'full'}
            num = 1 : prod(net.layers{layer}.OutputMapSize);
        case {'conv'}
            num = 1 : net.layers{layer}.filter.MapSize(1,3);
    end
else
    num = option.num;
end

kernel = [];
CostHistory = zeros(size(num,2),count);
MaxOutput = zeros(size(num,2),count);
tic;
for i = 1 : size(num,2)
    time1 = toc;
%     if strcmpi(option.display,'on')
%     end
    feature.num = num(1,i);
    [feature,FeatureData,net] = Net2Feature(net,feature);
    switch lower(option.VisualType)
        case {'erhan'}
            OutputData = normrnd(0,0.1,feature.layers{1}.OutputMapSize);
            OutputData = amplify * OutputData ./ sqrt(sum(OutputData(:) .^ 2));
        case {'karen'}
        case {'own'}        
            OutputData = 0.5 * ones(feature.layers{1}.OutputMapSize);
    end
    FeatureData.layers{1}.OutputData = OutputData;
    momentum = 0;
    for k = 1 : count 
        FeatureData = FeatureForward(feature,FeatureData);
        switch lower(option.VisualType)
            case {'erhan'}
                cost = FeatureData.layers{layer}.OutputData - penalty * ((sqrt(sum(FeatureData.layers{1}.OutputData(:) .^ 2)) - amplify) .^ 2);
                CostHistory(i,k) = cost;
                MaxOutput(i,k) = FeatureData.layers{layer}.OutputData;
                FeatureBackPro = FeatureBackForward(feature,FeatureData);
                factor = penalty * 2 * (sqrt(sum(FeatureData.layers{1}.OutputData(:) .^ 2)) - amplify) * ((sum(FeatureData.layers{1}.OutputData(:) .^ 2)) ^ (-0.5));
                grad = FeatureBackPro.layers{1}.sensitivity  - FeatureData.layers{1} .OutputData .* factor;
                momentum = 0.9 * momentum + LearningRate * grad;
                FeatureData.layers{1}.OutputData = FeatureData.layers{1}.OutputData + momentum;
            case {'karen'}
            case {'own'}
                cost = FeatureData.layers{layer}.OutputData - penalty * (sum(FeatureData.layers{1}.OutputData(:)) ./ numel(FeatureData.layers{1}.OutputData));
                CostHistory(i,k) = cost;
                MaxOutput(i,k) = FeatureData.layers{layer}.OutputData;
                FeatureBackPro = FeatureBackForward(feature,FeatureData);
                grad = FeatureBackPro.layers{1}.sensitivity  - penalty * (ones(size(FeatureData.layers{1}.OutputData)) ./ numel(FeatureData.layers{1}.OutputData));
                momentum = 0.9 * momentum + LearningRate * grad;
                OutputData = FeatureData.layers{1}.OutputData + momentum;
                max_output = max(OutputData(:));
                min_output = min(OutputData(:));
                OutputData = (OutputData - min_output) ./ (max_output - min_output);
                FeatureData.layers{1}.OutputData = OutputData;
        end
    end
%     MaxOutput(i,1) = FeatureData.layers{layer}.OutputData;
    if isempty(kernel)
        kernel = zeros([feature.layers{1}.OutputMapSize,size(num,2)]);
    end
    kernel(:,:,:,i) = FeatureData.layers{1}.OutputData;
    time2 = toc;
    display(['visualizing the ',num2str(i),'th neuron of the ',num2str(layer),'th layer takes ',num2str(time2 - time1),'s']);
end
if strcmpi(option.display,'on')
    edge_size = ceil(sqrt(size(kernel,4)));
    image = ones((size(kernel,1) + 1) * edge_size - 1,(size(kernel,2) + 1) * ...
        edge_size - 1,size(kernel,3));
    for i = 1 : size(kernel,4)
        x = floor((i - 1) / edge_size);
        y = mod(i - 1,edge_size);
        b = kernel(:,:,:,i);
        min_b = min(b(:));
        max_b = max(b(:));
        b = (b - min_b) ./ (max_b - min_b);
        image(x * (size(kernel,1) + 1) + 1 : x * (size(kernel,1) + 1) + size(kernel,1), ...
            y * (size(kernel,2) + 1) + 1 : y * (size(kernel,2) + 1) + size(kernel,2),:) ...
            = b;
    end
    figure;
    imshow(image);
    title(['feature maps of Layer ',num2str(layer)]);
    figure;
    plot(CostHistory');
    title(['CostHistory of Layer ',num2str(layer)]);
    figure;
    plot(MaxOutput');
    title(['MaxOutput of Layer ',num2str(layer)]);
    figure;
    plot((MaxOutput - CostHistory)');
    title(['penalty of Layer ',num2str(layer)]);
end
end
