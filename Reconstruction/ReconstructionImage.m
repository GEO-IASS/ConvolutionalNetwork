function [ReconsImage,ChannelImage] = ReconstructionImage(net,image,option,FeatureImage,MaxOutput)
if nargin < 3
    error('too few input arguments!')
end
if ~isfield(option,'display')
    option.display = 'off';
end
if ~isfield(option,'layer')
    error('you must specialize the iterm of ''layer'' in the option!');
end
if nargin == 3
    
end
data.layers{1}.OutputData = image;
data = NetForward(net,data,option);

if strcmpi(option.display,'on')
    Visual4DDatum(image);
end
switch lower(net.layers{option.layer}.type)
    case {'conv'}
        if strcmpi(option.display,'on')
            Visual3DDatum(data.layers{option.layer}.filter.OutputData);
        end
        stride = 1;
        for i = 1 : option.layer - 1
            if isfield(net.layers{i},'filter')
                stride = stride .* net.layers{i}.filter.stride;
            end
            if isfield(net.layers{i},'pool')
                stride = stride .* net.layers{i}.pool.stride;
            end
        end

        stride = stride .* net.layers{option.layer}.filter.stride;
        MapSize = net.layers{option.layer}.filter.MapSize;
        ChannelImage = zeros([(MapSize(1,1) - 1) * stride(1,1) + size(FeatureImage,1),...
            (MapSize(1,2) - 1) * stride(1,2) + size(FeatureImage,2),size(FeatureImage,3),...
            size(FeatureImage,4)]);
        for i = 1 : size(ChannelImage,4)
            Divide = zeros([(MapSize(1,1) - 1) * stride(1,1) + size(FeatureImage,1),...
                (MapSize(1,2) - 1) * stride(1,2) + size(FeatureImage,2),size(FeatureImage,3)]);
            for j = 1 : size(ChannelImage,3)
                for k = 1 : MapSize(1,2)
                    for l = 1 : MapSize(1,1)
                        ChannelImage((l - 1) * stride(1,1) + 1 : (l - 1) * stride + size(FeatureImage,1),...
                            (k - 1) * stride(1,2) + 1 : (k - 1) * stride(1,2) + size(FeatureImage,2),:,i) = ...
                            ChannelImage((l - 1) * stride(1,1) + 1 : (l - 1) * stride + size(FeatureImage,1),...
                            (k - 1) * stride(1,2) + 1 : (k - 1) * stride(1,2) + size(FeatureImage,2),:,i) + FeatureImage(:,:,:,i) .* ...
                            (data.layers{option.layer}.filter.OutputData(l,k,i) ./ MaxOutput(i,1)); %
                        Divide((l - 1) * stride(1,1) + 1 : (l - 1) * stride + size(FeatureImage,1),...
                            (k - 1) * stride(1,2) + 1 : (k - 1) * stride(1,2) + size(FeatureImage,2),:) = ...
                            Divide((l - 1) * stride(1,1) + 1 : (l - 1) * stride + size(FeatureImage,1),...
                            (k - 1) * stride(1,2) + 1 : (k - 1) * stride(1,2) + size(FeatureImage,2),:) + ones(size(FeatureImage(:,:,:,i)));
                    end
                end
            end
            ChannelImage(:,:,:,i) = ChannelImage(:,:,:,i) ./ Divide;
        end
        ReconsImage = sum(ChannelImage,4);
    case {'full'}
        ChannelImage = zeros(size(FeatureImage));
        ReconsImage = zeros(size(FeatureImage,1),size(FeatureImage,2),size(FeatureImage,3));
        for i = 1 : size(FeatureImage,4)
            ChannelImage(:,:,:,i) = FeatureImage(:,:,:,i) * data.layers{option.layer}.OutputData(i,1);
            ReconsImage = ReconsImage + FeatureImage(:,:,:,i) * (data.layers{option.layer}.OutputData(i,1) ./ MaxOutput(i,1));
        end
end
if strcmpi(option.display,'on')
    Visual4DDatum(ChannelImage);
    title('ChannelImage');
    Visual4DDatum(ReconsImage);
    title('ReconsImage');
end
end