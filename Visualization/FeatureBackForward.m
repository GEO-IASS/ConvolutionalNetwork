function FeatureBackPro = FeatureBackForward(feature,FeatureData)
layer = feature.layer;
switch lower(feature.layers{layer}.type)
    case {'full'}
        FeatureBackPro.layers{layer}.sensitivity = feature.layers{layer}.weight .* Difference(FeatureData.layers{layer}.OutputData,feature.layers{layer}.activation);% ...
            % .* sign(FeatureData.layers{layer}.OutputData);
        FeatureBackPro.layers{layer}.sensitivity = reshape(FeatureBackPro.layers{layer}.sensitivity,[feature.layers{layer - 1}.OutputMapSize,1]);
    case {'conv'}
        FeatureBackPro.layers{layer}.sensitivity = zeros(size(feature.layers{layer}.filter.kernel));
        for i = 1 : size(feature.layers{layer}.filter.kernel,3)
            FeatureBackPro.layers{layer}.sensitivity(:,:,i) = rot90(feature.layers{layer}.filter.kernel(:,:,i),2) .* Difference(FeatureData.layers{layer}.OutputData,feature.layers{layer}.activation);% ...
               %  .* sign(FeatureData.layers{layer}.OutputData);
        end
    case {'output'}
        FeatureBackPro.layers{layer}.sensitivity = FeatureData.layers{layer}.OutputData .* feature.layers{layer}.weight;
        FeatureBackPro.layers{layer}.sensitivity = reshape(FeatureBackPro.layers{layer}.sensitivity,[feature.layers{layer - 1}.OutputMapSize,1]);
end
%% the backforward of remaining layer
for i = layer - 1 : -1 : 1
    switch lower(feature.layers{i}.type)
        case {'full'}
%             FeatureBackPro.layers{i}.sensitivity = FeatureBackPro.layers{i + 1}.sensitivity;
            FeatureBackPro.layers{i}.sensitivity = feature.layers{i}.weight' * (FeatureBackPro.layers{i + 1}.sensitivity .* ...
                Difference(FeatureData.layers{i}.OutputData,feature.layers{i}.activation));
%             FeatureBackPro.layers{i}.sensitivity = feature.layers{i}.weight' * FeatureBackPro.layers{i}.sensitivity;
            FeatureBackPro.layers{i}.sensitivity = reshape(FeatureBackPro.layers{i}.sensitivity,[feature.layers{i - 1}.OutputMapSize,1]);  
        case {'conv'}
            FeatureBackPro.layers{i}.sensitivity = FeatureBackPro.layers{i + 1}.sensitivity(1 : feature.layers{i}.OutputMapSize(1,1),1 : feature.layers{i}.OutputMapSize(1,2),:);    
            
            if isfield(feature.layers{i},'pool') 
                switch lower(feature.layers{i}.pool.type)
                    case {'max'}
                        FeatureBackPro.layers{i}.sensitivity = DePool_cpp(feature.layers{i}.pool,FeatureBackPro.layers{i}.sensitivity,FeatureData.layers{i}.pool.mark);
                    case {'average','ave'}
                        FeatureBackPro.layers{i}.sensitivity = DePool_cpp(feature.layers{i}.pool,FeatureBackPro.layers{i}.sensitivity);
                end
                FeatureBackPro.layers{i}.sensitivity = FeatureBackPro.layers{i}.sensitivity(1 : feature.layers{i}.filter.MapSize(1,1),1 : feature.layers{i}.filter.MapSize(1,2),:,:);
            end
            
            if isfield(feature.layers{i},'LocalResponseNorm')
                FeatureBackPro.layers{i}.sensitivity = FeatureBackPro.layers{i}.sensitivity .* ...
                    DeLocalResponNormalize_cpp(feature.layers{i}.LocalResponseNorm,FeatureData.layers{i}.LocalResponseNorm.InputData);
            end    
            
            FeatureBackPro.layers{i}.sensitivity = FeatureBackPro.layers{i}.sensitivity(1 : feature.layers{i}.filter.MapSize(1,1),1 : feature.layers{i}.filter.MapSize(1,2),:);
            FeatureBackPro.layers{i}.sensitivity = FeatureBackPro.layers{i}.sensitivity .* Difference(FeatureData.layers{i}.filter.OutputData,feature.layers{i}.activation);
            
            FeatureBackPro.layers{i}.sensitivity = DeConvolution_cpp(feature.layers{i}.filter,FeatureBackPro.layers{i}.sensitivity);  
        case {'input'}
            FeatureBackPro.layers{i}.sensitivity = FeatureBackPro.layers{i + 1}.sensitivity(1 : feature.layers{i}.OutputMapSize(1,1),1 : feature.layers{i}.OutputMapSize(1,2),:);
    end
end
end