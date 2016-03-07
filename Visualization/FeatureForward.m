function FeatureData = FeatureForward(feature,FeatureData)
SampleNum = size(FeatureData.layers{1}.OutputData,4);
FeatureData.layers{1}.cost = 0;
for i = 1 : size(feature.layers,2);
    switch lower(feature.layers{i}.type)
        case {'conv'}
            FeatureData.layers{i}.filter.OutputData = Activation(Convolution_cpp(feature.layers{i}.filter,FeatureData.layers{i - 1}.OutputData),feature.layers{i}.activation);
            FeatureData.layers{i}.OutputData = FeatureData.layers{i}.filter.OutputData;
            if isfield(feature.layers{i},'LocalResponseNorm')
                FeatureData.layers{i}.LocalResponseNorm.InputData = FeatureData.layers{i}.OutputData;
                FeatureData.layers{i}.OutputData = LocalResponNormalize_cpp(feature.layers{i}.LocalResponseNorm,FeatureData.layers{i}.LocalResponseNorm.InputData);
            end
            
            if isfield(feature.layers{i},{'pool'})
                switch lower(feature.layers{i}.pool.type)
                    case {'max'}
                        switch(lower(feature.layers{i}.pool.VisualMark))
                            case {'free'}
                                [FeatureData.layers{i}.OutputData,FeatureData.layers{i}.pool.mark] = Pool_cpp(feature.layers{i}.pool,FeatureData.layers{i}.OutputData);
                            case {'fixed','random'}
                                FeatureData.layers{i}.OutputData = Pool_VisualMark_cpp(feature.layers{i}.pool,FeatureData.layers{i}.OutputData,FeatureData.layers{i}.pool.mark);
                        end
                    case {'average','ave'}
                        FeatureData.layers{i}.OutputData = Pool_cpp(feature.layers{i}.pool,FeatureData.layers{i}.OutputData);
                end
            end
            FeatureData.layers{i}.cost = sum((feature.layers{i}.filter.kernel(:)) .^ 2) + FeatureData.layers{i - 1}.cost;
        case {'full'}
             switch lower(feature.layers{i - 1}.type)
                case {'conv','input'}
                    MapSize = size(FeatureData.layers{i - 1}.OutputData);
                    FeatureData.layers{i}.InputData = reshape(FeatureData.layers{i - 1}.OutputData,prod(MapSize(1 : 3)),SampleNum);
                case {'full'}
                    FeatureData.layers{i}.InputData = FeatureData.layers{i - 1}.OutputData;                   
             end
            FeatureData.layers{i}.OutputData = Activation(bsxfun(@plus,feature.layers{i}.weight * FeatureData.layers{i}.InputData, ...
                feature.layers{i}.bias'),feature.layers{i}.activation);
            FeatureData.layers{i}.cost = sum((feature.layers{i}.weight(:)) .^ 2) + FeatureData.layers{i - 1}.cost;
        case {'output'}
    end
end
end