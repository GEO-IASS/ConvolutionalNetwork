clc;
clear all;
close all;
addpath(genpath('./'));
%% test the gradient

class_num = 2;
rows = 10;
columns = 10;
channels = 3;
i = 1;
net.layers{i}.type = 'input';
net.layers{i}.OutputMapSize = [rows,columns,channels];   

i = i + 1;
net.layers{i}.type = 'conv';
net.layers{i}.activation = 'reLu';
net.layers{i}.filter.size = [3 3 6];
net.layers{i}.filter.stride = [2 2];
net.layers{i}.filter.MapSize = [5,5,6];
net.layers{i}.LocalResponseNorm.k = 2;
net.layers{i}.LocalResponseNorm.n = 5;
net.layers{i}.LocalResponseNorm.alpha = 10e-4;
net.layers{i}.LocalResponseNorm.beta = 0.75;

i = i + 1;
net.layers{i}.type = 'conv';
net.layers{i}.activation = 'reLu';
net.layers{i}.filter.size = [3 3 12];
net.layers{i}.filter.stride = [2 2];
net.layers{i}.filter.MapSize = [2,2,12];

i = i + 1;
net.layers{i}.type = 'full';
net.layers{i}.activation = 'reLu';
net.layers{i}.NeuronNum = 6;

i = i + 1;
net.layers{i}.type = 'full';
net.layers{i}.activation = 'reLu';
net.layers{i}.NeuronNum = 6;

i = i + 1;
net.layers{i}.type = 'full';
net.layers{i}.activation = 'reLu';
net.layers{i}.NeuronNum = 6;

i = i + 1;
net.layers{i}.type = 'output';
net.layers{i}.classifier.type = 'softmax';
net.layers{i}.classifier.ClassNum = class_num;
net.layers{i}.CostFunction = 'entropy';

net = NetSetup(net);

layer = 4;
feature.layer = layer;
feature.num = 1;
feature = Net2Feature(net,feature);
OutputData = normrnd(0,0.1,feature.layers{1}.OutputMapSize);
amplify = 100;
OutputData = amplify * OutputData ./ sqrt(sum(OutputData(:) .^ 2));
FeatureData.layers{1}.cost = 0;
FeatureData.layers{1}.OutputData = OutputData;
FeatureData = FeatureForward(feature,FeatureData);
penalty = 50;
cost = FeatureData.layers{layer}.OutputData - penalty * ((sqrt(sum(FeatureData.layers{1}.OutputData(:) .^ 2)) - amplify) .^ 2);%
FeatureBackPro = FeatureBackForward(feature,FeatureData);

% test the gradient

old_image = OutputData;
epsilon = 1e-4;
WeightGrad = zeros(feature.layers{1}.OutputMapSize);
for i = 1 : feature.layers{1}.OutputMapSize(1,3)
    for j = 1 : feature.layers{1}.OutputMapSize(1,2)
        for m = 1 : feature.layers{1}.OutputMapSize(1,1)
            FeatureData.layers{1}.OutputData = old_image;
            FeatureData.layers{1}.OutputData(m,j,i) = FeatureData.layers{1}.OutputData(m,j,i) + epsilon;
            FeatureData = FeatureForward(feature,FeatureData);
            cost_1 = FeatureData.layers{layer}.OutputData - penalty * ((sqrt(sum(FeatureData.layers{1}.OutputData(:) .^ 2)) - amplify) .^ 2);%
            
            FeatureData.layers{1}.OutputData = old_image;
            FeatureData.layers{1}.OutputData(m,j,i) = FeatureData.layers{1}.OutputData(m,j,i) - epsilon;
            FeatureData = FeatureForward(feature,FeatureData);
            cost_2 = FeatureData.layers{layer}.OutputData - penalty * ((sqrt(sum(FeatureData.layers{1}.OutputData(:) .^ 2)) - amplify) .^ 2);%
            WeightGrad(m,j,i) = (cost_1 - cost_2) ./ (2 * epsilon);
            numGrad = WeightGrad(m,j,i);
            grad = FeatureBackPro.layers{1}.sensitivity(m,j,i);
            disp([numGrad grad]); 
            diff = norm(numGrad-grad)/norm(numGrad+grad);
            disp(['diff = ',num2str(diff)]); 
        end
    end
end
numGrad = WeightGrad(:);
FeatureData.layers{1}.OutputData = old_image;
factor = 2 * penalty * (sqrt(sum(FeatureData.layers{1}.OutputData(:) .^ 2)) - amplify) * ((sum(FeatureData.layers{1}.OutputData(:) .^ 2)) ^ (-0.5));
grad = FeatureBackPro.layers{1}.sensitivity - FeatureData.layers{1}.OutputData .* factor;%
grad = grad(:);
disp([numGrad grad]); 
diff = norm(numGrad-grad)/norm(numGrad+grad);
disp(['diff = ',num2str(diff)]); 

%% the real program
% load data_2014_5_21_drop_4class_desk
% layer = 3;
% num = 1;
% VisualFeature(net,layer,num);








