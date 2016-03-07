clc;
clear all;
close all;

addpath(genpath('../'));
%%
class_num = 2;
rows = 5;
columns = 5;
channels = 3;
sample_num = 2;
input_data = single(rand(rows,columns,channels,sample_num));
GroundTruth = single([1,0;0,1]);
% core_mode = 'cpu';

CostHistory = [];
Rate = [];
point = [];
point_sequence = [];
correct_sequence = 0;
% option.LearningRate = 0.01;
% option.MomentumRate = 0.9;
% option.WeightDecay = 5e-4;

i = 1;
net.layers{i}.type = 'input';
net.layers{i}.OutputMapSize = [rows,columns,channels];   
data.layers{i}.OutputData = input_data;

i = i + 1;
net.layers{i}.type = 'conv';
net.layers{i}.activation = 'tanh';
net.layers{i}.filter.size = [3 3 6];
% net.layers{i}.filter.stride = [2 2];
% net.layers{i}.filter.MapSize = [5,5,6];

i = i + 1;
net.layers{i}.type = 'conv';
net.layers{i}.activation = 'sigmoid';
net.layers{i}.filter.size = [3 3 12];
% net.layers{i}.filter.stride = [2 2];
% net.layers{i}.filter.MapSize = [2,2,12];
net.layers{i}.pool.type = 'max';
net.layers{i}.pool.size = [2 2];
net.layers{i}.OutputAlignment = 'random';
% net.layers{i}.pool.stride = [1 1];
% net.layers{i}.pool.MapSize = [1,1,12];

i = i + 1;
net.layers{i}.type = 'conv';
net.layers{i}.activation = 'linear';
net.layers{i}.filter.size = [1 1 24];

i = i + 1;
net.layers{i}.type = 'full';
net.layers{i}.activation = 'tanh';
net.layers{i}.NeuronNum = 2;

i = i + 1;
net.layers{i}.type = 'full';
net.layers{i}.activation = 'tanh';
net.layers{i}.NeuronNum = 2;

i = i + 1;
net.layers{i}.type = 'output';
net.layers{i}.CostFunction = 'entropy';
net.layers{i}.classifier.type = 'softmax';
net.layers{i}.classifier.ClassNum = class_num;
data.layers{i}.GroundTruth = GroundTruth;

%%
% net = NetSetup(net);
% layer = 3;
% old_weight = net.layers{layer}.weight;
% old_bias = net.layers{layer}.bias;
% [data,CostHistory] = NetForward(net,data,CostHistory);
% BackPro = NetBackPropagation(net,data);
% WeightGrad = zeros(size(net.layers{layer}.weight));
% epsilon = 10^(-4);
% for i = 1 : size(net.layers{layer}.weight,1)
%     for j = 1 : size(net.layers{layer}.weight,2)
%         for m = 1 : size(net.layers{layer}.weight,3)
%             for n = 1 : size(net.layers{layer}.weight,4)
%                 net.layers{layer}.weight = old_weight;
%                 net.layers{layer}.weight(i,j,m,n) = old_weight(i,j,m,n) + epsilon;
%                 [data,CostHistory] = NetForward(net,data,CostHistory);
%                 cost_1 = data.cost;
%                 net.layers{layer}.weight = old_weight;
%                 net.layers{layer}.weight(i,j,m,n) = old_weight(i,j,m,n) - epsilon;
%                 [data,CostHistory] = NetForward(net,data,CostHistory);
%                 cost_2 = data.cost;
%                 WeightGrad(i,j,m,n) = (cost_1 - cost_2) / (2 * epsilon);
%             end
%         end
%     end
% end
% numGrad = WeightGrad(:);
% grad = BackPro.layers{layer}.WeightGrad(:);
% disp([numGrad grad]); 
% diff = norm(numGrad-grad)/norm(numGrad+grad);
% disp(['diff = ',num2str(diff)]); 
% 
% 
% BiasGrad = zeros(size(old_bias));
% epsilon = 10^(-4);
% for i = 1 : size(net.layers{layer}.bias,2)
%         net.layers{layer}.bias = old_bias;
%         net.layers{layer}.bias(1,i) = old_bias(1,i) + epsilon;
%         [data,CostHistory] = NetForward(net,data,CostHistory);
%         cost_1 = data.cost;
%         
%         net.layers{layer}.bias = old_bias;
%         net.layers{layer}.bias(1,i) = old_bias(1,i) - epsilon;
%         [data,CostHistory] = NetForward(net,data,CostHistory);
%         cost_2 = data.cost;
%         BiasGrad(1,i) = (cost_1 - cost_2) / (2 * epsilon);
% end
% numGrad = BiasGrad(:);
% grad = BackPro.layers{layer}.BiasGrad(:);
% disp([numGrad grad]); 
% diff = norm(numGrad-grad)/norm(numGrad+grad);
% disp(['diff = ',num2str(diff)]); 



%%

net = NetSetup(net);
layer = 2;
old_weight = net.layers{layer}.filter.kernel;
data = NetForward(net,data,CostHistory);
BackPro = NetBackPropagation(net,data);
WeightGrad = single(zeros(size(net.layers{layer}.filter.kernel)));
epsilon = 10^(-1);
for i = 1 : size(net.layers{layer}.filter.kernel,1)
    for j = 1 : size(net.layers{layer}.filter.kernel,2)
        for m = 1 : size(net.layers{layer}.filter.kernel,3)
            for n = 1 : size(net.layers{layer}.filter.kernel,4)
                net.layers{layer}.filter.kernel = old_weight;
                net.layers{layer}.filter.kernel(i,j,m,n) = old_weight(i,j,m,n) + epsilon;
                data = NetForward(net,data);
                cost_1 = data.cost;
                net.layers{layer}.filter.kernel = old_weight;
                net.layers{layer}.filter.kernel(i,j,m,n) = old_weight(i,j,m,n) - epsilon;
                data = NetForward(net,data);
                cost_2 = data.cost;
                WeightGrad(i,j,m,n) = (cost_1 - cost_2) / (2 * epsilon);
            end
        end
    end
end
numGrad = WeightGrad(:);
grad = BackPro.layers{layer}.WeightGrad(:);
display('check gradient!');
disp([numGrad grad]); 
diff = norm(numGrad-grad)/norm(numGrad+grad);
disp(['diff = ',num2str(diff)]); 


old_bias = net.layers{layer}.filter.bias;
BiasGrad = single(zeros(size(old_bias)));
for i = 1 : size(net.layers{layer}.filter.bias,2)
        net.layers{layer}.filter.bias = old_bias;
        net.layers{layer}.filter.bias(1,i) = old_bias(1,i) + epsilon;
        data = NetForward(net,data);
        cost_1 = data.cost;
        
        net.layers{layer}.filter.bias = old_bias;
        net.layers{layer}.filter.bias(1,i) = old_bias(1,i) - epsilon;
        data = NetForward(net,data);
        cost_2 = data.cost;
        BiasGrad(1,i) = (cost_1 - cost_2) / (2 * epsilon);
end
numGrad = BiasGrad(:);
grad = BackPro.layers{layer}.BiasGrad(:);
disp([numGrad grad]); 
diff = norm(numGrad-grad)/norm(numGrad+grad);
disp(['diff = ',num2str(diff)]); 
