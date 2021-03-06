clc;
clear all;
close all;

addpath(genpath('../'));
%% data transfer
load mnist_uint8;

train_x = permute(double(reshape(train_x',28,28,1,60000))/255,[2,1,3,4]);
test_x = permute(double(reshape(test_x',28,28,1,10000))/255,[2,1,3,4]);
train_y = double(train_y');
test_y = double(test_y');

%%

i = 1;
net.layers{i}.type = 'input';
net.layers{i}.OutputMapSize = [28,28,1];

i = i + 1;
net.layers{i}.type = 'conv';
net.layers{i}.activation = 'tanh';
net.layers{i}.filter.size = [5 5 25];
% net.layers{i}.filter.stride = [2,2];
net.layers{i}.pool.type = 'max';
net.layers{i}.pool.size = [2 2];
net.layers{i}.OutputAlignment = 'random';

i = i + 1;
net.layers{i}.type = 'conv';
net.layers{i}.activation = 'tanh';
net.layers{i}.filter.size = [5 5 16];
% net.layers{i}.filter.stride = [2,2];
net.layers{i}.pool.type = 'max';
net.layers{i}.pool.size = [2 2];
% net.layers{i}.OutputAlignment = 'random';
% 
i = i + 1;
net.layers{i}.type = 'full';
net.layers{i}.activation = 'tanh';
net.layers{i}.NeuronNum = 64;
% net.layers{i}.DropoutRate = 0.5;

% i = i + 1;
% net.layers{i}.type = 'full';
% net.layers{i}.activation = 'reLu';
% net.layers{i}.NeuronNum = 10;
% net.layers{i}.DropoutRate = 0.2;

i = i + 1;
net.layers{i}.type = 'output';
net.layers{i}.classifier.type = 'softmax';
net.layers{i}.classifier.ClassNum = 10;
net.layers{i}.CostFunction = 'entropy';

net = NetSetup(net);
%%
option.display = 'on';
option.epoch = 1;
option.BatchSize = 500;
option.ForceCount = 50;
option.SaveInterval = 50;
option.LearningRate = 0.3;
% option.BatchCross = 1;


[net,TrainRecord] = NetTrain(net,train_x,train_y,option);
TestRecord = NetTest(net,test_x,test_y,option);
%%
option.layer = 4;
option.penalty = 0.5;
VisualFeature(net,option);
