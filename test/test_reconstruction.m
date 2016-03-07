clc;
clear all;
close all;

addpath(genpath('../'));
%% data transfer
load mnist_uint8;
load('4752-parameter-11-Dec-2014-num-3100.mat')

train_x = permute(double(reshape(train_x',28,28,1,60000))/255,[2,1,3,4]);
test_x = permute(double(reshape(test_x',28,28,1,10000))/255,[2,1,3,4]);
train_y = double(train_y');
test_y = double(test_y');

%%
data.layers{1}.OutputData = train_x(:,:,:,1);
option.layer = 2;
data = NetForward(net,data,option);