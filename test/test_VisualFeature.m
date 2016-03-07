clc;
clear all;
close all;

%%
addpath(genpath('./'));
load 1375-parameter-08-Sep-2014-num-2000
option.display = 'on';
option.layer = 2;
option.count = 100;
VisualFeature(net,option);
option.layer = 3;
VisualFeature(net,option);
% option.layer = 4;
% VisualFeature(net,option);