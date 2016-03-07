clc;
clear all;
close all;

option.display = 'on';
option.count = 200;

load 584-parameter-08-Sep-2014-num-2200
option.layer = 2;
VisualFeature(net,option);
option.layer = 3;
[feature,~,MaxOutputData] = VisualFeature(net,option);

%%
load mnist_uint8
train_x = permute(double(reshape(train_x',28,28,1,60000))/255,[2,1,3,4]);
test_x = permute(double(reshape(test_x',28,28,1,10000))/255,[2,1,3,4]);
train_y = double(train_y');
test_y = double(test_y');
data.layers{1}.OutputData = train_x(:,:,:,1 : 25);
option.layer = 3;
CostHistory = [];
% net.layers{2}.activation = 'linear';
% net.layers{3}.activation = 'linear';
data = NetForward(net,data,option);

% image = zeros(size(data.layers{1}.OutputData));
% for i = 1 : size(image,4)
%     for j = 1 : size(data.layers{option.layer}.OutputData,1)
%         image(:,:,:,i) = image(:,:,:,i) + ...
%             data.layers{option.layer}.OutputData(j,i) .* ...
%             feature(:,:,:,j);
%     end
% end
% VisualKernel(data.layers{option.layer}.OutputData);
active = data.layers{option.layer}.OutputData;
stride = 4;
image = zeros(29,29,size(active,4));
for m = 1 : size(active,4)
    for i = 1 : size(active,3)
        for j = 1 : size(active,2)
            for k = 1 : size(active,1)
                image((k - 1) * stride + 1 : (k - 1) * stride + size(feature,1), ...
                    (j - 1) * stride + 1 : (j - 1) * stride + size(feature,2),m) = ...
                    image((k - 1) * stride + 1 : (k - 1) * stride + size(feature,1), ...
                    (j - 1) * stride + 1 : (j - 1) * stride + size(feature,2),m) + active(k,j,i,m) / MaxOutputData(i,1) * feature(:,:,i);%
            end
        end
    end
end
Visual4DDatum(data.layers{1}.OutputData);
title('original image');
Visual3DDatum(image);
title('reconstruction image');
