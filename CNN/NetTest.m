function TestRecord = NetTest(net,test_x,test_y,option,TestRecord)
% this function implements the testing phase of the convolutional neural
% networks,and the parameters are defined as follow:
%   net: the structure of neural netorks
%   test_x: the testing data,3 dimensionality is allowed.
%   test_y: the labeled data corresponding to testing data.
%   option: specify the hyper parameters of this model,and which parameters could be assigned
%       are followed behind:
%       display
%       epoch
%       BatchSize
%       WeightDecay
%       OutputDecay
%       phase
%   TestRecord:this parameter records the intermedia outputs during the testing phase. 
% author: Tang Jianbo,all rights reserved,2014-08-01
if nargin <= 4
    TestRecord.CostHistory = [];
    TestRecord.Rate = [];
    TestRecord.correct_sequence = [];
    TestRecord.point_sequence = [];
    if nargin <= 3
        option = [];
    end
end
if ~isfield(option,'display')
    option.display = 'on';
end
if ~isfield(option,'epoch')
    option.epoch = 1;
end
if ~isfield(option,'BatchSize')
    option.BatchSize = size(test_x,4);
end
if ~isfield(option,'WeightDecay')
    option.WeightDecay = 5e-4; 
end
if ~isfield(option,'OutputDecay')
    option.OutputDecay = 5e-4; 
end
option.phase = 'test';


Rate = [];
CostHistory = [];
correct_sequence = [];

tic;
disp('========================= Test Starts =========================');
selecte_sequence = randperm(size(test_x,4));
test_x = test_x(:,:,:,selecte_sequence);
test_y = test_y(:,selecte_sequence); 
BatchNum = ceil(size(test_x,4) / option.BatchSize);
for k = 1 : BatchNum
    disp(['the ',num2str(k),'th batch starts']);
    time_1 = toc;
    data.layers{1}.OutputData = test_x(:,:,:,(k - 1) * option.BatchSize + 1 : min(end,k * option.BatchSize));
    data.layers{size(net.layers,2)}.GroundTruth = test_y(:,(k - 1) * option.BatchSize + 1 : min(end,k * option.BatchSize)); 
%     data.layers{1}.cost = 0;

    sample_num = size(data.layers{1}.OutputData,4);
    data = NetForward(net,data,option);
    CostHistory = [CostHistory,data.OutputCost];
    [~,loc_pro] = max(data.OutputProbability,[],1);
    [~,loc_label] = max(data.layers{size(net.layers,2)}.GroundTruth,[],1);
    correct_num = sum(loc_pro == loc_label);
    correct_sequence = [correct_sequence,correct_num];
    if strcmpi(option.display,'on')
        disp(['loop : ',num2str(size(correct_sequence,2))]);
        disp(['accuracy       : ',num2str(100 * correct_num / sample_num),'%']);
        disp(['OutputCost     : ',num2str(data.OutputCost)]);
    end
    time_4 = toc;
    disp(['time of a batch : ',num2str(time_4 - time_1),'s']);
end
disp('============================== test report =========================')
disp(['accuracy : ',num2str(100 * sum(correct_sequence) ./ size(test_x,4)),'%']);
TestRecord.correct_sequence = [TestRecord.correct_sequence;correct_sequence];
time = toc;
disp(['total time : ',num2str(time),'s']);

TestRecord.CostHistory = [TestRecord.CostHistory,CostHistory];
TestRecord.Rate = [TestRecord.Rate,Rate];
end