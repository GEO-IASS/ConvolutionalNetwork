function [net,TrainRecord,option] = NetTrain(net,train_x,train_y,option,TrainRecord)
% this function implements the training phase of the convolutional neural
% networks,and the input arguments are defined as follow:
%   net: the structure of neural netorks
%   train_x: the training data,3 dimensionality is allowed.
%   train_y: the labeled data corresponding to training data.
%   option: specify the hyper parameters of this model,and which parameters could be assigned
%       are followed behind:
%       display
%       epoch
%       BatchSize
%       ForceCount
%       BreakRate
%       AdjustCount
%       SaveInterval
%       LearningRate
%       MomentumRate
%       WeightDecay
%       OutputDecay
%       phase
%   TrainRecord:this parameter records the intermedia outputs during the training phase. 
% author: Tang Jianbo,all rights reserved,2014-08-01
if nargin <= 4
    TrainRecord.CostHistory = [];
    TrainRecord.Rate = [];
    TrainRecord.correct_sequence = [];
    TrainRecord.point_sequence = [];
    if nargin <= 3
        option = [];
    end
end

CostHistory = [];
Rate = [];
point_sequence = [];

if ~isfield(option,'display')
    option.display = 'off';
end

if ~isfield(option,'epoch')
    option.epoch = 1;
else
    option.epoch = single(option.epoch);
end

if ~isfield(option,'BatchSize')
    option.BatchSize = size(train_x,4);
else
    option.BatchSize = single(option.BatchSize);
end
if ~isfield(option,'ForceCount')
    option.ForceCount = 100;
else
    option.ForceCount = single(option.ForceCount);
end
if ~isfield(option,'BreakRate')
    option.BreakRate = 1;
else
    option.BreakRate = single(option.BreakRate);
end
if ~isfield(option,'AdjustCount')
    option.AdjustCount = 10;
else
    option.AdjustCount = single(option.AdjustCount);
end
if ~isfield(option,'SaveInterval')
    option.SaveInterval = 400;
else
    option.SaveInterval = single(option.SaveInterval);
end
if ~isfield(option,'LearningRate')
    option.LearningRate = 0.05;
else
    option.LearningRate = single(option.LearningRate);
end
if ~isfield(option,'MomentumRate')
    option.MomentumRate = 0;
else
    option.MomentumRate = single(option.MomentumRate);
end
if ~isfield(option,'WeightDecay')
    option.WeightDecay = 5e-4; 
else
    option.WeightDecay = single(option.WeightDecay); 
end
if ~isfield(option,'OutputDecay')
    option.OutputDecay = 5e-4; 
else
    option.OutputDecay = single(option.OutputDecay); 
end
if ~isfield(option,'BatchCross')
    option.BatchCross = 0; 
else
    option.BatchCross = single(option.BatchCross); 
end

option.phase = 'train';
point = single(size(TrainRecord.correct_sequence,2));
old_correct_num = single(0);
count = single(0);
mark = single(randi(10000));


tic;
disp('========================= Train Starts ====================');
for i = 1 : option.epoch
    time_0 = toc;
    disp(['the ',num2str(i),'th epoch starts']);
    selecte_sequence = randperm(size(train_x,4));
    train_x = train_x(:,:,:,selecte_sequence);
    train_y = train_y(:,selecte_sequence); 
    BatchNum = ceil(size(train_x,4) / option.BatchSize);
    for k = 1 : BatchNum
        BatchCross = option.BatchCross;
        disp(['the ',num2str(k),'th batch starts']);
        time_1 = toc;
%         data.layers{1}.WeightCost = 0;
        data.layers{1}.OutputData = train_x(:,:,:,(k - 1) * option.BatchSize + 1 : min(end,(k + BatchCross) * option.BatchSize));
        data.layers{size(net.layers,2)}.GroundTruth = train_y(:,(k - 1) * option.BatchSize + 1 : min(end,(k + BatchCross) * option.BatchSize)); 
        
        start_mark = 1;
        correct_num = 0;
        adjust_count = 0;
        sample_num = size(data.layers{1}.OutputData,4);
        change_percentage = zeros(1,size(net.layers,2));
        
        while correct_num < sample_num * option.BreakRate
            time_2 = toc;
            data = NetForward(net,data,option);
            CostHistory = [CostHistory,data.OutputCost];
            BackPro = NetBackPropagation(net,data,option);
            net = UpDate(BackPro,net);
            time_kernel = toc;
            [~,loc_pro] = max(data.OutputProbability,[],1);
            [~,loc_label] = max(data.layers{size(net.layers,2)}.GroundTruth,[],1);
            correct_num = sum(loc_pro == loc_label);
            
            if start_mark == 1
                first_rate = correct_num / sample_num;
                if size(TrainRecord.correct_sequence,2) >= 3
                    if abs(first_rate - TrainRecord.correct_sequence(1,end)) >= 0.1 && first_rate >= 0.8
                        BatchCross = BatchCross + 1;
                    end
                end
                start_mark = 0;
            end
            
            
            for j = 1 : size(net.layers,2)
                change_percentage(1,j) = net.layers{j}.ChangePercentage;
            end
            
            % change the learning rate
            TrainRecord.correct_sequence = [TrainRecord.correct_sequence,correct_num];
            if size(TrainRecord.correct_sequence,2) >= 3
                if TrainRecord.correct_sequence(1,end) < TrainRecord.correct_sequence(1,end - 1) && TrainRecord.correct_sequence(1,end - 1) <= TrainRecord.correct_sequence(1,end - 2)
                    adjust_count = adjust_count + 1;
                end
            end
            if adjust_count >= option.AdjustCount
                disp('the Learning Rate of this net is divided by 10!');
                option.LearningRate = option.LearningRate ./ 10;
                adjust_count = 0;
            end
            
            
            record_rate = sum(data.OutputProbability .* data.layers{size(net.layers,2)}.GroundTruth,2) ./ ...
                sum(data.layers{size(net.layers,2)}.GroundTruth,2);
            Rate = [Rate,record_rate];
            
            option.MomentumRate = min(1 - correct_num / sample_num,correct_num / sample_num);
            
            % save the parameter intervally
            if mod(size(TrainRecord.correct_sequence,2),option.SaveInterval) == 0
                disp('match the save interval condition!');
                save([num2str(mark),'-parameter-',date,'-num-',num2str(size(TrainRecord.correct_sequence,2))],'net','option','TrainRecord');
            end
            
            % break the training of this batch when the accuracy increases
            % more than a given percentage 
            if correct_num >= sample_num * min(first_rate + 0.1,option.BreakRate)
%                 BatchCross = BatchCross + 1;
                disp('matches the break condition,break!')
                disp(['first_rate : ',num2str(first_rate)]);
                disp(['BreakRate  : ',num2str(min(first_rate + 0.1,option.BreakRate))]);
                disp(['real Rate  : ',num2str(correct_num / sample_num)]);
                disp(['BatchCross : ',num2str(BatchCross)]);
                break;
            end
            
            % force to break out and change the dropout rate
            force_count = 0;
            if size(TrainRecord.correct_sequence,2) >= option.ForceCount
                force_count = sum(TrainRecord.correct_sequence(1,end - option.ForceCount + 1 : end) == correct_num);
                if force_count == option.ForceCount
                    disp('matches the Force break condition,break!');
                    option.ForceCount = floor(1.2 * option.ForceCount);
                    if correct_num / sample_num >= 0.9
                        option.BreakRate = correct_num / sample_num;
                    end
                    break;
                end
            end
            
            
            % display the parameter of training process
            point = point + 1;
            time_3 = toc;
            if strcmpi(option.display,'on')
                disp(['loop : ',num2str(size(TrainRecord.correct_sequence,2))]);
                disp(['kernel time  :  ',num2str(time_kernel - time_2),'s']);
                disp(['accuracy           : ',num2str(100 * correct_num / sample_num),'%']);
                disp(['Learning rate      : ',num2str(option.LearningRate)]);
                disp(['OutputCost         : ',num2str(data.OutputCost)]);
                disp(['WeightCost         : ',num2str(data.layers{size(net.layers,2)}.WeightCost)]);
                disp(['time               : ',num2str(time_3 - time_2),'s']);
                disp(['forece count       : ',num2str(force_count + (option.ForceCount == inf) * option.ForceCount)]);
                disp(['count              : ',num2str(count)]);
                disp(['old_correct_num    : ',num2str(old_correct_num)]);
                disp(['correct_num        : ',num2str(correct_num)]);
                disp(['change percentage  : ',num2str(change_percentage)]);
            end
            
%             if isfield(option,'LoopCount')
%                 if size(TrainRecord.correct_sequence,2) >= option.LoopCount
%                     break;
%                 end
%             end
        end
        point_sequence = [point_sequence,point];
        time_4 = toc;
        disp(['time of a batch : ',num2str(time_4 - time_1),'s']);
%         if isfield(option,'LoopCount')
%             if size(TrainRecord.correct_sequence,2) >= option.LoopCount
%                 break;
%             end
%         end
    end
    time_5 = toc;
    disp(['time of a epoch : ',num2str(time_5 - time_0),'s']);
%     if isfield(option,'LoopCount')
%         if size(TrainRecord.correct_sequence,2) >= option.LoopCount
%             break;
%         end
%     end
end
time = toc;
TrainRecord.CostHistory = [TrainRecord.CostHistory,CostHistory];
TrainRecord.Rate = [TrainRecord.Rate,Rate];
TrainRecord.point_sequence = [TrainRecord.point_sequence,point_sequence];
disp(['total time : ',num2str(time),'s']);
disp('========================= Train Finished =========================')
end