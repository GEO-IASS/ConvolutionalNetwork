function net = UpDate(BackPro,net)
% author: Tang Jianbo,all rights reserved,2014-08-01
for i = 1 : size(net.layers,2)
    switch lower(net.layers{i}.type)
        case {'output'}
            if isfield(net.layers{i},'classifier')
                net.layers{i}.weight = net.layers{i}.weight + BackPro.layers{i}.momentum.weight;
                net.layers{i}.ChangePercentage = sum(abs(BackPro.layers{i}.momentum.weight(:))) ./ sum(abs(net.layers{i}.weight(:)));
            end
        case {'conv'}
            net.layers{i}.filter.kernel = net.layers{i}.filter.kernel + BackPro.layers{i}.momentum.weight;
            net.layers{i}.ChangePercentage = sum(abs(BackPro.layers{i}.momentum.weight(:))) ./ sum(abs(net.layers{i}.filter.kernel(:)));
            net.layers{i}.filter.bias = net.layers{i}.filter.bias + BackPro.layers{i}.momentum.bias;
        case {'full'}
            net.layers{i}.weight = net.layers{i}.weight + BackPro.layers{i}.momentum.weight;
            net.layers{i}.ChangePercentage = sum(abs(BackPro.layers{i}.momentum.weight(:))) ./ sum(abs(net.layers{i}.weight(:)));
            net.layers{i}.bias = net.layers{i}.bias + BackPro.layers{i}.momentum.bias;
        otherwise
            net.layers{i}.ChangePercentage = 0;
    end
end
end