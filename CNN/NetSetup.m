function net = NetSetup(net) 
sigma = 0.01;
mean = 0;
for i = 1 : size(net.layers,2)
    switch lower(net.layers{i}.type)
        case {'conv'}
            if ~isfield(net.layers{i},'InputMapSize')
                net.layers{i}.InputMapSize = single(net.layers{i - 1}.OutputMapSize);
            else
                net.layers{i}.InputMapSize = single(net.layers{i}.InputMapSize);
            end
            
            if ~isfield(net.layers{i},'activation')
                net.layers{i}.activation = 'linear';
            end
            
            if ~isfield(net.layers{i}.filter,'stride')
                net.layers{i}.filter.stride = single([1,1]);
            else
                net.layers{i}.filter.stride = single(net.layers{i}.filter.stride);
            end
            
            if ~isfield(net.layers{i}.filter,'kernel')
                net.layers{i}.filter.kernel = single(normrnd(mean,sigma,[net.layers{i}.filter.size(1,1 : 2),net.layers{i}.InputMapSize(1,3), ...
                    net.layers{i}.filter.size(1,3)]));
            else
                net.layers{i}.filter.kernel = single(net.layers{i}.filter.kernel);
            end
            
            
            if ~isfield(net.layers{i}.filter,'bias')
                net.layers{i}.filter.bias = single(normrnd(mean,10 * sigma,[1,net.layers{i}.filter.size(1,3)])); 
            else
                net.layers{i}.filter.bias = single(net.layers{i}.filter.bias); 
            end
            
            if isfield(net.layers{i}.filter,'MapPad')
                net.layers{i}.filter.InputMapSize = single(net.layers{i}.InputMapSize + net.layers{i}.filter.MapPad);
            else
                net.layers{i}.filter.InputMapSize = single(net.layers{i}.InputMapSize);
            end
            
            net.layers{i}.filter.MapSize = single([ceil((net.layers{i}.filter.InputMapSize(1,1) - net.layers{i}.filter.size(1,1)) / net.layers{i}.filter.stride(1,1)) + 1, ...
                ceil((net.layers{i}.filter.InputMapSize(1,2) - net.layers{i}.filter.size(1,2)) / net.layers{i}.filter.stride(1,2)) + 1,net.layers{i}.filter.size(1,3)]);
            
            
            net.layers{i}.OutputMapSize = net.layers{i}.filter.MapSize;
            
            if isfield(net.layers{i},'pool')
                net.layers{i}.pool.size = single(net.layers{i}.pool.size);
                if ~isfield(net.layers{i}.pool,'stride')
                    net.layers{i}.pool.stride = single(net.layers{i}.pool.size);
                else
                    net.layers{i}.pool.stride = single(net.layers{i}.pool.stride);
                end
                net.layers{i}.pool.MapSize = single([ceil((net.layers{i}.OutputMapSize(1,1) - net.layers{i}.pool.size(1,1)) / net.layers{i}.pool.stride(1,1)) + 1, ...
                    ceil((net.layers{i}.OutputMapSize(1,2) - net.layers{i}.pool.size(1,2)) / net.layers{i}.pool.stride(1,2)) + 1,net.layers{i}.filter.size(1,3)]);
                net.layers{i}.OutputMapSize = net.layers{i}.pool.MapSize;
            end
            
            if isfield(net.layers{i},'LocalResponseNorm')
                if ~isfield(net.layers{i}.LocalResponseNorm,'k')
                    net.layers{i}.LocalResponseNorm.k = single(2);
                else
                    net.layers{i}.LocalResponseNorm.k = single(net.layers{i}.LocalResponseNorm.k);
                end
                
                if ~isfield(net.layers{i}.LocalResponseNorm,'n')
                    net.layers{i}.LocalResponseNorm.n = single(5);
                else
                    net.layers{i}.LocalResponseNorm.n = single(net.layers{i}.LocalResponseNorm.n);
                end
                
                if ~isfield(net.layers{i}.LocalResponseNorm,'alpha')
                    net.layers{i}.LocalResponseNorm.alpha = single(1.0000e-04);
                else
                    net.layers{i}.LocalResponseNorm.alpha = single(net.layers{i}.LocalResponseNorm.alpha);
                end
                
                if ~isfield(net.layers{i}.LocalResponseNorm,'beta')
                    net.layers{i}.LocalResponseNorm.beta = single(0.7500);
                else
                    net.layers{i}.LocalResponseNorm.beta = single(net.layers{i}.LocalResponseNorm.beta);
                end
            end
            
            if ~isfield(net.layers{i},'OutputAlignment')
                net.layers{i}.OutputAlignment = 'fixed';
            end
            
            if strcmpi(net.layers{i}.OutputAlignment,'random')
                net.layers{i}.ForwardAlignment = single(randperm(prod(net.layers{i}.OutputMapSize)));
                net.layers{i}.BackwardAlignment = single(zeros(1,prod(net.layers{i}.OutputMapSize)));
                for j  = 1 : prod(net.layers{i}.OutputMapSize)
                    net.layers{i}.BackwardAlignment(1,net.layers{i}.ForwardAlignment(1,j)) = single(j);
                end
            end
        case {'full'}
            if ~isfield(net.layers{i},'activation')
                net.layers{i}.activation = 'tanh';
            end
            
            if ~isfield(net.layers{i},'weight')
                net.layers{i}.weight = single(normrnd(mean,sigma,net.layers{i}.NeuronNum, ...
                    prod(net.layers{i - 1}.OutputMapSize)));
            else
                net.layers{i}.weight = single(net.layers{i}.weight);
            end
            
            if ~isfield(net.layers{i},'bias')
                net.layers{i}.bias = single(normrnd(mean,10 * sigma,[1,net.layers{i}.NeuronNum]));
            else
                net.layers{i}.bias = single(net.layers{i}.bias);
            end
            net.layers{i}.OutputMapSize = single(size(net.layers{i}.weight,1));
        case {'output'}
            if isfield(net.layers{i},'classifier')
                if ~isfield(net.layers{i},'weight')
                    net.layers{i}.weight = single(normrnd(mean,sigma,net.layers{i}.classifier.ClassNum, ...
                        prod(net.layers{i - 1}.OutputMapSize)));
                else
                    net.layers{i}.weight = single(net.layers{i}.weight);
                end
            end
    end
end
end
