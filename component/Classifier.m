function OutputData = Classifier(InputData,weight,type)
switch lower(type)
    case {'softmax'}
        M = weight * InputData;
        M1 = bsxfun(@minus,M,max(M,[],1));
        M2 = exp(M1);
        OutputData = bsxfun(@rdivide,M2,sum(M2,1));  
end
end