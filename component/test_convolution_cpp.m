clc;
close all;
clear all;

%%
mex Convolution_cpp.cpp COMPFLAGS="/openmp $COMPFLAGS"% -v LDFLAGS="\$LDFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp -O2 -funroll-loops" CFLAGS="\$CFLAGS -fopenmp -O2 -funroll-loops" COPTIMFLAGS="$COPTIMFLAGS -fopenmp" CC="gcc-4.8 -DOPENMP" CXX="g++-4.8 -DOPENMP" LD="g++-4.8" -largeArrayDims%(linux)
%COMPFLAGS="/openmp $COMPFLAGS"%(windows)
feature_num = 100;
last_feature_num = 300;
sample_num = 1;
data_size = 28;
filter_size = 3;
filter_stride = 2;
kernel_size = 3;

filter.size = single([filter_size,filter_size,feature_num]);
filter.stride = single([filter_stride,filter_stride]);
filter.kernel = single(rand(kernel_size,kernel_size,last_feature_num,feature_num));
filter.bias = single(rand(1,feature_num));

inputData = single(rand(data_size,data_size,last_feature_num,sample_num));
filter.MapSize = single([ceil((data_size - kernel_size) / filter_stride) + 1,ceil((data_size - kernel_size) / filter_stride) + 1,feature_num]);
tic;OutputData1 = Convolution_cpp(filter,inputData);toc;
tic;OutputData2 = Convolution(filter,inputData);toc;
display(norm(OutputData1(:) - OutputData2(:)) ./ norm(OutputData1(:) + OutputData2(:)));
%%
mex DeConvolution_cpp.cpp  -v LDFLAGS="\$LDFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp -O2 -funroll-loops" CFLAGS="\$CFLAGS -fopenmp -O2 -funroll-loops" COPTIMFLAGS="$COPTIMFLAGS -fopenmp" CC="gcc-4.8 -DOPENMP" CXX="g++-4.8 -DOPENMP" LD="g++-4.8" -largeArrayDims%(linux)
%COMPFLAGS="/openmp $COMPFLAGS"
tic;inputData1 = DeConvolution_cpp(filter,OutputData1);
toc;
tic;inputData2 = DeConvolution(filter,OutputData2);
toc;
display(norm(inputData1(:) - inputData2(:)) ./ norm(inputData1(:) + inputData2(:)));

mex CalculateWeightGradient_cpp.cpp -v LDFLAGS="\$LDFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp -O2 -funroll-loops" CFLAGS="\$CFLAGS -fopenmp -O2 -funroll-loops" COPTIMFLAGS="$COPTIMFLAGS -fopenmp" CC="gcc-4.8 -DOPENMP" CXX="g++-4.8 -DOPENMP" LD="g++-4.8" -largeArrayDims%(linux)
%COMPFLAGS="/openmp $COMPFLAGS"
sensitivity = OutputData1;
InputData = inputData;
tic;WeightGrad2 = CalculateWeightGradient_cpp(filter,sensitivity,InputData);
toc;
tic;WeightGrad1 = CalculateWeightGradient(filter,sensitivity,InputData);
toc;
display(norm(WeightGrad1(:) - WeightGrad2(:)) ./ norm(WeightGrad1(:) + WeightGrad2(:)));
