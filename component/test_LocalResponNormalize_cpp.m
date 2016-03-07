clc;
close all;
clear all;

%%
mex LocalResponNormalize_cpp.cpp -v LDFLAGS="\$LDFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp -O2 -funroll-loops" CFLAGS="\$CFLAGS -fopenmp -O2 -funroll-loops" COPTIMFLAGS="$COPTIMFLAGS -fopenmp" CC="gcc-4.8 -DOPENMP" CXX="g++-4.8 -DOPENMP" LD="g++-4.8" -largeArrayDims%(linux)
%(windows)COMPFLAGS="/openmp $COMPFLAGS"
LocalResponseNorm.k = single(2);
LocalResponseNorm.alpha = single(2);
LocalResponseNorm.n = single(3);
LocalResponseNorm.beta = single(3);
inputData = single(rand(9,9,400,10));
tic;output_1 = LocalResponNormalize_cpp(LocalResponseNorm,inputData);
toc;
tic;
output_2 = LocalResponNormalize(LocalResponseNorm,inputData);
toc;
display((norm(output_1(:)) - norm(output_2(:))) ./ (norm(output_1(:)) + norm(output_2(:))));
mex DeLocalResponNormalize_cpp.cpp -v LDFLAGS="\$LDFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp -O2 -funroll-loops" CFLAGS="\$CFLAGS -fopenmp -O2 -funroll-loops" COPTIMFLAGS="$COPTIMFLAGS -fopenmp" CC="gcc-4.8 -DOPENMP" CXX="g++-4.8 -DOPENMP" LD="g++-4.8" -largeArrayDims%(linux)
%(windows)COMPFLAGS="/openmp $COMPFLAGS"
tic;inputData1 = DeLocalResponNormalize_cpp(LocalResponseNorm,output_1);
toc;

tic;
inputData2 = DeLocalResponNormalize(LocalResponseNorm,output_2);
toc;
display((norm(inputData1(:)) - norm(inputData2(:))) ./ (norm(inputData1(:)) + norm(inputData2(:))));
