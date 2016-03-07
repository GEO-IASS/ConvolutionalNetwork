clc;
clear all;
close all;

%%
mex Pool_cpp.cpp  -v LDFLAGS="\$LDFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp -O2 -funroll-loops" CFLAGS="\$CFLAGS -fopenmp -O2 -funroll-loops" COPTIMFLAGS="$COPTIMFLAGS -fopenmp" CC="gcc-4.8 -DOPENMP" CXX="g++-4.8 -DOPENMP" LD="g++-4.8" -largeArrayDims%(linux)
%(windows)COMPFLAGS="/openmp $COMPFLAGS"
pool.type = 'max';
data_size = 8;
pool_size = 3;
pool_stride = 2;
channels = 100;
samples = 100;

a = single(rand(data_size,data_size,channels,samples));
pool.size = single([pool_size,pool_size]);
pool.stride = single([pool_stride,pool_stride]);
pool.MapSize = single([ceil((data_size - pool_size) / pool_stride) + 1,ceil((data_size - pool_size) / pool_stride) + 1,channels]);
tic;[e1,c1] = Pool_cpp(pool,a);toc;
tic;[e2,c2] = Pool(pool,a);toc;
display(norm(e1(:) - e2(:)) ./ norm(e1(:) + e2(:)));
% display(norm(c1(:) - c2(:)) / norm(c1(:) + c2(:)));
%%
sensitivity = e1;
mark = c1;
mex DePool_cpp.cpp -v LDFLAGS="\$LDFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp -O2 -funroll-loops" CFLAGS="\$CFLAGS -fopenmp -O2 -funroll-loops" COPTIMFLAGS="$COPTIMFLAGS -fopenmp" CC="gcc-4.8 -DOPENMP" CXX="g++-4.8 -DOPENMP" LD="g++-4.8" -largeArrayDims%(linux)
%(windows)COMPFLAGS="/openmp $COMPFLAGS"

tic;outputData1 = DePool_cpp(pool,e1,c1);toc;
tic;outputData2 = DePool(pool,e2,c2);toc;
display(norm(outputData1(:) - outputData2(:)) / norm(outputData1(:) + outputData2(:)));

%%
mex Pool_VisualMark_cpp.cpp -v LDFLAGS="\$LDFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp -O2 -funroll-loops" CFLAGS="\$CFLAGS -fopenmp -O2 -funroll-loops" COPTIMFLAGS="$COPTIMFLAGS -fopenmp" CC="gcc-4.8 -DOPENMP" CXX="g++-4.8 -DOPENMP" LD="g++-4.8" -largeArrayDims%(linux)
%(windows)COMPFLAGS="/openmp $COMPFLAGS"
tic;e4 = Pool_VisualMark_cpp(pool,a,c1);toc;
tic;e3 = Pool_VisualMark(pool,a,c1);toc;
display(norm(e1(:) - e3(:)) ./ norm(e1(:) + e3(:)));
display(norm(e2(:) - e4(:)) ./ norm(e2(:) + e4(:)));

%%

pool.type = 'average';
data_size = 8;
pool_size = 3;
pool_stride = 2;
channels = 10;
samples = 10;

a = single(rand(data_size,data_size,channels,samples));
pool.size = single([pool_size,pool_size]);
pool.stride = single([pool_stride,pool_stride]);
pool.MapSize = single([ceil((data_size - pool_size) / pool_stride) + 1,ceil((data_size - pool_size) / pool_stride) + 1,channels]);
tic;e1 = Pool_cpp(pool,a);toc;
tic;[e2,c2] = Pool(pool,a);toc;
display(norm(e1(:) - e2(:)) ./ norm(e1(:) + e2(:)));



tic;outputData1 = DePool_cpp(pool,e1);toc;
tic;outputData2 = DePool(pool,e2);toc;
display(norm(outputData1(:) - outputData2(:)) / norm(outputData1(:) + outputData2(:)));