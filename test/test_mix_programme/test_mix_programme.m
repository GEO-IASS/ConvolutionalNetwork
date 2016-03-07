clc;
clear all;
close all;

%%
mex test_mix_cpp.cpp
num0 = 5;
num1 = 5;
% num2 = 500;
a = single(rand(num0,num1));
b = single(rand(num0,num1));
tic;c1 = test_mix_cpp(a,b);toc;
tic;c2 = a ./ b;toc;
display(norm(c1(:) - c2(:)) ./ norm(c1(:) + c2(:)));
c3 = c1 ./ c2;

