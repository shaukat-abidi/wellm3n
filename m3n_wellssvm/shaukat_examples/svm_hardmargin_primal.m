clear all
close all
load('2D_data_hsvm.mat')
margin = 1;
m = size(samples,1);
n = size(samples,2);
x = samples;
clc
cvx_begin
variables w(n) b
dual variable alpha
minimize(0.5 * w' * w)
subject to
alpha: y .* (x * w  + b) >= margin
cvx_end

% plot hard margin svm
plot_classifier(x,y,w,b)