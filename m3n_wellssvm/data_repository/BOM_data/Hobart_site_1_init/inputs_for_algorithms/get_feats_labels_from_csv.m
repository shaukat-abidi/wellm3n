clear all
close all
clc

data = load('raw_files/data.csv');
gt_labels = load('raw_files/labels.txt');

assert(length(gt_labels) == size(data,1));

save('mat_files/data_labels.mat');
