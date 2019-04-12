close all 
clear all
clc

load('2D_data_hsvm.mat');
pos_samples=[];
neg_samples=[];

for i=1:size(samples,1)
    if (y(i) == 1)
            pos_samples = [pos_samples;samples(i,:)];
    else
            neg_samples = [neg_samples;samples(i,:)];
    end
end

plot(pos_samples(:,1),pos_samples(:,2),'r*');
hold on
plot(neg_samples(:,1),neg_samples(:,2),'b*');
