% It will take data.csv and rainfall_amount.csv as input, and write labels.txt 
% as an output file
clear all
close all
clc

data = load('data.csv');
labels = load('rainfall_amount.csv');
assert(length(labels) == size(data,1));
label_threshold = 2.0;


fName = strcat('labels.txt');
fid = fopen(fName,'w');
fprintf('Start writing results to file. \n');

%Discretize labels (True = 1; False = 2)
for iter_lab=1:length(labels)
    if(labels(iter_lab) <= label_threshold)
        labels(iter_lab) = 2;
    else
        labels(iter_lab) = 1;
    end

    %Write it to the file
    fprintf(fid,'%d\n',labels(iter_lab));
end

fclose(fid);
