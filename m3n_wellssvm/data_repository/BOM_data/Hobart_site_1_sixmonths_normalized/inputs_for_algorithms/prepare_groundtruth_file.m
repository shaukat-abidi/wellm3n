clear all
close all
clc

data = load('raw_files/data.csv');
labels = load('raw_files/labels.txt');
assert(length(labels) == size(data,1));

normalize_data = data;
scale_factor = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Writing Results to File %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fName = 'groundtruth_file_scale_100.txt';
fName = strcat('groundtruth_file_scale_', num2str(scale_factor), '.txt');

fid = fopen(fName,'w');

fprintf('Start writing results to file. \n');

rows = size(normalize_data,1);
cols = size(normalize_data,2);

qid = 1; %we have single sequence

for iter_rows=1:rows
    for iter_cols=1:cols
        feat_val = normalize_data(iter_rows,iter_cols);
        if (iter_cols == 1)
            fprintf(fid,'%d qid:1 %d:%f',labels(iter_rows),iter_cols,feat_val);
        else
            fprintf(fid,' %d:%f',iter_cols,feat_val);
        end
        
    end
    fprintf(fid,'\n');
    
    if(rem(iter_rows,50) == 0)
    fprintf('.');
    end
    
end
fprintf('\n');

fclose(fid);
