clear all
close all
clc

flag_normalize = 0;
flag_scaling = 1;

data = load('raw_files/data.csv');
labels = load('raw_files/labels.txt');

assert(length(labels) == size(data,1));

%Normalize data
normalize_data = data;
clear data
scale_factor = 1.0;
for i=1:size(normalize_data,1)
    for j=1:size(normalize_data,2)
        normalize_data(i,j) = normalize_data(i,j) * scale_factor;
    end    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Writing Results to File %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fName = 'groundtruth_file_scale_100.txt';
fName = strcat('groundtruth_file_scale_', num2str(scale_factor), '.txt');

fid = fopen(fName,'w');

fprintf('Start writing results to file. \n');

rows = size(normalize_data,1);
cols = size(normalize_data,2);


% Normalizing data
if(flag_normalize == 1)
    
    A = normalize_data;
    for iter_cols=1:cols
        std_dev = std(A(:,iter_cols));
        mean_val = mean(A(:,iter_cols));
        A(:,iter_cols) = A(:,iter_cols) - mean_val;
        A(:,iter_cols) = A(:,iter_cols) ./ std_dev;
    end
    clear normalize_data
    normalize_data = A;

end

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
