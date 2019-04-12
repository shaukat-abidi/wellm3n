clear all
close all
clc

normalize_data = load('data.csv');
rows = size(normalize_data,1);
cols = size(normalize_data,2);

% Normalizing data with 0-Mean and Std-Dev equals to 1
A = normalize_data;
epsilon = 0.000001;
for iter_cols=1:cols
	std_dev = std(A(:,iter_cols));
	mean_val = mean(A(:,iter_cols));
	A(:,iter_cols) = A(:,iter_cols) - mean_val;
	A(:,iter_cols) = A(:,iter_cols) ./ std_dev;
    
    % Verify normalization
    verify_mean = mean(A(:,iter_cols));
    verify_std = std(A(:,iter_cols));
    assert(abs(verify_mean - 0.0) <= epsilon)
    assert(abs(verify_std - 1.0) <= epsilon) 
end
clear normalize_data
normalize_data = A;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Writing Results to File %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fName = strcat('normalized_data.csv');

fid = fopen(fName,'w');

fprintf('Start writing results to file. \n');

for iter_rows=1:rows
    for iter_cols=1:cols
        feat_val = normalize_data(iter_rows,iter_cols);
        if (iter_cols < cols)
            fprintf(fid,'%f,',feat_val);
        else
            fprintf(fid,'%f',feat_val);
        end
        
    end
    fprintf(fid,'\n');
    
    if(rem(iter_rows,50) == 0)
    fprintf('.');
    end
    
end
fprintf('\n');

fclose(fid);
