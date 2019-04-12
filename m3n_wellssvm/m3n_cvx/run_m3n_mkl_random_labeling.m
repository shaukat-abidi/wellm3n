clear all
clc

%addpath('/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/functions/');

% RUN MKL (L=1)
base_path='/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/';
fname=strcat(base_path,'gt_0.txt');
sample = func_read_file( fname );

% alternate labels
% for iter_tok=1:sample.examples.lot
%     if (rem(iter_tok,2) == 0)
%         sample.examples.y(iter_tok) = 1;
%     else
%         sample.examples.y(iter_tok) = 2;
%         
%     end
% end

% random number
for iter_tok=1:sample.examples.lot
    ran_num = rand;
    if (rand_num <= 0.5)
        sample.examples.y(iter_tok) = 1;
    else
        sample.examples.y(iter_tok) = 2;
        
    end
end

% write gt file
fid = fopen('params/gt_1.txt','w');
tot_rows = size(sample.examples.x,1);
tot_cols = size(sample.examples.x,2);
for iter_tok=1:sample.examples.lot
    fprintf(fid,'%d 1',sample.examples.y(iter_tok));
    for iter_col=1:tot_cols
        fprintf(fid,' %f',sample.examples.x(iter_tok,iter_col));
    end    
    fprintf(fid,'\n');
end
fclose(fid);
