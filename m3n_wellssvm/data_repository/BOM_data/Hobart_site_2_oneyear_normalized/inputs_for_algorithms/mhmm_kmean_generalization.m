close all
clear all
load('mat_files/data_labels.mat');
clc

%Scale dataset by scale_factor
%scale_factor = 100;
%for i=1:size(data,1)
%    for j=1:size(data,2)
%        data(i,j) = data(i,j) * scale_factor;
%    end    
%end

% For Kmeans Data
kmeans_data = data;
labels = gt_labels;

clear gt_labels data


% For Kmeans Data
nex = 1; %One sample only

%Accumulate total-pos and tot-neg tokens
tot_pos_tokens=0;
tot_neg_tokens=0;
tot_tokens_in_dataset = 0;
for chosen_sample=1:nex
    gt_states = labels;
    tot_tok = length(gt_states);
    tot_tokens_in_dataset = tot_tokens_in_dataset + tot_tok;
    for iter_tok = 1:tot_tok
        if(gt_states(iter_tok) == 1)
            tot_pos_tokens = tot_pos_tokens + 1;
        else
            tot_neg_tokens = tot_neg_tokens + 1;
        end
        
    end
end

% K-Means
K=2;
[IDX, C] = kmeans(kmeans_data, K);
%IDX is the result of kmeans which assigns token to cluster (labels x sample)
%C is the centroid for clusters (cluster# x features)

%K-Means Accuracy
sum_true_predictions = 0;
tot_tokens = tot_tokens_in_dataset;
tp=0;
tn=0;
fp=0;
fn=0;
tok_in_sample=0;
for chosen_sample=1:nex
    inferred_state = IDX;
    gt_states = labels;
    % accuracy (%):
    correct_prediction = sum((gt_states == inferred_state));
    sum_true_predictions = sum_true_predictions + correct_prediction;
    tok_in_sample = length(inferred_state);
    for iter_tok=1:tok_in_sample
        if(inferred_state(iter_tok) == 1 && gt_states(iter_tok) == 1)
            tp = tp + 1;
        end
        if(inferred_state(iter_tok) == 1 && gt_states(iter_tok) == 2)
            fp = fp + 1;
        end
        if(inferred_state(iter_tok) == 2 && gt_states(iter_tok) == 1)
            fn = fn + 1;
        end
        if(inferred_state(iter_tok) == 2 && gt_states(iter_tok) == 2)
            tn = tn + 1;
        end
    end
    
end
% % accuracy (%):
accuracy_kmeans = sum_true_predictions / tot_tokens * 100;
fprintf('tot-tokens = %d tot-pos-tokens = %d tot-neg-tokens = %d\n',tot_tokens_in_dataset,tot_pos_tokens,tot_neg_tokens);
fprintf('tp=%d,tn=%d,fp=%d,fn=%d\n',tp,tn,fp,fn);
fprintf('%d %d %d %d\n',tp,tn,fp,fn);
fprintf('KMeans Accuracy = %f\n',accuracy_kmeans);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1_score = (2*precision*recall)/(precision+recall);
fprintf('precision = %f , recall = %f , f1_score = %f \n\n\n',precision,recall,f1_score);

if(accuracy_kmeans < 40)
    [rows,cols] = size(IDX);
    
    % Invert IDX
    for iter_rows=1:rows
        for iter_cols=1:cols
            if(IDX(iter_rows,iter_cols) == 1)
                IDX(iter_rows,iter_cols) = 2;
            else
                IDX(iter_rows,iter_cols) = 1;
            end
        end
    end
    
    %K-Means Accuracy
    sum_true_predictions = 0;
    tot_tokens = tot_tokens_in_dataset;
    tp=0;
    tn=0;
    fp=0;
    fn=0;
    tok_in_sample=0;
    
    for chosen_sample=1:nex
        inferred_state = IDX;
        gt_states = labels;
        % accuracy (%):
        correct_prediction = sum((gt_states == inferred_state));
        sum_true_predictions = sum_true_predictions + correct_prediction;
        tok_in_sample = length(inferred_state);
        for iter_tok=1:tok_in_sample
            if(inferred_state(iter_tok) == 1 && gt_states(iter_tok) == 1)
                tp = tp + 1;
            end
            if(inferred_state(iter_tok) == 1 && gt_states(iter_tok) == 2)
                fp = fp + 1;
            end
            if(inferred_state(iter_tok) == 2 && gt_states(iter_tok) == 1)
                fn = fn + 1;
            end
            if(inferred_state(iter_tok) == 2 && gt_states(iter_tok) == 2)
                tn = tn + 1;
            end
        end
        
    end
    % % accuracy (%):
    accuracy_kmeans = sum_true_predictions / tot_tokens * 100;
    fprintf('(RESULT INVERTED)tot-tokens = %d tot-pos-tokens = %d tot-neg-tokens = %d\n',tot_tokens_in_dataset,tot_pos_tokens,tot_neg_tokens);
    fprintf('(RESULT INVERTED)tp=%d,tn=%d,fp=%d,fn=%d\n',tp,tn,fp,fn);
    fprintf('(RESULT INVERTED)KMeans Accuracy = %f\n',accuracy_kmeans);
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    f1_score = (2*precision*recall)/(precision+recall);
    fprintf('(RESULT INVERTED)precision = %f , recall = %f , f1_score=%f \n',precision,recall,f1_score);

end

% clear X Y

% X = kmeans_data;
% Y = inferred_state;
% fid = fopen('kmeans_init_1.txt','w');
% qid = 1;
% 
% for iter_rows=1:size(X,1)
%     
%     fprintf(fid,'%d qid:1',Y(iter_rows) );
%     
%     for iter_cols=1:size(X,2)
%         fprintf(fid,' %d:%f',iter_cols, X(iter_rows,iter_cols) );        
%     end
%     
%     fprintf(fid,'\n');
% 
% end
% fclose(fid);
