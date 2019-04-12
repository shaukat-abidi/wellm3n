% addpath(genpath('/home/sabidi/Shaukat/WellSSVM/implementation/EM/MurphyToolbox/HMMall'))

%true labels=1,false labels=2
% close all
clear all
clc

% Load model file
load('/home/sabidi/Shaukat/m3n_wellssvm/data_repository/BOM_data/Hobart_site_1_oneyear_normalized/inputs_for_algorithms/kmeans_s1_y1_init_3.mat');
%load('/home/sabidi/Shaukat/m3n_wellssvm/data_repository/BOM_data/Hobart_site_2_oneyear_normalized/inputs_for_algorithms/kmeans_s2_y1_init_5.mat');

% Load validation dataset
% load('/home/sabidi/Shaukat/m3n_wellssvm/data_repository/BOM_data/Hobart_site_2_oneyear_normalized/inputs_for_algorithms/mat_files/data_labels.mat');
load('/home/sabidi/Shaukat/m3n_wellssvm/data_repository/BOM_data/Hobart_site_1_oneyear_normalized/inputs_for_algorithms/mat_files/data_labels.mat');

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

% Classify using Kmeans Centroids
IDX = kmeans_classify(C,kmeans_data);

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
fprintf('KMeans Accuracy = %f\n',accuracy_kmeans);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1_score = (2*precision*recall)/(precision+recall);
fprintf('precision = %f , recall = %f , f1_score = %f \n\n\n',precision,recall,f1_score);

if(accuracy_kmeans > 0)
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
