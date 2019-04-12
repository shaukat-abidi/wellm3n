function [ lot, H, tau, DELTA, best_score  ] = func_mvl_part_1( base_path, lambda_unary, lambda_pairwise, C  )

% MVL Part_1
% Output: lot, H, tau, DELTA
% Input: base_path, lambda_unary, lambda_pairwise, C

% clear all
%clc

% Generate file path
L_filename=strcat(base_path,'L.txt');
L = dlmread(L_filename);
assert(isscalar(L) == 1);

struct_y_binarized = repmat(struct('binarized_labeling',[]),L,1);
% for gt_0.txt
file_id = 0;
sample_filename = strcat(base_path,'gt_',num2str(file_id),'.txt');

fprintf('Reading %s ... ',sample_filename);
gt_sample(1) = func_read_file( sample_filename );
labeling_y = gt_sample(1).examples(1).y; % we have just one example
struct_y_binarized(1).binarized_labeling = func_get_binarized_labeling( labeling_y );
fprintf('Done.\n');
    
% Read ground-truth 
for iter_files=2:L
    file_id = iter_files - 1;
    sample_filename = strcat(base_path,'gt_',num2str(file_id),'.txt');
    gt_sample(iter_files) = gt_sample(1);
    
    fprintf('Reading labels from %s ... ',sample_filename);
    gt_labels_from_file = dlmread( sample_filename );
    fprintf('Done.\n');
    
    gt_sample(iter_files).examples.y = gt_labels_from_file;
    labeling_y = gt_sample(iter_files).examples(1).y; % we have just one example
    struct_y_binarized(iter_files).binarized_labeling = func_get_binarized_labeling( labeling_y );
    
end


% Get H, tau and DELTA
sample = gt_sample(1);
[H, tau, DELTA] = func_get_all_params_dual(sample, lambda_unary, lambda_pairwise, C);

% Get the best labeling among the groundtruth (working_set = C_w)
score = func_eval_negdual_score( H,tau,DELTA,struct_y_binarized(1).binarized_labeling );
score = -1 * score;

best_score = score;
best_ind = 1;
fprintf('gt_labeling: %d score:%f\n',1,score);

for iter_gt_labelings=2:L
    score = func_eval_negdual_score( H,tau,DELTA,struct_y_binarized(iter_gt_labelings).binarized_labeling );
    score = -1 * score;
    fprintf('gt_labeling: %d score:%f\n', iter_gt_labelings, score);
    
    if(score < best_score)
        best_score = score;
        best_ind = iter_gt_labelings;
    end
end

fprintf('best_gt_labeling: %d best_score:%f\n', best_ind, best_score);

% Generate Param_R
y_best = struct_y_binarized(best_ind).binarized_labeling;
paramR = func_get_paramR( H,tau,DELTA,y_best );

% write paramR to file for viterbi
lot = sample.examples.lot;
func_write_paramR_for_viterbi( paramR, lot );






end

