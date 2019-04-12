y_viterbi = dlmread('/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/viterbi_label.txt');
assert(length(y_viterbi) == lot);
y_viterbi_binarized = func_get_binarized_labeling( y_viterbi );

score_y_star = func_eval_negdual_score( H,tau,DELTA,y_viterbi_binarized );
score_y_star = -1 * score_y_star;

fprintf('score_of_best_groundtruth:%f score_of_viterbi_label:%f\n',best_score,score_y_star);

difference = 0.0;
stopping_criteria = 0.1;
difference = best_score - score_y_star;

fprintf('difference:%f\n',difference);

if(difference>stopping_criteria)
    % increase L
    L = L + 1;
    
    % init mu
    mu = 1/L * ones(L,1);
    
    % write mu, L to file
    func_write_mu_L( mu, L );
    
    % write gt file with labels only (labels returned by Viterbi Algorithm)
    func_write_gt_file( base_path, y_viterbi, L )
    
    % RUN script_m3n_mkl_part_2.m

end