clear all
clc

%addpath('/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/functions/');

% RUN MKL (L=1)
base_path='/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/';
fname=strcat(base_path,'gt_0.txt');
sample = func_read_file( fname );
C = 0.1;
[ lambda_unary, lambda_pairwise ] = func_run_cvx_single_labeling_no_quadprog( sample, C );
fprintf('************HIT ANY KEY TO CONTINUE***************\n');
pause;


% RUN MVL_PART 1
[lot, H, tau, DELTA, best_score] = func_mvl_part_1_L_1( C, lambda_unary, lambda_pairwise, base_path);
fprintf('************RUN VITERBI AND HIT ANY KEY TO CONTINUE***************\n');
pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOW RUN VITERBI THEN RUN PART 2 OF THIS SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CHECK whether we should continue or not
rerun = 0;
rerun  = func_mvl_part_2( base_path, lot, H, tau, DELTA, best_score);

while (rerun == 1)
    
    clear w_struct lambda_unary lambda_pairwise mu
    
    [ w_struct, mu, lambda_unary, lambda_pairwise ] = func_m3n_mkl_part_2_no_quadprog( base_path, C );
    fprintf('************HIT ANY KEY TO CONTINUE***************\n');
    pause;
    
    [ lot, H, tau, DELTA, best_score  ] = func_mvl_part_1( base_path, lambda_unary, lambda_pairwise, C  );
    fprintf('************RUN VITERBI AND HIT ANY KEY TO CONTINUE***************\n');
    pause;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NOW RUN VITERBI THEN RUN PART 2 OF THIS SCRIPT
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % CHECK whether we should continue or not
    rerun  = func_mvl_part_2( base_path, lot, H, tau, DELTA, best_score);
    fprintf('rerun = %d\n',rerun);
    fprintf('************HIT ANY KEY TO CONTINUE***************\n');
    pause;
    
end

% for classification
x = sample.examples.x;
lot = size(x,1);
L = size(mu,1);
[emit, trans] = func_get_emit_trans_vector( x, L, mu, w_struct );

% write to file
base_path = '/home/ssabidi/Shaukat/m3n_wellssvm/classification/hmmsvm_classification_modified/params/';
func_write_L_LOT_mu_trans_emit_for_classification( base_path, L, lot, trans, emit );

% NOW RUN CLASSIFICATION
fprintf('****************NOW RUN CLASSIFICATION MODULE*******************\n');

