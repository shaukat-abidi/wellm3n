clear all
close all
clc

%addpath('/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/functions/');

%fname='data_synthetic_5_frames.txt';
fname='/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/gt_0.txt';
sample = func_read_file( fname );

%delta = func_get_delta( sample.examples.y );

%psi = func_get_psi_vec(sample.examples.x,sample.examples.y)

%[node_marginals, pairwise_summations] = func_get_pairwise_constraints(sample);
%tot_pairwise_constraints = size(node_marginals,1);

%y_binarized = func_get_binarized_labeling( sample.examples.y );

% Check Delta
%lot = 5;
%lambda_unary=[1;2;30;40;5;6;70;80;9;10];
%DELTA = func_get_delta_dual( lambda_unary,lot );

% get h_matrix
%x = sample.examples.x;
%C=100;
%h_matrix = func_get_h_matrix(x,C);

% get hj_cd
% x = sample.examples.x;
% C=1;
% node_c=4;
% node_d=5;
% hj_cd = func_get_hj_cd_matrix(x,node_c,node_d,C);

% get kappa_sum for edge ab
%[ offset_unary, offset_pairwise ] = func_get_unary_pairwise_offset( sample );
% node_a = 9;
% node_b = 10;
% u_a = 2;
% u_b = 2;
% iter_ex=1;
% 
% unary_ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
% pairwise_ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset_pairwise );

% Getting tau
C=10;
[ lambda_unary, lambda_pairwise ] = func_run_cvx_single_labeling( sample, C );
%tau_fix  = func_get_tau_fix( sample, lambda_unary, lambda_pairwise, C  );
%tau = func_get_tau(sample, tau_fix, C);

% Get Delta (JUST FOR 1 example)
% lot = sample.examples(1).lot;
% DELTA = func_get_delta_dual( lambda_unary,lot );

% Get H (one example only)
% x = sample.examples(1).x;
% h_matrix = func_get_h_matrix(x,C);
% H = h_matrix * h_matrix';

%[H, tau, DELTA] = func_get_all_params_dual(sample, lambda_unary, lambda_pairwise, C);





