clear all
close all
clc

%addpath('/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/functions/');

fname='/home/ssabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/gt_0.txt';
sample = func_read_file( fname );

C=0.01;
[ lambda_unary, lambda_pairwise ] = func_run_cvx_single_labeling( sample, C );
