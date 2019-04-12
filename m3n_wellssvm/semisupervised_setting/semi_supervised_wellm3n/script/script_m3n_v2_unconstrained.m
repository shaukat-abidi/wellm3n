clear all
close all
clc
fname='matlab_3.txt';
sample = func_read_file( fname );

delta = func_get_delta( sample.examples.y );
psi = func_get_psi_vec(sample.examples.x,sample.examples.y);
H_psi = psi*psi';

T = sample.examples.lot;
unary_arg = 2*T;
pairwise_arg = 4*(T-1);  
C=10;

cvx_begin
variables lambda_unary(unary_arg) lambda_pairwise(pairwise_arg)
maximize ( (lambda_unary' * delta) - (0.5 * lambda_pairwise' * H_psi * lambda_pairwise) )
cvx_end
