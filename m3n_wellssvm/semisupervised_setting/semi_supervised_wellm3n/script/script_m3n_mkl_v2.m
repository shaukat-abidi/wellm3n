clear all
close all
clc

L = 3;
mu = zeros(L,1);


mu(1) = 1/3;
mu(2) = 1/3;
mu(3) = 1/3;

fname='params/gt_0.txt';
sample(1) = func_read_file( fname );

fname='params/gt_1.txt';
sample(2) = func_read_file( fname );

fname='params/gt_2.txt';
sample(3) = func_read_file( fname );


for repeat_mkl=1:14
    
    fprintf('Iter %d started ---------- mu=[%f %f %f]-----------\n',repeat_mkl,mu(1),mu(2),mu(3));
    
    % For L=1
    x = sample(1).examples.x;
    y = sample(1).examples.y;
    
    delta = mu(1) * func_get_delta_loss( y );
    delta_psi = sqrt(mu(1)) * func_get_delta_psi_vec(x,y);
    
    
    for iter_L=2:L
        x = sample(iter_L).examples.x;
        y = sample(iter_L).examples.y;
        
        delta_buffer = mu(iter_L) * func_get_delta_loss( y );
        delta_psi_buffer = sqrt(mu(iter_L)) * func_get_delta_psi_vec(x,y);
        
        
        delta = delta + delta_buffer;
        delta_psi = delta_psi + delta_psi_buffer;
    end
    
    H_psi = delta_psi*delta_psi';
    
    % Quick check
    %delta = [delta;delta];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % For SINGLE SEQUENCE ONLY
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    T = sample(1).examples.lot;
    unary_arg = 2*T;
    pairwise_arg = 4*(T-1);
    C=1;
    
    % Get pairwise constraints
    [node_marginals, pairwise_summations] = func_get_pairwise_constraints(sample(1));
    tot_pairwise_constraints = size(node_marginals,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CVX Optimization Program
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cvx_begin quiet
    variables lambda_unary(unary_arg) lambda_pairwise(pairwise_arg)
    maximize ( (lambda_unary' * delta) - (0.5 * lambda_pairwise' * H_psi * lambda_pairwise) )
    subject to
    for iter_const=1:tot_pairwise_constraints
        (lambda_pairwise(pairwise_summations(iter_const,1)) + lambda_pairwise(pairwise_summations(iter_const,2))) >= lambda_unary(node_marginals(iter_const));
        (lambda_pairwise(pairwise_summations(iter_const,1)) + lambda_pairwise(pairwise_summations(iter_const,2))) <= lambda_unary(node_marginals(iter_const));
    end
    lambda_pairwise >= 0;
    for iter_tok=1:T
        lambda_unary(2*iter_tok - 1) + lambda_unary(2*iter_tok) >= C;
        lambda_unary(2*iter_tok - 1) + lambda_unary(2*iter_tok) <= C;
    end
    cvx_end
    
    
    w_struct  = func_get_weight_vector( sample, lambda_pairwise, mu, L );
    tot_norm = 0;
    
    %     unscaled_norm=0;
    %     for iter_mu=1:L
    %         unscaled_norm = unscaled_norm + norm(w_struct(iter_mu).w) ;
    %     end
    %
%         for iter_mu=1:L
%             % Scale Norm
%             w_struct(iter_mu).w = mu(iter_L) * w_struct(iter_mu).w ;
%         end
    
    for iter_mu=1:L
        tot_norm = tot_norm + ( mu(iter_mu) * norm(w_struct(iter_mu).w) ) ;
    end
    for iter_mu=1:L
        mu(iter_mu) = ( mu(iter_mu) * norm(w_struct(iter_mu).w) )/tot_norm;
    end
    
    %clearvars -except mu sample L
    % clear lambda_unary lambda_pairwise
    %fprintf('Iter %d Finished ---------- mu(1)=%f, mu(2)=%f, unscaled norm=%f, scaled norm=%f-----------\n\n',repeat_mkl,mu(1),mu(2),unscaled_norm,tot_norm);
end


