function [ w_struct, mu, lambda_unary, lambda_pairwise ] = func_m3n_mkl_part_2_no_quadprog( base_path, C )

L_filename=strcat(base_path,'L.txt');
L = dlmread(L_filename);
assert(isscalar(L) == 1);

mu_filename=strcat(base_path,'mu.txt');
mu = dlmread(mu_filename);
assert(length(mu) == L);

% for gt_0.txt
file_id = 0;
sample_filename = strcat(base_path,'gt_',num2str(file_id),'.txt');

fprintf('Reading %s ... ',sample_filename);
sample(1) = func_read_file( sample_filename );
fprintf('Done.\n');
    

for iter_files=2:L
    file_id = iter_files - 1;
    sample_filename = strcat(base_path,'gt_',num2str(file_id),'.txt');
    sample(iter_files) = sample(1);
    
    fprintf('Reading labels from %s ... ',sample_filename);
    gt_labels_from_file = dlmread( sample_filename );
    fprintf('Done.\n');
    
    sample(iter_files).examples.y = gt_labels_from_file;
    
    
end



for repeat_mkl=1:5
    
    fprintf('Iter %d started ---------- mu=[%f ',repeat_mkl,mu(1));
    
    for iter_L=2:L
        fprintf('%f ',mu(iter_L));
    end
    
    fprintf(']------------\n');

    % For L=1
    x = sample(1).examples.x;
    y = sample(1).examples.y;
    
    delta = mu(1) * func_get_delta_loss( y );
    delta_psi = sqrt(mu(1)) * func_get_delta_psi_vec(x,y);
    % H_psi = delta_psi*delta_psi';

    
    for iter_L=2:L
        x = sample(iter_L).examples.x;
        y = sample(iter_L).examples.y;
        
        delta_current = mu(iter_L) * func_get_delta_loss( y );
        delta_psi_current = sqrt(mu(iter_L)) * func_get_delta_psi_vec(x,y);
        % H_psi_current = delta_psi_current * delta_psi_current';
        
        delta = delta + delta_current;
        delta_psi = delta_psi + delta_psi_current;
        % H_psi = H_psi + H_psi_current;
    end
    
    
    % Quick check
    %delta = [delta;delta];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % For SINGLE SEQUENCE ONLY
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    T = sample(1).examples.lot;
    unary_arg = 2*T;
    pairwise_arg = 4*(T-1);
    
    
    % Get pairwise constraints
    [node_marginals, pairwise_summations] = func_get_pairwise_constraints(sample(1));
    tot_pairwise_constraints = size(node_marginals,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CVX Optimization Program
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cvx_begin quiet
    variable lambda_unary(unary_arg) 
    variable lambda_pairwise(pairwise_arg)
    maximize ( (lambda_unary' * delta) - (0.5 * square_pos( norm(delta_psi' * lambda_pairwise) ) ) )
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
    
    for iter_mu=1:L
        tot_norm = tot_norm + ( norm(w_struct(iter_mu).w) ) ;
    end
    for iter_mu=1:L
        mu(iter_mu) = ( norm(w_struct(iter_mu).w) )/tot_norm;
    end
    
    %clearvars -except mu sample L
    % clear lambda_unary lambda_pairwise
    %fprintf('Iter %d Finished ---------- mu(1)=%f, mu(2)=%f, unscaled norm=%f, scaled norm=%f-----------\n\n',repeat_mkl,mu(1),mu(2),unscaled_norm,tot_norm);
end

% RUN script_m3n_mvl_part_1.m



end

