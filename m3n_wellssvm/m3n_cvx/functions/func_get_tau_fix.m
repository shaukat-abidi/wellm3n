function [ tau_fix ] = func_get_tau_fix( sample, lambda_unary, lambda_pairwise, C )

tot_ex = size(sample.tot_ex);
D = size(sample.examples(1).x,2);
tau_fix = zeros( (2*D) + 4 , 1);
[ offset_unary, offset_pairwise ] = func_get_unary_pairwise_offset( sample );
    
for iter_ex=1:tot_ex
    current_lot = sample.examples(iter_ex).lot;
    x = sample.examples(iter_ex).x;
    for iter_tok=1:current_lot-1
        node_a = iter_tok;
        node_b = iter_tok + 1;
        
        % Initialize kappa 
        kappa = zeros(6,1);

        % get hj_cd
        hj_cd = func_get_hj_cd_matrix(x,node_a,node_b,C);

        u_a = 1;
        u_b = 1;
        u_ab = [1;0;1;0;0;0];
        pairwise_ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset_pairwise );
        kappa = kappa + (lambda_pairwise(pairwise_ind) * u_ab);
        
        u_a = 1;
        u_b = 2;
        u_ab = [0;1;0;1;0;0];
        pairwise_ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset_pairwise );
        kappa = kappa + (lambda_pairwise(pairwise_ind) * u_ab);

        
        u_a = 2;
        u_b = 1;
        u_ab = [1;0;0;0;1;0];
        pairwise_ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset_pairwise );
        kappa = kappa + (lambda_pairwise(pairwise_ind) * u_ab);

        
        u_a = 2;
        u_b = 2;
        u_ab = [0;1;0;0;0;1];
        pairwise_ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset_pairwise );        
        kappa = kappa + (lambda_pairwise(pairwise_ind) * u_ab);
        
        tau_fix = tau_fix + (hj_cd' * kappa);
        
    end
    % for first node
    node_a = 1;
    u_a = 1;
    unary_ind_1 = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
    
    u_a = 2;
    unary_ind_2 = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
    
    x_tok_column_vec = x(node_a,:)';
    
    tau_fix = tau_fix + [lambda_unary(unary_ind_1)*C*x_tok_column_vec;lambda_unary(unary_ind_2)*C*x_tok_column_vec;0;0;0;0];
end

end

