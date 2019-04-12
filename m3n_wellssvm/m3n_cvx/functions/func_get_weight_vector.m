function [ w_struct ] = func_get_weight_vector( sample, lambda_pairwise, mu, L )

D = size(sample(1).examples.x,2);
vec_psi_dim = (2*D) + 4;
%w_buffer = zeros(1,vec_psi_dim);
w_struct = repmat(struct('w',zeros(vec_psi_dim,1)),L,1);
x = sample(1).examples.x;

for iter_L=1:L
    y = sample(iter_L).examples.y;
    w_buffer = zeros(1,vec_psi_dim);

    delta_psi = sqrt(mu(iter_L)) * func_get_delta_psi_vec(x,y);
    %delta_psi = func_get_delta_psi_vec(x,y);
    
    rows = size(delta_psi,1);
    for iter_rows=1:rows
        w_buffer = w_buffer + (lambda_pairwise(iter_rows)*delta_psi(iter_rows,:));
    end
    
    w_struct(iter_L).w = w_buffer';
    w_struct(iter_L).w = w_struct(iter_L).w * sqrt(mu(iter_L));
    assert(length(w_struct(iter_L).w) == vec_psi_dim);
end

end

