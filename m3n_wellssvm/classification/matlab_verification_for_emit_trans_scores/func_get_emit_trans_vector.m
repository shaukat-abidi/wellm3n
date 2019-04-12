function [ emit, trans ] = func_get_emit_trans_vector( x, L, mu, w_struct )

% For single example only
lot = size(x,1);
D = size(x,2);

trans = zeros(4,1);

% get trans vector
for iter_L = 1:L
    w = w_struct(iter_L).w;
    w_trans = w( (2*D)+1:(2*D)+4 );
    trans = trans + ( sqrt(mu(iter_L)) * w_trans );
end

% get emission vector
emit = zeros(lot,2);

for iter_tok=1:lot
    x_tok = x(iter_tok,:)';
    for iter_L = 1:L
        w = w_struct(iter_L).w;
        w_pos_emission = w( 1:D );
        w_neg_emission = w(D+1:2*D);
        
        emit(iter_tok,1) =  emit(iter_tok,1) + ( sqrt(mu(iter_L)) * (w_pos_emission' * x_tok) );
        emit(iter_tok,2) =  emit(iter_tok,2) + ( sqrt(mu(iter_L)) * (w_neg_emission' * x_tok) );
    end
end


end

