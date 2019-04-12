function [ psi_vector ] = func_psi_edge( x,node_a,node_b,u_a,u_b )

%            ---  node_a ----edge------ node_b ----
% x is the example
% node_a is the number of a^{th} node
% node_b is the number of b^{th} node
% u_a is the assignment of a^{th} node
% u_b is the assignment of b^{th} node

% func_psi_edge( x,node_a,node_b,u_a,u_b )will return psi_vector of
% dimension 2D+4. This function encodes emission of node_b and transition
% of node_a to node_b. If node_a turns out to be the first node of a
% sequence, then emissions of node_a and node_b will be included together.

% psi_vector = [emission_a,emission_b,t_11,t_12,t_21,t_22]'

tot_trans_feat = 4;
tot_emission_feat = size(x,2);
sizePsi = tot_trans_feat + (2*tot_emission_feat);
% psi_vector = zeros(sizePsi,1);

% Store emission features
x_t = x(node_b,:)';
if (u_b == 1)
    emit_c1 = x_t;
    emit_c2 = zeros(tot_emission_feat,1);
else
    emit_c2 = x_t;
    emit_c1 = zeros(tot_emission_feat,1);
end

% Special condition 
if (node_a == 1)
    x_t = x(node_a,:)';    
    if (u_a == 1)
        emit_c1 = emit_c1 + x_t;
    else
        emit_c2 = emit_c2 + x_t;
    end
end

% Get transition features    
if(u_a == 1 && u_b == 1)
    trans_feat=[1;0;0;0];
elseif(u_a == 1 && u_b == 2)
    trans_feat=[0;1;0;0];
elseif(u_a == 2 && u_b == 1)
    trans_feat=[0;0;1;0];
else
    trans_feat=[0;0;0;1];
end

% psi_vec
psi_vector=[emit_c1;emit_c2;trans_feat];
assert(length(psi_vector) == sizePsi);

end
