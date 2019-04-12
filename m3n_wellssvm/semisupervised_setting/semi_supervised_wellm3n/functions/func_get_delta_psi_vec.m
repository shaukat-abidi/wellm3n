function [ psi ] = func_get_delta_psi_vec( x,y_gt )
% Returns psi = 4*(lot-1) x (2D+4)
% where lot: total frames in x (1,...,lot)
% D: dimension of x_t
% y_gt: ground_truth labeling of x

lot = length(y_gt);
D = size(x,2);
psi=[];

for iter_tok=1:lot-1
    
    % Get psi for the groundtruth assignment
    node_a=iter_tok;
    node_b=iter_tok+1;
    u_a=y_gt(node_a);
    u_b=y_gt(node_b);
    psi_gt = func_psi_edge(x,node_a,node_b,u_a,u_b);
    
    % Get psi for edge (node_a -- node_b) with prediction = (1,1)
    u_a=1;
    u_b=1;
    psi_11 = func_psi_edge(x,node_a,node_b,u_a,u_b);
    % Get psi for edge (node_a -- node_b) with prediction = (1,2)
    u_a=1;
    u_b=2;
    psi_12 = func_psi_edge(x,node_a,node_b,u_a,u_b);
    % Get psi for edge (node_a -- node_b) with prediction = (2,1)
    u_a=2;
    u_b=1;
    psi_21 = func_psi_edge(x,node_a,node_b,u_a,u_b);
    % Get psi for edge (node_a -- node_b) with prediction = (2,2)
    u_a=2;
    u_b=2;
    psi_22 = func_psi_edge(x,node_a,node_b,u_a,u_b);
    
    psi = [psi;(psi_gt-psi_11)';(psi_gt-psi_12)';(psi_gt-psi_21)';(psi_gt-psi_22)'];
    
end

assert(size(psi,1) == 4*(lot-1));
assert(size(psi,2) == (2*D+ 4) );


end

