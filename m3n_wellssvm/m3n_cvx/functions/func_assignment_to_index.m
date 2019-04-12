function [ ind ] = func_assignment_to_index( node_a, node_b, u_a, u_b, sample_id, offset )
ind = 0;
ind = ( 4*(node_a - 1) ) + func_edge_assigment_to_index( u_a, u_b ) + (offset(sample_id));
end
