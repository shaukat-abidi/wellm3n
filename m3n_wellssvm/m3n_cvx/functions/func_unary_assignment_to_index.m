function [ ind ] = func_unary_assignment_to_index( node_a, u_a, sample_id, offset_unary )
ind = 0;
ind = offset_unary(sample_id) + (2*(node_a-1)) + u_a;
end