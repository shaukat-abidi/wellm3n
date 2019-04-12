function [ ind ] = func_edge_assigment_to_index( u_a, u_b )

% Get the position of vector entry corresponding to the current assignment     
if(u_a == 1 && u_b == 1)
    ind=1;
end

if(u_a == 1 && u_b == 2)
    ind=2;
end

if(u_a == 2 && u_b == 1)
    ind=3;
end

if(u_a == 2 && u_b == 2)
    ind=4;
end

end

