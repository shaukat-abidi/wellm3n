clear all
close all
clc
fname='matlab_3.txt';
sample = func_read_file( fname );

node_a=4;
node_b=5;
u_a=1;
u_b=1;


offset=zeros(sample.tot_ex,1);
end_ind = 0;

for iter_ex=1:sample.tot_ex
    start_ind=end_ind;
    offset(iter_ex) = start_ind;
    end_ind=start_ind+4*(sample.examples(iter_ex).lot-1);
end

offset_unary = zeros(sample.tot_ex,1);
end_ind = 0;

for iter_ex=1:sample.tot_ex
    start_ind=end_ind;
    offset_unary(iter_ex) = start_ind;
    end_ind=start_ind+2*(sample.examples(iter_ex).lot);
end


%psi = func_psi_edge(sample.examples(1).x,node_a,node_b,u_a,u_b);

% Printing indices for unary assignments
for iter_ex=1:sample.tot_ex
    for iter_tok=1:sample.examples(iter_ex).lot
        node_a=iter_tok;
        
        u_a=1;
        ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
        fprintf('sample_id:%d -- node_a:%d -- u_a:%d -- ind:%d\n',iter_ex,node_a,u_a,ind);
        
        u_a=2;
        ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
        fprintf('sample_id:%d -- node_a:%d -- u_a:%d -- ind:%d\n',iter_ex,node_a,u_a,ind);
        
        
    end
    fprintf('\n\n')
end



% Printing indices for pairwise assignments
for iter_ex=1:sample.tot_ex
    for iter_tok=1:sample.examples(iter_ex).lot-1
        node_a=iter_tok;
        node_b=iter_tok+1;
        
        u_a=1;
        u_b=1;
        ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
        fprintf('sample_id:%d -- node_a:%d -- node_b:%d -- u_a:%d -- u_b:%d ind:%d\n',iter_ex,node_a,node_b,u_a,u_b,ind);
        
        u_a=1;
        u_b=2;
        ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
        fprintf('sample_id:%d -- node_a:%d -- node_b:%d -- u_a:%d -- u_b:%d ind:%d\n',iter_ex,node_a,node_b,u_a,u_b,ind);
        
        u_a=2;
        u_b=1;
        ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
        fprintf('sample_id:%d -- node_a:%d -- node_b:%d -- u_a:%d -- u_b:%d ind:%d\n',iter_ex,node_a,node_b,u_a,u_b,ind);
        
        u_a=2;
        u_b=2;
        ind = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
        fprintf('sample_id:%d -- node_a:%d -- node_b:%d -- u_a:%d -- u_b:%d ind:%d\n',iter_ex,node_a,node_b,u_a,u_b,ind);
        
    end
    fprintf('\n\n')
end

