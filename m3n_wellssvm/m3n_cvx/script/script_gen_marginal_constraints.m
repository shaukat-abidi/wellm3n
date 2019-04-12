clear all
close all
clc
fname='matlab_3.txt';
sample = func_read_file( fname );


% Offset for pairwise constraints
offset=zeros(sample.tot_ex,1);
end_ind = 0;

for iter_ex=1:sample.tot_ex
    start_ind=end_ind;
    offset(iter_ex) = start_ind;
    end_ind=start_ind+4*(sample.examples(iter_ex).lot-1);
end

% Offset for unary constraints
offset_unary = zeros(sample.tot_ex,1);
end_ind = 0;

for iter_ex=1:sample.tot_ex
    start_ind=end_ind;
    offset_unary(iter_ex) = start_ind;
    end_ind=start_ind+2*(sample.examples(iter_ex).lot);
end

uc=[];
pc=[];

% Gotten the matrix for unary constraints
for iter_ex=1:sample.tot_ex
    lot = sample.examples(iter_ex).lot;
    for iter_tok=1:lot
        % get unary index
        if(iter_tok == 1 || iter_tok == lot)
            % Just two marginal constraints
            node_a = iter_tok;
            u_a=1;
            ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
            uc=[uc;ind];
            %fprintf('sample_id:%d -- node_a:%d -- u_a:%d -- ind:%d\n',iter_ex,node_a,u_a,ind);
            
            u_a=2;
            ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
            uc=[uc;ind];
            %fprintf('sample_id:%d -- node_a:%d -- u_a:%d -- ind:%d\n',iter_ex,node_a,u_a,ind);
        else
            % Four marginal constraints
            node_a = iter_tok;
            u_a=1;
            ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
            uc=[uc;ind;ind];
            %fprintf('sample_id:%d -- node_a:%d -- u_a:%d -- ind:%d\n',iter_ex,node_a,u_a,ind);
            
            node_a = iter_tok;
            u_a=2;
            ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
            uc=[uc;ind;ind];
            %fprintf('sample_id:%d -- node_a:%d -- u_a:%d -- ind:%d\n',iter_ex,node_a,u_a,ind);
        end
        
        
    end
    %fprintf('\n\n')
end



