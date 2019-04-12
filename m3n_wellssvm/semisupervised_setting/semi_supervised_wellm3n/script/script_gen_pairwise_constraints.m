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
            if(iter_tok == 1)
                % Marginalize node_b
                node_b = iter_tok+1;
                u_b=1;
                temp_a = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
                
                u_b=2;
                temp_b = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
                
                pc=[pc;temp_a temp_b];                
            else
                % Marginalize node_b
                % Remember, this is the last token of iter_ex thus
                % reversing node assignment (CHECK from notebook)
                tmp_node_a=iter_tok-1;
                tmp_node_b=iter_tok;
                tmp_ua=1;
                tmp_ub=u_a; %FIXED
                temp_a = func_assignment_to_index( tmp_node_a, tmp_node_b, tmp_ua, tmp_ub, iter_ex, offset );
                
                tmp_ua=2;
                temp_b = func_assignment_to_index( tmp_node_a, tmp_node_b, tmp_ua, tmp_ub, iter_ex, offset );
                                
                pc=[pc;temp_a temp_b];                
            end
            
            u_a=2;
            ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
            uc=[uc;ind];
            if(iter_tok == 1)
                % Marginalize node_b
                node_b = iter_tok+1;
                u_b=1;
                temp_a = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
                
                u_b=2;
                temp_b = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
                
                pc=[pc;temp_a temp_b];                
            else
                % Marginalize node_b
                % Remember, this is the last token of iter_ex thus
                % reversing node assignment (CHECK from notebook)
                tmp_node_a=iter_tok-1;
                tmp_node_b=iter_tok;
                tmp_ua=1;
                tmp_ub=u_a; %FIXED
                temp_a = func_assignment_to_index( tmp_node_a, tmp_node_b, tmp_ua, tmp_ub, iter_ex, offset );
                
                tmp_ua=2;
                temp_b = func_assignment_to_index( tmp_node_a, tmp_node_b, tmp_ua, tmp_ub, iter_ex, offset );
                                
                pc=[pc;temp_a temp_b];                
            end
        else
            % T = 2 ... lot-1
            
            % Four marginal constraints
            node_a = iter_tok;
            u_a=1;
            ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
            uc=[uc;ind;ind];
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             % Marginalize node_a EDGE(iter_tok-1,iter_tok)
            tmp_node_a=iter_tok-1;
            tmp_node_b=iter_tok;
            tmp_ua=1;
            tmp_ub=u_a; %FIXED
            temp_a = func_assignment_to_index( tmp_node_a, tmp_node_b, tmp_ua, tmp_ub, iter_ex, offset );
            
            tmp_ua=2;
            temp_b = func_assignment_to_index( tmp_node_a, tmp_node_b, tmp_ua, tmp_ub, iter_ex, offset );
            
            pc=[pc;temp_a temp_b];
            
            % Marginalize node_b EDGE(iter_tok,iter_tok+1)
            node_b = iter_tok+1;
            u_b=1;
            temp_a = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
            
            u_b=2;
            temp_b = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
            
            pc=[pc;temp_a temp_b];
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            node_a = iter_tok;
            u_a=2;
            ind = func_unary_assignment_to_index( node_a, u_a, iter_ex, offset_unary );
            uc=[uc;ind;ind];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             % Marginalize node_a EDGE(iter_tok-1,iter_tok)
            tmp_node_a=iter_tok-1;
            tmp_node_b=iter_tok;
            tmp_ua=1;
            tmp_ub=u_a; %FIXED
            temp_a = func_assignment_to_index( tmp_node_a, tmp_node_b, tmp_ua, tmp_ub, iter_ex, offset );
            
            tmp_ua=2;
            temp_b = func_assignment_to_index( tmp_node_a, tmp_node_b, tmp_ua, tmp_ub, iter_ex, offset );
            
            pc=[pc;temp_a temp_b];
            
            % Marginalize node_b EDGE(iter_tok,iter_tok+1)
            node_b = iter_tok+1;
            u_b=1;
            temp_a = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
            
            u_b=2;
            temp_b = func_assignment_to_index( node_a, node_b, u_a, u_b, iter_ex, offset );
            
            pc=[pc;temp_a temp_b];
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
        
        
    end
end

assert (size(uc,1) == size(pc,1))



