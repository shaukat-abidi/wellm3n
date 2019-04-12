clear all
clc

sample = func_read_file( 'a1.txt' );
lot = sample.examples.lot;
x = sample.examples.x;

L=3;
D = size(x,2);

mu(1)=0.650902;
mu(2)=0.349098;
mu(3)=0.000000;


w = dlmread('w.txt');

% hard coated w's
w1_trans = w(4:7);
w1_pos_emission = w(41:72);
w1_neg_emission = w(73:104);

w2_trans = w(108:111);
w2_pos_emission = w(145:176);
w2_neg_emission = w(177:208);

w3_trans = w(212:215);
w3_pos_emission = w(249:280);
w3_neg_emission = w(281:312);

w_struct = repmat(struct('w',[]),L,1);
w_struct(1).w = [w1_pos_emission;w1_neg_emission;w1_trans];
w_struct(2).w = [w2_pos_emission;w2_neg_emission;w2_trans];
w_struct(3).w = [w3_pos_emission;w3_neg_emission;w3_trans];

[emit_check, trans_check] = func_get_emit_trans_vector( x, L, mu, w_struct );

% prepare trans matrix
trans = ( sqrt(mu(1)) * w1_trans ) + ( sqrt(mu(2)) * w2_trans ) + ( sqrt(mu(3)) * w3_trans );

emit = zeros(lot,2);
for iter_tok=1:lot
    x_tok = x(iter_tok,:)';
    emit(iter_tok,1) =  ( sqrt(mu(1)) * (w1_pos_emission' * x_tok) ) + ( sqrt(mu(2)) * (w2_pos_emission' * x_tok) ) + ( sqrt(mu(3)) * (w3_pos_emission' * x_tok) );
    emit(iter_tok,2) =  ( sqrt(mu(1)) * (w1_neg_emission' * x_tok) ) + ( sqrt(mu(2)) * (w2_neg_emission' * x_tok) ) + ( sqrt(mu(3)) * (w3_neg_emission' * x_tok) );
end

% write to file
%base_path = '/home/ssabidi/Shaukat/m3n_wellssvm/classification/matlab_verification_for_emit_trans_scores/files_written/';
%func_write_L_LOT_mu_trans_emit_for_classification( base_path, L, lot, trans, emit );
