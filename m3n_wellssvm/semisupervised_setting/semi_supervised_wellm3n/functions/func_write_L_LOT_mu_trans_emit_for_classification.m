function func_write_L_LOT_mu_trans_emit_for_classification( base_path, L, lot, trans, emit )

filename_L = strcat(base_path,'L.txt');
filename_lot = strcat(base_path,'lot.txt');
filename_trans = strcat(base_path,'trans.txt');
filename_emit_pos = strcat(base_path,'emit_pos.txt');
filename_emit_neg = strcat(base_path,'emit_neg.txt');


fid_L = fopen(filename_L,'w');
fprintf(fid_L,'%d\n',L);
fclose(fid_L);

fid_lot = fopen(filename_lot,'w');
fprintf(fid_lot,'%d\n',lot);
fclose(fid_lot);

fid_trans = fopen(filename_trans,'w');
fprintf(fid_trans,'%f\n',trans(1));
fprintf(fid_trans,'%f\n',trans(2));
fprintf(fid_trans,'%f\n',trans(3));
fprintf(fid_trans,'%f\n',trans(4));
fclose(fid_trans);

fid_emit_pos = fopen(filename_emit_pos,'w');
for iter_tok=1:lot
    fprintf(fid_emit_pos,'%f\n',emit(iter_tok,1));
end
fclose(fid_emit_pos);

fid_emit_neg = fopen(filename_emit_neg,'w');
for iter_tok=1:lot
    fprintf(fid_emit_neg,'%f\n',emit(iter_tok,2));
end
fclose(fid_emit_neg);

end

