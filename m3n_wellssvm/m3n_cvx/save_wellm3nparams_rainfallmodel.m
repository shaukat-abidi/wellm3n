base_dir='/home/sabidi/Shaukat/m3n_wellssvm/data_repository/BOM_data/Hobart_site_2_oneyear_normalized/inputs_for_algorithms/';
model_name=strcat(base_dir, 'm3n_C', num2str(C), '_', 's2_y1_init_5.mat');
save(model_name,'L','mu','w_struct')
