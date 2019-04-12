clear all
clc

% addpath('/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/functions');

% Import trained Model
load('/home/sabidi/Shaukat/m3n_wellssvm/data_repository/BOM_data/Hobart_site_1_oneyear_normalized/inputs_for_algorithms/m3n_C0.001_s1_y1_init_3.mat')
% load('/home/sabidi/Shaukat/m3n_wellssvm/data_repository/BOM_data/Hobart_site_2_oneyear_normalized/inputs_for_algorithms/m3n_C0.001_s2_y1_init_5.mat')

% Test model on
base_path='/home/sabidi/Shaukat/m3n_wellssvm/models_saved/input_samples_for_wellm3n_classification/BOM/';
%fname=strcat(base_path,'s2_y1.txt'); %Data for site-2
fname=strcat(base_path,'s1_y1.txt'); % Data for site-1
sample = func_read_file(fname);

x = sample.examples.x;
lot = size(x,1);
L = size(mu,1);
[emit, trans] = func_get_emit_trans_vector( x, L, mu, w_struct );

% Write emission and transition weights for Viterbi
base_path = '/home/sabidi/Shaukat/m3n_wellssvm/classification/hmmsvm_classification_modified/params/';
func_write_L_LOT_mu_trans_emit_for_classification( base_path, L, lot, trans, emit );

% NOW RUN CLASSIFICATION
fprintf('****************NOW RUN CLASSIFICATION MODULE*******************\n');

