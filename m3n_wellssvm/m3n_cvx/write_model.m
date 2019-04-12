clear all
clc

% addpath('/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/functions');

% Import trained Model

% Import model trained on A1
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/a1/mosek_noQP/kmeans_init_2/m3n_wellssvm.mat')
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/a1/mosek_noQP/kmeans_init_3/m3n_wellssvm.mat')
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/a1/mosek_noQP/kmeans_init_4/m3n_wellssvm.mat')
load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/a1/mosek_noQP/kmeans_init_2/c0.01/m3n_wellssvm.mat')


% Import model trained on A2
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/a2/mosek_qp/kmeans_init_1/m3n_wellssvm.mat')

% Import model trained on A3
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/a3/mosek_noQP/kmeans_init_1/m3n_wellssvm.mat')

% Import model trained on B1
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/b1/mosek_QP/kmeans_init_1/m3n_wellssvm.mat')

% Import model trained on B3
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/b3/mosek_qp/kmeans_init_1/m3n_wellssvm.mat')

% Import model trained on C1
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/c1/mosek_qp/kmeans_init_3/m3n_wellssvm.mat')

% Import model trained on C3
% load('/home/sabidi/Shaukat/m3n_wellssvm/models_saved/c3/mosek_no_qp/kmeans_init_4/m3n_wellssvm.mat')

% Test model on
base_path='/home/sabidi/Shaukat/m3n_wellssvm/models_saved/input_samples_for_wellm3n_classification/';
fname=strcat(base_path,'c3.txt');
sample = func_read_file( fname );

x = sample.examples.x;
lot = size(x,1);
L = size(mu,1);
[emit, trans] = func_get_emit_trans_vector( x, L, mu, w_struct );

% Write emission and transition weights for Viterbi
base_path = '/home/sabidi/Shaukat/m3n_wellssvm/classification/hmmsvm_classification_modified/params/';
func_write_L_LOT_mu_trans_emit_for_classification( base_path, L, lot, trans, emit );

% NOW RUN CLASSIFICATION
fprintf('****************NOW RUN CLASSIFICATION MODULE*******************\n');

