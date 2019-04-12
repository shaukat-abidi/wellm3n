#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <time.h> 
#include <float.h>
#include "cblas.h"
#include <assert.h>

# define FNUM    int32_t     /* the type used for storing feature ids */
# define FVAL    float       /* the type used for storing feature values */
# define MAXFEATNUM 99999999 /* maximum feature number (must be in
			  	valid range of FNUM type and long int!) */

#ifdef NAN
/* NAN is supported */
#endif
#ifdef INFINITY
/* INFINITY is supported */
#endif

typedef struct type_supervised_labels{
	long tot_entries;
	long *seq_id;
	long *frame_id;
}TYPE_SUPLABELS;

typedef struct type_binlabels{
	long *bin_labels;
}TYPE_BINLABELS;
  	
typedef struct type_matrix{
	long rows;/*(Total Rows)1-Based*/
	long cols;/*(Total Columns)1-Based*/
	double *data;
}TYPE_MATRIX;

typedef struct type_h{
	/*for each example like h_1,h_2,...h_n*/
	long seq_id;/*0-Based*/
	long tot_tokens;/*1-Based*/
	TYPE_MATRIX *h_matrix;
}TYPE_H;

typedef struct type_block{
	/*for each pair of example like h_11,h_12,...h_1n down to h_nn*/
	long i;/*first example(0-Based)*/
	long j;/*second example(0-Based)*/
	long i_tokens;/*1-Based*/
	long j_tokens;/*1-Based*/
	TYPE_MATRIX *h_ij;
}TYPE_BLOCK;

typedef struct word {
  FNUM    wnum;	               /* word number */
  FVAL    weight;              /* word weight */
} WORD;

typedef struct svector {
  WORD    *words;              /* The features/values in the vector by
				  increasing feature-number. Feature
				  numbers that are skipped are
				  interpreted as having value zero. */
  double  twonorm_sq;          /* The squared euclidian length of the
                                  vector. Used to speed up the RBF kernel. */
  char    *userdefined;        /* You can put additional information
				  here. This can be useful, if you are
				  implementing your own kernel that
				  does not work with feature/values
				  representations (for example a
				  string kernel). By default,
				  svm-light will put here the string
				  after the # sign from each line of
				  the input file. */
  long    kernel_id;           /* Feature vectors with different
				  kernel_id's are orthogonal (ie. the
				  feature number do not match). This
				  is used for computing component
				  kernels for linear constraints which
				  are a sum of several different
				  weight vectors. (currently not
				  implemented). */
  struct svector *next;        /* Let's you set up a list of SVECTOR's
				  for linear constraints which are a
				  sum of multiple feature
				  vectors. List is terminated by
				  NULL. */
  double  factor;              /* Factor by which this feature vector
				  is multiplied in the sum. */
} SVECTOR;

typedef struct doc {
  long    docnum;              /* Document ID. This has to be the position of 
                                  the document in the training set array. */
  long    queryid;             /* for learning rankings, constraints are 
				  generated for documents with the same 
				  queryID. */
  double  costfactor;          /* Scales the cost of misclassifying this
				  document by this factor. The effect of this
				  value is, that the upper bound on the alpha
				  for this example is scaled by this factor.
				  The factors are set by the feature 
				  'cost:<val>' in the training data. */
  long    slackid;             /* Index of the slack variable
				  corresponding to this
				  constraint. All constraints with the
				  same slackid share the same slack
				  variable. This can only be used for
				  svm_learn_optimization. */
  long    kernelid;            /* Position in gram matrix where kernel
				  value can be found when using an
				  explicit gram matrix
				  (i.e. kernel_type=GRAM). */
  SVECTOR *fvec;               /* Feature vector of the example. The
				  feature vector can actually be a
				  list of feature vectors. For
				  example, the list will have two
				  elements, if this DOC is a
				  preference constraint. The one
				  vector that is supposed to be ranked
				  higher, will have a factor of +1,
				  the lower ranked one should have a
				  factor of -1. */
} DOC;

typedef struct pattern {
  /* this defines the x-part of a training example, e.g. the structure
     for storing a natural language sentence in NLP parsing */
  DOC     **tokens;
  long    length;
} PATTERN;

typedef struct label {
  /* this defines the y-part (the label) of a training example,
     e.g. the parse tree of the corresponding sentence. */
  long    *labels;
  long    *extended_labels;
  long    length;
  long    loet;//length-of-extended-labels
} LABEL;

typedef struct example {  /* an example is a pair of pattern and label */
  PATTERN x;
  LABEL y;
} EXAMPLE;

typedef struct sample { /* a sample is a set of examples */
  int     n;             /* n is the total number of examples */
  long    total_features; /* each token consists of (total_features) numbers */
  EXAMPLE *examples;
} SAMPLE;

typedef struct type_recover_labels {
  long sample_id;
  long tot_tokens;
  long *labels_recovered;
} TYPE_RECOVER_LABELS;

typedef struct type_binarized_labels {
  long sample_id;
  long tot_tokens;
  long tot_binary_entries;
  long *binary_labels;
} TYPE_BINARIZED_LABELS;

typedef struct type_common_input {
  
  TYPE_BINLABELS *ws_labels;
  TYPE_H *hi; 
  TYPE_H *hi_transpose;
  long *base_tab;
  long *lot;
  float *alpha;
  
  long tot_ex;
  long tot_ws;
  long tot_binfeats;
  long size_psi_min;
  double alphasum;
  
} TYPE_COMMON_INPUT;

typedef struct bvpath {
  /*Balanced-Viterbi Token representation*/
  long *path; /* Valid path from path[1] to path[path_length] */
  long isValid;
  long path_length; /*Inshort, it is 1-based index of current-token*/
  float score;
} BVPATH;

typedef struct viterbi_path {
  double    score;	               /* Viterbi score */
  long      label;                 /* label in current state */
  long      index;                 /* index that summarizes history of state */
  struct viterbi_path *backpointer;/* pointer to optimal VPATH element
				      in previous step that leads to current state */
} VPATH;

void *my_malloc(size_t size);
void set_matrix_element(TYPE_MATRIX *matrix,long i_row,long i_col,double set_value);
void initialize_matrix(TYPE_MATRIX *matrix,long total_rows,long total_columns);
void print_matrix_element(TYPE_MATRIX *matrix,long i_row,long i_col);
void print_matrix(TYPE_MATRIX *matrix);
void print_matrix_info(TYPE_MATRIX *matrix);
void get_H(TYPE_MATRIX *kernel_H,TYPE_BLOCK *all_blocks,long total_examples);
void   read_documents(char *, DOC ***, double **, long *, long *);
void   nol_ll(char *, long *, long *, long *);
void   read_input_parameters(int, char **, char *, char *);
void initialize_hi(TYPE_H *h_i,SAMPLE *samples,long current_example);
void initialize_hi_transpose(TYPE_H *h_i,TYPE_H *hi_transpose,SAMPLE *samples,long current_example);
void initialize_hij(TYPE_BLOCK *blk_ij,TYPE_H *h_i,long total_examples);
void get_base_indices(long *root_base,long *eb_even,long *eb_odd,long *tb,long *lot,long tot_ex);
void print_base_tables(long *eb_even,long *eb_odd,long *tb,long *lot,long tot_ex);
void fill_double_array(double *double_array,double fill_with,long tot_entries);
void fill_long_array(long *long_array,long fill_with,long tot_entries);
void clear_double_array(double *double_array,long tot_entries);
void clear_long_array(long *long_array,long tot_entries);
void clear_float_array(float *double_array,long tot_entries);
void print_double_array(double *double_array,long tot_entries);
void print_double_array_for_matlab(double *double_array,long tot_entries);
void print_double_array_chunk(double *double_array,long start_ind,long end_ind);
void print_double_array_chunk_matlab(double *double_array,long start_ind,long end_ind);
void print_float_array(float *float_array,long tot_entries);
void print_long_array(long *long_array,long total_entries);
void print_fvec(SVECTOR *fvec);
void free_svector(SVECTOR *vec);
void get_r(double *array_r,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,long *binarized_y);
void get_r_save_memory(double *array_r,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,long *binarized_y);
void get_param_delta(double *param_delta,double *alpha_i,long *lot,long *bin_labels,long tot_ex);
void get_param_tow(double *output_array,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,double *scaled_y);
void get_param_tow_save_memory(double *output_array,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,double *scaled_y);
void scale_y(double *bin_y,double *scale_factor, long *lot, long tot_ex);
void scale_double_array(double *array,double scale_factor, long tot_entries);
void copy_long_to_double(long *bin_y,double *double_bin_y, long *lot, long tot_ex);
void copy_long_array(long *copy_from,long *copy_inside, long total_entries);
void copy_inside_double_array(double *copy_to, double *copy_from,long ind_start,long ind_end,long tot_entries);
void get_param_R(double *out_array,double *param_a,double *param_b,double *param_c,long total_elements);
void get_max_label(long *token_label,double a,double b);
void get_best_labels(long *labels,double *r,long *eb,long *tb,long lot,long sample_id);
void get_best_labels_single_sequence(long *labels,double *r,long eb,long tb,long lot);
void get_best_labels_semisupervised(long *supervised_labels, long *labels,double *r,long *eb,long *tb,long lot,long sample_id);
void copy_inside_buffer(long *path, long *buffer);
void regenerate_path_vector(long *buffer, long *path,long add_entry);
void update_best_path_at_p(long *psi_p, long *psi_n, long best_label, long current_token, long lot);
void update_best_path_at_n(long *psi_p, long *psi_n, long best_label, long current_token, long lot);
void read_L(char *file,long *no_of_ws);
void read_alpha(char *file,float *alpha);
void read_paramR(char *file,double *alpha);
void read_labels(char *file,long *supervised_labels);
void compute_parameters(double *temp,double *K,double *tow,double *delta,TYPE_COMMON_INPUT *common_input);
void add_vector(double *sum,double *addendum,long tot_entries);
void sub_vector_from_constant(double *vector,double constant,long tot_entries);


SVECTOR *copy_svector(SVECTOR *vec);
SVECTOR *create_svector_shallow(WORD *words,char *userdefined,double factor);

void gen_binTS(long **labels, long *binarized_labels, long *lot, long tot_ex, long *total_binfeats,long *base_table);
void recover_labels(long *binarized_labels,long *recovered_labels,long lot,long sample_no,long *emit_base_table);
void print_binarized_array(long *binarized_labels,long total_entries);
void print_labels(SAMPLE dataset, long ex_num);
void print_recovered_array(long *recovered_labels,long lot);
void map_labelVector(long *labels, long *binarized_labels, long lot);
void fill_inTrans(long *labels,long cur_lab,long prev_lab,long start_ind,long end_ind);
long get_totBinFeats(long tot_ex,long *lot);
long get_H_dim(SAMPLE *sample);
void print_dataset(SAMPLE sample_dataset);

double *create_nvector(int n);
double *prod_matrix_vector(TYPE_MATRIX *A, double *v);

double get_matrix_element(TYPE_MATRIX *matrix,long i_row,long i_col);
double sprod_ss(SVECTOR *a, SVECTOR *b);
double compute_dotproduct(long *a,double *b,long total_entries);

int    space_or_null(int);
int    parse_document(char *, WORD *, double *, long *, long *, double *, long *, long, char **);
int    read_word(char *in, char *out);


long get_linear_index(long total_rows,long total_columns,long i_row,long i_col);
SAMPLE      read_struct_examples(char *file);

SVECTOR *create_svector(WORD *, char *, double);
SVECTOR* multadd_ss_r(SVECTOR *a,SVECTOR *b,double fa, double fb,
		      double min_non_zero);
DOC    *create_example(long, long, long, double, SVECTOR *);

/*write to file*/
void write_vl_to_file(char *filename,SAMPLE *ts);
void write_L_to_file(char *filename,long L);
void write_mu_to_file(char *filename,long total_mu);
void write_labels_only_to_file(char *filename,long *labels,long lot);

/* read file */
void read_supervised_label_file(char *file, TYPE_SUPLABELS *sup_labels); 

/*Balance Viterbi Implementation*/
void copy_BV_path_inside_longArray(BVPATH *copy_from,long *returned_label);
long check_validity( long t,long P,long y_t );
void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to);
void copy_arg2to_bvpath(BVPATH *arg1, long arg2);
void allocate_initialize_2darray_bvpath(BVPATH **arg1, long token_ind);
void free_BVPATH_2DArray(BVPATH **arg1);
void free_BVPATH(BVPATH *arg1);
BVPATH **copy_BVPATH_arg1to_arg2(BVPATH **arg1,BVPATH **arg2);
void call_BViterbi(long *labels,double *r,long *eb,long *tb,long lot,long sample_id);
void print_BVPATH(BVPATH **arg1);
void fill_path_vector(long *long_array,long fill_with,long tot_tokens);
void print_path_vector_BV(long *long_array,long tot_tokens);
BVPATH **allocate_init_2darray_bvpath(long token_ind);
BVPATH *allocate_init_bvpath(long token_ind);

/* Joachims Viterbi Implementation */
long *viterbi(double *trans, double **emit, long length, long states, 
              long trans_order, long emit_order, double *score);
VPATH *viterbi_forward_order1(double *trans, double **emit, long emit_order,
			      long start, long end, long states, VPATH **path);
long *viterbi_semisupervised(long *supervised_labels,double *trans, double **emit, long length, long states, 
              long trans_order, long emit_order, double *score);
VPATH *viterbi_forward_order1_semisupervised(long *supervised_labels,double *trans, double **emit, long emit_order,
			      long start, long end, long states, VPATH **path);     
