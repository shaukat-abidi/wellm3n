/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.h                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include "svm_struct_api_types.h"
#include "svm_struct/svm_struct_common.h"

#ifndef svm_struct_api
#define svm_struct_api

void        svm_struct_learn_api_init(int argc, char* argv[]);
void        svm_struct_learn_api_exit();
void        svm_struct_classify_api_init(int argc, char* argv[]);
void        svm_struct_classify_api_exit();
SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm);
void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm);
CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm);
LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm);
LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm);
LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm);
int         empty_label(LABEL y);
SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm, 
	        STRUCT_LEARN_PARM *sparm);
SVECTOR     *psi_atomic(PATTERN x, LABEL y, STRUCTMODEL *sm, 
	        STRUCT_LEARN_PARM *sparm);
	        
double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm);
int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm);
void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm);
void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm,
				       STRUCT_TEST_STATS *teststats);
void        eval_prediction(long exnum, EXAMPLE ex, LABEL prediction, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,
			    STRUCT_TEST_STATS *teststats);
void        write_struct_model(char *file,STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm);
STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm);
void        write_label(FILE *fp, LABEL y);
void        free_pattern(PATTERN x);
void        free_label(LABEL y);
void        free_struct_model(STRUCTMODEL sm);
void        free_struct_sample(SAMPLE s);
void        print_struct_help();
void        parse_struct_parameters(STRUCT_LEARN_PARM *sparm);
void        print_struct_help_classify();
void        parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm);
void        svm_learn_struct_joint_custom(SAMPLE sample, 
				   STRUCT_LEARN_PARM *sparm,
				   LEARN_PARM *lparm, KERNEL_PARM *kparm, 
				   STRUCTMODEL *sm);
void   fill_in_labels(LABEL *y,int T);
void   find_start_end_indices(LABEL y, int slice_index, int *s_ind, int *e_ind);
void   populate_extended_labels(SAMPLE* training_sample,STRUCTMODEL *sm,SAMPLE* gt_samples);
void   populate_extended_labels_y(LABEL *y,STRUCTMODEL *sm);
void   initialize_buffer_labels(SAMPLE* training_sample,STRUCTMODEL *sm);
void   copy_labels_inside_buffer(LABEL *from,LABEL *to);
void   print_buffer_labels(LABEL *y);
void print_classified_labels(LABEL *y_gt,LABEL *y_classified);
void update_stats_vars(LABEL *y_gt,LABEL *y_classified, long *total_pos, long *total_neg, long *pos_predictions, long *neg_predictions);
void invert_classified_labels(LABEL *y_classified);

void   print_wvec(STRUCTMODEL *);
void   print_cset(CONSTSET *,double *);

/*MS:file-operation*/
void read_L(char *file,int *model_L);
void read_mu(char *file,STRUCTMODEL *sm,float *model_mu);
void get_w_endpoints(STRUCTMODEL *sm,int *s_ind, int *e_ind,int slice_index);
double get_l2norm(double *vec,int s_ind, int e_ind);
void read_labels(char *file,long *labels);
void print_sample_info(SAMPLE *training_sample);
void write_mu_to_file(char *filename,STRUCTMODEL *sm);
void write_ws_to_file(char *filename,SAMPLE *ts);
void write_alphas_to_file(char *filename,double *alphas,long total_alphas);
void write_totws_to_file(char *filename,long total_ws);
double      loss_inside_classification(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm);
void normalize_temp(double *temp,int len);
void normalize_emit(double **emit,int tot_tok);

#endif
