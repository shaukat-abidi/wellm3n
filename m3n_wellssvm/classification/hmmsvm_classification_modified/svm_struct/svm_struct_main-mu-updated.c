/***********************************************************************/
/*                                                                     */
/*   svm_struct_main.c                                                 */
/*                                                                     */
/*   Command line interface to the alignment learning module of the    */
/*   Support Vector Machine.                                           */
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


/* the following enables you to use svm-learn out of C++ */
#ifdef __cplusplus
extern "C" {
#endif
#include "../svm_light/svm_common.h"
#include "../svm_light/svm_learn.h"
#ifdef __cplusplus
}
#endif
# include "svm_struct_learn.h"
# include "svm_struct_common.h"
# include "../svm_struct_api.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

/* } */

char trainfile[200];           /* file with training examples */
char modelfile[200];           /* file for resulting classifier */

void   read_input_parameters(int, char **, char *, char *,long *, long *,
			     STRUCT_LEARN_PARM *, LEARN_PARM *, KERNEL_PARM *,
			     int *);
void   wait_any_key();
void   print_help();

/* Added Functions */
void   populate_extended_labels_T_times(SAMPLE* training_sample,int T);
void get_w_endpoints(STRUCTMODEL *sm,int *s_ind, int *e_ind,int slice_index);
double get_l2norm(double *vec,int s_ind, int e_ind);

int main (int argc, char* argv[])
{  
  SAMPLE sample;  /* training sample */
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  STRUCT_LEARN_PARM struct_parm;
  STRUCTMODEL structmodel;
  int alg_type;
  
  /**************************/
  long total_iterations;
  int iter_tokens,iter_ex,T,iter_mu,iter_models;
  int s_ind,e_ind,slice_index;//Start-End index
  /*MS: Mu-Update Variables*/
  double norm_full_w;
  double *norm_wi=NULL;
  
  /********Initializations******/
  total_iterations = 0;
  structmodel.L = 5; 
  norm_full_w = 0.0;
  T=structmodel.L;
  
  structmodel.mu = (float *)my_malloc(sizeof(float) * structmodel.L);
  norm_wi = (double *)my_malloc(sizeof(double) * structmodel.L);
  
  for(iter_mu=0;iter_mu<structmodel.L;iter_mu++)
  {
	  /* mu and norm_wi have same number of instances */
	  structmodel.mu[iter_mu]=1000/structmodel.L;
	  norm_wi[iter_mu] = 0.0;
  }
  
  
  /****Important Note*********/
  //T is 1-based
  //slice-index is 0-based
  /**************************/
  
  svm_struct_learn_api_init(argc,argv);

  read_input_parameters(argc,argv,trainfile,modelfile,&verbosity,
			&struct_verbosity,&struct_parm,&learn_parm,
			&kernel_parm,&alg_type);

  if(struct_verbosity>=1) {
    printf("Reading training examples..."); fflush(stdout);
  }
  /* read the training examples */
  sample=read_struct_examples(trainfile,&struct_parm);
  if(struct_verbosity>=1) {
    printf("done\n"); fflush(stdout);
  }

  //printf("value of sv: %d\n", structmodel.svm_model->sv_num);
  printf("ALG_TYPE: %d\n", alg_type);
  
  /*Shaukat:Experiment Area*/
  printf("sample.n = %d\n",sample.n);
  
  //Tokens in 1st-Example
  //printf("example[0].y.length = %ld\n",sample.examples[0].y.length); 
  
  //Printing labels for first example
  //for(iter_tokens=0;iter_tokens<sample.examples[0].y.length;iter_tokens++)
  //{
  //	    printf("token#:%d label = %ld\n",iter_tokens,sample.examples[0].y.labels[iter_tokens]); 
  //}
  
  //Populating Extended Labels Datastructure
  printf("Populating Extended Labels Datastructure... ");
  populate_extended_labels(&sample,&structmodel);
  printf("done.\n");
  
  
  //Printing extended-labels for 1st example
  //for(iter_tokens=0;iter_tokens<sample.examples[0].y.length;iter_tokens++)
  //{
  //	    printf("token#:%d extended-label = %ld\n",iter_tokens,sample.examples[0].y.extended_labels[iter_tokens]); 
  //}
  
  
  /**********Populating Extended Labels-T-Times : T >=2 (should be)*********/
  /*
   * Description:
   * This function will not be used because it was assumed
   * that T will grow. Infact, the assumption that was made
   * was not right.
   *  
   */
  //printf("Populating Extended Labels-T-Times ... ");
  //populate_extended_labels_T_times(&sample,T);
  //printf("done.\n");
  
  
  /******extended-labels count for all-examples ********/
  //for(iter_ex=0;iter_ex<sample.n;iter_ex++)
  //{
		//ELC=Extended label count
    	//printf("ELC for ex# %d = %ld (org=%ld) \n",iter_ex,sample.examples[iter_ex].y.loet,
    	//sample.examples[iter_ex].y.length); 
  //}
  
  //Printing start-end index for desired slice-index (all-examples)
  //for(iter_ex=0;iter_ex<sample.n;iter_ex++)
  //{ 
	    //find_start_end_indices(sample.examples[iter_ex].y, 0, &s_ind, &e_ind);
	    //find_start_end_indices(sample.examples[iter_ex].y, 1, &s_ind, &e_ind);
	    //find_start_end_indices(sample.examples[iter_ex].y, 2, &s_ind, &e_ind);
	    //find_start_end_indices(sample.examples[iter_ex].y, 3, &s_ind, &e_ind);
	    //find_start_end_indices(sample.examples[iter_ex].y, 4, &s_ind, &e_ind);
		//printf("ex#: %ld , %d -- %d \n",iter_ex,s_ind,e_ind); 
  //}  
  
  
  
  /*************************Debugging Psi****************************************/
  //init_struct_model(sample,&structmodel,&struct_parm,&learn_parm,&kernel_parm); 
  //structmodel.sizePsi = structmodel.sizePsi+1;          /* sm must contain size of psi on return */
  //printf("sizePsi:%ld\n",structmodel.sizePsi);
  //assert(structmodel.sizePsi == 284);
  //print_fvec(psi(sample.examples[46].x,sample.examples[46].y,&structmodel,&struct_parm));
  
  /*************************Debugging append_Psi****************************************/
  //SVECTOR *fy=NULL;
  //init_struct_model(sample,&structmodel,&struct_parm,&learn_parm,&kernel_parm); 
  //structmodel.sizePsi = structmodel.sizePsi+1;          /* sm must contain size of psi on return */
  //printf("sizePsi:%ld\n",structmodel.sizePsi);
  //assert(structmodel.sizePsi == 284);
  //fy=psi(sample.examples[46].x,sample.examples[46].y,&structmodel,&struct_parm);
  //print_fvec(fy);	    
  /********************************************************************************************/
  
  /*************************Debugging append_Psi****************************************/
  //init_struct_model(sample,&structmodel,&struct_parm,&learn_parm,&kernel_parm); 
  //assert(structmodel.sizePsi == 14*structmodel.L);
  //LABEL y_todiscard;
  //y_todiscard = find_most_violated_constraint_marginrescaling(sample.examples[0].x, sample.examples[0].y, 
  //						     &structmodel, &struct_parm);
  /********************************************************************************************/
  

  /* Do the learning and return structmodel. */
  while(total_iterations<6)
  {
	  printf("*******starting iteration # %ld  .... ******\n",total_iterations);

	  if(alg_type == 4)
		{
			  /*Print mu_s*/
			  for(iter_models=0;iter_models<structmodel.L;iter_models++)
			  {
				  printf("mu[%ld] = %f\n",iter_models,structmodel.mu[iter_models]);fflush(stdout);
			  }
			  
			  svm_learn_struct_joint(sample,&struct_parm,&learn_parm,&kernel_parm,&structmodel,ONESLACK_DUAL_CACHE_ALG);
			  
			  /****MS: After Learning---Update Mu's***********/
			  printf("*******iteration # %ld finished .... ******\n",total_iterations);
			  
			  
			  norm_full_w = 0.0; /*set it to zero. Its denominator of MU-Update Equation*/
			  
			  /*Obtain Norms*/
			  for(iter_models=0;iter_models<structmodel.L;iter_models++)
			  {
				  s_ind = -1;
				  e_ind = -1;
				  
				  /*Get start-end indices of w-vector*/
				  get_w_endpoints(&structmodel,&s_ind,&e_ind,iter_models);
				  
				  /*get L2-Norm*/
				  norm_wi[iter_models] = get_l2norm(structmodel.w,s_ind,e_ind);
				  printf("start_ind=%d,end_index=%d,norm=%f\n",
				  s_ind,e_ind,norm_wi[iter_models]);fflush(stdout);
				  
				  /*Scale Norm by mu*/
				  norm_wi[iter_models] = norm_wi[iter_models] * structmodel.mu[iter_models];
				  norm_full_w += norm_wi[iter_models];
			  }
			  printf("norm_full_w = %f\n",norm_full_w);
			  
			  /*Recompute mu*/
			  for(iter_models=0;iter_models<structmodel.L;iter_models++)
			  {
				  structmodel.mu[iter_models] = 0.0;
				  structmodel.mu[iter_models] = norm_wi[iter_models]/norm_full_w;
			  }
			  /************************************************/
		}
		total_iterations++;
	}
  
  
  /* Warning: The model contains references to the original data 'docs'.
     If you want to free the original data, and only keep the model, you 
     have to make a deep copy of 'model'. */
  if(struct_verbosity>=1) {
    printf("Writing learned model...");fflush(stdout);
  }
  write_struct_model(modelfile,&structmodel,&struct_parm);
  if(struct_verbosity>=1) {
    printf("done\n");fflush(stdout);
  }
   
  printf("freeing sample...\n");fflush(stdout); 
  free_struct_sample(sample);
  
  printf("freeing model...\n");fflush(stdout); 
  free_struct_model(structmodel);

  printf("api_exit...\n");fflush(stdout); 
  svm_struct_learn_api_exit();

  return 0;
}

/*---------------------------------------------------------------------------*/


void populate_extended_labels_T_times(SAMPLE* training_sample,int T)
{
	
	/* *************************************************************
	 * 1) Call this function when T >=2
	 * 2) T: 1-based Index 
	 * 
	 * Description: 
	 * This function will re-allocate memory for y.extended_labels
	 * according to T. Then it will be filled by y.labels i.e. we 
	 * will have T-1 copies of y.labels inside y.extended_labels.
	 * 
	 * y.extended_labels = [--y.labels--,--y.labels--,...,y.labels] 
	 * 
	 * *************************************************************
	 *
	 */
	   /*
	   * Description:
	   * This function will not be used because it was assumed
	   * that T will grow. Infact, the assumption that was made
	   * was not right.
	   */
	
	if(T>=2)
	{
		long lot;//length of tokens
		long i_ex,i_tok;//iterate examples,iterate tokens
		int iter_slices;
		
		for(i_ex = 0;i_ex<training_sample->n;i_ex++)
		{
			lot = training_sample->examples[i_ex].y.length;
			training_sample->examples[i_ex].y.extended_labels = 
												(long *)realloc(training_sample->examples[i_ex].y.extended_labels,
												 sizeof(long *) * lot * T);
	  
			 //printf("lot = %ld\n",lot); 
			for(iter_slices = 0;iter_slices<T;iter_slices++) 
			{	
				for(i_tok = 0;i_tok<lot;i_tok++)
				{
					training_sample->examples[i_ex].y.extended_labels[(iter_slices*lot) + i_tok] = 	
									training_sample->examples[i_ex].y.labels[i_tok];
					training_sample->examples[i_ex].y.loet = (iter_slices*lot) + i_tok;													
				}	
			}
		 training_sample->examples[i_ex].y.loet++;//It was Zero-Based before this increment								
		}
	}
}


void populate_extended_labels_initialize(SAMPLE* training_sample)
{
	/* *************************************************************
	 * 1) T: 1-based Index 
	 * 
	 * Description: 
	 * This function will allocate memory for y.extended_labels
	 * Then it will be filled by y.labels i.e. we 
	 * will have 1 copy of y.labels inside y.extended_labels.
	 * 
	 * y.extended_labels = [--y.labels--] 
	 * 
	 * *************************************************************
	 *
	 */
	
	long lot;//length of tokens
	long i_ex,i_tok;
	
	for(i_ex = 0;i_ex<training_sample->n;i_ex++)
	{
		lot = training_sample->examples[i_ex].y.length;
		training_sample->examples[i_ex].y.extended_labels = 
											(long *)my_malloc(sizeof(long *) * lot);
  
         //printf("lot = %ld\n",lot); 
	
		for(i_tok = 0;i_tok<lot;i_tok++)
		{
			training_sample->examples[i_ex].y.extended_labels[i_tok] = 	
							training_sample->examples[i_ex].y.labels[i_tok];							
		}									
	}
}

void read_input_parameters(int argc,char *argv[],char *trainfile,
			   char *modelfile,
			   long *verbosity,long *struct_verbosity, 
			   STRUCT_LEARN_PARM *struct_parm,
			   LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,
			   int *alg_type)
{
  long i;
  char type[100];
  
  /* set default */
  (*alg_type)=DEFAULT_ALG_TYPE;
  struct_parm->C=-0.01;
  struct_parm->slack_norm=1;
  struct_parm->epsilon=DEFAULT_EPS;
  struct_parm->custom_argc=0;
  struct_parm->loss_function=DEFAULT_LOSS_FCT;
  struct_parm->loss_type=DEFAULT_RESCALING;
  struct_parm->newconstretrain=100;
  struct_parm->ccache_size=5;
  struct_parm->batch_size=100;

  strcpy (modelfile, "svm_struct_model");
  strcpy (learn_parm->predfile, "trans_predictions");
  strcpy (learn_parm->alphafile, "");
  (*verbosity)=0;/*verbosity for svm_light*/
  (*struct_verbosity)=1; /*verbosity for struct learning portion*/
  learn_parm->biased_hyperplane=1;
  learn_parm->remove_inconsistent=0;
  learn_parm->skip_final_opt_check=0;
  learn_parm->svm_maxqpsize=10;
  learn_parm->svm_newvarsinqp=0;
  learn_parm->svm_iter_to_shrink=-9999;
  learn_parm->maxiter=100000;
  learn_parm->kernel_cache_size=40;
  learn_parm->svm_c=99999999;  /* overridden by struct_parm->C */
  learn_parm->eps=0.001;       /* overridden by struct_parm->epsilon */
  learn_parm->transduction_posratio=-1.0;
  learn_parm->svm_costratio=1.0;
  learn_parm->svm_costratio_unlab=1.0;
  learn_parm->svm_unlabbound=1E-5;
  learn_parm->epsilon_crit=0.001;
  learn_parm->epsilon_a=1E-10;  /* changed from 1e-15 */
  learn_parm->compute_loo=0;
  learn_parm->rho=1.0;
  learn_parm->xa_depth=0;
  kernel_parm->kernel_type=0;
  kernel_parm->poly_degree=3;
  kernel_parm->rbf_gamma=1.0;
  kernel_parm->coef_lin=1;
  kernel_parm->coef_const=1;
  strcpy(kernel_parm->custom,"empty");
  strcpy(type,"c");

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case '?': print_help(); exit(0);
      case 'a': i++; strcpy(learn_parm->alphafile,argv[i]); break;
      case 'c': i++; struct_parm->C=atof(argv[i]); break;
      case 'p': i++; struct_parm->slack_norm=atol(argv[i]); break;
      case 'e': i++; struct_parm->epsilon=atof(argv[i]); break;
      case 'k': i++; struct_parm->newconstretrain=atol(argv[i]); break;
      case 'h': i++; learn_parm->svm_iter_to_shrink=atol(argv[i]); break;
      case '#': i++; learn_parm->maxiter=atol(argv[i]); break;
      case 'm': i++; learn_parm->kernel_cache_size=atol(argv[i]); break;
      case 'w': i++; (*alg_type)=atol(argv[i]); break;
      case 'o': i++; struct_parm->loss_type=atol(argv[i]); break;
      case 'n': i++; learn_parm->svm_newvarsinqp=atol(argv[i]); break;
      case 'q': i++; learn_parm->svm_maxqpsize=atol(argv[i]); break;
      case 'l': i++; struct_parm->loss_function=atol(argv[i]); break;
      case 'f': i++; struct_parm->ccache_size=atol(argv[i]); break;
      case 'b': i++; struct_parm->batch_size=atof(argv[i]); break;
      case 't': i++; kernel_parm->kernel_type=atol(argv[i]); break;
      case 'd': i++; kernel_parm->poly_degree=atol(argv[i]); break;
      case 'g': i++; kernel_parm->rbf_gamma=atof(argv[i]); break;
      case 's': i++; kernel_parm->coef_lin=atof(argv[i]); break;
      case 'r': i++; kernel_parm->coef_const=atof(argv[i]); break;
      case 'u': i++; strcpy(kernel_parm->custom,argv[i]); break;
      case '-': strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);i++; strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);break; 
      case 'v': i++; (*struct_verbosity)=atol(argv[i]); break;
      case 'y': i++; (*verbosity)=atol(argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if(i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  strcpy (trainfile, argv[i]);
  if((i+1)<argc) {
    strcpy (modelfile, argv[i+1]);
  }
  if(learn_parm->svm_iter_to_shrink == -9999) {
    learn_parm->svm_iter_to_shrink=100;
  }

  if((learn_parm->skip_final_opt_check) 
     && (kernel_parm->kernel_type == LINEAR)) {
    printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
    learn_parm->skip_final_opt_check=0;
  }    
  if((learn_parm->skip_final_opt_check) 
     && (learn_parm->remove_inconsistent)) {
    printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
    wait_any_key();
    print_help();
    exit(0);
  }    
  if((learn_parm->svm_maxqpsize<2)) {
    printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
    wait_any_key();
    print_help();
    exit(0);
  }
  if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
    printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
    printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->svm_iter_to_shrink<1) {
    printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
    wait_any_key();
    print_help();
    exit(0);
  }
  if(struct_parm->C<0) {
    printf("\nYou have to specify a value for the parameter '-c' (C>0)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(((*alg_type) < 0) || (((*alg_type) > 5) && ((*alg_type) != 9))) {
    printf("\nAlgorithm type must be either '0', '1', '2', '3', '4', or '9'!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->transduction_posratio>1) {
    printf("\nThe fraction of unlabeled examples to classify as positives must\n");
    printf("be less than 1.0 !!!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->svm_costratio<=0) {
    printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(struct_parm->epsilon<=0) {
    printf("\nThe epsilon parameter must be greater than zero!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if((struct_parm->ccache_size<=0) && ((*alg_type) == 4)) {
    printf("\nThe cache size must be at least 1!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(((struct_parm->batch_size<=0) || (struct_parm->batch_size>100))  
     && ((*alg_type) == 4)) {
    printf("\nThe batch size must be in the interval ]0,100]!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if((struct_parm->slack_norm<1) || (struct_parm->slack_norm>2)) {
    printf("\nThe norm of the slacks must be either 1 (L1-norm) or 2 (L2-norm)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if((struct_parm->loss_type != SLACK_RESCALING) 
     && (struct_parm->loss_type != MARGIN_RESCALING)) {
    printf("\nThe loss type must be either 1 (slack rescaling) or 2 (margin rescaling)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->rho<0) {
    printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
    printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
    printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
    printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
    printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
    printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
    wait_any_key();
    print_help();
    exit(0);
  }

  parse_struct_parameters(struct_parm);
}

void wait_any_key()
{
  printf("\n(more)\n");
  (void)getc(stdin);
}

void print_help()
{
  printf("\nSVM-struct learning module: %s, %s, %s\n",INST_NAME,INST_VERSION,INST_VERSION_DATE);
  printf("   includes SVM-struct %s for learning complex outputs, %s\n",STRUCT_VERSION,STRUCT_VERSION_DATE);
  printf("   includes SVM-light %s quadratic optimizer, %s\n",VERSION,VERSION_DATE);
  copyright_notice();
  printf("   usage: svm_struct_learn [options] example_file model_file\n\n");
  printf("Arguments:\n");
  printf("         example_file-> file with training data\n");
  printf("         model_file  -> file to store learned decision rule in\n");

  printf("General Options:\n");
  printf("         -?          -> this help\n");
  printf("         -v [0..3]   -> verbosity level (default 1)\n");
  printf("         -y [0..3]   -> verbosity level for svm_light (default 0)\n");
  printf("Learning Options:\n");
  printf("         -c float    -> C: trade-off between training error\n");
  printf("                        and margin (default 0.01)\n");
  printf("         -p [1,2]    -> L-norm to use for slack variables. Use 1 for L1-norm,\n");
  printf("                        use 2 for squared slacks. (default 1)\n");
  printf("         -o [1,2]    -> Rescaling method to use for loss.\n");
  printf("                        1: slack rescaling\n");
  printf("                        2: margin rescaling\n");
  printf("                        (default %d)\n",DEFAULT_RESCALING);
  printf("         -l [0..]    -> Loss function to use.\n");
  printf("                        0: zero/one loss\n");
  printf("                        ?: see below in application specific options\n");
  printf("                        (default %d)\n",DEFAULT_LOSS_FCT);
  printf("Optimization Options (see [2][5]):\n");
  printf("         -w [0,..,9] -> choice of structural learning algorithm (default %d):\n",(int)DEFAULT_ALG_TYPE);
  printf("                        0: n-slack algorithm described in [2]\n");
  printf("                        1: n-slack algorithm with shrinking heuristic\n");
  printf("                        2: 1-slack algorithm (primal) described in [5]\n");
  printf("                        3: 1-slack algorithm (dual) described in [5]\n");
  printf("                        4: 1-slack algorithm (dual) with constraint cache [5]\n");
  printf("                        9: custom algorithm in svm_struct_learn_custom.c\n");
  printf("         -e float    -> epsilon: allow that tolerance for termination\n");
  printf("                        criterion (default %f)\n",DEFAULT_EPS);
  printf("         -k [1..]    -> number of new constraints to accumulate before\n"); 
  printf("                        recomputing the QP solution (default 100) (-w 0 and 1 only)\n");
  printf("         -f [5..]    -> number of constraints to cache for each example\n");
  printf("                        (default 5) (used with -w 4)\n");
  printf("         -b [1..100] -> percentage of training set for which to refresh cache\n");
  printf("                        when no epsilon violated constraint can be constructed\n");
  printf("                        from current cache (default 100%%) (used with -w 4)\n");
  printf("SVM-light Options for Solving QP Subproblems (see [3]):\n");
  printf("         -n [2..q]   -> number of new variables entering the working set\n");
  printf("                        in each svm-light iteration (default n = q). \n");
  printf("                        Set n < q to prevent zig-zagging.\n");
  printf("         -m [5..]    -> size of svm-light cache for kernel evaluations in MB\n");
  printf("                        (default 40) (used only for -w 1 with kernels)\n");
  printf("         -h [5..]    -> number of svm-light iterations a variable needs to be\n"); 
  printf("                        optimal before considered for shrinking (default 100)\n");
  printf("         -# int      -> terminate svm-light QP subproblem optimization, if no\n");
  printf("                        progress after this number of iterations.\n");
  printf("                        (default 100000)\n");
  printf("Kernel Options:\n");
  printf("         -t int      -> type of kernel function:\n");
  printf("                        0: linear (default)\n");
  printf("                        1: polynomial (s a*b+c)^d\n");
  printf("                        2: radial basis function exp(-gamma ||a-b||^2)\n");
  printf("                        3: sigmoid tanh(s a*b + c)\n");
  printf("                        4: user defined kernel from kernel.h\n");
  printf("         -d int      -> parameter d in polynomial kernel\n");
  printf("         -g float    -> parameter gamma in rbf kernel\n");
  printf("         -s float    -> parameter s in sigmoid/poly kernel\n");
  printf("         -r float    -> parameter c in sigmoid/poly kernel\n");
  printf("         -u string   -> parameter of user defined kernel\n");
  printf("Output Options:\n");
  printf("         -a string   -> write all alphas to this file after learning\n");
  printf("                        (in the same order as in the training set)\n");
  printf("Application-Specific Options:\n");
  print_struct_help();
  wait_any_key();

  printf("\nMore details in:\n");
  printf("[1] T. Joachims, Learning to Align Sequences: A Maximum Margin Aproach.\n");
  printf("    Technical Report, September, 2003.\n");
  printf("[2] I. Tsochantaridis, T. Joachims, T. Hofmann, and Y. Altun, Large Margin\n");
  printf("    Methods for Structured and Interdependent Output Variables, Journal\n");
  printf("    of Machine Learning Research (JMLR), Vol. 6(Sep):1453-1484, 2005.\n");
  printf("[3] T. Joachims, Making Large-Scale SVM Learning Practical. Advances in\n");
  printf("    Kernel Methods - Support Vector Learning, B. Schölkopf and C. Burges and\n");
  printf("    A. Smola (ed.), MIT Press, 1999.\n");
  printf("[4] T. Joachims, Learning to Classify Text Using Support Vector\n");
  printf("    Machines: Methods, Theory, and Algorithms. Dissertation, Kluwer,\n");
  printf("    2002.\n");
  printf("[5] T. Joachims, T. Finley, Chun-Nam Yu, Cutting-Plane Training of Structural\n");
  printf("    SVMs, Machine Learning Journal, to appear.\n");
}
void get_w_endpoints(STRUCTMODEL *sm,int *s_ind, int *e_ind,int slice_index)
{
	/* *********************************************************
	 * Description: 
	 * w = [--slice 0--,--slice 1--,...,--slice T-1--]
	 *
	 * This Function will return starting\ending indices for 
	 * desired slice-id inside s_ind\e_ind.
	 * 
	 * slice_index: desired slice-id(0,...T-1)
	 * s_ind      :	returned index indicating start of slice-id   
	 * e_ind      :	returned index indicating end of slice-id   
	 * *********************************************************
	 *
	 */
	
	*(s_ind) = slice_index * sm->sizePsi_atomic;
	*(e_ind) = *(s_ind) + ( sm->sizePsi_atomic - 1 );
	//printf("length:%d  slice_id:%d start:%d -- end:%d \n",y.length,slice_index,*(s_ind),*(e_ind)); 

}
double get_l2norm(double *vec,int start_ind, int end_ind)
{
	long iter_vec;
	double norm,sum;
	
	iter_vec = 0;
	norm = 0.0;
	sum = 0.0;
	
	for(iter_vec = start_ind; iter_vec<=end_ind; iter_vec++)
	{
		sum += (vec[iter_vec] * vec[iter_vec]); 
	}
	
	norm = sqrt(sum);
	
	//printf("(Inside function--line %d)norm=%f\n",__LINE__,norm);fflush(stdout);
	return(norm);
}

