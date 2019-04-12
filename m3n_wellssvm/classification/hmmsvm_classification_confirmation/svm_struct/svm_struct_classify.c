/***********************************************************************/
/*                                                                     */
/*   svm_struct_classify.c                                             */
/*                                                                     */
/*   Classification module of SVM-struct.                              */
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
/************************************************************************/

#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "../svm_light/svm_common.h"
#ifdef __cplusplus
}
#endif
#include "../svm_struct_api.h"
#include "svm_struct_common.h"

char testfile[200];
char modelfile[200];
char predictionsfile[200];

void read_input_parameters(int, char **, char *, char *, char *, 
			   STRUCT_LEARN_PARM *, long*, long *);
void print_help(void);
void printw_for_matlab(STRUCTMODEL model);
void get_full_stats(LABEL *gt_y, LABEL *ybar, long *total_pos, long *total_neg, long *tp, long *tn,long *fp, long *fn);


int main (int argc, char* argv[])
{
  long T,iter_mu,iter_w;
  long correct=0,incorrect=0,no_accuracy=0;
  long i;
  double t1,runtime=0;
  double avgloss=0,l=0,accuracy=0;
  double precision=0,recall=0,f1_score=0;

  FILE *predfl;
  STRUCTMODEL model; 
  STRUCT_LEARN_PARM sparm;
  STRUCT_TEST_STATS teststats;
  SAMPLE testsample;
  LABEL y;

  svm_struct_classify_api_init(argc,argv);

  read_input_parameters(argc,argv,testfile,modelfile,predictionsfile,&sparm,
			&verbosity,&struct_verbosity);

  if(struct_verbosity>=1) {
    printf("Reading model..."); fflush(stdout);
  }
  model=read_struct_model(modelfile,&sparm);

  if(struct_verbosity>=1) {
    fprintf(stdout, "done.\n");
  }

  /***********************************************************************/
  /*MS: Receiving segmentation fault due to uninitialized fields of model*/
  /*Read L from File*/
  read_L("/home/ssabidi/Shaukat/m3n_wellssvm/classification/hmmsvm_classification_confirmation/params/L.txt",&model.L);
  T=model.L;/*same as in Main of learning module*/ 
  printf("Inside classification module for WellSSVM\n");
  printf("L=%d\n",model.L);
  
  
  /**********************************************************************/

  /* Shaukat: Very Hacky Fix 
   * Decreasing value of model.L by 1 because it was increased when MVL
   * suggested us to stop.
   */
  
    /**********************************************************************/
   T = T - 1; /* Uncomment it for running full experiment with automated script*/
   model.L = model.L - 1; /* Uncomment it for running full experiment with automated script*/
    /**********************************************************************/
  printf("(changed)L=%d\n",model.L);
  /* Shaukat: Very Hacky Fix 
   * Decreasing value of model.L by 1 because it was increased when MVL
   * suggested us to stop.
   */
   
   
  /* Allocate memory for mu's and reset its values*/
  model.mu = (float *)my_malloc(sizeof(float) * model.L);
  
  for(iter_mu=0;iter_mu<model.L;iter_mu++)
  {model.mu[iter_mu]=0.0;}
  
  /* Reading mu from file*/
  read_mu("/home/ssabidi/Shaukat/m3n_wellssvm/classification/hmmsvm_classification_confirmation/params/mu_converged.txt",&model,model.mu);
  
  /*print mu's*/
  for(iter_mu=0;iter_mu<model.L;iter_mu++)
  {printf("mu[%ld]=%f\n",iter_mu,model.mu[iter_mu]);fflush(stdout);}
	
  model.sizePsi_atomic = 0; /*Reset it*/
  model.sizePsi_atomic = model.sizePsi/model.L; /*Setting it with its proper value*/
  
  printf("model.sizePsi_atomic  = %ld\n",model.sizePsi_atomic);fflush(stdout);
  printf("model.sizePsi  = %ld\n",model.sizePsi);fflush(stdout);
  /***********************************************************************/
  if(model.svm_model->kernel_parm.kernel_type == LINEAR) { /* linear kernel */
    /* compute weight vector */
    add_weight_vector_to_linear_model(model.svm_model);
    model.w=model.svm_model->lin_weights;
  }
  
  /***********************************************************************/
   /* Print weight vector */
   for(iter_w=0;iter_w<model.svm_model->totwords;iter_w++)
   {
	   printf("w[%d]=%f\n",iter_w,model.w[iter_w]);fflush(stdout);
   }
   
   /*for Debugging*/
   printw_for_matlab(model);
  /***********************************************************************/

  if(struct_verbosity>=1) {
    printf("Reading test examples..."); fflush(stdout);
  }
  testsample=read_struct_examples(testfile,&sparm);
  if(struct_verbosity>=1) {
    printf("done.\n"); fflush(stdout);
  }

  if(struct_verbosity>=1) {
    printf("Classifying test examples...\n"); fflush(stdout);
  }

  if ((predfl = fopen (predictionsfile, "w")) == NULL)
  { perror (predictionsfile); exit (1); }

  /*WellSSVM-Prediction Variables*/
  long total_tokens;
  double total_loss;
  long pos_toks_in_sample,neg_toks_in_sample,tp_sample,fp_sample,fn_sample,tn_sample;
  long pos_toks_in_dataset,neg_toks_in_dataset,true_pos_pred_in_dataset,true_neg_pred_in_dataset,false_pos_pred_in_dataset,false_neg_pred_in_dataset;
  total_tokens = 0;
  total_loss = 0.0;
  pos_toks_in_dataset=0;
  neg_toks_in_dataset=0;
  true_pos_pred_in_dataset=0;
  true_neg_pred_in_dataset=0;
  false_pos_pred_in_dataset=0;
  false_neg_pred_in_dataset=0;
  
  for(i=0;i<testsample.n;i++) {
	
	/* initialize variables for this sample */
	pos_toks_in_sample = 0; neg_toks_in_sample = 0; tp_sample = 0; fp_sample = 0; fn_sample = 0; tn_sample = 0;
	  
    y=classify_struct_example(testsample.examples[i].x,&model,&sparm);
    printf("printing predicition for sample #%ld\n",i);
    print_classified_labels(&testsample.examples[i].y,&y);
    //update_stats_vars(&testsample.examples[i].y,&y,&pos_toks_in_sample,&neg_toks_in_sample,&true_pos_pred_in_sample,&true_neg_pred_in_sample);
  	get_full_stats(&testsample.examples[i].y,&y,&pos_toks_in_sample,&neg_toks_in_sample,&tp_sample,&tn_sample,&fp_sample,&fn_sample);


    //l=loss(testsample.examples[i].y,y,&sparm);
    l=loss_inside_classification(testsample.examples[i].y,y,&sparm);
    printf("sample-%ld: tp/totpos_toks_in_sample=%ld/%ld\n",i,tp_sample,pos_toks_in_sample);
    printf("sample-%ld: tn/totneg_toks_in_sample=%ld/%ld\n",i,tn_sample,neg_toks_in_sample);
    printf("sample-%ld: fp=%ld\n",i,fp_sample);
    printf("sample-%ld: fn=%ld\n",i,fn_sample);
    
    //printf("sample-%ld: Incorrect Predicted-Labels:%f (out of %ld)\n",i,l,y.length);
    
    pos_toks_in_dataset+=pos_toks_in_sample;
    neg_toks_in_dataset+=neg_toks_in_sample;
    true_pos_pred_in_dataset+=tp_sample;
    true_neg_pred_in_dataset+=tn_sample;
    false_pos_pred_in_dataset+=fp_sample;
    false_neg_pred_in_dataset+=fn_sample;
  
    total_tokens +=  y.length;
    total_loss += l;
    printf("total-loss till sample-%ld:%f\n",i,total_loss);

    if(empty_label(testsample.examples[i].y)) 
      {    
		  printf("test-data is not labeled\n");
		  no_accuracy=1;
      } /* test data is not labeled */
    
    
    
    free_label(y);
  }  
  accuracy = (double)(total_tokens-total_loss)/(total_tokens);
  
  printf("******************************************************\n");
  printf("tot_tokens=%ld,tot_pos_tokens=%ld,tot_neg_tokens=%ld\n",total_tokens,pos_toks_in_dataset,neg_toks_in_dataset);
  printf("tp=%ld,tn=%ld,fp=%ld,fn=%ld\n",true_pos_pred_in_dataset,true_neg_pred_in_dataset,false_pos_pred_in_dataset,false_neg_pred_in_dataset);
  printf("Total Loss(Tokens with wrong labels):%f\n",total_loss);
  printf("Accuracy = %f\n",accuracy);
  printf("Accuracy = %f,precision=%f,recall=%f\n",accuracy,(double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_pos_pred_in_dataset),(double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_neg_pred_in_dataset));
  precision = (double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_pos_pred_in_dataset);
  recall = (double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_neg_pred_in_dataset);
  f1_score = (2*precision*recall)/(precision+recall);
  printf("f1_score=%f\n",f1_score);
  
  /*If Accuracy < 0.5*/ 
    if(accuracy>0) //if(accuracy<0.5)
  {
	 /*Inverting results*/ 
     /*Calling Classification again*/
     
     /**************Classification Start******************/
	  total_tokens = 0;
	  total_loss = 0.0;
	  pos_toks_in_dataset=0;
	  neg_toks_in_dataset=0;
	  true_pos_pred_in_dataset=0;
	  true_neg_pred_in_dataset=0;
	  false_pos_pred_in_dataset=0;
	  false_neg_pred_in_dataset=0;
	  
	  for(i=0;i<testsample.n;i++) {
		
		/* initialize variables for this sample */
	    pos_toks_in_sample = 0; neg_toks_in_sample = 0; tp_sample = 0; fp_sample = 0; fn_sample = 0; tn_sample = 0;
		  
		y=classify_struct_example(testsample.examples[i].x,&model,&sparm);
		invert_classified_labels(&y);
		printf("printing inverted predicition for sample #%ld\n",i);
		print_classified_labels(&testsample.examples[i].y,&y);
		//update_stats_vars(&testsample.examples[i].y,&y,&pos_toks_in_sample,&neg_toks_in_sample,&true_pos_pred_in_sample,&true_neg_pred_in_sample);
        get_full_stats(&testsample.examples[i].y,&y,&pos_toks_in_sample,&neg_toks_in_sample,&tp_sample,&tn_sample,&fp_sample,&fn_sample);

	    //l=loss(testsample.examples[i].y,y,&sparm);
        l=loss_inside_classification(testsample.examples[i].y,y,&sparm);
	    printf("sample-%ld: tp/totpos_toks_in_sample=%ld/%ld\n",i,tp_sample,pos_toks_in_sample);
	    printf("sample-%ld: tn/totneg_toks_in_sample=%ld/%ld\n",i,tn_sample,neg_toks_in_sample);
	    printf("sample-%ld: fp=%ld\n",i,fp_sample);
	    printf("sample-%ld: fn=%ld\n",i,fn_sample);
	    
	    //printf("sample-%ld: Incorrect Predicted-Labels:%f (out of %ld)\n",i,l,y.length);
	    
	    pos_toks_in_dataset+=pos_toks_in_sample;
	    neg_toks_in_dataset+=neg_toks_in_sample;
	    true_pos_pred_in_dataset+=tp_sample;
	    true_neg_pred_in_dataset+=tn_sample;
	    false_pos_pred_in_dataset+=fp_sample;
	    false_neg_pred_in_dataset+=fn_sample;
	  
	    total_tokens +=  y.length;
	    total_loss += l;
	    printf("total-loss till sample-%ld:%f\n",i,total_loss);
	
	    if(empty_label(testsample.examples[i].y)) 
	      {    
			  printf("test-data is not labeled\n");
			  no_accuracy=1;
	      } /* test data is not labeled */
	    
	    
	    free_label(y);

	  }
	  
	  accuracy = (double)(total_tokens-total_loss)/(total_tokens);
      printf("******************************************************\n");
      printf("(RESULT INVERTED)tot_tokens=%ld,tot_pos_tokens=%ld,tot_neg_tokens=%ld\n",total_tokens,pos_toks_in_dataset,neg_toks_in_dataset);
      printf("(RESULT INVERTED)tp=%ld,tn=%ld,fp=%ld,fn=%ld\n",true_pos_pred_in_dataset,true_neg_pred_in_dataset,false_pos_pred_in_dataset,false_neg_pred_in_dataset);
      printf("(RESULT INVERTED)Total Loss(Tokens with wrong labels):%f\n",total_loss);
      printf("(RESULT INVERTED)Accuracy = %f\n",accuracy);	  
      printf("(RESULT INVERTED)Accuracy = %f,precision=%f,recall=%f\n",accuracy,(double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_pos_pred_in_dataset),(double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_neg_pred_in_dataset));
      precision = (double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_pos_pred_in_dataset);
	  recall = (double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_neg_pred_in_dataset);
      f1_score = (2*precision*recall)/(precision+recall);
      printf("f1_score=%f\n",f1_score);
	  
     /**************Classification End********************/
     
  }
  
  
  free_struct_sample(testsample);
  free_struct_model(model);

  svm_struct_classify_api_exit();

  return(0);
}

void read_input_parameters(int argc,char *argv[],char *testfile,
			   char *modelfile,char *predictionsfile,
			   STRUCT_LEARN_PARM *struct_parm,
			   long *verbosity,long *struct_verbosity)
{
  long i;
  
  /* set default */
  strcpy (modelfile, "svm_model");
  strcpy (predictionsfile, "svm_predictions"); 
  (*verbosity)=0;/*verbosity for svm_light*/
  (*struct_verbosity)=1; /*verbosity for struct learning portion*/
  struct_parm->custom_argc=0;

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case 'h': print_help(); exit(0);
      case '?': print_help(); exit(0);
      case '-': strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);i++; strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);break; 
      case 'v': i++; (*struct_verbosity)=atol(argv[i]); break;
      case 'y': i++; (*verbosity)=atol(argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if((i+1)>=argc) {
    printf("\nNot enough input parameters!\n\n");
    print_help();
    exit(0);
  }
  strcpy (testfile, argv[i]);
  strcpy (modelfile, argv[i+1]);
  if((i+2)<argc) {
    strcpy (predictionsfile, argv[i+2]);
  }

  parse_struct_parameters_classify(struct_parm);
}

void print_help(void)
{
  printf("\nSVM-struct classification module: %s, %s, %s\n",INST_NAME,INST_VERSION,INST_VERSION_DATE);
  printf("   includes SVM-struct %s for learning complex outputs, %s\n",STRUCT_VERSION,STRUCT_VERSION_DATE);
  printf("   includes SVM-light %s quadratic optimizer, %s\n",VERSION,VERSION_DATE);
  copyright_notice();
  printf("   usage: svm_struct_classify [options] example_file model_file output_file\n\n");
  printf("options: -h         -> this help\n");
  printf("         -v [0..3]  -> verbosity level (default 2)\n\n");

  print_struct_help_classify();
}

void printw_for_matlab(STRUCTMODEL model)
{
	long iter_w;
	
	for(iter_w=0;iter_w<model.svm_model->totwords;iter_w++)
      printf("%f\n",model.w[iter_w]);fflush(stdout);
   
}
void get_full_stats(LABEL *gt_y, LABEL *ybar, long *total_pos, long *total_neg, long *tp, long *tn,long *fp, long *fn)
{
	
	long tot_tok,iter_elems=0;
	long pos_lab,neg_lab;

	pos_lab=1;
	neg_lab=2;
    tot_tok=gt_y->length;

	while(iter_elems < tot_tok)
	{
		/*gt=1*/
		if(gt_y->labels[iter_elems] == pos_lab ){
			(*total_pos)=(*total_pos) + 1;
			}
		
		/*gt=2*/
		if(gt_y->labels[iter_elems] == neg_lab ){
			(*total_neg) = (*total_neg) + 1;
		}
		iter_elems++;
	}
    
    /* Reset iter_elems again and collect tp,tn,fp,fn*/
    iter_elems=0;
    while(iter_elems < tot_tok)
	{
		/*pred=1 and gt=1*/
		if(ybar->labels[iter_elems] == 1 && gt_y->labels[iter_elems] == 1){
			 (*tp)	= (*tp) + 1;
			}
		
		/*pred=1 and gt=2*/
		if(ybar->labels[iter_elems] == 1 && gt_y->labels[iter_elems] == 2){
			 (*fp)	= (*fp) + 1;
			}
		
		/*pred=2 and gt=1*/
		if(ybar->labels[iter_elems] == 2 && gt_y->labels[iter_elems] == 1){
			 (*fn)	= (*fn) + 1;
			}
		
		/*pred=2 and gt=2*/
		if(ybar->labels[iter_elems] == 2 && gt_y->labels[iter_elems] == 2){
			 (*tn)	= (*tn) + 1;
			}
		
		iter_elems++;
	}
    
    
}


