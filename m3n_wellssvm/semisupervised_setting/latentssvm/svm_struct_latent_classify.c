/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_classify.c                                       */
/*                                                                      */
/*   Classification Code for Latent SVM^struct                          */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 9.Nov.08                                                     */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include "svm_struct_latent_api.h"


void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile,char *predictionsfile, STRUCT_LEARN_PARM *sparm);
void update_stats_vars(STRUCT_LEARN_PARM *sparm,LATENT_VAR hbar, long *total_pos, long *total_neg, long *true_pos_pred, long *true_neg_pred);
void get_full_stats(STRUCT_LEARN_PARM *sparm,LATENT_VAR hbar, long *total_pos, long *total_neg, long *tp, long *tn,long *fp, long *fn);

int main(int argc, char* argv[]) {
  int j;
  long i;
  long iter_h;
  long total_tokens;
  long f_pred,t_pred;
  long pos_toks_in_sample,neg_toks_in_sample,tp_sample,fp_sample,fn_sample,tn_sample;
  long pos_toks_in_dataset,neg_toks_in_dataset,true_pos_pred_in_dataset,true_neg_pred_in_dataset,false_pos_pred_in_dataset,false_neg_pred_in_dataset;
  double individual_loss,accuracy,total_loss;
  double precision,recall,f1_score;
  
  char testfile[1024];
  char modelfile[1024];
  char predictionsfile[1024];

  FILE *predfl;
  STRUCTMODEL model;
  STRUCT_LEARN_PARM sparm;
  LEARN_PARM lparm;
  KERNEL_PARM kparm;

  SAMPLE testsample;
  LABEL y;
  LATENT_VAR h; 

  total_tokens = 0;
  f_pred = 0;
  pos_toks_in_dataset=0;
  neg_toks_in_dataset=0;
  true_pos_pred_in_dataset=0;
  true_neg_pred_in_dataset=0;
  false_pos_pred_in_dataset=0;
  false_neg_pred_in_dataset=0;
  accuracy = 0.0;
  total_loss = 0.0;
  
  /* read input parameters */
  read_input_parameters(argc,argv,testfile,modelfile,predictionsfile,&sparm);

  /* read model file */
  printf("Reading model..."); fflush(stdout);
  model = read_struct_model(modelfile, &sparm);
  printf("done.\n");

  /* read test examples */
  printf("Reading test examples..."); fflush(stdout);
  testsample = read_struct_examples(testfile,&sparm);
  printf("done.\n");

  init_struct_model(testsample,&model,&sparm,&lparm,&kparm);
  
  
  /*We should read states inside sparm, because our loss depends on h now*/
  /* impute latent variable for first iteration */
  init_latent_variables(&testsample,&lparm,&model,&sparm);
  
  for (i=0;i<testsample.n;i++) {	
	// printf("Number of classes %d\n",sparm.num_classes);
	/*addition*/
    sparm.lot = testsample.examples[i].x.length;
    sparm.sample_index = i;
    
    /* initialize variables for this sample */
	pos_toks_in_sample = 0; neg_toks_in_sample = 0; tp_sample = 0; fp_sample = 0; fn_sample = 0; tn_sample = 0;
	
    classify_struct_example(testsample.examples[i].x,&y,&h,&model,&sparm);
    //printf("Entering loss\n");
    individual_loss=loss_for_classification(h,&sparm);
    //loss_for_classification_test(h,&sparm,&individual_loss);
    //printf("sample:%ld loss=%f\n",i,individual_loss);
    //exit(0);
    
    // Prints states value ---------------------------
	printf("printing predicition for sample #%ld\n",i);
    for (j=0;j<(testsample.examples[i].x.length);j++) {
        printf("gt_tok_label[%d]=%ld  predicted_tok_label[%d]=%ld\n",j,sparm.h_init[i].states[j],j,h.states[j]);
	}	  
	//------------------------------------------------  
	//update_stats_vars(&sparm,h,&pos_toks_in_sample,&neg_toks_in_sample,&true_pos_pred_in_sample,&true_neg_pred_in_sample);
	 get_full_stats(&sparm,h,&pos_toks_in_sample,&neg_toks_in_sample,&tp_sample,&tn_sample,&fp_sample,&fn_sample);
	
	

    printf("sample-%ld: tp/totpos_toks_in_sample=%ld/%ld\n",i,tp_sample,pos_toks_in_sample);
    printf("sample-%ld: tn/totneg_toks_in_sample=%ld/%ld\n",i,tn_sample,neg_toks_in_sample);
    printf("sample-%ld: fp=%ld\n",i,fp_sample);
    printf("sample-%ld: fn=%ld\n",i,fn_sample);
    
    pos_toks_in_dataset+=pos_toks_in_sample;
    neg_toks_in_dataset+=neg_toks_in_sample;
    true_pos_pred_in_dataset+=tp_sample;
    true_neg_pred_in_dataset+=tn_sample;
    false_pos_pred_in_dataset+=fp_sample;
    false_neg_pred_in_dataset+=fn_sample;
  
    total_tokens +=  sparm.lot;
    total_loss += individual_loss;
    printf("false_prediction till here:%f\n",total_loss);
    
	free_label(y);
    free_latent_var(h);
	  
  }
  
  accuracy = (total_tokens-total_loss)/(total_tokens);
  printf("******************************************************\n");
  printf("tot_tokens=%ld,tot_pos_tokens=%ld,tot_neg_tokens=%ld\n",total_tokens,pos_toks_in_dataset,neg_toks_in_dataset);
  printf("tp=%ld,tn=%ld,fp=%ld,fn=%ld\n",true_pos_pred_in_dataset,true_neg_pred_in_dataset,false_pos_pred_in_dataset,false_neg_pred_in_dataset);
  printf("Total Loss(Tokens with wrong labels):%f\n",total_loss);
  precision = (double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_pos_pred_in_dataset);
  recall = (double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_neg_pred_in_dataset);
  f1_score = (2*precision*recall)/(precision+recall);
  printf("precision = %f,recall = %f\n",(double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_pos_pred_in_dataset),(double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_neg_pred_in_dataset));
  printf("Accuracy = %f,F1-Score=%f\n",accuracy,f1_score);
  
  if(accuracy > 0){
	  total_tokens = 0;
	  f_pred = 0;
	  pos_toks_in_dataset=0;
	  neg_toks_in_dataset=0;
	  true_pos_pred_in_dataset=0;
	  true_neg_pred_in_dataset=0;
	  false_pos_pred_in_dataset=0;
	  false_neg_pred_in_dataset=0;
	  accuracy = 0.0;
      total_loss = 0.0;

    for (i=0;i<testsample.n;i++) {
		sparm.lot = testsample.examples[i].x.length;
	    sparm.sample_index = i;
	    
	    /* initialize variables for this sample */
		pos_toks_in_sample = 0; neg_toks_in_sample = 0; tp_sample = 0; fp_sample = 0; fn_sample = 0; tn_sample = 0;
		
	    classify_struct_example(testsample.examples[i].x,&y,&h,&model,&sparm);
	    /*Invert h (START)*/
	    for(iter_h=0;iter_h<testsample.examples[i].x.length;iter_h++){
			if(h.states[iter_h] == 1){h.states[iter_h]=2;}
			else{h.states[iter_h]=1;}			
		}
	    /*Invert h (START)*/
	    individual_loss=loss_for_classification(h,&sparm);
		
		// Prints states value ---------------------------
		printf("printing predicition for sample #%ld\n",i);
	    for (j=0;j<(testsample.examples[i].x.length);j++) {
	        printf("gt_tok_label[%d]=%ld  predicted_tok_label[%d]=%ld\n",j,sparm.h_init[i].states[j],j,h.states[j]);
		}	  
	    //------------------------------------------------  
	    get_full_stats(&sparm,h,&pos_toks_in_sample,&neg_toks_in_sample,&tp_sample,&tn_sample,&fp_sample,&fn_sample);
	
	

	    printf("sample-%ld: tp/totpos_toks_in_sample=%ld/%ld\n",i,tp_sample,pos_toks_in_sample);
	    printf("sample-%ld: tn/totneg_toks_in_sample=%ld/%ld\n",i,tn_sample,neg_toks_in_sample);
	    printf("sample-%ld: fp=%ld\n",i,fp_sample);
	    printf("sample-%ld: fn=%ld\n",i,fn_sample);
	    
	    pos_toks_in_dataset+=pos_toks_in_sample;
	    neg_toks_in_dataset+=neg_toks_in_sample;
	    true_pos_pred_in_dataset+=tp_sample;
	    true_neg_pred_in_dataset+=tn_sample;
	    false_pos_pred_in_dataset+=fp_sample;
	    false_neg_pred_in_dataset+=fn_sample;
	  
	    total_tokens +=  sparm.lot;
	    total_loss += individual_loss;
	    printf("false_prediction till here:%f\n",total_loss);
	    
		free_label(y);
	    free_latent_var(h);
	}
	
	  accuracy = (total_tokens-total_loss)/(total_tokens);
	  printf("(RESULT INVERTED)******************************************************\n");
	  printf("(RESULT INVERTED)tot_tokens=%ld,tot_pos_tokens=%ld,tot_neg_tokens=%ld\n",total_tokens,pos_toks_in_dataset,neg_toks_in_dataset);
	  printf("(RESULT INVERTED)tp=%ld,tn=%ld,fp=%ld,fn=%ld\n",true_pos_pred_in_dataset,true_neg_pred_in_dataset,false_pos_pred_in_dataset,false_neg_pred_in_dataset);
	  printf("(RESULT INVERTED)Total Loss(Tokens with wrong labels):%f\n",total_loss);
	  precision = (double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_pos_pred_in_dataset);
      recall = (double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_neg_pred_in_dataset);
      f1_score = (2*precision*recall)/(precision+recall);
	  printf("(RESULT INVERTED)precision = %f,recall = %f\n",(double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_pos_pred_in_dataset),(double)true_pos_pred_in_dataset/(true_pos_pred_in_dataset+false_neg_pred_in_dataset));
      printf("(RESULT INVERTED)Accuracy = %f,F1-Score=%f\n",accuracy,f1_score);
	  
		
  }
  
  
  
  
  free_label(y);
  free_latent_var(h); 
  
  free_struct_sample(testsample);
  free_struct_model(model,&sparm);

  return(0);

}

void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile,char *predictionsfile, STRUCT_LEARN_PARM *sparm) {

  long i;
  
  /* set default */
  strcpy(modelfile, "svm_model");
  strcpy (predictionsfile, "svm_predictions");
  sparm->custom_argc = 0;

  for (i=1;(i<argc)&&((argv[i])[0]=='-');i++) {
    switch ((argv[i])[1]) {
      case '-': strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);i++; strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);break;  
      default: printf("\nUnrecognized option %s!\n\n",argv[i]); exit(0);    
    }
  }

  if (i+1>=argc) {
    printf("\nNot enough input parameters!\n\n");
    exit(0);
  }

  strcpy(testfile, argv[i]);
  strcpy(modelfile, argv[i+1]);
  if((i+2)<argc) {
    strcpy (predictionsfile, argv[i+2]);
  }

  parse_struct_parameters(sparm);

}
void update_stats_vars(STRUCT_LEARN_PARM *sparm,LATENT_VAR hbar, long *total_pos, long *total_neg, long *true_pos_pred, long *true_neg_pred)
{
	
	long tot_tok,iter_elems=0;
    long sample_id;
	long pos_lab,neg_lab;

    sample_id=sparm->sample_index;
	pos_lab=1;
	neg_lab=2;
    tot_tok=sparm->lot;

	while(iter_elems < tot_tok)
	{
		/*gt=1 and prediction=1*/
		if(sparm->h_init[sample_id].states[iter_elems] == pos_lab ){
			(*total_pos)=(*total_pos) + 1;
			if(sparm->h_init[sample_id].states[iter_elems] == hbar.states[iter_elems]){
			 (*true_pos_pred)	= (*true_pos_pred) + 1;
			}
		}
		
		/*gt=2 and prediction=2*/
		if(sparm->h_init[sample_id].states[iter_elems] == neg_lab ){
			(*total_neg) = (*total_neg) + 1;
			if(sparm->h_init[sample_id].states[iter_elems] == hbar.states[iter_elems]){
				(*true_neg_pred) = (*true_neg_pred) + 1;
			}
		}
		
		
		iter_elems++;
	}
     
     
}

void get_full_stats(STRUCT_LEARN_PARM *sparm,LATENT_VAR hbar, long *total_pos, long *total_neg, long *tp, long *tn,long *fp, long *fn)
{
	
	long tot_tok,iter_elems=0;
    long sample_id;
	long pos_lab,neg_lab;

    sample_id=sparm->sample_index;
	pos_lab=1;
	neg_lab=2;
    tot_tok=sparm->lot;

	while(iter_elems < tot_tok)
	{
		/*gt=1*/
		if(sparm->h_init[sample_id].states[iter_elems] == pos_lab ){
			(*total_pos)=(*total_pos) + 1;
			}
		
		/*gt=2*/
		if(sparm->h_init[sample_id].states[iter_elems] == neg_lab ){
			(*total_neg) = (*total_neg) + 1;
		}
		iter_elems++;
	}
    
    /* Reset iter_elems again and collect tp,tn,fp,fn*/
    iter_elems=0;
    while(iter_elems < tot_tok)
	{
		/*pred=1 and gt=1*/
		if(hbar.states[iter_elems] == 1 && sparm->h_init[sample_id].states[iter_elems] == 1){
			 (*tp)	= (*tp) + 1;
			}
		
		/*pred=1 and gt=2*/
		if(hbar.states[iter_elems] == 1 && sparm->h_init[sample_id].states[iter_elems] == 2){
			 (*fp)	= (*fp) + 1;
			}
		
		/*pred=2 and gt=1*/
		if(hbar.states[iter_elems] == 2 && sparm->h_init[sample_id].states[iter_elems] == 1){
			 (*fn)	= (*fn) + 1;
			}
		
		/*pred=2 and gt=2*/
		if(hbar.states[iter_elems] == 2 && sparm->h_init[sample_id].states[iter_elems] == 2){
			 (*tn)	= (*tn) + 1;
			}
		
		iter_elems++;
	}
    
    
    
    
    
    
}
