#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "mvl_common.h"


int main (int argc, char* argv[])
{
    
    /*Declaration*/
    SAMPLE supervised_sample,supervised_sample_buffer;  /* training sample with supervised labels */
    TYPE_SUPLABELS sup_labels;
    
    char selected_supervision_file[1024]; /* File having indices of supervised frames */
    char real_groundtruth_file[1024]; /* Real ground-truth training file used for labeling selected frames ONLY */
    
    read_input_parameters(argc,argv,real_groundtruth_file,selected_supervision_file);
    /* Print everything to check validity of files */ 
    printf("Reading ground-truth file ... : %s\n\n",real_groundtruth_file);
    printf("Reading file for selecting groundtruth labels for semi-supervised learning... : %s\n\n",selected_supervision_file);
    
    /********* Read important parameters from files *************/
	long lot,tot_entries;
	long iter_paramR, iter_tok;
	double *paramR=NULL;	
	read_L("/home/ssabidi/Shaukat/m3n_wellssvm/semisupervised_setting/semi_supervised_wellm3n/params/lot.txt",&lot);
	tot_entries = (2*lot) + (4*(lot-1));
	paramR = (double *)my_malloc(sizeof(double) * tot_entries);
	
	/*Initialize paramR*/
	for(iter_paramR=0;iter_paramR<tot_entries;iter_paramR++)
	 {paramR[iter_paramR] = 0.0;}
	
	/*Read paramR from file*/
    read_paramR("/home/ssabidi/Shaukat/m3n_wellssvm/semisupervised_setting/semi_supervised_wellm3n/params/paramR.txt",paramR);
    
    /*Print paramR*/
	for(iter_paramR=0;iter_paramR<tot_entries;iter_paramR++)
	{
	  printf("paramR[%ld] = %lf\n",iter_paramR,paramR[iter_paramR]);fflush(stdout);
	}
	/********* Read important parameters from files (End) *************/
	
	/* **************Selected Supervision*******************(Start) */
  long iter_entries, iter_ex;
  long tok_len,ex_num,tok_index;
   
  /* Read groundtruth file again */ 
  printf("Reading real ground-truth file for supervised labels ... \n"); fflush(stdout);
  supervised_sample=read_struct_examples(real_groundtruth_file);
  supervised_sample_buffer = read_struct_examples(real_groundtruth_file); 
  
  /* Reading supervised labels */
  read_supervised_label_file(selected_supervision_file,&sup_labels);
 
  /* Read file for supervised labels*/
  printf("Done.\n");
  
  /* prepare supervised sample here (Start) */
  /* Step-1: Change all labels to zero */
  for(iter_ex=0;iter_ex<supervised_sample.n;iter_ex++)
  {
	  tok_len = supervised_sample.examples[iter_ex].y.length;
	  for(iter_tok=0;iter_tok<tok_len;iter_tok++){
		  /* change all labels to zero */
		  supervised_sample.examples[iter_ex].y.labels[iter_tok]=0;
	  }
  }
  
  /* Step-2: Copy supervised labels from buffer groundtruth sample 
   * to supervised sample */
  
  for(iter_entries=0;iter_entries<sup_labels.tot_entries;iter_entries++)
  {
	ex_num = sup_labels.seq_id[iter_entries] - 1; /*1-based index*/
	tok_index = sup_labels.frame_id[iter_entries] - 1; /*1-based index*/   
	supervised_sample.examples[ex_num].y.labels[tok_index] = supervised_sample_buffer.examples[ex_num].y.labels[tok_index];	  
  }
  
  /* Step-3: Print supervised dataset for verification */
  //print_dataset(supervised_sample);
  
  /* prepare supervised sample here (End) */
  
  /* **************Selected Supervision*******************(END) */ 
	
	
	/**************************Viterbi Start******************************************/
    long *viterbi_label;
	long eb,tb;
	
	/*Malloc And Clear*/
	viterbi_label = (long *)my_malloc(lot * sizeof(long));
	clear_long_array(viterbi_label,lot);
	eb = 0;
	tb = (2*lot);
	//printf("eb: %ld, tb:%ld \n",eb,tb);
	
	/* Viterbi Semi-Supervised */
	get_best_labels_single_sequence_semisupervised(supervised_sample.examples[0].y.labels,viterbi_label,paramR,eb,tb,lot); 

   /* Print Viterbi labeling */
   printf("Printing Viterbi Output\n");
   
   for(iter_tok=0;iter_tok<lot;iter_tok++)
   {
	  printf("%ld\n",viterbi_label[iter_tok]);
   }
   
   /*write labeling to file*/
   write_labels_only_to_file("/home/ssabidi/Shaukat/m3n_wellssvm/semisupervised_setting/semi_supervised_wellm3n/params/viterbi_label.txt",viterbi_label,lot);

 	/**************************Viterbi End******************************************/

   
}







