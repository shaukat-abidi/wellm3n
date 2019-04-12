#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "mvl_common.h"


int main (int argc, char* argv[])
{
    /*Declaration*/
    
    /********* Get no of working set and set of its Alphas *************/
	long lot,tot_entries;
	long iter_paramR, iter_tok;
	double *paramR=NULL;	
	read_L("/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/lot.txt",&lot);
	tot_entries = (2*lot) + (4*(lot-1));
	paramR = (double *)my_malloc(sizeof(double) * tot_entries);
	
	/*Initialize paramR*/
	for(iter_paramR=0;iter_paramR<tot_entries;iter_paramR++)
	 {paramR[iter_paramR] = 0.0;}
	
	/*Read paramR from file*/
    read_paramR("/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/paramR.txt",paramR);
    
    /*Print paramR*/
	for(iter_paramR=0;iter_paramR<tot_entries;iter_paramR++)
	{
	  printf("paramR[%ld] = %lf\n",iter_paramR,paramR[iter_paramR]);fflush(stdout);
	}
	/********* Get no of working set and set of its Alphas(End) *************/
	
	
	
	/**************************Viterbi Start******************************************/
    long *viterbi_label;
	long eb,tb;
	
	/*Malloc And Clear*/
	viterbi_label = (long *)my_malloc(lot * sizeof(long));
	clear_long_array(viterbi_label,lot);
	eb = 0;
	tb = (2*lot);
	//printf("eb: %ld, tb:%ld \n",eb,tb);
	
	/*Viterbi*/
	get_best_labels_single_sequence(viterbi_label,paramR,eb,tb,lot);
	

   /* Print Viterbi labeling */
   printf("Printing Viterbi Output\n");
   
   for(iter_tok=0;iter_tok<lot;iter_tok++)
   {
	  printf("%ld\n",viterbi_label[iter_tok]);
   }
   
   /*write labeling to file*/
   write_labels_only_to_file("/home/sabidi/Shaukat/m3n_wellssvm/m3n_cvx/params/viterbi_label.txt",viterbi_label,lot);

 	/**************************Viterbi End******************************************/

   
}







