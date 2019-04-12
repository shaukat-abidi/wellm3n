#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "mvl_common.h"


int main (int argc, char* argv[])
{
    /*Declaration*/
    SAMPLE sample;  /* training sample */
    char trainfile[200];           /* file with training examples */
    char matlab_trainfile[200];           /* file for matlab */

    long  tot_ex,tot_feats;
    read_input_parameters(argc,argv,trainfile,matlab_trainfile);

	/****************Read 1st labeling from working_set again ******************************/
	/* read the training examples */
    printf("Reading %s \n",trainfile); fflush(stdout);
    sample=read_struct_examples(trainfile);
    printf("Done.\n");
    printf("Finsihed reading %s\n",trainfile); fflush(stdout);
   
   /*Initialization*/
   tot_ex = sample.n;
   tot_feats = sample.total_features;
   
   /****************Read 1st ground-truth again(END) ******************************/

    
   /* write for Matlab */
	char filename[255];
	snprintf(filename,255,matlab_trainfile);
    /*writing viterbi labelings*/
    write_matlabfile(filename,&sample);
    
    
}







