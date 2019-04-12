/*****************************************************************************/
/*                                                                           */
/*   svm_struct_latent_api.c                                                 */
/*                                                                           */
/*   API function definitions for HCRF in Latent SVM^struct                  */
/*                                                                           */
/*   Authors: Ehsan Zare Borzeshi & Massimo Piccardi                         */
/*   Date: Feb. 2013                                                         */
/*                                                                           */
/*   This code is based on the code for SVM-HMM from Thorsten Joachims.      */
/*   We have re-used the code wherever possible, simplifying index           */
/*   expressions and implementing only order 1 transitions and order 0       */
/*   emissions.                                                              */
/*                                                                           */
/*   This software is available for non-commercial use only. The authors     */
/*   are not responsible for implications from the use of this software.     */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/* Important notes:                                                          */ 
/*                                                                           */
/* The HCRF is the graphical model described in Quattoni et al. PAMI 2007    */
/* It consists of a top variable (class) specifying the action out of {1,M}, */
/* a time chain of intermediate variables (states) out of {1,N} each         */
/* and a time sequence of multidimensional observations (numf dimensions).   */  
/*                                                                           */
/* The parameters include the usual transition and emission parameters and   */
/* the parameters between class and states, called compatibility parameters. */
/*                                                                           */
/* All parameters are kept in a single vector, w, with these ranges of       */
/* indices:                                                                  */
/* index 0: unused (has a reserved meaning in some functions)                */
/* indices 1:N*N = transition parameters, in previous-state major order      */
/* indices N*N + 1:N*numf = emission parameters, state major order           */
/* indices N*N + N*numf + 1:M*N = compatibility parameters, class major order*/
/*                                                                           */
/* Throughout the code, we uses some pointers and indices to refer to these  */
/* ranges:                                                                   */
/* transbase: points to element of index (1)                                 */
/* emitbase: index (N*N)                                                     */
/* compbase: points to element of index (N*N + N*numf + 1)                   */
/* compbaseindex: index (N*N + N*numf)                                       */
/*                                                                           */
/* The code calls a Viterbi decoder in three functions:                      */
/* - classify_struct_example                                                 */
/* - find_most_violated_constraint_marginrescaling                           */
/* - infer_latent_variables                                                  */
/*                                                                           */
/* The Viterbi decoder is identical to that of HMM, but the emission scores  */
/* are modified prior to invokation to include the compatibility             */
/* parameters and the loss function as needed.                               */  
/*                                                                           */
/* Compared to the hmmsvm code from Joachims, we have used 0-based indices   */
/* and vectors/matrices of minimal size everywhere. Class and state labels   */
/* are instead maintained as 1-based. Given that these decisions are         */
/* internal to this module, we see no impact on the learner and classifier.  */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <assert.h>
#include "svm_struct_latent_api_types.h"

#define OUT
#define MAX(x,y)      ((x) < (y) ? (y) : (x))

/* TYPEDEF'S: ****************************************************************/

typedef struct viterbi_path {
  double    score;	               /* Viterbi score */
  long      label;                 /* label in current state */
  long      index;                 /* index that summarizes history of state */
  struct viterbi_path *backpointer;/* pointer to optimal VPATH element
				      in previous step that leads to current state */
} VPATH;

/* PROTOTYPES: ***************************************************************/

long *viterbi(double *trans, double **emit, long length, long states, 
              long trans_order, long emit_order, double *score);
long *viterbi_semisupervised(long *supervised_labels,double *trans, double **emit, long length, long states, 
              long trans_order, long emit_order, double *score);

VPATH *viterbi_forward_order1(double *trans, double **emit, long emit_order,
			      long start, long end, long states, VPATH **path);
VPATH *viterbi_forward_order1_semisupervised(long *supervised_labels,double *trans, double **emit, long emit_order,
			      long start, long end, long states, VPATH **path);
			      
double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm);
void loss_for_classification_test(LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm, double *calc_loss);
void print_dataset(SAMPLE sample_dataset);

/* read file */
void read_supervised_label_file(char *file, TYPE_SUPLABELS *sup_labels); 

/* FUNCTIONS: ****************************************************************/

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

/****************************************************************************/

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

/****************************************************************************/

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}

/****************************************************************************/
SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples=NULL;
  long     n;       /* number of examples */
  DOC      **examples_flat=NULL;
  double   *labels_flat=NULL;
  long     totwords=0,tottokens=0,i,queryid,maxlabel=0;

  /* Read as one set of examples using the read_documents function
     from SVM-light */
  read_documents(file,&examples_flat,&labels_flat,&totwords,&tottokens);

  /* Create examples by splitting the input into sequences of tokens. */
  queryid=-1;
  n=0;
  for(i=0;i<tottokens;i++) {
    if(queryid < (long)examples_flat[i]->queryid) {
      queryid=(long)examples_flat[i]->queryid;
      n++;
      examples=(EXAMPLE *)realloc(examples,sizeof(EXAMPLE)*(n));
      examples[n-1].x.length=1;
      examples[n-1].y.length=1;
      examples[n-1].x.tokens=(DOC **)my_malloc(sizeof(DOC **));
      examples[n-1].y.label_vector=(long *)my_malloc(sizeof(long *));
    }
    else if(queryid > (long)examples_flat[i]->queryid) {
      printf("ERROR (Line %ld): qid example ID's have to be in increasing order.\n",i+1);
      exit(1);
    }
    else {
      examples[n-1].x.length++;
      examples[n-1].y.length++;
      examples[n-1].x.tokens=(DOC **)realloc(examples[n-1].x.tokens,
				       sizeof(DOC **)*examples[n-1].x.length);
      examples[n-1].y.label_vector=(long *)realloc(examples[n-1].y.label_vector,
				       sizeof(long *)*examples[n-1].y.length);
    }
    examples[n-1].x.tokens[examples[n-1].x.length-1]=examples_flat[i];
    examples[n-1].y.label_vector[examples[n-1].y.length-1]=(long)labels_flat[i];
    /* the labels for the sequence are all the same, so we can as well keep the last: */
    examples[n-1].y.label=(long)labels_flat[i];    
    if(labels_flat[i] < 0) {
      printf("ERROR (Line %ld): Token label IDs cannot be negative.\n",i+1);
      exit(1);
    }
    maxlabel=MAX(maxlabel,(long)labels_flat[i]);
  }

  printf(" %ld examples, %ld tokens, %ld features, %ld classes...\n",n,tottokens,totwords,maxlabel);
  sparm->num_features=totwords;  	
  printf("sparm->num_features: %ld\n",sparm->num_features);
  
  
  free(examples_flat); 
  free(labels_flat);

  sample.n=n;
  sample.examples=examples;
  return(sample);
}

/****************************************************************************/

SAMPLE read_struct_examples_original_HCRF(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y follows the definition in 
  svm_struct_latent_api_types.h. The latent variables, h, are NOT initialized 
  in this function, but by the solver at the first iteration.
  This function has been adapted and tested over the specific data.
*/
  SAMPLE sample;
  EXAMPLE  *examples=NULL;
  long     n;       /* number of examples */
  DOC      **examples_flat=NULL;
  double   *labels_flat=NULL;
  long     totwords=0,tottokens=0,i,queryid,maxlabel=0;

  /* Read as one set of examples using the read_documents function from SVM-light */

  read_documents(file,&examples_flat,&labels_flat,&totwords,&tottokens);

  /* Create examples by splitting the input into sequences of tokens. */
	
  queryid=-1;
  n=0;
  for(i=0;i<tottokens;i++) {
    if(queryid < (long)examples_flat[i]->queryid) { /* first time index in the sequence */	
      queryid=(long)examples_flat[i]->queryid;
      n++;
      examples=(EXAMPLE *)realloc(examples,sizeof(EXAMPLE)*(n));
      examples[n-1].x.length=1;      
      examples[n-1].x.tokens=(DOC **)my_malloc(sizeof(DOC **));      
    }
    else if(queryid > (long)examples_flat[i]->queryid) { /* wrong file format */
      printf("ERROR (Line %ld): qid example ID's have to be in increasing order.\n",i+1);
      exit(1);
    }
    else { /* next time index in the sequence */	
      examples[n-1].x.length++;      
      examples[n-1].x.tokens=(DOC **)realloc(examples[n-1].x.tokens,
				       sizeof(DOC **)*examples[n-1].x.length);
    }
    examples[n-1].x.tokens[examples[n-1].x.length-1]=examples_flat[i];
	/* the labels for the sequence are all the same, so we can as well keep the last: */
    examples[n-1].y.label=(int)labels_flat[i]; 
    
    if(labels_flat[i] < 0) {
      printf("ERROR (Line %ld): Token label IDs cannot be negative.\n",i+1);
      exit(1);
    }    
  }

  free(examples_flat); 
  free(labels_flat);
  sample.n=n;
  sample.examples=examples;

  return(sample);
}

/****************************************************************************/

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
/*
  Initializes parameters in STRUCTMODEL *sm: 
  - sets the dimension of the feature space sm->sizePsi;
  - does not set the initial weights sm->w.
  Initializes some parameters in STRUCT_LEARN_PARM *sparm.	 
*/
  long   sizePsi;

  EXAMPLE  *examples=sample.examples;  
  sparm->num_classes=1;	
  sparm->num_states=2;
  
  /* ****************** */

  /* sparm->num_features has been initialized in read_struct_model */ 

  //sparm->num_features=72;  	/*Change it with different samples*/

  /* ****************** */
  
  sparm->hmm_trans_order=1;
  sparm->hmm_emit_order=0;  

  sm->svm_model = (MODEL *) my_malloc(sizeof(MODEL));  	  
  sm->svm_model->kernel_parm = *kparm;
  
  sizePsi= 1 +
		 + (sparm->num_states * sparm->num_states)
         + (sparm->num_states * sparm->num_features)
		 + (sparm->num_states * sparm->num_classes);

  sm->sizePsi=sizePsi;  
  sm->walpha=1;
  
  printf("In init_struct_example \n");
  printf("sparm->num_features: %ld\n",sparm->num_features);
  printf("sm->sizePsi: %ld\n",sm->sizePsi);
}

/****************************************************************************/

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/ 
  int i; 
  int j;

  // printf("states: %d\n\n", sparm->num_states);
  //srand(time(0));
  sparm->h_init = (LATENT_VAR *)my_malloc(sizeof(LATENT_VAR) * sample->n );	
  
  for (i=0;i<sample->n;i++) { 
	  sample->examples[i].h.states = (long *)my_malloc(sizeof(long)*sample->examples[i].x.length);
	  sparm->h_init[i].states = (long *)my_malloc(sizeof(long)*sample->examples[i].x.length);
	  // printf("\n Initial state values of test sample %d\n", i);
	  
	  for (j=0;j<sample->examples[i].x.length;j++) { 	  

      // method 1: alternately adds 0 and 1 to the class label
	  //sample->examples[i].h.states[j] = sample->examples[i].y.label + j % 2;       
	  sample->examples[i].h.states[j] = sample->examples[i].y.label_vector[j];       
	  sparm->h_init[i].states[j] = sample->examples[i].y.label_vector[j];
	  
	  /*Set y.label = 1*/
	  sample->examples[i].y.label = 1;        

      // method 2: alternates all state values with different rates
	  // sample->examples[i].h.states[j] = j % sparm->num_states + 1;

      // method 3: random initialization
	  // sample->examples[i].h.states[j] = random() % sparm->num_states + 1; 

      // method 4: no transitions
	  // sample->examples[i].h.states[j] = sample->examples[i].y.label; 

      // method 5: alternately adds 0 and 1 to the class label
	  // sample->examples[i].h.states[j] = sample->examples[i].y.label + (j + sample->examples[i].y.label) % 2; 

	  // method 6: alternately adds 0 and 1 to the class label
	  // sample->examples[i].h.states[j] = sample->examples[i].y.label + (j/2) % 2;

	  /*
	  // method 7: initial possitive samples (y = 1) with 1 and negative sample (y = 2) with random variable (2+)	
	  if (sample->examples[i].y.label == 1){
		 sample->examples[i].h.states[j] = sample->examples[i].y.label; 
	  };
	  int rn = random() % (sparm->num_states - 1);	  
	  if (sample->examples[i].y.label == 2){  
		      sample->examples[i].h.states[j] = rn + 2; 
	  }; 
	  */

	  /*	  
	  // method 8: initial possitive samples (y = 1) with 2 and negative sample (y = 2) with random variable (3+) and leave state 1 for future use	
	  if (sample->examples[i].y.label == 1){
		 sample->examples[i].h.states[j] = sample->examples[i].y.label+1; 
	  };
	  int rn = random() % (sparm->num_states - 2);	  
	  if (sample->examples[i].y.label == 2){		  
		      sample->examples[i].h.states[j] = rn + 3; 
	  };
	  */
		  
	  // Prints initial states values --------------------	   	    	  
      // printf(" %d  ", sample->examples[i].h.states[j]);	  
	  //-------------------------------------------------- 		  
		  
	  };
  };
  printf("Latent variables initialized.\n");
  for (i=0;i<sample->n;i++) {
	  for (j=0;j<sample->examples[i].x.length;j++){
		  printf("ex:%d,tok:%d,class:%ld,state:%ld\n",i+1,j+1,sample->examples[i].y.label,sparm->h_init[i].states[j]);
		  //printf("ex:%d,tok:%d,class:%d,state:%d\n",i+1,j+1,sample->examples[i].y.label,sample->examples[i].h.states[j]);
		  }
	  	  printf("----------------------\n");
	  }
};

/****************************************************************************/

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 

  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here). 
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight 
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. */

  SVECTOR *fvec=NULL,*fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes; 
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long i,back,order;
  double val;
  long emitbase, compbaseindex;
  double *fvec_n;
  WORD tfeat[2];
	
  /* printf("Interim highest feature number %d\n",sm->svm_model->totwords); */

  /****** create transition features: ******/
	
  tfeat[1].wnum=0;   /* ! this second element of tfeat is created only as terminator of the list of words */
  tfeat[0].weight=1; /* only this element (tfeat[0]) "counts" */
  fvec=NULL;

  /* in each SVECTOR we store the value of an indicator function.
	 The value (field tfeat[0].weight) is equal to 1,
	 and the feature number (field tfeat[0].wnum) encodes indices i, j linearised */
	
  for(i=1;i<length;i++) {
	tfeat[0].wnum = states * (h.states[i-1] - 1) + h.states[i];
	fshift=create_svector(tfeat,NULL,1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
	append_svector_list(fshift,fvec);
	fvec=fshift;
    /* printf("tfeat[0].wnum: %d\n", tfeat[0].wnum); */
  }

  /* retrieves the index at which the emission parameters start: */	
  emitbase = states * states;

  /* create emission features: */
  for(i=0;i<length;i++) {
    fshift=shift_s(x.tokens[i]->fvec,emitbase + fnum * (h.states[i] - 1));
    /* the above function creates an SVECTOR and copies the tokens in it.
       Such tokens are one observation, "o_t", dimension by dimension.
       However, it also shifts the index of each dimension (originally,
       from 1 to fnum) to make it unique for a particular state and
       also unique in the parameter vector */
    append_svector_list(fshift,fvec);
    fvec=fshift;
  }

  /* retrieves the index at which the compatibility parameters start: */	
  compbaseindex = emitbase + fnum * states;

  /* create compatibility features: */
  for(i=0;i<length;i++) {
	tfeat[0].wnum = compbaseindex + states * (y.label - 1) + h.states[i];
	fshift=create_svector(tfeat,NULL,1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
    append_svector_list(fshift,fvec);
    fvec=fshift;
  }

  return(fvec);
}

/****************************************************************************/

void classify_struct_example(PATTERN x, LABEL* y, LATENT_VAR* h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {

/*
  Classifies input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)> and returning only y. 
*/  

  double *score;
  long length=x.length;  	 
  long classes=sparm->num_classes;
  long states = sparm->num_states; 
  long fnum=sparm->num_features;
  double *trans;
  double **emit,*emiti,**emitcopy;
  double *comp;
  long i,j;
  long index,index_h; 
  double *transbase=sm->w + 1;  /* transition weights start at index 1 */
  double *compbase;
  long emitbase;
  double max;
  long max_index;
  
  score = (double *)my_malloc (sizeof(double) * classes);

  emitbase = states * states;
  
  /* creates emit, a (length x states) matrix: */
  emit=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emit[i]=(double *)my_malloc(sizeof(double)*(states));
    emiti=emit[i];

	/* computes the emission scores in each row: */
    for(index=0; index < states; index++) 
	emiti[index] = sprod_ns(&(sm->w[emitbase + fnum*index]),x.tokens[i]->fvec);
	/* how does sprod_ns know the length of the vectors it multiplies?:
	   The second argument points to an array of WORD which is explictely 
	   terminated (by a '0' wnum field) */
  }
	
  trans=(double *)my_malloc(sizeof(double)*(states*states));
  /* copy the transition weights: */
  index_h = states*states;
  for(index = 0; index < index_h; index++) {
      trans[index]=transbase[index];
  }

  compbase = sm->w + (states * states) + (states * fnum) + 1;

  comp=(double *)my_malloc(sizeof(double)*(states*classes));
  /* copy the compatibility weights: */
  index_h = states*classes;
  for(index = 0; index < index_h; index++) {
      comp[index] = compbase[index];
  }

  /* creates an auxiliary (length x states) matrix: */
	
  emitcopy=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emitcopy[i]=(double *)my_malloc(sizeof(double)*(states));
    }

  for(index = 0; index < classes; index++) { /* this loop calls Viterbi for each class */
	  
  /* copies emission scores and adds the compatibility scores for class index: */

	for(i=0;i<length;i++) {
	  for(j=0;j<states;j++) {
		  emitcopy[i][j] = emit[i][j] + comp[j + index * states];
	  }
    }
    
    h->states = viterbi(trans,emitcopy,length,states,
                     sparm->hmm_trans_order,sparm->hmm_emit_order, &score[index]);
  }

  /* searches for the label with the highest score: */
    	
  max = score[0];
  max_index = 0;
  
  /* Block it because we have one class problem	
  for (index = 1; index < classes; index++) {
	  if (score[index] > max) {
		  max = score[index];
		  max_index = index;		  
	  }
  }	
  */
  y->label =  max_index + 1; /* classes are in range 1... */

  // just compute the probability of max class	
  /* We dont need prob
  double denom = exp(score[0]);
  for (index = 1; index < classes; index++) {	
      denom = denom + exp(score[index]);
  }
  */ 	  
  //y->prob = 0;
  /*y->prob = (exp(max)/denom); We dont need prob */	
  //printf("Probablity of winning class: %.12f\n", y->prob); 


  for(i=0;i<length;i++) {
	 for(j=0;j<states;j++) {
		  emitcopy[i][j] = emit[i][j] + comp[j + max_index * states];
	  }
	}

  h->states = viterbi(trans,emitcopy,length,states,
                     sparm->hmm_trans_order,sparm->hmm_emit_order, &score[max_index]); 
	
  for(i=0;i<length;i++) 
    free(emit[i]);
  free(emit);
  for(i=0;i<length;i++) 
    free(emitcopy[i]);
  free(emitcopy);
  free(trans);

}

/****************************************************************************/

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {

  /*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar.

  Important note: in the HCRF, we assume the number of classes to be relatively small.
  We compute the score and the best hbar for each of them, we add the appropriate loss,
  and compare.
	  
*/

  LATENT_VAR h;
  LABEL aux;
  double *score;
  long length=x.length;
  long classes=sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  double *trans;
  double **emit,*emiti,**emitcopy;
  double *comp;
  long i,j;
  long index,index_h;
  double *transbase=sm->w + 1;  /* transition weights start at index 1 */
  double *compbase;
  long emitbase;
  double max;
  long max_index;
  long **ptr; /* array of pointers to arrays of long (state labels) */
  
  
  
  score = (double *)my_malloc (sizeof(double) * classes);
  ptr = (long **)my_malloc (sizeof(long *) * classes);

  emitbase = states * states;
	
  /* creates emit, a (length x states) matrix: */
  emit=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emit[i]=(double *)my_malloc(sizeof(double)*(states));
    emiti=emit[i];

	/* computes the emission scores in each row: */
    for(index=0; index < states; index++) 
	emiti[index] = sprod_ns(&(sm->w[emitbase + fnum*index]),x.tokens[i]->fvec);
	/* how does sprod_ns know the length of the vectors it multiplies?:
	   The second argument points to an array of WORD which is explictely 
	   terminated (by a '0' wnum field) */
  }
	
  trans=(double *)my_malloc(sizeof(double)*(states*states));
  /* copy the transition weights: */
  index_h = states*states;
  for(index = 0; index < index_h; index++) {
      trans[index]=transbase[index];
  }
   
  compbase = sm->w + (states * states) + (states * fnum) + 1;

  comp=(double *)my_malloc(sizeof(double)*(states*classes));
  /* copy the compatibility weights: */
  index_h = states*classes;
  for(index = 0; index < index_h; index++) {
      comp[index]=compbase[index];
  }

  /* OPTIONAL: adds a prior over some states by adding to comp: */
  // 1 = no prior
 for(index = 0; index < classes; index++) {
     comp[index*states] *= 1; // increases the first state's probability	 
	 // printf("class is %d and transition for its first (reserved) state is %d\n", index, comp[index*states]);
  }
 	
  /* creates an auxiliary (length x states) matrix: */
	
  emitcopy=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emitcopy[i]=(double *)my_malloc(sizeof(double)*(states));
    }
   
  for(index = 0; index < classes; index++) { /* this loop calls Viterbi for each class */
	  
  /* copies emission scores and adds the compatibility scores for class index: */
    {		
	for(i=0;i<length;i++) {   		
	  for(j=0;j<states;j++) 		  	 
		  emitcopy[i][j] = emit[i][j] + comp[j + index * states];
	  }
    }
    
    ptr[index] = viterbi(trans,emitcopy,length,states,
                     sparm->hmm_trans_order,sparm->hmm_emit_order, &score[index]);
    /* NB: function viterbi allocates memory for the returned array */
  }
 
  /* searches for the label with the highest score + loss: */
  //printf("line:%ld,file:%s\n",__LINE__,__FILE__);
 	
  aux.label = 0;
  
  (*hbar).states =  ptr[0];/*Added by MS*/
  max = score[0] + loss(y,aux,*hbar,sparm);
  max_index = 0;
  for (index = 1; index < classes; index++) {
	  (*hbar).states =  ptr[index];/*Added by MS*/
      aux.label = index;
      /*Changed by MS: loss(y,aux,*hbar,sparm)*/
	  if (score[index] + loss(y,aux,*hbar,sparm) > max) {
		  max = score[index] + loss(y,aux,*hbar,sparm);
		  max_index = index;
	  }
  }
  //printf("line:%ld,file:%s\n",__LINE__,__FILE__);

  /* assigns the output parameters: */
  (*ybar).label =  1; /* classes are in range 1... */
  //(*ybar).label =  max_index + 1; /* classes are in range 1... */
  //(*hbar).states =  ptr[max_index];
		
  for(i = 0; i < classes; i++)
    if (i != max_index)
	  free(ptr[i]); /* frees the memory allocated by the Viterbi calls
	                   which are not the best hbar */
  free(ptr);

  for(i=0;i<length;i++) 
    free(emit[i]);
  free(emit);

  for(i=0;i<length;i++) 
    free(emitcopy[i]);
  free(emitcopy);

  free(trans);
	
  free(comp);
	
}

/****************************************************************************/

LATENT_VAR infer_latent_variables_original(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Compute the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  LATENT_VAR h;
  double score;
  long length=x.length;
  long classes=sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  double *trans;
  double **emit,*emiti;
  double *comp;
  long i,j;
  long index,index_h;
  double *transbase=sm->w + 1;  /* first weights in w are the transition scores */
  double *compbase;
  long emitbase;

  emitbase = states * states;

  /* creates emit, a (length x states) matrix: */
  emit=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emit[i]=(double *)my_malloc(sizeof(double)*(states));
    emiti=emit[i];

	/* computes the emission scores in each row: */
    for(index=0; index < states; index++) 
	emiti[index] = sprod_ns(&(sm->w[emitbase + fnum*index]),x.tokens[i]->fvec);
	/* how does sprod_ns know the length of the vectors it multiplies?:
	   The second argument points to an array of WORD which is explictely 
	   terminated (by a '0' wnum field) */
  }

  trans=(double *)my_malloc(sizeof(double)*(states*states));
  /* copy the transition weights: */
  index_h = states*states;
  for(index = 0; index < index_h; index++) {
      trans[index]=transbase[index];
  }

  compbase = sm->w + (states * states) + (states * fnum) + 1;

  comp=(double *)my_malloc(sizeof(double)*(states*classes));
  /* copy the compatibility weights: */
  index_h = states*classes;
  for(index = 0; index < index_h; index++) {
      comp[index]=compbase[index];
  }

  /* adds compatibility scores to the emission scores: */

  for(i=0;i<length;i++) {
	  for(j=0;j<states;j++) {
		  emit[i][j] += comp[j + (y.label - 1) * states];
	  }
  }
  
  h.states = viterbi(trans,emit,length,states,
                     sparm->hmm_trans_order,sparm->hmm_emit_order, &score);
  
  printf("Inside InferLatent-Vars\n");
  long indexi;
  for (indexi = 0; indexi < length; indexi++)
	  {printf("%ld ", h.states[indexi]);}
  printf("\n");
	
  for(i=0;i<length;i++) 
    free(emit[i]);
  free(emit);
  free(trans);

  return(h); 
}

/****************************************************************************/
LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, long *supervised_labels ,STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  /* Copied from Find_MVC_MR*/
  LATENT_VAR h;
  LABEL aux;
  double *score;
  long length=x.length;
  long classes=sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  double *trans;
  double **emit,*emiti,**emitcopy;
  double *comp;
  long i,j;
  long index,index_h;
  double *transbase=sm->w + 1;  /* transition weights start at index 1 */
  double *compbase;
  long emitbase;
  double max;
  long max_index;
  long **ptr; /* array of pointers to arrays of long (state labels) */
  
  
  
  score = (double *)my_malloc (sizeof(double) * classes);
  ptr = (long **)my_malloc (sizeof(long *) * classes);

  emitbase = states * states;
	
  /* creates emit, a (length x states) matrix: */
  emit=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emit[i]=(double *)my_malloc(sizeof(double)*(states));
    emiti=emit[i];

	/* computes the emission scores in each row: */
    for(index=0; index < states; index++) 
	emiti[index] = sprod_ns(&(sm->w[emitbase + fnum*index]),x.tokens[i]->fvec);
	/* how does sprod_ns know the length of the vectors it multiplies?:
	   The second argument points to an array of WORD which is explicitely 
	   terminated (by a '0' wnum field) */
  }
	
  trans=(double *)my_malloc(sizeof(double)*(states*states));
  /* copy the transition weights: */
  index_h = states*states;
  for(index = 0; index < index_h; index++) {
      trans[index]=transbase[index];
  }
   
  compbase = sm->w + (states * states) + (states * fnum) + 1;

  comp=(double *)my_malloc(sizeof(double)*(states*classes));
  /* copy the compatibility weights: */
  index_h = states*classes;
  for(index = 0; index < index_h; index++) {
      comp[index]=compbase[index];
  }

  /* OPTIONAL: adds a prior over some states by adding to comp: */
  // 1 = no prior
 for(index = 0; index < classes; index++) {
     comp[index*states] *= 1; // increases the first state's probability	 
	 // printf("class is %d and transition for its first (reserved) state is %d\n", index, comp[index*states]);
  }
 	
  /* creates an auxiliary (length x states) matrix: */
	
  emitcopy=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emitcopy[i]=(double *)my_malloc(sizeof(double)*(states));
    }
   
  for(index = 0; index < classes; index++) { /* this loop calls Viterbi for each class */
	  
  /* copies emission scores and adds the compatibility scores for class index: */
    {		
	for(i=0;i<length;i++) {   		
	  for(j=0;j<states;j++) 		  	 
		  emitcopy[i][j] = emit[i][j] + comp[j + index * states];
	  }
    }
    
    /* Replacing it with semi-supervised version of Viterbi Algorithm */
    //ptr[index] = viterbi(trans,emitcopy,length,states,sparm->hmm_trans_order,sparm->hmm_emit_order, &score[index]);
    
    ptr[index] = viterbi_semisupervised(supervised_labels,trans,emitcopy,length,states,sparm->hmm_trans_order,sparm->hmm_emit_order, &score[index]);
    printf("Viterbi has been executed ! \n");
    //exit(0);                 
                     
    /* NB: function viterbi allocates memory for the returned array */
  }
 
  /* searches for the label with the highest score + loss: */
  //printf("line:%ld,file:%s\n",__LINE__,__FILE__);
 	
  aux.label = 0;
  
  h.states =  ptr[0];/*Added by MS*/
  max = score[0];
  max_index = 0;
  
  printf("Inside InferLatent-Vars (total_classes:%ld Sample:%ld)\n",classes,sparm->sample_index);
  long indexi;
  for (indexi = 0; indexi < length; indexi++)
	  {printf("%ld ",h.states[indexi]);}
  printf("\n");
  	
  for(i = 0; i < classes; i++)
    if (i != max_index)
	  free(ptr[i]); /* frees the memory allocated by the Viterbi calls
	                   which are not the best hbar */
  free(ptr);

  for(i=0;i<length;i++) 
    free(emit[i]);
  free(emit);

  for(i=0;i<length;i++) 
    free(emitcopy[i]);
  free(emitcopy);

  free(trans);
	
  free(comp);


  return(h); 
}
/****************************************************************************/

LATENT_VAR infer_latent_variables_unsupervised_viterbi(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  /* Copied from Find_MVC_MR*/
  LATENT_VAR h;
  LABEL aux;
  double *score;
  long length=x.length;
  long classes=sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  double *trans;
  double **emit,*emiti,**emitcopy;
  double *comp;
  long i,j;
  long index,index_h;
  double *transbase=sm->w + 1;  /* transition weights start at index 1 */
  double *compbase;
  long emitbase;
  double max;
  long max_index;
  long **ptr; /* array of pointers to arrays of long (state labels) */
  
  
  
  score = (double *)my_malloc (sizeof(double) * classes);
  ptr = (long **)my_malloc (sizeof(long *) * classes);

  emitbase = states * states;
	
  /* creates emit, a (length x states) matrix: */
  emit=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emit[i]=(double *)my_malloc(sizeof(double)*(states));
    emiti=emit[i];

	/* computes the emission scores in each row: */
    for(index=0; index < states; index++) 
	emiti[index] = sprod_ns(&(sm->w[emitbase + fnum*index]),x.tokens[i]->fvec);
	/* how does sprod_ns know the length of the vectors it multiplies?:
	   The second argument points to an array of WORD which is explicitely 
	   terminated (by a '0' wnum field) */
  }
	
  trans=(double *)my_malloc(sizeof(double)*(states*states));
  /* copy the transition weights: */
  index_h = states*states;
  for(index = 0; index < index_h; index++) {
      trans[index]=transbase[index];
  }
   
  compbase = sm->w + (states * states) + (states * fnum) + 1;

  comp=(double *)my_malloc(sizeof(double)*(states*classes));
  /* copy the compatibility weights: */
  index_h = states*classes;
  for(index = 0; index < index_h; index++) {
      comp[index]=compbase[index];
  }

  /* OPTIONAL: adds a prior over some states by adding to comp: */
  // 1 = no prior
 for(index = 0; index < classes; index++) {
     comp[index*states] *= 1; // increases the first state's probability	 
	 // printf("class is %d and transition for its first (reserved) state is %d\n", index, comp[index*states]);
  }
 	
  /* creates an auxiliary (length x states) matrix: */
	
  emitcopy=(double **)my_malloc(sizeof(double*)*length);
  for(i=0;i<length;i++) {
    emitcopy[i]=(double *)my_malloc(sizeof(double)*(states));
    }
   
  for(index = 0; index < classes; index++) { /* this loop calls Viterbi for each class */
	  
  /* copies emission scores and adds the compatibility scores for class index: */
    {		
	for(i=0;i<length;i++) {   		
	  for(j=0;j<states;j++) 		  	 
		  emitcopy[i][j] = emit[i][j] + comp[j + index * states];
	  }
    }
    
    
    ptr[index] = viterbi(trans,emitcopy,length,states,
                     sparm->hmm_trans_order,sparm->hmm_emit_order, &score[index]);
                     
                     
    /* NB: function viterbi allocates memory for the returned array */
  }
 
  /* searches for the label with the highest score + loss: */
  //printf("line:%ld,file:%s\n",__LINE__,__FILE__);
 	
  aux.label = 0;
  
  h.states =  ptr[0];/*Added by MS*/
  max = score[0];
  max_index = 0;
  
  printf("Inside InferLatent-Vars (total_classes:%ld Sample:%ld)\n",classes,sparm->sample_index);
  long indexi;
  for (indexi = 0; indexi < length; indexi++)
	  {printf("%ld ",h.states[indexi]);}
  printf("\n");
  	
  for(i = 0; i < classes; i++)
    if (i != max_index)
	  free(ptr[i]); /* frees the memory allocated by the Viterbi calls
	                   which are not the best hbar */
  free(ptr);

  for(i=0;i<length;i++) 
    free(emit[i]);
  free(emit);

  for(i=0;i<length;i++) 
    free(emitcopy[i]);
  free(emitcopy);

  free(trans);
	
  free(comp);


  return(h); 
}

/****************************************************************************/



double loss_original(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {

/*
  Computes the loss of prediction ybar against the correct label y.
  This function seems part of the interface (must exist, and with this proptotype)
*/

    if (y.label==ybar.label) {
    return(0);
  } else {
    return(1);
  }
}

double loss_semisupervised_backup(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {

/*
  Computes the loss of prediction ybar against the correct label y.
  This function seems part of the interface (must exist, and with this proptotype)
*/  
     long iter_tok;
     long tot_tok;
     long sample_id;
     double calc_loss;
     
     tot_tok=sparm->lot;
     sample_id=sparm->sample_index;
     calc_loss=0.0;
     
         
     for(iter_tok=0;iter_tok<tot_tok;iter_tok++){
          //printf("tok:%ld -- qid:%ld  -- gt:%ld -- pred:%ld \n",iter_tok,sample_id,sparm->supervised_sample.examples[sample_id].y.label_vector[iter_tok],hbar.states[iter_tok]);

         //printf("sample:%ld total_tok=%ld tok=%ld hbar.states[%ld]:%ld sparm->h_init[%ld].states[%ld]:%ld\n"
         //,sample_id,tot_tok,iter_tok,iter_tok,hbar.states[iter_tok],sample_id,iter_tok,sparm->h_init[sample_id].states[iter_tok]);

		 if(sparm->supervised_sample.examples[sample_id].y.label_vector[iter_tok] != 0 ){
			 /* Supervised token; hence loss must be measured against it */
			 if(hbar.states[iter_tok] != sparm->supervised_sample.examples[sample_id].y.label_vector[iter_tok]){
				 calc_loss += (float)(1.0/tot_tok);
				 }
		 }
		 
	}
     
     printf("Loss Function: qid:%ld tot_loss=%f\n",sample_id,calc_loss);
     return(calc_loss);
    
}

double loss_unsupervised(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {

/*******************************************************/
/* This is the loss function for unsupervised learning */
/*******************************************************/

/*
  Computes the loss of prediction ybar against the correct label y.
  This function seems part of the interface (must exist, and with this proptotype)
*/  
     long iter_tok;
     long tot_tok;
     long sample_id;
     double calc_loss;
     
     tot_tok=sparm->lot;
     sample_id=sparm->sample_index;
     calc_loss=0.0;
     
     for(iter_tok=0;iter_tok<tot_tok;iter_tok++){
         //printf("sample:%ld total_tok=%ld tok=%ld hbar.states[%ld]:%ld sparm->h_init[%ld].states[%ld]:%ld\n"
         //,sample_id,tot_tok,iter_tok,iter_tok,hbar.states[iter_tok],sample_id,iter_tok,sparm->h_init[sample_id].states[iter_tok]);

		 if(hbar.states[iter_tok]!=sparm->h_init[sample_id].states[iter_tok])
		 	 calc_loss += (float)(1.0/tot_tok);
		 
		 }
     
     //printf("sample:%ld loss=%f\n",sample_id,calc_loss);
     return(calc_loss);
    
}

double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {

/*******************************************************/
/* This is the loss function for semisupervised learning */
/*******************************************************/

/*
  Computes the loss of prediction ybar against the correct label y.
  This function seems part of the interface (must exist, and with this proptotype)
*/  
     long iter_tok;
     long tot_tok;
     long sample_id;
     double calc_loss;
     
     tot_tok=sparm->lot;
     sample_id=sparm->sample_index;
     calc_loss=0.0;
     
         
     for(iter_tok=0;iter_tok<tot_tok;iter_tok++){
          //printf("tok:%ld -- qid:%ld  -- gt:%ld -- pred:%ld ",iter_tok,sample_id,sparm->supervised_sample.examples[sample_id].y.label_vector[iter_tok],hbar.states[iter_tok]);

         //printf("sample:%ld total_tok=%ld tok=%ld hbar.states[%ld]:%ld sparm->h_init[%ld].states[%ld]:%ld\n"
         //,sample_id,tot_tok,iter_tok,iter_tok,hbar.states[iter_tok],sample_id,iter_tok,sparm->h_init[sample_id].states[iter_tok]);

		 if(sparm->supervised_sample.examples[sample_id].y.label_vector[iter_tok] != 0 ){
			 /* Supervised token; hence loss must be measured against it */
			 if(hbar.states[iter_tok] != sparm->supervised_sample.examples[sample_id].y.label_vector[iter_tok]){
				 calc_loss += (float)(1.0/tot_tok);
   	             //printf(" .. loss added: %f\n",calc_loss);
				 }
		 }
		 
          //printf("\n");

	}
     
     printf("Loss Function: qid:%ld tot_loss=%f\n",sample_id,calc_loss);
     return(calc_loss);
    
}

double loss_for_classification(LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {

/*
  Computes the loss of prediction ybar against the correct label y.
  This function seems part of the interface (must exist, and with this proptotype)
*/  
     double calc_loss;
     long iter_tok;
     long tot_tok;
     long sample_id;
     
     tot_tok=sparm->lot;
     sample_id=sparm->sample_index;
     calc_loss=0;
     
     for(iter_tok=0;iter_tok<tot_tok;iter_tok++){
         
		 if(hbar.states[iter_tok]!=sparm->h_init[sample_id].states[iter_tok])
		 	 {
				 calc_loss = (calc_loss+1);
			 }
			 
		 //printf("sample:%ld total_tok=%ld tok=%ld hbar.states[%ld]:%ld sparm->h_init[%ld].states[%ld]:%ld loss=%f\n"
         //,sample_id,tot_tok,iter_tok,iter_tok,hbar.states[iter_tok],sample_id,iter_tok,sparm->h_init[sample_id].states[iter_tok]
         //,calc_loss);

		 
		 }
     
     //printf("inside loss function --- sample:%ld loss=%f\n",sample_id,calc_loss);
     return calc_loss;    
}

void loss_for_classification_test(LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm, double *calc_loss) {

/*
  Computes the loss of prediction ybar against the correct label y.
  This function seems part of the interface (must exist, and with this proptotype)
*/  
     long iter_tok;
     long tot_tok;
     long sample_id;
     
     tot_tok=sparm->lot;
     sample_id=sparm->sample_index;
     *calc_loss=0;
     
     for(iter_tok=0;iter_tok<tot_tok;iter_tok++){
         
		 if(hbar.states[iter_tok]!=sparm->h_init[sample_id].states[iter_tok])
		 	 {
				 *calc_loss = (*calc_loss+1);
			 }
			 
		 printf("sample:%ld total_tok=%ld tok=%ld hbar.states[%ld]:%ld sparm->h_init[%ld].states[%ld]:%ld loss=%f\n"
         ,sample_id,tot_tok,iter_tok,iter_tok,hbar.states[iter_tok],sample_id,iter_tok,sparm->h_init[sample_id].states[iter_tok]
         ,*calc_loss);

		 
		 }
     
     printf("inside loss function --- sample:%ld loss=%f\n",sample_id,*calc_loss);
     
}
/****************************************************************************/
void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {

  // Writes the learned weight vector sm->w to file file after training. 

  FILE *modelfl;  
         
  printf("Start writing model file ...\n"); fflush(stdout);

  if ((modelfl = fopen (file, "w")) == NULL)
  { perror (file); exit (1); } 

  fprintf(modelfl,"HCRF - parameters follow :\n");	  
  fprintf(modelfl,"%ld # kernel type\n", sm->svm_model->kernel_parm.kernel_type);
  fprintf(modelfl,"%ld # kernel parameter -d \n", sm->svm_model->kernel_parm.poly_degree);
  fprintf(modelfl,"%.8g # kernel parameter -g \n", sm->svm_model->kernel_parm.rbf_gamma);
  fprintf(modelfl,"%.8g # kernel parameter -s \n", sm->svm_model->kernel_parm.coef_lin);
  fprintf(modelfl,"%.8g # kernel parameter -r \n", sm->svm_model->kernel_parm.coef_const);  
  fprintf(modelfl,"%ld # number of emission features \n",sparm->num_features);
  fprintf(modelfl,"%ld # number of classes \n",sparm->num_classes);
  fprintf(modelfl,"%ld # number of states \n",sparm->num_states);	
  fprintf(modelfl,"%ld # HMM order of transitions \n",sparm->hmm_trans_order);
  fprintf(modelfl,"%ld # HMM order of emissions \n",sparm->hmm_emit_order);
  fprintf(modelfl,"%d # loss function \n",sparm->loss_function);  
  fprintf(modelfl,"%ld # Size of Psi \n", sm->sizePsi);	
  fwrite ((void *)sm->w, sizeof(double), sm->sizePsi, modelfl);
	
  fclose(modelfl);

  printf("Finished writing model file.\n"); fflush(stdout);
}

/****************************************************************************/
STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Reads in the learned model parameters from file file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
  This function is used only in the prediction module, not in the learning module.
*/

  FILE *modelfl;
  long iter_w;
  STRUCTMODEL sm;
	
  sm.svm_model = (MODEL *) my_malloc(sizeof(MODEL));    
  iter_w = 1;	

  if ((modelfl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }

  fscanf(modelfl,"HCRF - parameters follow :\n");
  fscanf(modelfl,"%ld%*[^\n]\n", &sm.svm_model->kernel_parm.kernel_type);	
  fscanf(modelfl,"%ld%*[^\n]\n", &sm.svm_model->kernel_parm.poly_degree);
  fscanf(modelfl,"%lf%*[^\n]\n", &sm.svm_model->kernel_parm.rbf_gamma);
  fscanf(modelfl,"%lf%*[^\n]\n", &sm.svm_model->kernel_parm.coef_lin);
  fscanf(modelfl,"%lf%*[^\n]\n", &sm.svm_model->kernel_parm.coef_const);
  fscanf(modelfl,"%ld%*[^\n]\n", &sparm->num_features);
  fscanf(modelfl,"%ld%*[^\n]\n", &sparm->num_classes);
  fscanf(modelfl,"%ld%*[^\n]\n", &sparm->num_states);
  fscanf(modelfl,"%ld%*[^\n]\n", &sparm->hmm_trans_order);
  fscanf(modelfl,"%ld%*[^\n]\n", &sparm->hmm_emit_order);
  fscanf(modelfl, "%d%*[^\n]\n", &sparm->loss_function);  
  fscanf(modelfl,"%ld%*[^\n]\n", &sm.sizePsi);  
  sm.w = (double*) my_malloc(sizeof(double) * sm.sizePsi);	
  fread((void *)sm.w, sizeof(double), sm.sizePsi, modelfl);

  
  printf("Kernel type %ld\n",sm.svm_model->kernel_parm.kernel_type);
  printf("Poly degree %ld\n",sm.svm_model->kernel_parm.poly_degree);
  printf("RBF gamma %lf\n"  ,sm.svm_model->kernel_parm.rbf_gamma);	
  printf("Coef Lin %lf\n"  ,sm.svm_model->kernel_parm.coef_lin);	
  printf("Coef cost %lf\n"  ,sm.svm_model->kernel_parm.coef_const);
  printf("Number of features %ld\n" ,sparm->num_features);	
  printf("Number of classes %ld\n" ,sparm->num_classes);	
  printf("Number of states %ld\n" ,sparm->num_states);
  printf("Size of Psi %ld\n" ,sm.sizePsi);
  printf("Weight %lf\n" ,sm.w[1]);	
  printf("test\n");	
  
  for (iter_w=1; iter_w <= sm.sizePsi; iter_w++)
  {
  printf ("w[%ld]=%lf\n", iter_w, sm.w[iter_w]);
  }
  
  //exit(0);	
  return(sm);
}

/****************************************************************************/
void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */  
    fprintf(fp,"%ld %.12f\n",y.label,y.prob);
} 

/****************************************************************************/

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/  
  free(sm.w);
}

/****************************************************************************/

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/
  long i;
  for(i=0;i<x.length;i++)
  free_example(x.tokens[i],1);
  free(x.tokens);
}

/****************************************************************************/

void free_label(LABEL y) {
/*
  NB: no memory needs to be malloc'ed when creating label y, which is a static field.
*/
} 

/****************************************************************************/

void free_latent_var(LATENT_VAR h) {
/*
  NB: no memory is malloc'ed to the states as it's not needed.
*/
}

/****************************************************************************/

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);
}

/****************************************************************************/

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;  
  /* set default */  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */		
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }	 
}

/****************************************************************************/

long *viterbi(double *trans, double **emit, long length, long states, 
              long trans_order, long emit_order, double *score)

 /* Computes the best state path for given transition weights and emission scores.
 /* NB! For the HCRF, the Viterbi function is identical as for a common HMM.
	Prior to invocation, one has to add the compatibility scores to the emission 
	scores. All the rest is unvaried */

{
  VPATH  **path,*best,*pathbuf;
  long   *bestpathlabel;
  double *bestpathscore;
  long   location;
  long   i;

  if(length==0) {
    return(NULL);
  }
  
  /* Creates path, a (length x states) matrix of VPATH elements. */
  path=(VPATH **)my_malloc(sizeof(VPATH *)*length);
  pathbuf=(VPATH *)my_malloc(sizeof(VPATH)*length*(states));
  for(i=0;i<length;i++) 
    path[i]=pathbuf + i * states;

  /* Do forward pass of Viterbi. */    
   best=viterbi_forward_order1(trans,emit,emit_order,0,length-1,states,path);
  
  /* Trace back the highest scoring path. */
  bestpathscore=(double *)my_malloc(sizeof(long)*length);
  bestpathlabel=(long *)my_malloc(sizeof(long)*length);	
  for(location=length-1;location>=0;location--) {		 	  
    bestpathlabel[location]=best->label;  /* not strictly needed to store them all */
	bestpathscore[location]=best->score;  /* not strictly needed to store them all */
    best=best->backpointer;
  }	  
  *score = bestpathscore[length - 1]; /* the score for the whole sequence
	                                    should be in its last element */	
  free(pathbuf);
  free(path);
  return(bestpathlabel);
}

/****************************************************************************/

/****************************************************************************/

long *viterbi_semisupervised(long *supervised_labels, double *trans, double **emit, long length, long states, 
              long trans_order, long emit_order, double *score)

 /* Computes the best state path for given transition weights and emission scores.
 /* NB! For the HCRF, the Viterbi function is identical as for a common HMM.
	Prior to invocation, one has to add the compatibility scores to the emission 
	scores. All the rest is unvaried */

{
  VPATH  **path,*best,*pathbuf;
  long   *bestpathlabel;
  double *bestpathscore;
  long   location;
  long   i;

  if(length==0) {
    return(NULL);
  }
  
  /* Creates path, a (length x states) matrix of VPATH elements. */
  path=(VPATH **)my_malloc(sizeof(VPATH *)*length);
  pathbuf=(VPATH *)my_malloc(sizeof(VPATH)*length*(states));
  for(i=0;i<length;i++) 
    path[i]=pathbuf + i * states;

  /* Do forward pass of Viterbi. */    
  best=viterbi_forward_order1_semisupervised(supervised_labels,trans,emit,emit_order,0,length-1,states,path);
  printf("Viterbi: Forward pass computed! \n");
  
  /* Trace back the highest scoring path. */
  bestpathscore=(double *)my_malloc(sizeof(long)*length);
  bestpathlabel=(long *)my_malloc(sizeof(long)*length);	
  for(location=length-1;location>=0;location--) {		 	  
    bestpathlabel[location]=best->label;  /* not strictly needed to store them all */
	bestpathscore[location]=best->score;  /* not strictly needed to store them all */
    best=best->backpointer;
  }	  
  *score = bestpathscore[length - 1]; /* the score for the whole sequence
	                                    should be in its last element */	
  free(pathbuf);
  free(path);
  return(bestpathlabel);
}

/****************************************************************************/
/****************************************************************************/
VPATH *viterbi_forward_order1_semisupervised(long *supervised_states, double *trans, double **emit, long emit_order,
			      long start, long end, long states, VPATH **path)

/* Computes order 1 forward pass of Viterbi for given transition weights
	and emission scores. Assumes that the sequence is at least 2
	tokens long. */
{
  /* supervised_states: States labeled as {1,2} must not change during Viterbi */
  
						  /*********************************/
						  /* MUST NOT SUPERVISE LAST STATE */	
						  /*********************************/
						  
  
  double smax,s;
  long   p;
  int    lastpos;
  long   location,ci,si;
  double *emit_location,*emit_ptr,*trans_ptr;
  VPATH  *current,*backpointer,*path_end=NULL,*path_location;
  VPATH  *path_end_extended=NULL;
  
  /* Semi-supervised extension */
  //long   supervised_states[12]= {1,1,1,1,1,0,0,0,0,0,1,0};/* change it */
  //double emit_score,trans_score;
  
  printf("Inside viterbi_forward_order1_semisupervised ... \n");
  printf("emit_order:%ld, start:%ld, end:%ld, states:%ld \n",emit_order,start,end,states);
  
   
  /* For the first time frame: */ 	
  for(si=0;si<states;si++) {
    current=&path[start][si];
    current->label=si+1;
    current->score=emit[start][si];
    current->backpointer=NULL;
  }

  smax=0;

  for(location=start+1;location<=end;location++) 
  {
  emit_location=emit[location]; /* a pointer to the row of scores */
  path_location=path[location];
  backpointer=path[location-1]; /* at the first iteration, points to "current" */
  lastpos=(location==end);
	  for(p = 0; p < states; p++) 
	  { /* source state, h_{t-1} */
		current=path_location;  /* MS: pointer for current path */
		emit_ptr=emit_location; /* MS: respective location for emit score for current token */
		trans_ptr=&trans[p*states]; /* MS: respective location for trans score for current token */
		/* MS - Notes 
		 * p=0: trans_ptr=&trans[0]
		 * p=1: trans_ptr=&trans[2]
		 * trans = [w_11,w_12,w_21,w_22]
		 * */
			for(si=0;si<states;si++) 
			{  		   /* destination state, h_t */
				       /* MS - Notes : 
						* si=0: h_{t-1} = 1
						* si=1: h_{t-1} = 2
						* p=0,si=0: h_{t-1} = 1 and h_{t} = 1
						* p=0,si=1: h_{t-1} = 1 and h_{t} = 2
						* p=1,si=0: h_{t-1} = 2 and h_{t} = 1
						* p=1,si=1: h_{t-1} = 2 and h_{t} = 2 
						* */	 
					   if( (supervised_states[location-1] == 1) || (supervised_states[location-1] == 2) )
					   {
   					       if( (p == 0) && (supervised_states[location-1] == 1)  )
						   {
							   /*
							   * p=0,si=0: h_{t-1} = 1 and h_{t} = 1 (trans[0], path={11})
							   * p=0,si=1: h_{t-1} = 1 and h_{t} = 2 (trans[1], path={12})
							   */
							   backpointer=path[location-1]; /* Assign backpointer to the correct previous frame */
   					       	   s=backpointer->score+(*(trans_ptr++))+(*(emit_ptr++)); /* score */	
							   
							   current->label=si+1;   /* state labels are 1...N */
							   current->backpointer=backpointer;
							   current->score=s;
							   
							   if( (lastpos) && ((smax<s) || (!path_end)) ) 
									 { 
									      /* When last state is not supervised*/
										   path_end=current;
										   smax=current->score;
									   
									 }
								
							  current++; /* MS: current is moved to the next state */
					      
						   }
						   else if( (p == 1) && (supervised_states[location-1] == 2) )
						   {
							   /*
							   * p=1,si=0: h_{t-1} = 2 and h_{t} = 1 (trans[3], path={21})
							   * p=1,si=1: h_{t-1} = 2 and h_{t} = 2 (trans[4], path={22}) 
							   */
							   backpointer=path[location-1]; /* Assign backpointer to the correct previous frame */
   					           backpointer++; /* MS: backpointer is now made to point to the next state */
							   s=backpointer->score+(*(trans_ptr++))+(*(emit_ptr++)); /* score */	
							   
							   current->label=si+1;   /* state labels are 1...N */
							   current->backpointer=backpointer;
							   current->score=s;
							   if((lastpos) && ((smax<s) || (!path_end))) 
									 {
									   path_end=current;
									   smax=current->score;
									 }
								
							  current++; /* MS: current is moved to the next state */
							   
						   }
						   else
						   {
							   continue;
						   }
						   					   						   						   
					   }
					   else
					   {
				           /* If previous state is not supervised, then continue with normal Viterbi Algorithm */
						   s=backpointer->score+(*(trans_ptr++))+(*(emit_ptr++)); /* score */	
						   if((p==0) || (current->score<s)) 
						   {
							 current->label=si+1;   /* state labels are 1...N */
							 current->backpointer=backpointer;
							 current->score=s;
							 
							 if((lastpos) && ((smax<s) || (!path_end))) 
								 {
								   path_end=current;
								   smax=current->score;
								 }
							
							}
						  current++; /* MS: current is moved to the next state */
					   }
			 }		 
		backpointer++; /* MS: backpointer is now made to point to the next state of the current token */
		}
  }
  
  
  /* For the last-Frame */
  
  
  /* printf("Address: %d\n", path_end); */
  return(path_end);
}
/****************************************************************************/

VPATH *viterbi_forward_order1(double *trans, double **emit, long emit_order,
			      long start, long end, long states, VPATH **path)

/* Computes order 1 forward pass of Viterbi for given transition weights
	and emission scores. Assumes that the sequence is at least 2
	tokens long. */
{
  double smax,s;
  long   p;
  int    lastpos;
  long   location,ci,si;
  double *emit_location,*emit_ptr,*trans_ptr;;
  VPATH  *current,*backpointer,*path_end=NULL,*path_location;

   
  /* For the first time frame: */ 	
  for(si=0;si<states;si++) {
    current=&path[start][si];
    current->label=si+1;
    current->score=emit[start][si];
    current->backpointer=NULL;
  }

  smax=0;

  for(location=start+1;location<=end;location++) {
  emit_location=emit[location]; /* a pointer to the row of scores */
  path_location=path[location];
  backpointer=path[location-1]; /* at the first iteration, points to "current" */
  lastpos=(location==end);
  for(p = 0; p < states; p++) { /* destination state, h_t */
    current=path_location;
	emit_ptr=emit_location;
    trans_ptr=&trans[p*states];
    for(si=0;si<states;si++) {  /* source state, h_t-1 */ 
	   s=backpointer->score+(*(trans_ptr++))+(*(emit_ptr++));	
	   if((p==0) || (current->score<s)) {
	     current->label=si+1;   /* state labels are 1...N */
	     current->backpointer=backpointer;
	     current->score=s;
	     if((lastpos) && ((smax<s) || (!path_end))) {
	       path_end=current;
	       smax=current->score;
	       }
	    }
	  current++;
      }
    backpointer++;
    }
  }
  /* printf("Address: %d\n", path_end); */
  return(path_end);
}
void read_supervised_label_file(char *file, TYPE_SUPLABELS *sup_labels) 
{
  /* Read example number and supervised labels */

  FILE *fl;
  long qid,frame_index,iter_rows;
  long tot_rows; /* to be read from file */
  
  //printf("inside file-1\n");

  if ((fl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }
  
  /*read total rows*/
  fscanf(fl,"%ld",&tot_rows);
  printf("Total rows read from file = %ld \n",tot_rows);
  //printf("inside file-2\n");
  
  /* Populate sup_labels entries */
  iter_rows=0;
  sup_labels->tot_entries = tot_rows;
  sup_labels->seq_id = (long *)my_malloc(sizeof(long)*sup_labels->tot_entries);
  sup_labels->frame_id = (long *)my_malloc(sizeof(long)*sup_labels->tot_entries);
    
  while(fscanf(fl,"%ld %ld",&qid,&frame_index) != EOF) {
	sup_labels->seq_id[iter_rows]=qid;  
	sup_labels->frame_id[iter_rows]=frame_index;  
    //printf("%ld,%ld\n",qid,frame_index);
    printf("%ld,%ld\n",sup_labels->seq_id[iter_rows],sup_labels->frame_id[iter_rows]);
    iter_rows++;    
  }
  assert(iter_rows == tot_rows);
  printf("Check Successful (%ld = %ld)\n", iter_rows,tot_rows);
  fclose(fl);
}
void print_dataset(SAMPLE sample_dataset)
{
	
  EXAMPLE *ex;
  int tot_ex;
  long lot;
  long iter_ex,iter_tok;
  
  ex = sample_dataset.examples;
  tot_ex = sample_dataset.n;
  
  for(iter_ex=0;iter_ex<tot_ex;iter_ex++)
  {
	  lot = ex[iter_ex].y.length;
	  for(iter_tok=0;iter_tok<lot;iter_tok++){
		  printf("%ld qid:%ld\n",ex[iter_ex].y.label_vector[iter_tok],iter_ex+1);
	  }
  }
}
