#include <stdio.h>
#include <stdlib.h>
#include "mvl_common.h"
#include "cblas.h"

#define MAX(x,y)      ((x) < (y) ? (y) : (x))
#define min(x,y) (((x) < (y)) ? (x) : (y))

void get_H(TYPE_MATRIX *kernel_H,TYPE_BLOCK *all_blocks,long total_examples)
{
	/*Declaration of Variables*/
	long rows_H,cols_H;
	long iter_ex,iter_blocks;
	
	/*Initialize Variables*/
	rows_H = 0;
	cols_H = 0;
	
	for(iter_ex=0;iter_ex<total_examples;iter_ex++)
	{
		rows_H += (all_blocks[iter_ex].h_ij)->rows; 
	}
	
	cols_H = rows_H; /*As H is symmetric*/
	
	/*Set H's parameters*/
	kernel_H->rows = rows_H;
	kernel_H->cols = cols_H;
	
	/*Allocate memory for kernel_H->data*/
	kernel_H->data = (double *)my_malloc(rows_H*cols_H*sizeof(double));
	
	/*Populate H appropriately with all_blocks*/
	
	/*************Matlab's Implementation*************************************/
	/*
	function [ H,iter_H ] = function_fill_H( H,blocks,tot_ex )
	%UNTITLED6 Summary of this function goes here
	%   Detailed explanation goes here
	current_block = 0;
	iter_H=0;
	for i=1:tot_ex
		first_block_id=tot_ex*(i-1) + 1;
		rows_to_go = blocks(first_block_id).rows;
		for iter_rows=1:rows_to_go   
			for j=1:tot_ex
			   current_block = (tot_ex * (i-1) ) + ( j );        
			   for iter_cols=1:blocks(current_block).cols
				   iter_H=iter_H+1;
				   H(iter_H) = blocks(current_block).data(iter_rows,iter_cols);               
			   end
			end
		end 
	end
	end
	*/
   /***************************************************************************/
   
   /*Conversion from Matlab into C*/
   long current_block,iter_H;
   long first_block_id,rows_to_go;
   long iter_rows,iter_cols,j;
   double desired_value;
   
   current_block = 0;
   iter_H = 0;
   first_block_id = -1;
   rows_to_go = -1;
   iter_rows = 0;
   iter_cols = 0;
   j=0;
   desired_value = 0.0;
   
   for(iter_ex = 0;iter_ex<total_examples;iter_ex++)
   {
	   first_block_id=total_examples*(iter_ex);
	   rows_to_go = (all_blocks[first_block_id].h_ij)->rows;
	   
	   for(iter_rows=0;iter_rows<rows_to_go;iter_rows++)
	   {
		   for(j=0;j<total_examples;j++)
		   {
			   current_block = (total_examples * iter_ex) + (j);
			   
			   for(iter_cols=0;iter_cols<(all_blocks[current_block].h_ij)->cols;iter_cols++)
			   {
				   //get_matrix_element(TYPE_MATRIX *matrix,long i_row,long i_col)
				   desired_value = get_matrix_element(all_blocks[current_block].h_ij,iter_rows,iter_cols);
				   kernel_H->data[iter_H] = desired_value;
				   iter_H++;
			   }
		   }		   
	   }
	   
   }
	
}
void initialize_matrix(TYPE_MATRIX *matrix,long total_rows,long total_columns)
{
	long iter_rows,iter_cols;
	
	matrix->rows = total_rows;
	matrix->cols = total_columns;
	matrix->data = (double *)my_malloc(total_rows*total_columns*sizeof(double));
	
	/*Initialize matrix data header with 0's*/
	for(iter_rows = 0;iter_rows < total_rows;iter_rows++){
		for(iter_cols = 0;iter_cols < total_columns;iter_cols++){
			set_matrix_element(matrix,iter_rows,iter_cols,0.0);
		}
	}
	
}
void set_matrix_element(TYPE_MATRIX *matrix,long i_row,long i_col,double set_value)
{
	long current_index;
	
	current_index = -1; /*initialize*/
	
	//printf("a\n");
	current_index = get_linear_index(matrix->rows,matrix->cols,i_row,i_col);
	//printf("b\n");
	//printf("current_index = %ld, set_value = %f\n",current_index,set_value);
	matrix->data[current_index] = set_value;
	
}
void print_matrix_element(TYPE_MATRIX *matrix,long i_row,long i_col)
{
	long current_index;
	double return_value;
	
	current_index = -1; /*initialize*/
	return_value = -1; /*initialize*/
	
	current_index = get_linear_index(matrix->rows,matrix->cols,i_row,i_col);
	return_value = matrix->data[current_index];
	
	printf("matrix[%ld][%ld] = %f\n",i_row,i_col,return_value);
}
void print_matrix_info(TYPE_MATRIX *matrix)
{
	printf("rows = %ld;columns = %ld\n",matrix->rows,matrix->cols);
}
void print_matrix(TYPE_MATRIX *matrix)
{
	long iter_rows,iter_cols;
	long total_rows,total_columns,current_index;
	
	
	current_index = -1; /*initialize*/
	total_rows = matrix->rows;
	total_columns = matrix->cols;
	
	printf("[\n");
	/*Print whole matrix*/
	for(iter_rows = 0;iter_rows < total_rows;iter_rows++){
		for(iter_cols = 0;iter_cols < total_columns;iter_cols++){
			current_index = get_linear_index(matrix->rows,matrix->cols,iter_rows,iter_cols);
			printf("%f,",matrix->data[current_index]);
		}
		printf("\n");
	}
	printf("]\n");
}

double get_matrix_element(TYPE_MATRIX *matrix,long i_row,long i_col)
{
	long current_index;
	double return_value;
	
	current_index = -1; /*initialize*/
	return_value = -1; /*initialize*/
	
	current_index = get_linear_index(matrix->rows,matrix->cols,i_row,i_col);
	return_value = matrix->data[current_index];
	
	return(return_value);
}
long get_linear_index(long total_rows,long total_columns,long i_row,long i_col)
{
	/*total_rows   : total rows in Matrix*/
	/*total_columns: total columns in Matrix*/
	/*i_rows       : current row of Matrix*/
	/*i_cols       : current column in Matrix*/
	
	/*This function returns index of desired element inside linear array
	 * as that array was seen as 2D-array. Indices are imputed
	 * using an assumption of row-major order*/
	
	long desired_index;
	desired_index = -1;
	
	desired_index = i_col + (i_row * total_columns);
	return(desired_index);
}
void *my_malloc(size_t size)
{
  void *ptr;
  if(size<=0) size=1; /* for AIX compatibility */
  ptr=(void *)malloc(size);
  if(!ptr) { 
    perror ("Out of memory!\n"); 
    exit (1); 
  }
  return(ptr);
}
SAMPLE      read_struct_examples(char *file)
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
      examples[n-1].y.labels=(long *)my_malloc(sizeof(long *));
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
      examples[n-1].y.labels=(long *)realloc(examples[n-1].y.labels,
				       sizeof(long *)*examples[n-1].y.length);
    }
    examples[n-1].x.tokens[examples[n-1].x.length-1]=examples_flat[i];
    examples[n-1].y.labels[examples[n-1].y.length-1]=(long)labels_flat[i];
    if(labels_flat[i] < 0) {
      printf("ERROR (Line %ld): Token label IDs cannot be negative.\n",i+1);
      exit(1);
    }
    maxlabel=MAX(maxlabel,(long)labels_flat[i]);
  }

  printf(" %ld examples, %ld tokens, %ld features, %ld classes... ",n,tottokens,totwords,maxlabel);

  free(examples_flat); 
  free(labels_flat);

  sample.n=n;
  sample.total_features = totwords;
  sample.examples=examples;
  return(sample);
}

void read_documents(char *docfile, DOC ***docs, double **label, 
		    long int *totwords, long int *totdoc)
{
  char *line,*comment;
  WORD *words;
  long dnum=0,wpos,dpos=0,dneg=0,dunlab=0,queryid,slackid,max_docs;
  long max_words_doc, ll;
  double doc_label,costfactor;
  FILE *docfl;

  printf("Scanning examples..."); fflush(stdout);
  
  nol_ll(docfile,&max_docs,&max_words_doc,&ll); /* scan size of input file */
  max_words_doc+=2;
  ll+=2;
  max_docs+=2;
  
  printf("done\n"); fflush(stdout);
  

  (*docs) = (DOC **)my_malloc(sizeof(DOC *)*max_docs);    /* feature vectors */
  (*label) = (double *)my_malloc(sizeof(double)*max_docs); /* target values */
  line = (char *)my_malloc(sizeof(char)*ll);

  if ((docfl = fopen (docfile, "r")) == NULL)
  { perror (docfile); exit (1); }

  words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc+10));
  
  printf("Reading examples into memory..."); fflush(stdout);
  
  dnum=0;
  (*totwords)=0;
  while((!feof(docfl)) && fgets(line,(int)ll,docfl)) {
    if(line[0] == '#') continue;  /* line contains comments */
    if(!parse_document(line,words,&doc_label,&queryid,&slackid,&costfactor,
		       &wpos,max_words_doc,&comment)) {
      printf("\nParsing error in line %ld!\n%s",dnum,line);
      exit(1);
    }
    (*label)[dnum]=doc_label;
    /* printf("docnum=%ld: Class=%f ",dnum,doc_label); */
    if(doc_label > 0) dpos++;
    if (doc_label < 0) dneg++;
    if (doc_label == 0) dunlab++;
    if((wpos>1) && ((words[wpos-2]).wnum>(*totwords))) 
      (*totwords)=(words[wpos-2]).wnum;
    if((*totwords) > MAXFEATNUM) {
      printf("\nMaximum feature number exceeds limit defined in MAXFEATNUM!\n");
      printf("LINE: %s\n",line);
      exit(1);
    }
    (*docs)[dnum] = create_example(dnum,queryid,slackid,costfactor,
				   create_svector(words,comment,1.0));
    /* printf("\nNorm=%f\n",((*docs)[dnum]->fvec)->twonorm_sq);  */
    dnum++;
    
    /*  
		if(verbosity>=1) {
		  if((dnum % 100) == 0) {
		printf("%ld..",dnum); fflush(stdout);
		  }
    }
    */
  } 
  fclose(docfl);
  free(line);
  free(words);
  
  fprintf(stdout, "OK. (%ld examples read)\n", dnum);
  
  (*totdoc)=dnum;
}
void nol_ll(char *file, long int *nol, long int *wol, long int *ll) 
     /* Grep through file and count number of lines, maximum number of
        spaces per line, and longest line. */
{
  FILE *fl;
  int ic;
  char c;
  long current_length,current_wol;

  if ((fl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }
  current_length=0;
  current_wol=0;
  (*ll)=0;
  (*nol)=1;
  (*wol)=0;
  while((ic=getc(fl)) != EOF) {
    c=(char)ic;
    current_length++;
    if(space_or_null((int)c)) {
      current_wol++;
    }
    if(c == '\n') {
      (*nol)++;
      if(current_length>(*ll)) {
	(*ll)=current_length;
      }
      if(current_wol>(*wol)) {
	(*wol)=current_wol;
      }
      current_length=0;
      current_wol=0;
    }
  }
  fclose(fl);
}
int space_or_null(int c) {
  if (c==0)
    return 1;
  return isspace((unsigned char)c);
}
int parse_document(char *line, WORD *words, double *label,
		   long *queryid, long *slackid, double *costfactor,
		   long int *numwords, long int max_words_doc,
		   char **comment)
{
  register long wpos,pos;
  long wnum;
  double weight;
  char featurepair[1000],junk[1000];

  (*queryid)=0;
  (*slackid)=0;
  (*costfactor)=1;

  pos=0;
  (*comment)=NULL;
  while(line[pos] ) {      /* cut off comments */
    if((line[pos] == '#') && (!(*comment))) {
      line[pos]=0;
      (*comment)=&(line[pos+1]);
    }
    if(line[pos] == '\n') { /* strip the CR */
      line[pos]=0;
    }
    pos++;
  }
  if(!(*comment)) (*comment)=&(line[pos]);
  /* printf("Comment: '%s'\n",(*comment)); */

  wpos=0;
  /* check, that line starts with target value or zero, but not with
     feature pair */
  if(sscanf(line,"%s",featurepair) == EOF) return(0);
  pos=0;
  while((featurepair[pos] != ':') && featurepair[pos]) pos++;
  if(featurepair[pos] == ':') {
	perror ("Line must start with label or 0!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
  }
  /* read the target value */
  if(sscanf(line,"%lf",label) == EOF) return(0);
  pos=0;
  while(space_or_null((int)line[pos])) pos++;
  while((!space_or_null((int)line[pos])) && line[pos]) pos++;
  while((pos+=read_word(line+pos,featurepair)) &&
	(featurepair[0]) && 
	(wpos<max_words_doc)) {
    /* printf("%s\n",featurepair); */
    if(sscanf(featurepair,"qid:%ld%s",&wnum,junk)==1) {
      /* it is the query id */
      (*queryid)=(long)wnum;
    }
    else if(sscanf(featurepair,"sid:%ld%s",&wnum,junk)==1) {
      /* it is the slack id */
      if(wnum > 0) 
	(*slackid)=(long)wnum;
      else {
	perror ("Slack-id must be greater or equal to 1!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
      }
    }
    else if(sscanf(featurepair,"cost:%lf%s",&weight,junk)==1) {
      /* it is the example-dependent cost factor */
      (*costfactor)=(double)weight;
    }
    else if(sscanf(featurepair,"%ld:%lf%s",&wnum,&weight,junk)==2) {
      /* it is a regular feature */
      if(wnum<=0) { 
	perror ("Feature numbers must be larger or equal to 1!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
      }
      if((wpos>0) && ((words[wpos-1]).wnum >= wnum)) { 
	perror ("Features must be in increasing order!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
      }
      (words[wpos]).wnum=wnum;
      (words[wpos]).weight=(FVAL)weight; 
      wpos++;
    }
    else {
      perror ("Cannot parse feature/value pair!!!\n"); 
      printf("'%s' in LINE: %s\n",featurepair,line);
      exit (1); 
    }
  }
  (words[wpos]).wnum=0;
  (*numwords)=wpos+1;
  return(1);
}
SVECTOR *create_svector(WORD *words,char *userdefined,double factor)
{
  SVECTOR *vec;
  long    fnum,i;

  fnum=0;
  while(words[fnum].wnum) {
    fnum++;
  }
  fnum++;
  vec = (SVECTOR *)my_malloc(sizeof(SVECTOR));
  vec->words = (WORD *)my_malloc(sizeof(WORD)*(fnum));
  for(i=0;i<fnum;i++) { 
      vec->words[i]=words[i];
  }
  vec->twonorm_sq=-1;

  if(userdefined) {
    vec->userdefined=(char *)my_malloc(sizeof(char)*(strlen(userdefined)+1));
    strcpy(vec->userdefined,userdefined);
  }
  else 
    vec->userdefined = NULL;

  vec->kernel_id=0;
  vec->next=NULL;
  vec->factor=factor;
  return(vec);
}
DOC *create_example(long docnum, long queryid, long slackid, 
		    double costfactor, SVECTOR *fvec)
{
  DOC *example;
  example = (DOC *)my_malloc(sizeof(DOC));
  example->docnum=docnum;
  example->kernelid=docnum;
  example->queryid=queryid;
  example->slackid=slackid;
  example->costfactor=costfactor;
  example->fvec=fvec;
  return(example);
}
int read_word(char *in, char *out) {
  int found=0;
  while(isspace((int)(*in)) && (*in)) { /* skip over whitespace */
    in++;
    found++;
  }
  while(!space_or_null((int)(*in))) {   /* read non-whitespace string */
       (*out)=(*in);
    in++;
    found++;
    out++;
  }
  (*out)=0;
  return(found);
}
void read_input_parameters(int argc,char *argv[],char *trainfile, char *matlab_trainfile)
{
  strcpy (trainfile, argv[1]);
  strcpy (matlab_trainfile, argv[2]);
}
void initialize_hi(TYPE_H *h_i,SAMPLE *samples,long current_example)
{
	long iter_rows,iter_cols,iter_hi,iter_trans_row;
	long t_i,total_rows,total_columns;
	long D;
	long current_token;
	
	EXAMPLE *ex=NULL;
	TYPE_MATRIX *h_mat=NULL;
	SVECTOR *fvec = NULL;
	ex = samples->examples;
	
	h_i[current_example].seq_id = current_example;
	h_i[current_example].tot_tokens = ex[current_example].x.length;
	t_i = h_i[current_example].tot_tokens;
	D = samples->total_features;
	
	/*Allocate Memory for h_matrix*/
	h_i[current_example].h_matrix = (TYPE_MATRIX *)my_malloc(sizeof(TYPE_MATRIX));
	h_mat = h_i[current_example].h_matrix; /*pointer or a reference*/
	total_rows = (2*t_i) + (4*(t_i-1));
	total_columns = (2*D) + 4; 
	initialize_matrix(h_mat,total_rows,total_columns);/*allocate memory and initializes data matrix with zeros*/
	
	/*Fill h_mat with appropriate values*/
	iter_hi = 0;
	current_token = -1; /*trick to find current token using our encoding*/
	
	/*Fill Emission Part*/
	for(iter_rows = 0; iter_rows < (2*t_i) ;iter_rows++)
	{
		if( iter_rows%2 == 0) 
			{
				current_token++; /*current token will be increased from here ONLY*/
				fvec = ex[current_example].x.tokens[current_token]->fvec;
				/*iter_rows is even--fill it with true class (10)*/
				
				for(iter_cols = 0;iter_cols<total_columns;iter_cols++)
				{
					if(iter_cols < D)
					{
						h_mat->data[iter_hi] = fvec->words[iter_cols].weight;
						iter_hi++;
					}
					else
					{
						h_mat->data[iter_hi] = 0.0;
						iter_hi++;
					}					
				}
				
			}
			
		else
		{
			for(iter_cols = 0;iter_cols<total_columns;iter_cols++)
				{
					if(iter_cols < D)
					{
						h_mat->data[iter_hi] = 0.0;
						iter_hi++;
					}
					else if(iter_cols >=D && iter_cols < 2*D )
					{
						h_mat->data[iter_hi] = fvec->words[iter_cols-D].weight;
						iter_hi++;
					}
					else
					{
						h_mat->data[iter_hi] = 0.0;
						iter_hi++;
					}					
				}
			
		}
		
	}
		
	/*Fill Transition Part*/
	for(iter_rows = (2*t_i); iter_rows < (4*(t_i-1)) ;iter_rows++)
	{
		for(iter_cols = 0;iter_cols<total_columns;iter_cols++)
				{
					if(iter_cols < 2*D - 4)
					{
						h_mat->data[iter_hi] = 0.0;
						iter_hi++;
					}					
				}
	}
	
	/*Pick respective element and change its value*/
	iter_trans_row = -1;
	for(iter_rows = (2*t_i); iter_rows < (2*t_i) + 4*(t_i-1) ;iter_rows++)
	{
		iter_trans_row++;
		
		if(iter_trans_row == 0)
		{
			set_matrix_element(h_mat,iter_rows,2*D,1.0);
		}
		else if(iter_trans_row == 1)
		{
			set_matrix_element(h_mat,iter_rows, (2*D) + 1,1.0);
		}
		else if(iter_trans_row == 2)
		{
			set_matrix_element(h_mat,iter_rows, (2*D) + 2,1.0);
		}		
		else
		{
			set_matrix_element(h_mat,iter_rows, (2*D) + 3,1.0);
		}
		
		if(iter_trans_row == 3)
		{iter_trans_row = -1;}
	}
}
void initialize_hi_transpose(TYPE_H *h_i,TYPE_H *hi_transpose,SAMPLE *samples,long current_example)
{
	long iter_rows,iter_cols,iter_hi,iter_trans_row,hi_rows,hi_cols;
	long t_i,total_rows,total_columns;
	long D;
	long current_token;
	double current_value;
	
	EXAMPLE *ex=NULL;
	TYPE_MATRIX *h_mat=NULL;
	SVECTOR *fvec = NULL;
	ex = samples->examples;
	
	hi_transpose[current_example].seq_id = current_example;
	hi_transpose[current_example].tot_tokens = ex[current_example].x.length;
	t_i = h_i[current_example].tot_tokens;
	D = samples->total_features;
	
	/*Allocate Memory for h_matrix*/
	hi_transpose[current_example].h_matrix = (TYPE_MATRIX *)my_malloc(sizeof(TYPE_MATRIX));
	h_mat = hi_transpose[current_example].h_matrix; /*pointer or a reference*/
	total_rows = (2*D) + 4;
	total_columns = (2*t_i) + (4*(t_i-1));
	initialize_matrix(h_mat,total_rows,total_columns);/*allocate memory and initializes data matrix with zeros*/
	
	/*Get Transpose*/
	hi_rows = h_i[current_example].h_matrix->rows;
	hi_cols = h_i[current_example].h_matrix->cols;
	
	for(iter_rows=0;iter_rows<hi_rows;iter_rows++){
		for(iter_cols=0;iter_cols<hi_cols;iter_cols++){
			/* get value from h_i.matrix(row,col) */
			current_value = get_matrix_element(h_i[current_example].h_matrix,iter_rows,iter_cols);
			
			/* store value in hi_transpose.matrix(col,row)*/
			set_matrix_element(h_mat,iter_cols,iter_rows,current_value);
		
		}
	}
}

void initialize_hij(TYPE_BLOCK *blk_ij,TYPE_H *h_i,long total_examples)
{
	/*Initialization*/
    long iter_outer_block,iter_inner_block;
    long block_id;
    long hij_rows,hij_cols,t_i,t_j;
    
    /*Variables for C-BLAS*/
    double alpha,beta;
    long m,k,n;
    double *A,*B,*C;
    
    block_id=0;
    hij_rows = 0;
    hij_cols = 0;
    t_i=0;
    t_j=0;
    alpha = 1.0;
    beta = 0.0;
    A=NULL;
    B=NULL;
    C=NULL;
    
    /*total_blocks = total_examples * total_examples */

    /*Initialize all matrices inside blocks blk_ij*/
    for(iter_outer_block = 0;iter_outer_block<total_examples;iter_outer_block++)
	{
		for(iter_inner_block = 0;iter_inner_block<total_examples;iter_inner_block++)
		{
			block_id = iter_inner_block + (total_examples * iter_outer_block); /*current block*/
			blk_ij[block_id].h_ij = (TYPE_MATRIX *)my_malloc(sizeof(TYPE_MATRIX));
			//printf("Memory allocated for blk_ij[%ld].h_ij  \n",block_id);

		}
	}
    
    /*Initialize blocks*/
	for(iter_outer_block = 0;iter_outer_block<total_examples;iter_outer_block++)
	{
		for(iter_inner_block = 0;iter_inner_block<total_examples;iter_inner_block++)
		{
			block_id = iter_inner_block + (total_examples * iter_outer_block); /*current block*/
			blk_ij[block_id].i = iter_outer_block;    
			blk_ij[block_id].j = iter_inner_block;    
			blk_ij[block_id].i_tokens = h_i[iter_outer_block].tot_tokens;    
			blk_ij[block_id].j_tokens = h_i[iter_inner_block].tot_tokens; 	
			
			/*Initialize and fill h_ij*/
			t_i = blk_ij[block_id].i_tokens;
			t_j = blk_ij[block_id].j_tokens;
			
			/* h_ij = [( 2Ti + 4(Ti-1) ) X (2D+4)] X [(2D+4) X ( 2Tj + 4(Tj-1) ) ] */
			hij_rows = (2*t_i) + (4*(t_i-1));
			hij_cols = (2*t_j) + (4*(t_j-1));
			
			
			
			/*Always assign memory to those fields that require malloc
			* before passing them to functions*/		
			initialize_matrix(blk_ij[block_id].h_ij,hij_rows,hij_cols);
			
			/*h_ij =  (h_i) * (h_j)^T */
			//printf (" Computing matrix product using dgemm function via CBLAS interface \n\n");
			/*Just for C-Blas*/
			/* A (mxk) */
			/* B^T (kxn) */ /*our B (nxk)*/
			/* C (m*n) */ 
			/* A = (h_i[iter_outer_block].h_matrix)->data*/
			/* B = (h_i[iter_inner_block].h_matrix)->data*/
			/* C = (blk_ij[block_id].h_ij)->data */
			m = (h_i[iter_outer_block].h_matrix)->rows;
			k = (h_i[iter_outer_block].h_matrix)->cols;
			n = (h_i[iter_inner_block].h_matrix)->rows;
			
			A = (h_i[iter_outer_block].h_matrix)->data;
			B = (h_i[iter_inner_block].h_matrix)->data;
			C = (blk_ij[block_id].h_ij)->data; 
			
			/*cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
							m, n, k, alpha, A, k, B, n, beta, C, n);*/
			
			/*cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
			  A, no_of_columns_in_A, B, no_of_columns_in_B_without_its_transpose, beta, C, no_of_columns_in_C);*/
							
			//printf("i:%ld j:%ld blk:%ld (A:mxk):%ldx%ld (B:nxk):%ldx%ld (C:mxn):%ldx%ld\n",iter_outer_block,iter_inner_block,block_id,m,k,n,k,m,n);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                m, n, k, alpha, A, k, B,k, beta,C, n);
			
			/*
			 * Visit bottom address for more info 
			 * https://software.intel.com/sites/products/documentation/doclib/mkl_sa/
			 * 11/tutorials/mkl_mmx_c/GUID-36BFBCE9-EB0A-43B0-ADAF-2B65275726EA.htm
			 *
			 */
			
			//printf ("\n Computations completed.\n\n");
			printf ("h_%ld_%ld imputed\n",iter_outer_block,iter_inner_block);
			
		}
		
	}
	
	
}


long get_totBinFeats(long tot_ex,long *lot)
{
	long iter_ex;
	long tot_binfeats;
	tot_binfeats = 0;
	
	for(iter_ex=0;iter_ex<tot_ex;iter_ex++)
	{
		tot_binfeats += ( 6*lot[iter_ex] - 4 );
	}
	return(tot_binfeats);
}
void gen_binTS(long **labels, long *binarized_labels, long *lot, long tot_ex, long *total_binfeats,long *base_table)
{
	/*gen_binTS       : generate Binary Training set (Emission+Transition)*/
	/*labels          : labels for every training sample inside labels[training_example#][token_id]*/
	/*binarized_labels: returned (single) binarized label vector of all training samples*/
	/*lot             : vector containing lengths of tokens for each training sample*/
	/*base_table      : stored here are the indices that points to the start of emission label for each example 
	                    It is useful for recovering labels(1,2) from binarized_labels   */
	
	long iter_ex,iter_ind;
	long start_ind,end_ind,ind_tbc;
	/*ind_tbc: indices to be copied*/
	
	long *buffer_labels=NULL;
	
	start_ind = 0;/*initialize*/
	end_ind = -1;/*initialize*/
		
	for (iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		/*Step-1: Get desired range of indices for binarized_labels */
		start_ind = end_ind+1;/*Verified*/
		end_ind = start_ind + (6*lot[iter_ex]-4) -1; /*Verified*/
		/*Added*/
		base_table[iter_ex]=start_ind;
		//printf("(ex#%ld) start_ind:%ld end_ind:%ld\n",iter_ex,start_ind,end_ind);
		
		/*Step-2: Fill-up buffer_labels with map_labelVector(Will be slow process)*/
		buffer_labels = (long *)my_malloc( sizeof(long) * (6*lot[iter_ex]-4) );
 	    map_labelVector(labels[iter_ex],buffer_labels,lot[iter_ex]);
        
		
		/*Step-3: Copy buffer_labels inside appropriate location of binarized_labels*/
		ind_tbc = 0;
		for(iter_ind=start_ind;iter_ind<=end_ind;iter_ind++)
		{
			binarized_labels[iter_ind] = buffer_labels[ind_tbc];
			ind_tbc++;
			*(total_binfeats) = *(total_binfeats) + 1;
		}
		assert( ind_tbc ==  (6*lot[iter_ex]-4)   );
		
		/*Step-4: Free buffer_labels*/
		if(buffer_labels)
			free(buffer_labels);		
				
	}
	
	assert(end_ind == ( *(total_binfeats) - 1 ));
		
}
void recover_labels(long *binarized_labels,long *recovered_labels,long lot,long sample_no,long *emit_base_table)
{
	/*sample-no: Example # (Zero-Based)*/
	/*lot: length of tokens in sample_no*/
	/*binarized_labels: binarized labels passed to function*/
	/*recovered_labels: recovered labels thrown by function (1,2)*/
	
	long iter_tok,base_ind,exact_loc;
	base_ind = emit_base_table[sample_no];
	//printf("lot = %ld base_index: %ld\n",lot,base_ind);
	for (iter_tok = 0;iter_tok<lot;iter_tok++)
	{
		exact_loc = base_ind + (2*iter_tok);
		
		if(binarized_labels[exact_loc] == 1 && binarized_labels[exact_loc+1] == 0 )
		{
			recovered_labels[iter_tok] = 1;
			//printf("%ld",1);
			//printf("tok:%ld exact_location:%ld bin_feat:%ld%ld\n",iter_tok,exact_loc,binarized_labels[exact_loc],binarized_labels[exact_loc + 1]);
		}
		else
		{
			recovered_labels[iter_tok] = 2;
			//printf("tok:%ld exact_location:%ld bin_feat:%ld%ld\n",iter_tok,exact_loc,binarized_labels[exact_loc],binarized_labels[exact_loc + 1]);
			//printf("%ld",2);
		}		
	}
	//printf("\n");
}
void print_binarized_array(long *binarized_labels,long total_entries)
{
	long iter_entries;
	printf("Binarized_label (1x%ld): \n",total_entries);
	
	for(iter_entries = 0;iter_entries<total_entries;iter_entries++)
	{
		printf("%d ",binarized_labels[iter_entries]);
	}
	printf("\n");
}
void print_recovered_array(long *recovered_lables,long lot)
{
	long iter_tok;

	printf("recovered_label:");
	for(iter_tok = 0;iter_tok<lot;iter_tok++)
	{
		printf("%d",recovered_lables[iter_tok]);
	}
	printf("\n");
}
void print_long_array(long *long_array,long total_entries)
{
	long iter_entries;

	for(iter_entries = 0;iter_entries<total_entries;iter_entries++)
	{
		printf("%ld",long_array[iter_entries]);
	}
	printf("\n");
}
void map_labelVector(long *labels, long *binarized_labels, long lot)
{
	/*labels: input labels (1,2)*/
	/*binarized_lables: binary labels thrown by this function (emission+transition)*/
	/*lot: length of tokens or total labels inside labels*/
	
	long emit_start,emit_end,trans_start,trans_end;
	long iter_tok;	
	long current_label,previous_label;
	
	for(iter_tok=0;iter_tok<lot;iter_tok++)
    {
		/*Write binarized emission labels*/
		emit_start = 2 * iter_tok; 
		emit_end = emit_start + 1;
		
		if(labels[iter_tok] == 1)
		{
			binarized_labels[emit_start] = 1;
			binarized_labels[emit_end] = 0;		
		}
		else
		{
			binarized_labels[emit_start] = 0;
			binarized_labels[emit_end] = 1;
		}
		
		/*Write Binarized Transition labels*/
		if(iter_tok > 0)
		{
			trans_start = (2*lot) + 4*(iter_tok-1);
			trans_end = trans_start + 3;
			current_label = labels[iter_tok];
			previous_label = labels[iter_tok-1];
			fill_inTrans(binarized_labels,current_label
			,previous_label,trans_start,trans_end);
		}
		
	}
	
	
}

void fill_inTrans(long *labels,long cur_lab,long prev_lab,long start_ind,long end_ind)
{
	/* Original Function used in all experiments */
	 
	/*labels: target array for binary labels*/
	/*cur_lab: current label*/
	/*prev_lab: previous label*/
	/*start_ind: values will be written from labels[start_ind] */
	/*end_ind: last value will be written at labels[end_ind] */
	
	if(prev_lab == 2 && cur_lab == 2)
	{
		labels[start_ind] = 0;
		labels[start_ind+1] = 0;
		labels[start_ind+2] = 0;
		labels[start_ind+3] = 1;
	}
	else if(prev_lab == 2 && cur_lab == 1)
	{
		labels[start_ind]   = 0;
		labels[start_ind+1] = 0;
		labels[start_ind+2] = 1;
		labels[start_ind+3] = 0;
	}
	else if(prev_lab == 1 && cur_lab == 2)
	{
		labels[start_ind]   = 0;
		labels[start_ind+1] = 1;
		labels[start_ind+2] = 0;
		labels[start_ind+3] = 0;
	}
	else
	{
		labels[start_ind]   = 1;
		labels[start_ind+1] = 0;
		labels[start_ind+2] = 0;
		labels[start_ind+3] = 0;
	}
	
}

void fill_inTrans_new(long *labels,long cur_lab,long prev_lab,long start_ind,long end_ind)
{
    /* New Function : Dated 4 Dec 2015 */

	/*labels: target array for binary labels*/
	/*cur_lab: current label*/
	/*prev_lab: previous label*/
	/*start_ind: values will be written from labels[start_ind] */
	/*end_ind: last value will be written at labels[end_ind] */
	
	if(prev_lab == 1 && cur_lab == 1)
	{
		labels[start_ind] = 0;
		labels[start_ind+1] = 0;
		labels[start_ind+2] = 0;
		labels[start_ind+3] = 1;
	}
	else if(prev_lab == 1 && cur_lab == 2)
	{
		labels[start_ind]   = 0;
		labels[start_ind+1] = 0;
		labels[start_ind+2] = 1;
		labels[start_ind+3] = 0;
	}
	else if(prev_lab == 2 && cur_lab == 1)
	{
		labels[start_ind]   = 0;
		labels[start_ind+1] = 1;
		labels[start_ind+2] = 0;
		labels[start_ind+3] = 0;
	}
	else
	{
		labels[start_ind]   = 1;
		labels[start_ind+1] = 0;
		labels[start_ind+2] = 0;
		labels[start_ind+3] = 0;
	}
	
}

long get_H_dim(SAMPLE *sample)
{
	long iter_ex;
	long bin_feats,h_dim,t_i;
	h_dim = 0;
	bin_feats = 0;
	
	for(iter_ex = 0; iter_ex<sample->n ; iter_ex++)
	{
		bin_feats = 0;
		t_i = sample->examples[iter_ex].y.length;
		bin_feats = (2 * t_i) + ( 4 * (t_i-1) );
		printf("ex:%ld,tok:%ld,bin_feats:%ld\n",iter_ex,t_i,bin_feats);
		h_dim += bin_feats;  
	}
	
	return(h_dim);
}
void get_base_indices(long *root_base,long *eb_even,long *eb_odd,long *tb,long *lot,long tot_ex)
{
	/* root_base : copy it inside eb_even*/
	/* eb_even   : emit_base_table_even  */
	/* eb_odd    : emit_base_table_odd   */
	/* tb        : trans_base_table      */
	
	long iter_ex;
	iter_ex = 0;
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		/*initialize*/
		eb_even[iter_ex] = 0;
		eb_odd[iter_ex] = 0;
		tb[iter_ex] = 0;
		
		/*fill it*/
		eb_even[iter_ex] = root_base[iter_ex];
		eb_odd[iter_ex] = eb_even[iter_ex] + 1;
		tb[iter_ex] = eb_even[iter_ex] + ( 2 * lot[iter_ex] ) ;
	}
}
void print_base_tables(long *eb_even,long *eb_odd,long *tb,long *lot,long tot_ex)
{
	long iter_ex = 0;
	
	for(iter_ex=0;iter_ex<tot_ex;iter_ex++)
   {
	   printf("------------------------\n");
	   printf("ex:%ld,lot:%ld\n",iter_ex,lot[iter_ex]);
	   printf("emit_base_even[%ld] = %ld,\n",iter_ex,eb_even[iter_ex]);
	   printf("emit_base_odd[%ld] = %ld,\n",iter_ex,eb_odd[iter_ex]);
	   printf("trans[%ld] = %ld,\n",iter_ex,tb[iter_ex]);
	   printf("------------------------\n");
   }
   
}
void clear_double_array(double *double_array,long tot_entries)
{
	long iter_entries;
	for(iter_entries = 0;iter_entries<tot_entries;iter_entries++)
		{double_array[iter_entries] = 0.0;}
}
void clear_float_array(float *float_array,long tot_entries)
{
	long iter_entries;
	for(iter_entries = 0;iter_entries<tot_entries;iter_entries++)
		{float_array[iter_entries] = 0.0;}
}
void clear_long_array(long *long_array,long tot_entries)
{
	long iter_entries;
	for(iter_entries = 0;iter_entries<tot_entries;iter_entries++)
		{long_array[iter_entries] = 0;}
}
void fill_long_array(long *long_array,long fill_with,long tot_entries)
{
	long iter_entries;
	for(iter_entries = 0;iter_entries<tot_entries;iter_entries++)
		{long_array[iter_entries] = fill_with;}
}
void fill_double_array(double *double_array,double fill_with,long tot_entries)
{
	long iter_entries;
	for(iter_entries = 0;iter_entries<tot_entries;iter_entries++)
		{double_array[iter_entries] = fill_with;}
}
void print_double_array(double *double_array,long tot_entries)
{
	long iter_entries;
	for(iter_entries = 0;iter_entries<tot_entries;iter_entries++)
		{ printf("array[%ld] = %f\n",iter_entries,double_array[iter_entries]);}
}
void print_double_array_chunk(double *double_array,long start_ind,long end_ind)
{
	long iter_entries;
	for(iter_entries = start_ind;iter_entries<=end_ind;iter_entries++)
		{printf("array[%ld] = %f\n",iter_entries,double_array[iter_entries]);}
}
void print_double_array_chunk_matlab(double *double_array,long start_ind,long end_ind)
{
	long iter_entries;
	printf("[\n");
	for(iter_entries = start_ind;iter_entries<=end_ind;iter_entries++)
		{printf("%f\n",iter_entries,double_array[iter_entries]);}
	printf("]\n");
}
void print_double_array_for_matlab(double *double_array,long tot_entries)
{
	long iter_entries;
	printf("[\n");
	for(iter_entries = 0;iter_entries<tot_entries;iter_entries++)
		{ printf("%f,\n",iter_entries,double_array[iter_entries]);}
	printf("]\n");
}
void print_float_array(float *float_array,long tot_entries)
{
	long iter_entries;
	for(iter_entries = 0;iter_entries<tot_entries;iter_entries++)
		{ printf("array[%ld] = %f\n",iter_entries,float_array[iter_entries]);}
}
double sprod_ss(SVECTOR *a, SVECTOR *b) 
     /* compute the inner product of two sparse vectors */
{
    register double sum=0;
    register WORD *ai,*bj;
    ai=a->words;
    bj=b->words;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	ai++;
      }
      else {
	sum+=(ai->weight) * (bj->weight);
	ai++;
	bj++;
      }
    }
    return((double)sum);
}
SVECTOR *copy_svector(SVECTOR *vec)
{
  SVECTOR *newvec=NULL;
  if(vec) {
    newvec=create_svector(vec->words,vec->userdefined,vec->factor);
    newvec->kernel_id=vec->kernel_id;
    newvec->next=copy_svector(vec->next);
  }
  return(newvec);
}
void free_svector(SVECTOR *vec)
{
  SVECTOR *next;
  while(vec) {
    if(vec->words)
      free(vec->words);
    if(vec->userdefined)
      free(vec->userdefined);
    next=vec->next;
    free(vec);
    vec=next;
  }
}
void print_fvec(SVECTOR *fvec)
{
	SVECTOR *debug_vec=NULL;
    debug_vec =  copy_svector(fvec);//SVECTOR *copy_svector(SVECTOR *vec)
    long iter_words = 0;
	
	while(debug_vec != NULL){
		iter_words=0;
		while(debug_vec->words[iter_words].wnum)
		{
		  printf("iter_words = %ld wnum=%d , weight=%f ",iter_words,debug_vec->words[iter_words].wnum,
		  debug_vec->words[iter_words].weight);
		  //printf("%f\n",debug_vec->words[iter_words].weight);
		  iter_words++;
		  printf("(increased)iter_words = %ld \n",iter_words);
		}
		debug_vec=debug_vec->next;
	}
	free_svector(debug_vec);
}
SVECTOR* multadd_ss_r(SVECTOR *a,SVECTOR *b,double fa, double fb,
		      double min_non_zero) 
     /* compute fa*a+fb*b of two sparse vectors */
     /* Note: SVECTOR lists are not followed, but only the first
	SVECTOR is used */
{
    SVECTOR *vec;
    register WORD *sum,*sumi;
    register WORD *ai,*bj;
    long veclength;
    double weight;
  
    ai=a->words;
    bj=b->words;
    veclength=0;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	veclength++;
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	veclength++;
	ai++;
      }
      else {
	veclength++;
	ai++;
	bj++;
      }
    }
    while (bj->wnum) {
      veclength++;
      bj++;
    }
    while (ai->wnum) {
      veclength++;
      ai++;
    }
    veclength++;

    sum=(WORD *)my_malloc(sizeof(WORD)*veclength);
    sumi=sum;
    ai=a->words;
    bj=b->words;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	(*sumi)=(*bj);
	sumi->weight*=fb;
	sumi++;
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	(*sumi)=(*ai);
	sumi->weight*=fa;
	sumi++;
	ai++;
      }
      else {
	weight=fa*(double)ai->weight+fb*(double)bj->weight;
	if((weight<-min_non_zero) || (weight>min_non_zero)) {
	  sumi->wnum=ai->wnum;
	  sumi->weight=weight;
	  sumi++;
	}
	ai++;
	bj++;
      }
    }
    while (bj->wnum) {
      (*sumi)=(*bj);
      sumi->weight*=fb;
      sumi++;
      bj++;
    }
    while (ai->wnum) {
      (*sumi)=(*ai);
      sumi->weight*=fa;
      sumi++;
      ai++;
    }
    sumi->wnum=0;

    if(1) { /* potentially this wastes some memory, but saves malloc'ing */
      vec=create_svector_shallow(sum,NULL,1.0);
    }
    else {  /* this is more memory efficient */
      vec=create_svector(sum,NULL,1.0);
      free(sum);
    }
    return(vec);
}
SVECTOR *create_svector_shallow(WORD *words,char *userdefined,double factor)
     /* unlike 'create_svector' this does not copy words and userdefined */
{
  SVECTOR *vec;

  vec = (SVECTOR *)my_malloc(sizeof(SVECTOR));
  vec->words = words;
  vec->twonorm_sq=-1;
  vec->userdefined=userdefined;
  vec->kernel_id=0;
  vec->next=NULL;
  vec->factor=factor;
  return(vec);
}

void get_r(double *array_r,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,long *binarized_y)
{
	long iter_ex,iter_tok,iter_inner_ex,iter_outer_ex,iter_outer_tokens,iter_inner_tokens;
	long tot_ex,y_ind,desired_ind;
	float token_weight;
	double fa,fb;
	
	SVECTOR *emit_a = NULL;
	SVECTOR *emit_b = NULL;
	SVECTOR *result_a = NULL;
	SVECTOR *accumulated_result_a = NULL;
	SVECTOR *result_b = NULL;
	
	SVECTOR *fvec = NULL;
	
	tot_ex = samples->n;
	y_ind = 0;
	token_weight = 0.0;
	fa = 0.0;
	fb = 0.0;
	desired_ind = 0;
	
	/*get emit_a */
    //printf("-----------------emita Starting--------------------------------\n");
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = even_ebt[iter_ex];
		emit_a = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = (double)binarized_y[desired_ind];
			fb = 0.0;
			result_a = multadd_ss_r(emit_a,emit_a,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_a);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = even_ebt[iter_ex] + (2*iter_tok);
			emit_a = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_a);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_a);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	//printf("-----------------emit_b Starting--------------------------------\n");
	/*get emit_b */
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = odd_ebt[iter_ex];
		emit_b = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = (double)binarized_y[desired_ind];
			fb = 0.0;
			result_b = multadd_ss_r(emit_b,emit_b,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_b);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = odd_ebt[iter_ex] + (2*iter_tok);
			emit_b = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_b);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_b);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	
	/***Generate Trans features***/
	long iter_trans_ind;
	double *trans;
	trans = (double *)my_malloc(4*sizeof(double));
	clear_double_array(trans,4);
	
	iter_trans_ind=0;
	
	//printf("------------------trans start-------------------------------------\n");
	
	for(iter_trans_ind = 0;iter_trans_ind<4;iter_trans_ind++)
	{
		for(iter_ex=0;iter_ex< tot_ex ;iter_ex++)
		{
			desired_ind = trans_bt[iter_ex] + iter_trans_ind;
			for(iter_tok = 0;iter_tok < (lot[iter_ex]-1) ;iter_tok++)
			{
				//printf("%ld,",desired_ind);
				trans[iter_trans_ind] += (double)binarized_y[desired_ind];
				desired_ind += 4;
			}			
		}
		
		//printf("\n");

	}
    //print_double_array(trans,4);
	//printf("------------------trans end-------------------------------------\n");
	
	/*****************************/
	long iter_r,iter_emitfeats,iter_transfeats,iter_translayer;
	long current_token,current_token_occurence;
	
	iter_r = 0;
	iter_emitfeats = 0;
	iter_translayer = 0;
	current_token = -1; /*trick to find current token using our encoding*/
	current_token_occurence = 0;
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		current_token = -1; /*trick to find current token using our encoding*/
		
		for(iter_emitfeats = 0;iter_emitfeats < 2*lot[iter_ex];iter_emitfeats++)
		{
			if(iter_emitfeats % 2 == 0)
			{
				current_token++;
				current_token_occurence=1;
				
			}
			
			printf("r[%ld] -- ex:%ld tok:%ld tok_occ:%ld \n",iter_r,iter_ex,current_token,current_token_occurence);
			fvec = samples->examples[iter_ex].x.tokens[current_token]->fvec;
			
			/*select result_a or result_b*/
			if(current_token_occurence == 1)
			{array_r[iter_r] = sprod_ss(fvec,result_a);}
			else
			{array_r[iter_r] = sprod_ss(fvec,result_b);}

			iter_r++;
			current_token_occurence++;
		}
		
		for(iter_translayer = 0; iter_translayer < lot[iter_ex] -1 ; iter_translayer++ )
		{
			for(iter_transfeats = 0;iter_transfeats < 4;iter_transfeats++)
			{
				array_r[iter_r] = trans[iter_transfeats];
				printf("r[%ld] ex:%ld trans_layer:%ld trans[%ld] \n",iter_r,iter_ex,iter_translayer,iter_transfeats);
				//fvec_inner = samples->examples[iter_inner_ex].x.tokens[iter_inner_tokens]->fvec;
				iter_r++;
			}
			
			
		}
		
	}
		
}

void get_r_save_memory(double *array_r,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,long *binarized_y)
{
	long iter_ex,iter_tok,iter_inner_ex,iter_outer_ex,iter_outer_tokens,iter_inner_tokens;
	long tot_ex,y_ind,desired_ind;
	float token_weight;
	double fa,fb;
	
	SVECTOR *emit_a = NULL;
	SVECTOR *emit_b = NULL;
	SVECTOR *old_result = NULL;
	SVECTOR *result_a = NULL;
	SVECTOR *accumulated_result_a = NULL;
	SVECTOR *result_b = NULL;
	
	SVECTOR *fvec = NULL;
	
	tot_ex = samples->n;
	y_ind = 0;
	token_weight = 0.0;
	fa = 0.0;
	fb = 0.0;
	desired_ind = 0;
	
	/*get emit_a */
    //printf("-----------------emita Starting--------------------------------\n");
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = even_ebt[iter_ex];
		emit_a = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = (double)binarized_y[desired_ind];
			fb = 0.0;
			result_a = multadd_ss_r(emit_a,emit_a,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			old_result = result_a;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			free_svector(old_result);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_a);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = even_ebt[iter_ex] + (2*iter_tok);
			emit_a = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			old_result = result_a;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			free_svector(old_result);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_a);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_a);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	//printf("-----------------emit_b Starting--------------------------------\n");
	/*get emit_b */
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = odd_ebt[iter_ex];
		emit_b = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = (double)binarized_y[desired_ind];
			fb = 0.0;
			result_b = multadd_ss_r(emit_b,emit_b,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			old_result = result_b;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			free_svector(old_result);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_b);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = odd_ebt[iter_ex] + (2*iter_tok);
			emit_b = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			old_result = result_b;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			free_svector(old_result);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_b);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_b);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	
	/***Generate Trans features***/
	long iter_trans_ind;
	double *trans;
	trans = (double *)my_malloc(4*sizeof(double));
	clear_double_array(trans,4);
	
	iter_trans_ind=0;
	
	//printf("------------------trans start-------------------------------------\n");
	
	for(iter_trans_ind = 0;iter_trans_ind<4;iter_trans_ind++)
	{
		for(iter_ex=0;iter_ex< tot_ex ;iter_ex++)
		{
			desired_ind = trans_bt[iter_ex] + iter_trans_ind;
			for(iter_tok = 0;iter_tok < (lot[iter_ex]-1) ;iter_tok++)
			{
				//printf("%ld,",desired_ind);
				trans[iter_trans_ind] += (double)binarized_y[desired_ind];
				desired_ind += 4;
			}			
		}
		
		//printf("\n");

	}
    //print_double_array(trans,4);
	//printf("------------------trans end-------------------------------------\n");
	
	/*****************************/
	long iter_r,iter_emitfeats,iter_transfeats,iter_translayer;
	long current_token,current_token_occurence;
	
	iter_r = 0;
	iter_emitfeats = 0;
	iter_translayer = 0;
	current_token = -1; /*trick to find current token using our encoding*/
	current_token_occurence = 0;
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		current_token = -1; /*trick to find current token using our encoding*/
		
		for(iter_emitfeats = 0;iter_emitfeats < 2*lot[iter_ex];iter_emitfeats++)
		{
			if(iter_emitfeats % 2 == 0)
			{
				current_token++;
				current_token_occurence=1;
				
			}
			
			//printf("r[%ld] -- ex:%ld tok:%ld tok_occ:%ld \n",iter_r,iter_ex,current_token,current_token_occurence);
			fvec = samples->examples[iter_ex].x.tokens[current_token]->fvec;
			
			/*select result_a or result_b*/
			if(current_token_occurence == 1)
			{array_r[iter_r] = sprod_ss(fvec,result_a);}
			else
			{array_r[iter_r] = sprod_ss(fvec,result_b);}

			iter_r++;
			current_token_occurence++;
		}
		
		for(iter_translayer = 0; iter_translayer < lot[iter_ex] -1 ; iter_translayer++ )
		{
			for(iter_transfeats = 0;iter_transfeats < 4;iter_transfeats++)
			{
				array_r[iter_r] = trans[iter_transfeats];
				//printf("r[%ld] ex:%ld trans_layer:%ld trans[%ld] \n",iter_r,iter_ex,iter_translayer,iter_transfeats);
				//fvec_inner = samples->examples[iter_inner_ex].x.tokens[iter_inner_tokens]->fvec;
				iter_r++;
			}
			
			
		}
		
	}
		
}

void get_param_tow(double *output_array,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,double *scaled_y)
{
	long iter_ex,iter_tok,iter_inner_ex,iter_outer_ex,iter_outer_tokens,iter_inner_tokens;
	long tot_ex,y_ind,desired_ind;
	float token_weight;
	double fa,fb;
	
	SVECTOR *emit_a = NULL;
	SVECTOR *emit_b = NULL;
	SVECTOR *result_a = NULL;
	SVECTOR *accumulated_result_a = NULL;
	SVECTOR *result_b = NULL;
	
	SVECTOR *fvec = NULL;
	
	tot_ex = samples->n;
	y_ind = 0;
	token_weight = 0.0;
	fa = 0.0;
	fb = 0.0;
	desired_ind = 0;
	
	/*get emit_a */
    //printf("-----------------emita Starting--------------------------------\n");
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = even_ebt[iter_ex];
		emit_a = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = scaled_y[desired_ind];
			fb = 0.0;
			result_a = multadd_ss_r(emit_a,emit_a,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = scaled_y[desired_ind];
			fb = 1.0;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_a);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = even_ebt[iter_ex] + (2*iter_tok);
			emit_a = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = scaled_y[desired_ind];
			fb = 1.0;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_a);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_a);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	//printf("-----------------emit_b Starting--------------------------------\n");
	/*get emit_b */
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = odd_ebt[iter_ex];
		emit_b = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = scaled_y[desired_ind];
			fb = 0.0;
			result_b = multadd_ss_r(emit_b,emit_b,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = scaled_y[desired_ind];
			fb = 1.0;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_b);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = odd_ebt[iter_ex] + (2*iter_tok);
			emit_b = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = scaled_y[desired_ind];
			fb = 1.0;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_b);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_b);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	
	/***Generate Trans features***/
	long iter_trans_ind;
	double *trans;
	trans = (double *)my_malloc(4*sizeof(double));
	clear_double_array(trans,4);
	
	iter_trans_ind=0;
	
	//printf("------------------trans start-------------------------------------\n");
	
	for(iter_trans_ind = 0;iter_trans_ind<4;iter_trans_ind++)
	{
		for(iter_ex=0;iter_ex< tot_ex ;iter_ex++)
		{
			desired_ind = trans_bt[iter_ex] + iter_trans_ind;
			for(iter_tok = 0;iter_tok < (lot[iter_ex]-1) ;iter_tok++)
			{
				//printf("%ld,",desired_ind);
				trans[iter_trans_ind] += scaled_y[desired_ind];
				desired_ind += 4;
			}			
		}
		
		//printf("\n");

	}
    //print_double_array(trans,4);
	//printf("------------------trans end-------------------------------------\n");
	
	/*****************************/
	long iter_array,iter_emitfeats,iter_transfeats,iter_translayer;
	long current_token,current_token_occurence;
	
	iter_array = 0;
	iter_emitfeats = 0;
	iter_translayer = 0;
	current_token = -1; /*trick to find current token using our encoding*/
	current_token_occurence = 0;
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		current_token = -1; /*trick to find current token using our encoding*/
		
		for(iter_emitfeats = 0;iter_emitfeats < 2*lot[iter_ex];iter_emitfeats++)
		{
			if(iter_emitfeats % 2 == 0)
			{
				current_token++;
				current_token_occurence=1;
				
			}
			
			printf("r[%ld] -- ex:%ld tok:%ld tok_occ:%ld \n",iter_array,iter_ex,current_token,current_token_occurence);
			fvec = samples->examples[iter_ex].x.tokens[current_token]->fvec;
			
			/*select result_a or result_b*/
			if(current_token_occurence == 1)
			{output_array[iter_array] = sprod_ss(fvec,result_a);}
			else
			{output_array[iter_array] = sprod_ss(fvec,result_b);}

			iter_array++;
			current_token_occurence++;
		}
		
		for(iter_translayer = 0; iter_translayer < lot[iter_ex] -1 ; iter_translayer++ )
		{
			for(iter_transfeats = 0;iter_transfeats < 4;iter_transfeats++)
			{
				output_array[iter_array] = trans[iter_transfeats];
				printf("r[%ld] ex:%ld trans_layer:%ld trans[%ld] \n",iter_array,iter_ex,iter_translayer,iter_transfeats);
				//fvec_inner = samples->examples[iter_inner_ex].x.tokens[iter_inner_tokens]->fvec;
				iter_array++;
			}
			
			
		}
		
	}
		
}

void get_param_tow_save_memory(double *output_array,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,double *scaled_y)
{
	long iter_ex,iter_tok,iter_inner_ex,iter_outer_ex,iter_outer_tokens,iter_inner_tokens;
	long tot_ex,y_ind,desired_ind;
	float token_weight;
	double fa,fb;
	
	SVECTOR *emit_a = NULL;
	SVECTOR *emit_b = NULL;
	SVECTOR *result_a = NULL;
	SVECTOR *old_result = NULL;
	SVECTOR *accumulated_result_a = NULL;
	SVECTOR *result_b = NULL;
	
	SVECTOR *fvec = NULL;
	
	tot_ex = samples->n;
	y_ind = 0;
	token_weight = 0.0;
	fa = 0.0;
	fb = 0.0;
	desired_ind = 0;
	
	/*get emit_a */
    //printf("-----------------emita Starting--------------------------------\n");
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = even_ebt[iter_ex];
		emit_a = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = scaled_y[desired_ind];
			fb = 0.0;
			result_a = multadd_ss_r(emit_a,emit_a,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = scaled_y[desired_ind];
			fb = 1.0;
			old_result = result_a;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			free_svector(old_result);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_a);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = even_ebt[iter_ex] + (2*iter_tok);
			emit_a = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = scaled_y[desired_ind];
			fb = 1.0;
			old_result = result_a;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			free_svector(old_result);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_a);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_a);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	//printf("-----------------emit_b Starting--------------------------------\n");
	/*get emit_b */
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = odd_ebt[iter_ex];
		emit_b = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = scaled_y[desired_ind];
			fb = 0.0;
			result_b = multadd_ss_r(emit_b,emit_b,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = scaled_y[desired_ind];
			fb = 1.0;
			old_result = result_b;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			free_svector(old_result);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_b);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = odd_ebt[iter_ex] + (2*iter_tok);
			emit_b = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = scaled_y[desired_ind];
			fb = 1.0;
			old_result = result_b;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			free_svector(old_result);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_b);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_b);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	
	/***Generate Trans features***/
	long iter_trans_ind;
	double *trans;
	trans = (double *)my_malloc(4*sizeof(double));
	clear_double_array(trans,4);
	
	iter_trans_ind=0;
	
	//printf("------------------trans start-------------------------------------\n");
	
	for(iter_trans_ind = 0;iter_trans_ind<4;iter_trans_ind++)
	{
		for(iter_ex=0;iter_ex< tot_ex ;iter_ex++)
		{
			desired_ind = trans_bt[iter_ex] + iter_trans_ind;
			for(iter_tok = 0;iter_tok < (lot[iter_ex]-1) ;iter_tok++)
			{
				//printf("%ld,",desired_ind);
				trans[iter_trans_ind] += scaled_y[desired_ind];
				desired_ind += 4;
			}			
		}
		
		//printf("\n");

	}
    //print_double_array(trans,4);
	//printf("------------------trans end-------------------------------------\n");
	
	/*****************************/
	long iter_array,iter_emitfeats,iter_transfeats,iter_translayer;
	long current_token,current_token_occurence;
	
	iter_array = 0;
	iter_emitfeats = 0;
	iter_translayer = 0;
	current_token = -1; /*trick to find current token using our encoding*/
	current_token_occurence = 0;
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		current_token = -1; /*trick to find current token using our encoding*/
		
		for(iter_emitfeats = 0;iter_emitfeats < 2*lot[iter_ex];iter_emitfeats++)
		{
			if(iter_emitfeats % 2 == 0)
			{
				current_token++;
				current_token_occurence=1;
				
			}
			
			printf("r[%ld] -- ex:%ld tok:%ld tok_occ:%ld \n",iter_array,iter_ex,current_token,current_token_occurence);
			fvec = samples->examples[iter_ex].x.tokens[current_token]->fvec;
			
			/*select result_a or result_b*/
			if(current_token_occurence == 1)
			{output_array[iter_array] = sprod_ss(fvec,result_a);}
			else
			{output_array[iter_array] = sprod_ss(fvec,result_b);}

			iter_array++;
			current_token_occurence++;
		}
		
		for(iter_translayer = 0; iter_translayer < lot[iter_ex] -1 ; iter_translayer++ )
		{
			for(iter_transfeats = 0;iter_transfeats < 4;iter_transfeats++)
			{
				output_array[iter_array] = trans[iter_transfeats];
				printf("r[%ld] ex:%ld trans_layer:%ld trans[%ld] \n",iter_array,iter_ex,iter_translayer,iter_transfeats);
				//fvec_inner = samples->examples[iter_inner_ex].x.tokens[iter_inner_tokens]->fvec;
				iter_array++;
			}
			
			
		}
		
	}
		
}

void get_param_delta(double *param_delta,double *alpha_i,long *lot,long *bin_labels,long tot_ex)
{
	long iter_ex,iter_entries,delta_index;
	iter_ex = 0;
	iter_entries = 0; /*iterate through binary entries of bin_labels*/
	delta_index = 0;  /*write result to param_delta[delta_index]*/
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		for(iter_entries = 0;iter_entries < (6 * lot[iter_ex] - 4);iter_entries++)
		{
			param_delta[delta_index] = alpha_i[iter_ex] * ( 1 - (2*bin_labels[delta_index]) );
			delta_index++;
		}
	}
}

void get_r_backup(double *array_r,SAMPLE *samples,long *even_ebt,long *odd_ebt,long *trans_bt,long *lot,long *binarized_y)
{
	/*Original Function*/
	
	long iter_ex,iter_tok,iter_inner_ex,iter_outer_ex,iter_outer_tokens,iter_inner_tokens;
	long tot_ex,y_ind,desired_ind;
	float token_weight;
	double fa,fb;
	
	SVECTOR *emit_a = NULL;
	SVECTOR *emit_b = NULL;
	SVECTOR *result_a = NULL;
	SVECTOR *accumulated_result_a = NULL;
	SVECTOR *result_b = NULL;
	
	SVECTOR *fvec = NULL;
	
	tot_ex = samples->n;
	y_ind = 0;
	token_weight = 0.0;
	fa = 0.0;
	fb = 0.0;
	desired_ind = 0;
	
	/*get emit_a */
    //printf("-----------------emita Starting--------------------------------\n");
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = even_ebt[iter_ex];
		emit_a = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = (double)binarized_y[desired_ind];
			fb = 0.0;
			result_a = multadd_ss_r(emit_a,emit_a,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_a);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = even_ebt[iter_ex] + (2*iter_tok);
			emit_a = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			result_a = multadd_ss_r(emit_a,result_a,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_a);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_a);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	//printf("-----------------emit_b Starting--------------------------------\n");
	/*get emit_b */
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		desired_ind = odd_ebt[iter_ex];
		emit_b = samples->examples[iter_ex].x.tokens[0]->fvec;
		if(iter_ex == 0)
		{
			fa = (double)binarized_y[desired_ind];
			fb = 0.0;
			result_b = multadd_ss_r(emit_b,emit_b,fa,fb,0); 
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
        else
        {
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,0,desired_ind,fa);
		}
		//print_fvec(emit_b);

        for(iter_tok = 1;iter_tok<lot[iter_ex];iter_tok++)
		{
			desired_ind = odd_ebt[iter_ex] + (2*iter_tok);
			emit_b = samples->examples[iter_ex].x.tokens[iter_tok]->fvec; 
			fa = (double)binarized_y[desired_ind];
			fb = 1.0;
			result_b = multadd_ss_r(emit_b,result_b,fa,fb,0);
			//printf("ex:%ld,tok:%ld,y[%ld]:%f \n",iter_ex,iter_tok,desired_ind,fa);
			//print_fvec(emit_b);

		}
		//printf("accumulated result till ex:%ld---start\n",iter_ex);
		//print_fvec(result_b);
		//printf("accumulated result till ex:%ld---end\n",iter_ex);
	
	}
	
	/***Generate Trans features***/
	long iter_trans_ind;
	double *trans;
	trans = (double *)my_malloc(4*sizeof(double));
	clear_double_array(trans,4);
	
	iter_trans_ind=0;
	
	//printf("------------------trans start-------------------------------------\n");
	
	for(iter_trans_ind = 0;iter_trans_ind<4;iter_trans_ind++)
	{
		for(iter_ex=0;iter_ex< tot_ex ;iter_ex++)
		{
			desired_ind = trans_bt[iter_ex] + iter_trans_ind;
			for(iter_tok = 0;iter_tok < (lot[iter_ex]-1) ;iter_tok++)
			{
				//printf("%ld,",desired_ind);
				trans[iter_trans_ind] += (double)binarized_y[desired_ind];
				desired_ind += 4;
			}			
		}
		
		//printf("\n");

	}
    //print_double_array(trans,4);
	//printf("------------------trans end-------------------------------------\n");
	
	/*****************************/
	long iter_r,iter_emitfeats,iter_transfeats,iter_translayer;
	long current_token,current_token_occurence;
	
	iter_r = 0;
	iter_emitfeats = 0;
	iter_translayer = 0;
	current_token = -1; /*trick to find current token using our encoding*/
	current_token_occurence = 0;
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		current_token = -1; /*trick to find current token using our encoding*/
		
		for(iter_emitfeats = 0;iter_emitfeats < 2*lot[iter_ex];iter_emitfeats++)
		{
			if(iter_emitfeats % 2 == 0)
			{
				current_token++;
				current_token_occurence=1;
				
			}
			
			printf("r[%ld] -- ex:%ld tok:%ld tok_occ:%ld \n",iter_r,iter_ex,current_token,current_token_occurence);
			fvec = samples->examples[iter_ex].x.tokens[current_token]->fvec;
			
			/*select result_a or result_b*/
			if(current_token_occurence == 1)
			{array_r[iter_r] = sprod_ss(fvec,result_a);}
			else
			{array_r[iter_r] = sprod_ss(fvec,result_b);}

			iter_r++;
			current_token_occurence++;
		}
		
		for(iter_translayer = 0; iter_translayer < lot[iter_ex] -1 ; iter_translayer++ )
		{
			for(iter_transfeats = 0;iter_transfeats < 4;iter_transfeats++)
			{
				array_r[iter_r] = trans[iter_transfeats];
				printf("r[%ld] ex:%ld trans_layer:%ld trans[%ld] \n",iter_r,iter_ex,iter_translayer,iter_transfeats);
				//fvec_inner = samples->examples[iter_inner_ex].x.tokens[iter_inner_tokens]->fvec;
				iter_r++;
			}
			
			
		}
		
	}
		
}
void copy_long_to_double(long *bin_y,double *double_bin_y, long *lot, long tot_ex)
{
	/*copy long values from bin_y[index] to double_bin_y[index] in double format*/
	
	long iter_ex,iter_entries,y_index;
	iter_ex = 0;
	iter_entries = 0; /*iterate through binary entries of bin_labels*/
	y_index = 0;      /*current index of bin_y*/
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		for(iter_entries = 0;iter_entries < (6 * lot[iter_ex] - 4);iter_entries++)
		{
			double_bin_y[y_index] = (double)bin_y[y_index];
			y_index++;
		}
	}
}

void copy_long_array(long *copy_from, long *copy_inside, long total_entries)
{
	/*copy long values from copy_from[iter_entries] to copy_inside[iter_entries]*/
	
	long iter_entries;
	
	for(iter_entries = 0;iter_entries<total_entries;iter_entries++)
		copy_inside[iter_entries]=copy_from[iter_entries];
}

void scale_y(double *bin_y,double *scale_factor, long *lot, long tot_ex)
{
	/*Scale binarized entries of each sample by its corresponding alpha*/
	/*Convert bin_y to double and then pass it to function*/
	
	long iter_ex,iter_entries,y_index;
	iter_ex = 0;
	iter_entries = 0; /*iterate through binary entries of bin_labels*/
	y_index = 0;      /*current index of bin_y and value stored here will be scaled*/
	
	for(iter_ex = 0;iter_ex<tot_ex;iter_ex++)
	{
		for(iter_entries = 0;iter_entries < (6 * lot[iter_ex] - 4);iter_entries++)
		{
			bin_y[y_index] *= scale_factor[iter_ex];
			y_index++;
		}
	}
}
void scale_double_array(double *array,double scale_factor, long tot_entries)
{
	/* Scale all entries of array */
	
	long iter_entries,ind;
	ind = -1;
	
	for(iter_entries = 0;iter_entries < tot_entries;iter_entries++){
		ind++;
		array[ind] *= scale_factor;
	}
	
	assert(ind == (tot_entries-1) );
}
void get_param_R(double *out_array,double *param_a,double *param_b,double *param_c,long total_elements)
{
	long iter_elements;
	iter_elements = 0;
	
	for(iter_elements=0;iter_elements<total_elements;iter_elements++)
	{
		out_array[iter_elements] = param_a[iter_elements]  
								   + param_b[iter_elements]
								   + param_c[iter_elements];
	}	
}
void get_max_label(long *token_label,double a,double b)
{
	if(a>b)
		*(token_label) = 1;
	else
		*(token_label) = 2;
}
void get_best_labels(long *labels,double *r,long *eb,long *tb,long lot,long sample_id)
{
	/* TBR         : to-be-returned                                                              */
	/* labels(TBR) : contain labels obtained by an application of Viterbi algorithm              */
	/* r           : Hy+tow+delta 													             */
	/* eb          : array that contains starting indices of emission feature for every-sample   */
	/* tb          : array that contains starting indices of transition feature for every-sample */
	/* lot         : length of token in ex_num(sample_id)			 		                     */
	/* sample_id   : current sample/example number   							                 */
	
	
	/* FOR-loop variables */
	long iter_ex,iter_tok,iter_states;
	
	
	/* Viterbi variables */
	long base_index; /* starting index current sample's feature inside "r"*/
	long best_label; /* best label for previous tokens that leads to current state*/
	long ind_w_pp,ind_w_np,ind_w_pn,ind_w_nn; /* transition weight indices for current and previous token inside r*/
	double w_pp,w_np,w_pn,w_nn; /* transition weights for current and previous token*/
	long   *psi_p,*psi_n;	
	double *score_history_p,*score_history_n,*current_scores,*previous_scores;
	
	/* Initialization */
	psi_p = (long *)my_malloc(lot * sizeof(long)); /* best path if current token ends with label "P"*/  
	psi_n = (long *)my_malloc(lot * sizeof(long)); /* best path if current token ends with label "N"*/  
	score_history_p = (double *)my_malloc(lot * sizeof(double)); /* best scores for tokens with label "P"*/
	score_history_n = (double *)my_malloc(lot * sizeof(double)); /* best scores for tokens with label "N"*/
	current_scores  = (double *)my_malloc(3 * sizeof(double));      /*scores for current token:
																	  current_scores[0]: empty 	 					
																	  current_scores[1]: If current token have label "P" 	 					
																	  current_scores[2]: If current token have label "N"	 					
																	*/
	previous_scores = (double *)my_malloc(3 * sizeof(double));      /*scores for last token:
																	  previous_scores[0]: empty 	 					
																	  previous_scores[1]: If last token have label "P" 	 					
																	  previous_scores[2]: If last token have label "N"	 					
																	*/
	
	
	/* Clear Arrays */
	clear_long_array(psi_p,lot);
	clear_long_array(psi_n,lot);
	clear_double_array(score_history_p,lot);																
	clear_double_array(score_history_n,lot);																
	clear_double_array(current_scores,3);																
	clear_double_array(previous_scores,3);		
	
	/* For first-token */
	base_index = eb[sample_id];
	score_history_p[0] = r[base_index];
	score_history_n[0] = r[base_index + 1];
	
	/* Copy transition indices and weights*/
	ind_w_pp = tb[sample_id];
	ind_w_np = tb[sample_id]+2;
	ind_w_pn = tb[sample_id]+1;
	ind_w_nn = tb[sample_id]+3;
	
	w_pp = r[ind_w_pp];
	w_np = r[ind_w_np];
	w_pn = r[ind_w_pn];
	w_nn = r[ind_w_nn];
    
    /***Debugging Code***/
    //printf("-- Printing param_R from inside Viterbi (sample_id: %ld,tokens: %ld) --\n",sample_id,lot);
    //long deb_start_ind,deb_end_ind;
    //deb_start_ind = base_index;/*inclusive*/
    //deb_end_ind = base_index + ( ( (6*lot) - 4 ) - 1 ); /*inclusive*/
    //print_double_array_chunk(r,deb_start_ind,deb_end_ind);
    //print_double_array_chunk_matlab(r,deb_start_ind,deb_end_ind);
    /*******************/
    
	for(iter_tok = 1;iter_tok<lot;iter_tok++)
	{
		base_index += 2; /* base-index for this token inside r */
		
		if(iter_tok >=2){
			/*Update transition indices*/
			ind_w_pp += 4;
			ind_w_np += 4;
			ind_w_pn += 4;
			ind_w_nn += 4;
			/*Update transition weights*/
			w_pp = r[ind_w_pp];
			w_np = r[ind_w_np];
			w_pn = r[ind_w_pn];
			w_nn = r[ind_w_nn];
		}
		//printf("tok:%ld,bi:%ld,w_pp:%f,w_np:%f,w_pn:%f,w_nn:%f\n",iter_tok,base_index,w_pp,w_np,w_pn,w_nn);fflush(stdout);
		for(iter_states=1;iter_states<=2;iter_states++)
		{
			if(iter_states==1)
			{
				current_scores[1] = w_pp + score_history_p[iter_tok-1];
				current_scores[2]=  w_np + score_history_n[iter_tok-1];
				

				/* get label of best score from above expression */
				get_max_label(&best_label,current_scores[1],current_scores[2]);
								
				/* impute score_history_p[iter_tok] */
				score_history_p[iter_tok] = r[base_index] + current_scores[best_label];
				//printf("tot:%f,r=%f,cs[1]=%f,cs[2]=%f,bl_s1=%ld\n",score_history_p[iter_tok],r[base_index],current_scores[1],current_scores[2],best_label);fflush(stdout);
				
				/* store the best path which leads to current token in state "P" */
				if(iter_tok==1)
				{psi_p[iter_tok-1] = best_label;}
				else
				{update_best_path_at_p(psi_p, psi_n, best_label, iter_tok, lot);}
				//printf("bl_s1=%ld\n",best_label);fflush(stdout);
				//printf("psi_p...\n");fflush(stdout);
				//print_long_array(psi_p,lot);
			}
			else
			{
				current_scores[1] = w_pn + score_history_p[iter_tok-1];
				current_scores[2]=  w_nn + score_history_n[iter_tok-1];
				
				/* get label of best score from above expression */
				get_max_label(&best_label,current_scores[1],current_scores[2]);
				
				/* impute score_history_n[iter_tok] */
				score_history_n[iter_tok] = r[base_index+1] + current_scores[best_label];
				//printf("tot:%f,r=%f,cs[1]=%f,cs[2]=%f,bl_s2=%ld\n",score_history_n[iter_tok],r[base_index+1],current_scores[1],current_scores[2],best_label);fflush(stdout);
				
				/* store best path which leads to current token in state "N" */
				if(iter_tok==1)
				{psi_n[iter_tok-1] = best_label;}
				else
				{update_best_path_at_n(psi_p, psi_n, best_label, iter_tok, lot);}
				//printf("bl_s2=%ld\n",best_label);fflush(stdout);
				//printf("psi_n...\n");fflush(stdout);
				//print_long_array(psi_n,lot);
			}
		}
	}
	
	/* Final Step */
	if(score_history_p[lot-1] > score_history_n[lot-1])
	{
		psi_p[lot-1] = 1;
		copy_long_array(psi_p,labels,lot); 
		//printf("Viterbi bestscore(ending in P) = %f\n",score_history_p[lot-1]);
	}
	else
	{
		psi_n[lot-1] = 2;
		copy_long_array(psi_n,labels,lot);
		//printf("Viterbi bestscore(ending in N) = %f\n",score_history_n[lot-1]);
	}
	
	/*debug*/
	//void print_double_array(double *double_array,long tot_entries)
	//print_long_array(long *long_array,long total_entries)
	//printf("history_p...\n");fflush(stdout);
	//print_double_array(score_history_p,lot);
	//printf("history_n...\n");fflush(stdout);
	//print_double_array(score_history_n,lot);
	//printf("labels...\n");fflush(stdout);
    //print_long_array(labels,lot);
    
	/* Free variables */
	/*free(psi_p);
	free(psi_n);
	free(score_history_p);
	free(score_history_n);
	free(current_scores);
	free(previous_scores);*/
}
void update_best_path_at_p(long *psi_p, long *psi_n, long best_label, long current_token, long lot)
{
	/* best_label is the label of current_token-1's token that has lead to current
	 * token in P */
	
	long buffer_width;
	buffer_width =  current_token; /* current_token is 0-based. Its sufficient size 
	is current_token-1 but since we want to copy it inside psi_p or psi_n therefore 
	we will insert 0 at its end */
	
	long buffer_vector[buffer_width]; /*Its last entry is zero */
	
	/*Fill buffer vector with 5 and last element with 0*/
	fill_long_array(buffer_vector,5,buffer_width);
	buffer_vector[buffer_width-1] = 0; /*terminating array*/
	
	/* If best_label is "P" or "1"*/
	if(best_label == 1)
	{
		/*Step-1*/
		/*copy first "buffer_width" elements of psi_p inside buffer_vector*/
		/*void copy_inside_buffer(long *path, long *buffer)*/
		copy_inside_buffer(psi_p,buffer_vector); 
		
		/*Step-2*/
		/*psi_p = [buffer_width,best_label]*/
		/*regenerate_path_vector(long *buffer, long *path,long add_entry)*/
		regenerate_path_vector(buffer_vector,psi_p,best_label);
		
	}
	else
	{  
		/*Step-1*/
		/*copy first "buffer_width" elements of psi_n inside buffer_vector*/
		/*void copy_inside_buffer(long *path, long *buffer)*/
		copy_inside_buffer(psi_n,buffer_vector); 
		
		/*Step-2*/
		/*psi_p = [buffer_width,best_label]*/
		/*regenerate_path_vector(long *buffer, long *path,long add_entry)*/
		regenerate_path_vector(buffer_vector,psi_p,best_label);
	}	
	//printf("psi_p...best_label:%ld\n",best_label);fflush(stdout);
	//print_long_array(psi_p,lot);
}
void update_best_path_at_n(long *psi_p, long *psi_n, long best_label, long current_token, long lot)
{
	/* best_label is the label of current_token-1's token that has lead to current
	 * token in N */
	
	long buffer_width;
	buffer_width =  current_token; /* current_token is 0-based. Its sufficient size 
	is current_token-1 but since we want to copy it inside psi_p or psi_n therefore 
	we will insert 0 at its end */
	
	long buffer_vector[buffer_width]; /*Its last entry is zero */
	
	/*Fill buffer vector with 5 and last element with 0*/
	fill_long_array(buffer_vector,5,buffer_width);
	buffer_vector[buffer_width-1] = 0; /*terminating array*/
	
	/* If best_label is "P" or "1"*/
	if(best_label == 1)
	{
		/*Step-1*/
		/*copy first "buffer_width" elements of psi_p inside buffer_vector*/
		/*void copy_inside_buffer(long *path, long *buffer)*/
		copy_inside_buffer(psi_p,buffer_vector); 
		
		/*Step-2*/
		/*psi_n = [buffer_width,best_label]*/
		/*regenerate_path_vector(long *buffer, long *path,long add_entry)*/
		regenerate_path_vector(buffer_vector,psi_n,best_label);
		
	}
	else
	{  
		/*Step-1*/
		/*copy first "buffer_width" elements of psi_n inside buffer_vector*/
		/*void copy_inside_buffer(long *path, long *buffer)*/
		copy_inside_buffer(psi_n,buffer_vector); 
		
		/*Step-2*/
		/*psi_n = [buffer_width,best_label]*/
		/*regenerate_path_vector(long *buffer, long *path,long add_entry)*/
		regenerate_path_vector(buffer_vector,psi_n,best_label);
	}	
	//printf("psi_n...,best_label:%ld\n",best_label);fflush(stdout);
	//print_long_array(psi_n,lot);
}
void copy_inside_buffer(long *path, long *buffer)
{
	/*copy long values from path[iter_entries] to buffer[iter_entries]*/
	
	long iter_entries;
	iter_entries=0;
	
	while(buffer[iter_entries]!=0)
		{
			buffer[iter_entries]=path[iter_entries];
			iter_entries++;
		}
}
void regenerate_path_vector(long *buffer, long *path,long add_entry)
{
	/*copy long values from buffer[iter_entries] to path[iter_entries]
	  and, it will insert add_entry at the end of path*/
	
	long iter_entries;
	iter_entries=0;
	
	while(buffer[iter_entries]!=0)
		{
			path[iter_entries]=buffer[iter_entries];
			iter_entries++;
		}
	
	/* Update path*/
	path[iter_entries] = add_entry;
}

void read_L(char *file,long *no_of_ws)
{
	   FILE * fp;
       
       if ((fp = fopen (file, "r")) == NULL)
		{ perror (file); exit (1); }
	  
	   fscanf(fp,"%ld",no_of_ws);
	    
       fclose(fp);
}
void read_alpha(char *file,float *alpha)
{
	   FILE * fp;
       long iter_alpha;
       iter_alpha = 0;
       
       if ((fp = fopen (file, "r")) == NULL)
		{ perror (file); exit (1); }
	  
	   while(!feof(fp)) {
		fscanf(fp,"%f\n",&alpha[iter_alpha]);
	    iter_alpha++;
	  } 
	  
	  fclose(fp);
}
double *create_nvector(int n)
{
  double *vector;
  
  vector=(double *)my_malloc(sizeof(double)*(n));
  
  return(vector);
}
double *prod_matrix_vector(TYPE_MATRIX *A, double *v)
/* For column vector v and matrix A (assumed to match in size), computes w=A*v */
{
  int i,j;
  double sum,element_value;
  double *w;
  
  w=create_nvector(A->rows);
  clear_double_array(w,A->rows);/*fill it with 0.0*/
  
  for (i=0;i<A->rows;i++) {
    sum=0.0;
    for (j=0;j<A->cols;j++) {
	  element_value = get_matrix_element(A,i,j); 	
      sum+=v[j]*element_value;
    }
    w[i]=sum;
  }

  return(w);
}

void add_vector(double *sum,double *addendum,long tot_entries)
{
	/* vec_sum[i] = vec_sum[i] + addendum[i] */
	
	long iter_entries;
	for(iter_entries=0;iter_entries<tot_entries;iter_entries++){
		sum[iter_entries] += addendum[iter_entries];
	}	
}
void sub_vector_from_constant(double *vector,double constant,long tot_entries)
{
	/* vector[i] = constant -  vector[i] */
	
	long iter_entries;
	for(iter_entries=0;iter_entries<tot_entries;iter_entries++){
		vector[iter_entries] = constant - vector[iter_entries];
	}	
}

void copy_vector_chunk(double *to,double *from,long start,long end,long dim_to_vec)
{
	/*copy from[start to end] inside to[0-(end-start-1)]*/
	/*dim_to_vec: dimension for vector "to"*/
	long iter_entries,ind;
	ind=-1;
	for(iter_entries = start;iter_entries<=end;iter_entries++){
		ind++;
		to[ind] = from[iter_entries];
	}
	//printf("ind = %ld,dim_to_vec = %ld\n",ind,dim_to_vec);
	assert( ind == (dim_to_vec-1) );
}
void copy_inside_double_array(double *copy_to, double *copy_from,long ind_start,long ind_end,long tot_entries)
{
	/*copy values from copy_from[0:tot_entries-1] inside copy_to[ind_start:ind_end]*/
	
	long iter_entries,current_ind;
	current_ind=-1;
	
	for(iter_entries = ind_start; iter_entries <= ind_end ;iter_entries++){
		current_ind++;
		copy_to[iter_entries] = copy_from[current_ind];
	}
	
	assert(current_ind == (tot_entries-1) );
	
}
void compute_parameters(double *temp,double *K,double *tow,double *delta,TYPE_COMMON_INPUT *common_input)
{
	TYPE_BINLABELS *ws_labels = NULL; 
	TYPE_H *hi                = NULL; /*h_i for every sample*/
   	TYPE_H *hi_transpose      = NULL;
	float *alpha              = NULL; 
	long *base_tab            = NULL;
	long *lot                 = NULL;
	
	long tot_ex;
	long tot_ws; 
	long tot_binfeats;
   	long size_psi_minimum;
    double alphasum;
     
	ws_labels = common_input->ws_labels;
	alpha = common_input->alpha;
	base_tab = common_input->base_tab;
	lot = common_input->lot;
	tot_ex = common_input->tot_ex;
	tot_ws = common_input->tot_ws;
	tot_binfeats = common_input->tot_binfeats;
	size_psi_minimum = common_input->size_psi_min;
    hi                = common_input->hi; 
   	hi_transpose      = common_input->hi_transpose;
   	alphasum          = common_input->alphasum;
	
	/*Input*/
	/* tot_binfeats : dimension of temp(long array of labels in binarized form) */
	/* alpha        : alpha for each working set                                */
	/* tot_ws       : total number of working set (total no. of alphas)         */
	/* base_tab     : stored here are the indices that points to the start of
					  emission label for each example    	                   */
	/* lot          : vector containing lengths of tokens for each training sample*/
	
	/*OUTPUT*/
	/*temp,K,tau*/
	
	/***************************STEP-1: Impute Temp***************************/
	long iter_ex,iter_ws,iter_feats;
	
  	clear_double_array(temp,tot_binfeats);/*filling temp with 0.0*/

	for(iter_feats=0;iter_feats<tot_binfeats;iter_feats++){
		for(iter_ws=0;iter_ws<tot_ws;iter_ws++){
			temp[iter_feats] += (alpha[iter_ws] *  ws_labels[iter_ws].bin_labels[iter_feats]); /*I need to Check it*/
		}		
	}
	/***************************STEP-1: Impute Temp(END)***************************/

	/*************************STEP-2: Impute K*******************************/
	/* K = ( h_1^T * temp(1) ) + ( h_2^T * temp(2) ) + ... + ( h_n^T * temp(n) )  */
	/* K = ( h_1^T * a1 ) + ( h_2^T * a2 ) + ... + ( h_n^T * a(n) )  */
	clear_double_array(K,size_psi_minimum);
	
	long start_index,end_index,tot_a;/*indices for desired sample labels inside temp*/
	double *prod = NULL;
	double *a=NULL;
	tot_a = 0;
	
	for(iter_ex=0;iter_ex<tot_ex;iter_ex++){
		//printf("tok %ld,base_tab:%ld\n",lot[iter_ex],base_tab[iter_ex]);
		start_index = base_tab[iter_ex];                         /*inclusive*/
		end_index = start_index + ( (6*lot[iter_ex]) - 4 ) - 1; /*inclusive*/
		tot_a = end_index - start_index + 1; /*total entires for a*/
		
		a = (double *)my_malloc(tot_a * sizeof(double));
		clear_double_array(a,tot_a);
		
		/*copy iter_ex's entries from temp to a*/
		copy_vector_chunk(a,temp,start_index,end_index,tot_a);
		
		/*compute prod = h_i^T * temp(i) */		
		prod = prod_matrix_vector(hi_transpose[iter_ex].h_matrix, a);
        
        //printf("printing matrix (h_%ld_t) \n",iter_ex);
        //print_matrix(hi_transpose[iter_ex].h_matrix);
        //printf("printing a (temp[%ld])\n",iter_ex);
        //print_double_array_for_matlab(a,tot_a);
		//printf("prod (h_%ld_t x temp[%ld]) \n",iter_ex,iter_ex);
        //print_double_array_for_matlab(prod,hi_transpose[iter_ex].h_matrix->rows);
		
		/*Add prod and K*/
		add_vector(K,prod,size_psi_minimum);/*K=K+prod*/
		
		/*free vectors*/
	    //a=NULL;
	    //prod=NULL;
		free(a);
	    free(prod);
		
	}
    /************************STEP-2: Impute K(END)*************************/
    
	
	/*****STEP-3: Impute tow = [tow(1),tow(2),...,tow(n)] *************/
	/* tow_i = alphasum * h_i * K */	
    
    /*clear tow*/
    clear_double_array(tow,tot_binfeats);/*filling tow with 0.0*/
    
    for(iter_ex=0;iter_ex<tot_ex;iter_ex++){
		//printf("tok %ld,base_tab:%ld\n",lot[iter_ex],base_tab[iter_ex]);
		start_index = base_tab[iter_ex];                         /*inclusive*/
		end_index = start_index + ( (6*lot[iter_ex]) - 4 ) - 1; /*inclusive*/
		tot_a = end_index - start_index + 1; /*total entires for tow(i)*/
		
		/*compute prod = h_i * K */
		/* h_i = |y_i|  x |2D+4|  */		
		/* K   = |2D+4| x 1      */		
		prod = prod_matrix_vector(hi[iter_ex].h_matrix, K);
        
        //printf("printing matrix (h_%ld_t) \n",iter_ex);
        //print_matrix(hi_transpose[iter_ex].h_matrix);
        //printf("printing a (temp[%ld])\n",iter_ex);
        //print_double_array_for_matlab(a,tot_a);
		//printf("prod (h_%ld_t x temp[%ld]) \n",iter_ex,iter_ex);
        //print_double_array_for_matlab(prod,hi_transpose[iter_ex].h_matrix->rows);
		
		/*copy prod inside tow: from tow(start_index,end_index)*/
        copy_inside_double_array(tow, prod,start_index,end_index,tot_a);
        
        //printf("---prod(%ld)-------\n",iter_ex);
        //print_double_array(prod,tot_a);		
		
		/*free vectors*/
	    free(prod);
		
	}
	
	/*Scale tow by alphasum*/
	scale_double_array(tow,alphasum,tot_binfeats);
    /*****STEP-3: Impute tow = [tow(1),tow(2),...,tow(n)](END) *************/
    
    /*****STEP-4: Impute delta = [delta(1),delta(2),...,delta(n)] ******************/
     /*clear delta*/
    clear_double_array(delta,tot_binfeats);/*filling delta with 0.0*/
    
    for(iter_feats=0;iter_feats<tot_binfeats;iter_feats++){		
		/*copy temp inside delta*/
		delta[iter_feats] = temp[iter_feats];        
	}
    
    /*Scale delta by 2*/
   	scale_double_array(delta,2.0,tot_binfeats);
    
    /*Get delta*/ 
    sub_vector_from_constant(delta,alphasum,tot_binfeats);

    /*****STEP-4: Impute delta = [delta(1),delta(2),...,delta(n)](End) *************/

}
double compute_dotproduct(long *a,double *b,long total_entries)
{
	long iter_entries;
	double dot_product;
	dot_product= 0.0;
	
	for(iter_entries=0;iter_entries<total_entries;iter_entries++){
		dot_product += (a[iter_entries]*b[iter_entries]);
	}
	
	return(dot_product);
}
void write_matlabfile(char *filename,SAMPLE *ts)
{
	FILE *fp;
	long iter_ex,iter_tok;
	long tot_ex;
	/*
	  WORD *w1;
	  int j=0;
	  for(i=0;i<sample.n;i++)
	  {
		  w1=sample.examples[i].x.doc->fvec->words;
		  printf("%d ",ex[i].y.class);
		  for(j=0;j<=22; j++){
		  printf("%d:%f ",w1[j].wnum,w1[j].weight);	  
		  }
		  printf("\n");
	  } 
  */
    WORD *sample_words=NULL;    
	EXAMPLE *ex;
	
	tot_ex=ts->n;
	ex = ts->examples;
	
	if( (fp = fopen(filename,"w")) == NULL )
	{ perror (filename); exit(1);}
	
    
	for(iter_ex=0;iter_ex<tot_ex;iter_ex++){
		for(iter_tok=0;iter_tok<ex[iter_ex].y.length;iter_tok++){
			sample_words = ex[iter_ex].x.tokens[iter_tok]->fvec->words;
			fprintf(fp,"%ld %ld",ex[iter_ex].y.labels[iter_tok],iter_ex+1);
			while(sample_words->wnum != 0){
				fprintf(fp," %f",sample_words->wnum,sample_words->weight);
				sample_words++;
			}
			fprintf(fp,"\n");
		}
	}
	
	fclose(fp);
	
}
void write_vl_to_file(char *filename,SAMPLE *ts)
{
	FILE *fp;
	long iter_ex,iter_tok;
	long tot_ex;
	/*
	  WORD *w1;
	  int j=0;
	  for(i=0;i<sample.n;i++)
	  {
		  w1=sample.examples[i].x.doc->fvec->words;
		  printf("%d ",ex[i].y.class);
		  for(j=0;j<=22; j++){
		  printf("%d:%f ",w1[j].wnum,w1[j].weight);	  
		  }
		  printf("\n");
	  } 
  */
    WORD *sample_words=NULL;    
	EXAMPLE *ex;
	
	tot_ex=ts->n;
	ex = ts->examples;
	
	if( (fp = fopen(filename,"w")) == NULL )
	{ perror (filename); exit(1);}
	
    
	for(iter_ex=0;iter_ex<tot_ex;iter_ex++){
		for(iter_tok=0;iter_tok<ex[iter_ex].y.length;iter_tok++){
			sample_words = ex[iter_ex].x.tokens[iter_tok]->fvec->words;
			fprintf(fp,"%ld qid:%ld",ex[iter_ex].y.labels[iter_tok],iter_ex+1);
			while(sample_words->wnum != 0){
				fprintf(fp," %d:%f",sample_words->wnum,sample_words->weight);
				sample_words++;
			}
			fprintf(fp,"\n");
		}
	}
	
	fclose(fp);
	
}
void write_L_to_file(char *filename,long L)
{
	FILE *fp;
	
	if( (fp = fopen(filename,"w")) == NULL )
	{ perror (filename); exit(1);}
	
	fprintf(fp,"%ld\n",L);	
	
	fclose(fp);
	
}
void write_mu_to_file(char *filename,long total_mu)
{
	FILE *fp;
	long iter_mu;
	
	if( (fp = fopen(filename,"w")) == NULL )
	{ perror (filename); exit(1);}
	
    
	for(iter_mu=0;iter_mu<total_mu;iter_mu++)
	{fprintf(fp,"%.8g\n",(double)(1.0/total_mu));}
	
	
	fclose(fp);
	
}
/* Balance Viterbi Algorithm*/
/* IMPORTANT: INDICES IN BALANCED VITERBI IS 1-BASED*/
void copy_BV_path_inside_longArray(BVPATH *copy_from,long *returned_label)
{
	/* This function is called at the end of balanced viterbi 
	 * It will copy best balanced path inside (long *) which makes
	 * integration of current implementation of balanced viterbi seamless
	 * with our previous implementation  
	 */
	long iter_entries,tot_tokens;
	long copyto_index;
	tot_tokens = copy_from->path_length;
	
	copyto_index=0;/*our MVL code assumes labels are stored in 0-Based array*/
	/*iter_entries = 0 will point to -1*/
	for(iter_entries = 1;iter_entries<=tot_tokens;iter_entries++){
		returned_label[copyto_index] = copy_from->path[iter_entries];
		copyto_index++;
	}
}

void fill_path_vector_BV(long *long_array,long fill_with,long tot_tokens)
{
	long iter_entries;
	for(iter_entries = 0;iter_entries<=tot_tokens;iter_entries++)
		{long_array[iter_entries] = fill_with;}
}
void print_path_vector_BV(long *long_array,long tot_tokens)
{
	long iter_entries;

	for(iter_entries = 0;iter_entries<=tot_tokens;iter_entries++){
		printf("%ld",long_array[iter_entries]);
	}
	printf("\n");
}
long check_validity( long t,long P,long y_t )
{
	/* CHECK_VALIDITY : Given current token, number of positive labels upto
	  current token and the state of current token, this function will return 1
	  if the corresponding sequence with given attributes is valid 
	  val_isValid: -1 - Invalid sequence, 1 - Valid sequence
	  t          : Current token (1-based index)
	  P          : Number of positive labels upto this token
	  y_t        : state of current token (either 1 or 2)
	*/

	long val_isValid;
	val_isValid = 1;

	/* Rule - a */
	if(P<0){val_isValid = -1;}
	

	/* Rule - b */
	if(P == 0 && y_t == 1){val_isValid = -1;}

	/* Rule - c */
	if(P == t && y_t == 2){val_isValid = -1;}

	/* Rule - d */
	if(P>t){val_isValid = -1;}
	
	return(val_isValid);

}
void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)
{
	/*copy all long values from copy_from->path[iter_entries] to copy_to->[iter_entries]*/
	/*copy path from argument#1->path to argument#2->path*/
	
	/*
	 *  Note: copy_from->path_length =  copy_to->path_length - 1  
	 */
	 
	long tot_entries_to_copy;
	long iter_entries;
	iter_entries=0;
	tot_entries_to_copy = copy_from->path_length;
	
	for(iter_entries=1;iter_entries<=tot_entries_to_copy;iter_entries++){
		copy_to->path[iter_entries] = copy_from->path[iter_entries]; 
		}
}
void copy_arg2to_bvpath(BVPATH *arg1, long arg2)
{
	/* arg1->path[arg1->path_length] = arg2 */ 
	long append_pos;
	append_pos = arg1->path_length;
	arg1->path[append_pos] = arg2;	
}
void allocate_initialize_2darray_bvpath(BVPATH **arg1, long token_ind)
{
	/* Allocate memory to 2D array using pointer to pointer*/
	/* Then initializes 2D array */
	/* token_ind is the current token number [1,...,lot]*/
	
	/*****************	(Start) Malloc P X 3 ***************************/
	/* Step-1*/
	/* Initialize P X 3 */
	/* arg1 was freed before it arrives here */
	
	/* if token_ind = 2, then initialize arg[P=0][0,1,2],arg[P=1][0,1,2]
	 * and arg[P=2][0,1,2]*/
	   
	   long arg1_col;
	   long iter_rows,iter_p,iter_states;
	   
	   arg1_col=3;/*columns of arg1[][0,1,2]*/
	   
	   /*Check if arg1 is null*/
	   if(arg1 != NULL){arg1=NULL;}
	   
	   /*Allocating rows*/
	   arg1 = (BVPATH **)my_malloc(sizeof(BVPATH *) * (token_ind+1) );
	   for(iter_rows=0;iter_rows<=token_ind;iter_rows++){
		   /*Allocating columns*/
	      arg1[iter_rows] = (BVPATH *)my_malloc(sizeof(BVPATH)*arg1_col);
	   }
	   /*****************	(End) Malloc P X 3 ***************************/

	   /*****************	(Start) Initialize P X 3 ***************************/
	   /* Step-2*/
  	   /* Initializing 2D array of BVPATH */
  	   /* arg1[iter_p][iter_states]*/
   	   for(iter_p=0;iter_p<=token_ind;iter_p++){
		   for(iter_states=1;iter_states<=2;iter_states++){
			   arg1[iter_p][iter_states].path_length = token_ind;
			   arg1[iter_p][iter_states].score = -INFINITY; /*System dependent: It should
															 be -Infinity*/
			   arg1[iter_p][iter_states].isValid = -1;
			   
			   /*path is 1-based: path[0] is nothing
			    * if token_ind = 5 then 
			    * path[1],path[2],path[3],path[4],path[5] is valid path
			    */
			   arg1[iter_p][iter_states].path =
			    (long *)my_malloc( sizeof(long) * (token_ind + 1) );
		   }

	   }
       print_BVPATH(arg1);
 
  	   /*****************	(End) Initialize P X 3 ***************************/
    
	
}
BVPATH *allocate_init_bvpath(long token_ind)
{
	   /* Returns a pointer to BVPATH-object*/
	   BVPATH *arg1=NULL;
	   
	   /*Allocation of memory*/
	   arg1 = (BVPATH *)my_malloc( sizeof(BVPATH) );
	   
	   /*****************	(Start) Initialize *************************/
	    arg1->path_length = token_ind;
		arg1->score = -INFINITY; /*System dependent: It should be -Infinity*/
		arg1->isValid = -1;
			   
		/*path is 1-based: path[0] is nothing
		 * if token_ind = 5 then 
		 * path[1],path[2],path[3],path[4],path[5] is valid path
		*/
		arg1->path = (long *)my_malloc( sizeof(long) * (token_ind + 1) );
			    
		/*fill path with -1*/
		//void fill_path_vector_BV(long *long_array,long fill_with,long tot_tokens);
		fill_path_vector_BV(arg1->path,-1,token_ind);

       //print_BVPATH(arg1);
 
  	   /*****************	(End) Initialize P X 3 ***************************/
    return(arg1);	
}
BVPATH **allocate_init_2darray_bvpath(long token_ind)
{
	/* Allocate memory to 2D array using pointer to pointer*/
	/* Then initializes 2D array */
	/* token_ind is the current token number [1,...,lot]*/
	
	/*****************	(Start) Malloc P X 3 ***************************/
	/* Step-1*/
	/* Initialize P X 3 */
	/* arg1 was freed before it arrives here */
	
	/* if token_ind = 2, then initialize arg[P=0][0,1,2],arg[P=1][0,1,2]
	 * and arg[P=2][0,1,2]*/
	   BVPATH **arg1=NULL;
	   long arg1_col;
	   long iter_rows,iter_p,iter_states;
	   
	   /*Check if arg1 is null*/
	   
	   
	   arg1_col=3;/*columns of arg1[][0,1,2]*/
	   
	   
	   /*Allocating rows*/
	   arg1 = (BVPATH **)my_malloc(sizeof(BVPATH *) * (token_ind+1) );
	   for(iter_rows=0;iter_rows<=token_ind;iter_rows++){
		   /*Allocating columns*/
	      arg1[iter_rows] = (BVPATH *)my_malloc(sizeof(BVPATH)*arg1_col);
	   }
	   /*****************	(End) Malloc P X 3 ***************************/

	   /*****************	(Start) Initialize P X 3 ***************************/
	   /* Step-2*/
  	   /* Initializing 2D array of BVPATH */
  	   /* arg1[iter_p][iter_states]*/
   	   for(iter_p=0;iter_p<=token_ind;iter_p++){
		   for(iter_states=1;iter_states<=2;iter_states++){
			   arg1[iter_p][iter_states].path_length = token_ind;
			   arg1[iter_p][iter_states].score = -INFINITY; /*System dependent: It should
															 be -Infinity*/
			   arg1[iter_p][iter_states].isValid = -1;
			   
			   /*path is 1-based: path[0] is nothing
			    * if token_ind = 5 then 
			    * path[1],path[2],path[3],path[4],path[5] is valid path
			    */
			   arg1[iter_p][iter_states].path =
			    (long *)my_malloc( sizeof(long) * (token_ind + 1) );
			    
			    /*fill path with -1*/
			    //void fill_path_vector_BV(long *long_array,long fill_with,long tot_tokens);
			    fill_path_vector_BV(arg1[iter_p][iter_states].path,-1,token_ind);

		   }

	   }
       //print_BVPATH(arg1);
 
  	   /*****************	(End) Initialize P X 3 ***************************/
    return(arg1);
	
}
void free_BVPATH_2DArray(BVPATH **arg1)
{
	/************free arg1(Start)***************/
	long tot_token;
	long iter_P,iter_states;
	tot_token = arg1[0][2].path_length; /*total tokens upto current point*/
	   for(iter_P=0;iter_P<=tot_token;iter_P++){
		   for(iter_states=1;iter_states<=2;iter_states++){
			   /*free path vector*/
			   if(arg1[iter_P][iter_states].path)
					free(arg1[iter_P][iter_states].path);
		   }
		   /*free arg1[iter_P]*/
		   free(arg1[iter_P]);				   
	   }
	   /*free arg1*/
	   free(arg1);
	 /************free arg1(End)***************/
}
void free_BVPATH(BVPATH *arg1)
{
	/************free arg1(Start)***************/
	   free(arg1->path);				   
	   
	   /*free arg1*/
	   free(arg1);
	 /************free arg1(End)***************/
}
BVPATH **copy_BVPATH_arg1to_arg2(BVPATH **arg1,BVPATH **arg2)
{
	/* arg2 = arg1 (Equivalent to Matlab) */
	/* where arg1 and arg2 are 2D arrays of BVPATH*/
	
	long tot_tokens;
	long iter_P,iter_states;
	
	/************free arg2(Start)***************/
	free_BVPATH_2DArray(arg2);
	 /************free arg2(End)***************/
	
	/************Reallocate and initialize arg2(Start)***************/
    tot_tokens = arg1[0][2].path_length; /*total tokens upto current point*/
    //allocate_initialize_2darray_bvpath(arg2,tot_tokens);
   	arg2=allocate_init_2darray_bvpath(tot_tokens);
    /*Here dimensionality of arg1 and arg2 becomes equal*/
	/************Reallocate arg2(End)***************/
	
	/************Start: Copy arg1 to arg2 ****************************/
	for(iter_P=0;iter_P<=tot_tokens;iter_P++){
		for(iter_states=1;iter_states<=2;iter_states++){
			/*Copy path_length*/
			arg2[iter_P][iter_states].path_length = arg1[iter_P][iter_states].path_length;
			
			/*Copy score*/
			arg2[iter_P][iter_states].score = arg1[iter_P][iter_states].score;
			
			/*Copy path_length*/
			arg2[iter_P][iter_states].isValid = arg1[iter_P][iter_states].isValid;
			
			/*Copy Path*/
			//copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to);
			copy_path_arg12(&arg1[iter_P][iter_states], &arg2[iter_P][iter_states] );

		}		
	}	
	/************End: Copy arg1 to arg2 ****************************/
	return(arg2);
}

void print_BVPATH(BVPATH **arg1)
{
	long tot_tokens;
	long iter_P,iter_states;
	
	tot_tokens = arg1[0][2].path_length;
	printf("(inside print_BVPATH)tot_tokens:%ld\n",tot_tokens);
	/************Start: Print arg1  ****************************/
	if(arg1 != NULL){
		for(iter_P=0;iter_P<=tot_tokens;iter_P++){
			for(iter_states=1;iter_states<=2;iter_states++){
				/*Print path_length*/
				printf("arg1[%ld][%ld].path_length=%ld ----- ",iter_P,iter_states,arg1[iter_P][iter_states].path_length);
				//void print_path_vector_BV(long *long_array,long tot_tokens);
				print_path_vector_BV(arg1[iter_P][iter_states].path,tot_tokens);

				/*Print score*/
				printf("arg1[%ld][%ld].score=%f\n",iter_P,iter_states,arg1[iter_P][iter_states].score);
				
				/*Print isValid*/
				printf("arg1[%ld][%ld].isValid=%ld\n",iter_P,iter_states,arg1[iter_P][iter_states].isValid);
							
			}		
		}	
	}
	/************End: Print arg1  ****************************/
	
}

void call_BViterbi(long *labels,double *r,long *eb,long *tb,long lot,long sample_id)
{
	/* TBR         : to-be-returned                                                              */
	/* labels(TBR) : contain labels obtained by an application of Viterbi algorithm              */
	/* r           : Hy+tow+delta 													             */
	/* eb          : array that contains starting indices of emission feature for every-sample   */
	/* tb          : array that contains starting indices of transition feature for every-sample */
	/* lot         : length of token in ex_num(sample_id)			 		                     */
	/* sample_id   : current sample/example number   							                 */
	
	
	/* FOR-loop variables */
	long iter_tok,iter_P,iter_states;
	
	
	/* Viterbi variables */
	BVPATH **psi_prev_tok=NULL,**psi_current_tok=NULL;
	BVPATH *temp_a=NULL,*temp_b=NULL;
	long big_P,current_token,isValidPath; /*current_token:1-based id of current token*/
	long base_index; /* starting index current sample's feature inside "r"*/
	long best_label; /* best label for previous tokens that leads to current state*/
	long ind_w_pp,ind_w_np,ind_w_pn,ind_w_nn; /* transition weight indices for current and previous token inside r*/
	float w_pp,w_np,w_pn,w_nn; /* transition weights for current and previous token*/
	
	/**Balance Viterbi Final Step**/
	long best_label_P_ind;/*Balance-Viterbi final step*/
	long best_label_state_ind;/*Balance-Viterbi final step*/
	long min_balanced,max_balanced;/*Balance Constraints*/
	float beta_BV;/*beta for balanced viterbi*/
	float best_score_BV,temp_score_BV;/*Balance-Viterbi final step*/
	/* Initialization */
	isValidPath = 0;
	
	/* For first-token */
	current_token = 1;
	base_index = eb[sample_id];
	
	//printf("file:%s,line:%d\n",__FILE__,__LINE__);fflush(stdout);
	psi_prev_tok=allocate_init_2darray_bvpath(1);
    
    /*Assign appropriate weights to 1st-Token*/
	psi_prev_tok[1][1].isValid = 1;
	psi_prev_tok[1][1].score = (float)r[base_index];
	psi_prev_tok[1][1].path[current_token] = 1;
	psi_prev_tok[0][2].isValid = 1;
	psi_prev_tok[0][2].score = (float)r[base_index + 1];
    psi_prev_tok[0][2].path[current_token] = 2;
    //print_BVPATH(psi_prev_tok);

    
	/* Copy transition indices and weights*/
	ind_w_pp = tb[sample_id];
	ind_w_np = tb[sample_id]+2;
	ind_w_pn = tb[sample_id]+1;
	ind_w_nn = tb[sample_id]+3;
	
	
   for(iter_tok = 2;iter_tok<=lot;iter_tok++)
	{
		base_index += 2; /* base-index for this token inside r */
		
		/*****(Start) Transition scores are updated here******/
		/*****(WE are zeroing out the effect of transitions)*/
		w_pp = (float)r[ind_w_pp];
		w_np = (float)r[ind_w_np];
		w_pn = (float)r[ind_w_pn];
		w_nn = (float)r[ind_w_nn];
		
		/*****(End) Transition scores are updated here******/
		
		/********(Start)Initialize ( 2D-Array ) psi_current_tok *************/
		/*BVPATH **allocate_init_2darray_bvpath(long token_ind)*/
	    psi_current_tok=allocate_init_2darray_bvpath(iter_tok);	
		/********(End)Initialize ( 2D-Array ) psi_current_tok *************/
		
		/******(Start) Normal Balanced Viterbi Algorithm (Main-Loop)*******/
	    for(iter_P=0;iter_P<=iter_tok;iter_P++){
			for(iter_states=1;iter_states<=2;iter_states++){
				/* Check if psi(iter_P,y_t=iter_states) is valid*/
				/*long check_validity( long t,long P,long y_t )*/
				isValidPath = check_validity(iter_tok,iter_P,iter_states);
				
				/***** (Start) Continue calculation with this state if isValidPath==1****/
				if(isValidPath == 1){
					
					/****(Start) If iter_states == 1*****/
					if(iter_states == 1){
						/*Initialize temp_a and temp_b*/
						/*BVPATH *allocate_init_bvpath(long token_ind)*/
						temp_a = allocate_init_bvpath(iter_tok);
						temp_b = allocate_init_bvpath(iter_tok);

						/*Check Validity of temp_a and temp_b*/
						temp_a->isValid = check_validity(iter_tok-1,iter_P-1,2);
						temp_b->isValid = check_validity(iter_tok-1,iter_P-1,1);
						
						/*Calculate score of temp_a and temp_b*/
						if(temp_a->isValid == 1){
							temp_a->score = psi_prev_tok[iter_P-1][2].score + w_np + ((float)r[base_index]);
							
							/*void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)*/
							copy_path_arg12(&psi_prev_tok[iter_P-1][2], temp_a);
							
							/*Append 1 at the end of temp_a*/
							/*void copy_arg2to_bvpath(BVPATH *arg1, long arg2)*/
							copy_arg2to_bvpath(temp_a, 1);  	
						}
						if(temp_b->isValid == 1){
							temp_b->score = psi_prev_tok[iter_P-1][1].score + w_pp + ((float)r[base_index]);
							
							/*void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)*/
							copy_path_arg12(&psi_prev_tok[iter_P-1][1], temp_b);
							
							/*Append 1 at the end of temp_b*/
							/*void copy_arg2to_bvpath(BVPATH *arg1, long arg2)*/
							copy_arg2to_bvpath(temp_b, 1); 
						}
						/* Compare temp_a and temp_b and 
						 * Update psi_current_tok[iter_P][iter_states]
						 */
						if(temp_a->score > temp_b->score){
							psi_current_tok[iter_P][iter_states].score = temp_a->score;
							copy_path_arg12(temp_a,&psi_current_tok[iter_P][iter_states]);
							psi_current_tok[iter_P][iter_states].isValid = temp_a->isValid;
						}
						else{
							psi_current_tok[iter_P][iter_states].score = temp_b->score;
							copy_path_arg12(temp_b,&psi_current_tok[iter_P][iter_states]);
							psi_current_tok[iter_P][iter_states].isValid = temp_b->isValid;
						}						
						/*Free temp_a and temp_b*/
						//free_BVPATH(BVPATH *arg1);
						free_BVPATH(temp_a);
						free_BVPATH(temp_b);
					}
					/****(End) If iter_states == 1*****/
					
					
					/****(Start) If iter_states == 2*****/
					if(iter_states == 2){
						/*Initialize temp_a and temp_b*/
						/*BVPATH *allocate_init_bvpath(long token_ind)*/
						temp_a = allocate_init_bvpath(iter_tok);
						temp_b = allocate_init_bvpath(iter_tok);

						/*Check Validity of temp_a and temp_b*/
						temp_a->isValid = check_validity(iter_tok-1,iter_P,2);
						temp_b->isValid = check_validity(iter_tok-1,iter_P,1);
						
						/*Calculate score of temp_a and temp_b*/
						if(temp_a->isValid == 1){
							temp_a->score = psi_prev_tok[iter_P][2].score + w_nn + ((float)r[base_index+1]);
							
							/*void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)*/
							copy_path_arg12(&psi_prev_tok[iter_P][2], temp_a);
							
							/*Append 2 at the end of temp_a*/
							/*void copy_arg2to_bvpath(BVPATH *arg1, long arg2)*/
							copy_arg2to_bvpath(temp_a, 2);  	
						}
						if(temp_b->isValid == 1){
							temp_b->score = psi_prev_tok[iter_P][1].score + w_pn + ((float)r[base_index+1]);
							
							/*void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)*/
							copy_path_arg12(&psi_prev_tok[iter_P][1], temp_b);
							
							/*Append 2 at the end of temp_b*/
							/*void copy_arg2to_bvpath(BVPATH *arg1, long arg2)*/
							copy_arg2to_bvpath(temp_b, 2); 
						}
						/* Compare temp_a and temp_b and 
						 * Update psi_current_tok[iter_P][iter_states]
						 */
						if(temp_a->score > temp_b->score){
							psi_current_tok[iter_P][iter_states].score = temp_a->score;
							copy_path_arg12(temp_a,&psi_current_tok[iter_P][iter_states]);
 						    psi_current_tok[iter_P][iter_states].isValid = temp_a->isValid;
						}
						else{
							psi_current_tok[iter_P][iter_states].score = temp_b->score;
							copy_path_arg12(temp_b,&psi_current_tok[iter_P][iter_states]);
 						    psi_current_tok[iter_P][iter_states].isValid = temp_b->isValid;

						}						
						/*Free temp_a and temp_b*/
						//free_BVPATH(BVPATH *arg1);
						free_BVPATH(temp_a);
						free_BVPATH(temp_b);
					}
					
					/****(End) If iter_states == 2*****/
					
				}
				/***** (End) Continue calculation with this state if isValidPath==1*****/
			}
		}
		/******(End) Normal Balanced Viterbi Algorithm(Main-Loop)*******/
		
		
		/****(Start) Copy psi_current_tok[][] into psi_prev_tok[][] and free psi_current_token****/
		/* Free psi_prev_tok*/
		free_BVPATH_2DArray(psi_prev_tok);
		/* Initialize psi_prev_tok*/
		psi_prev_tok=allocate_init_2darray_bvpath(iter_tok);
		/* Copy psi_current_tok into psi_prev_tok*/
		psi_prev_tok = copy_BVPATH_arg1to_arg2(psi_current_tok,psi_prev_tok);
		/* Free psi_current_tok*/
	    free_BVPATH_2DArray(psi_current_tok);
		/****(End) Copy psi_current_tok[][] into psi_prev_tok[][] and free psi_current_token****/
		
		/*******Update transition indices for next token***********/
		ind_w_pp += 4;
		ind_w_np += 4;
		ind_w_pn += 4;
		ind_w_nn += 4;
		/***************************************************/	
		
	}
	//print_BVPATH(psi_prev_tok);
	
	/**Return Best Label**/
	/**Balance Viterbi Final Step**/
	//long best_label_P_ind;/*Balance-Viterbi final step*/
	//long best_label_state_ind;/*Balance-Viterbi final step*/
	//float best_score_BV,temp_score_BV;/*Balance-Viterbi final step*/
	
	/****(Start) Normal Viterbi Label ***********/
	/*
	best_score_BV=-INFINITY;
	for(iter_P=0;iter_P<=lot;iter_P++){
		for(iter_states=1;iter_states<=2;iter_states++){
			if(psi_prev_tok[iter_P][iter_states].isValid == 1){
				temp_score_BV = psi_prev_tok[iter_P][iter_states].score;
				// Compare if best score is beaten
				if(temp_score_BV > best_score_BV){
					best_label_P_ind = iter_P;
					best_label_state_ind = iter_states;
					best_score_BV = temp_score_BV;
				}				
			}
		}
	}
	*/ 
    //printf("best_score = %f,psi_previous_tok[%ld][%ld] :\n",best_score_BV,best_label_P_ind,best_label_state_ind);
	
	/****(End) Normal Viterbi Label ***********/
	
	/****(Start) Balanced Viterbi Label ***********/
	//long best_label_P_ind;/*Balance-Viterbi final step*/
	//long best_label_state_ind;/*Balance-Viterbi final step*/
	//long min_balanced,max_balanced;/*Balance Constraints*/
	//float beta_BV;/*beta for balanced viterbi*/
	//float best_score_BV,temp_score_BV;/*Balance-Viterbi final step*/
	
	beta_BV = 0.48;
	min_balanced = beta_BV * lot;
	//max_balanced = (1-beta_BV) * lot;
	max_balanced = lot - min_balanced;
	//printf("beta=%f,min_balanced(discarding decimal places)=%ld,max_balanced(discarding decimal places)=%ld\n",beta_BV,min_balanced,max_balanced);
	best_score_BV=-INFINITY;
	for(iter_P=min_balanced;iter_P<=max_balanced;iter_P++){
		for(iter_states=1;iter_states<=2;iter_states++){
			if(psi_prev_tok[iter_P][iter_states].isValid == 1){
				temp_score_BV = psi_prev_tok[iter_P][iter_states].score;
				// Compare if best score is beaten
				if(temp_score_BV > best_score_BV){
					best_label_P_ind = iter_P;
					best_label_state_ind = iter_states;
					best_score_BV = temp_score_BV;
				}				
			}
		}
	}
    //printf("best_score = %f,psi_previous_tok[%ld][%ld] :\n",best_score_BV,best_label_P_ind,best_label_state_ind);
    /****(End) Balanced Viterbi Label ***********/
	
	/*****(Start) Copy best labels obtained in labels *****/
	//copy_BV_path_inside_longArray(BVPATH *copy_from,long *returned_label)
	copy_BV_path_inside_longArray(&psi_prev_tok[best_label_P_ind][best_label_state_ind],labels);
	
	/*****(End)   Copy best labels obtained in labels *****/	
}

void call_BViterbi_fully_operational_backup(long *labels,double *r,long *eb,long *tb,long lot,long sample_id)
{
	/* TBR         : to-be-returned                                                              */
	/* labels(TBR) : contain labels obtained by an application of Viterbi algorithm              */
	/* r           : Hy+tow+delta 													             */
	/* eb          : array that contains starting indices of emission feature for every-sample   */
	/* tb          : array that contains starting indices of transition feature for every-sample */
	/* lot         : length of token in ex_num(sample_id)			 		                     */
	/* sample_id   : current sample/example number   							                 */
	
	
	/* FOR-loop variables */
	long iter_tok,iter_P,iter_states;
	
	
	/* Viterbi variables */
	BVPATH **psi_prev_tok=NULL,**psi_current_tok=NULL;
	BVPATH *temp_a=NULL,*temp_b=NULL;
	long big_P,current_token,isValidPath; /*current_token:1-based id of current token*/
	long base_index; /* starting index current sample's feature inside "r"*/
	long best_label; /* best label for previous tokens that leads to current state*/
	long ind_w_pp,ind_w_np,ind_w_pn,ind_w_nn; /* transition weight indices for current and previous token inside r*/
	float w_pp,w_np,w_pn,w_nn; /* transition weights for current and previous token*/
	
	/**Balance Viterbi Final Step**/
	long best_label_P_ind;/*Balance-Viterbi final step*/
	long best_label_state_ind;/*Balance-Viterbi final step*/
	long min_balanced,max_balanced;/*Balance Constraints*/
	float beta_BV;/*beta for balanced viterbi*/
	float best_score_BV,temp_score_BV;/*Balance-Viterbi final step*/
	/* Initialization */
	isValidPath = 0;
	
	/* For first-token */
	current_token = 1;
	base_index = eb[sample_id];
	
	//printf("file:%s,line:%d\n",__FILE__,__LINE__);fflush(stdout);
	psi_prev_tok=allocate_init_2darray_bvpath(1);
    
    /*Assign appropriate weights to 1st-Token*/
	psi_prev_tok[1][1].isValid = 1;
	psi_prev_tok[1][1].score = (float)r[base_index];
	psi_prev_tok[1][1].path[current_token] = 1;
	psi_prev_tok[0][2].isValid = 1;
	psi_prev_tok[0][2].score = (float)r[base_index + 1];
    psi_prev_tok[0][2].path[current_token] = 2;
    //print_BVPATH(psi_prev_tok);

    
	/* Copy transition indices and weights*/
	ind_w_pp = tb[sample_id];
	ind_w_np = tb[sample_id]+2;
	ind_w_pn = tb[sample_id]+1;
	ind_w_nn = tb[sample_id]+3;
	
	
   for(iter_tok = 2;iter_tok<=lot;iter_tok++)
	{
		base_index += 2; /* base-index for this token inside r */
		
		/*****(Start) Transition scores are updated here******/
		w_pp = (float)r[ind_w_pp];
		w_np = (float)r[ind_w_np];
		w_pn = (float)r[ind_w_pn];
		w_nn = (float)r[ind_w_nn];			
		/*****(End) Transition scores are updated here******/
		
		/********(Start)Initialize ( 2D-Array ) psi_current_tok *************/
		/*BVPATH **allocate_init_2darray_bvpath(long token_ind)*/
	    psi_current_tok=allocate_init_2darray_bvpath(iter_tok);	
		/********(End)Initialize ( 2D-Array ) psi_current_tok *************/
		
		/******(Start) Normal Balanced Viterbi Algorithm (Main-Loop)*******/
	    for(iter_P=0;iter_P<=iter_tok;iter_P++){
			for(iter_states=1;iter_states<=2;iter_states++){
				/* Check if psi(iter_P,y_t=iter_states) is valid*/
				/*long check_validity( long t,long P,long y_t )*/
				isValidPath = check_validity(iter_tok,iter_P,iter_states);
				
				/***** (Start) Continue calculation with this state if isValidPath==1****/
				if(isValidPath == 1){
					
					/****(Start) If iter_states == 1*****/
					if(iter_states == 1){
						/*Initialize temp_a and temp_b*/
						/*BVPATH *allocate_init_bvpath(long token_ind)*/
						temp_a = allocate_init_bvpath(iter_tok);
						temp_b = allocate_init_bvpath(iter_tok);

						/*Check Validity of temp_a and temp_b*/
						temp_a->isValid = check_validity(iter_tok-1,iter_P-1,2);
						temp_b->isValid = check_validity(iter_tok-1,iter_P-1,1);
						
						/*Calculate score of temp_a and temp_b*/
						if(temp_a->isValid == 1){
							temp_a->score = psi_prev_tok[iter_P-1][2].score + w_np + ((float)r[base_index]);
							
							/*void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)*/
							copy_path_arg12(&psi_prev_tok[iter_P-1][2], temp_a);
							
							/*Append 1 at the end of temp_a*/
							/*void copy_arg2to_bvpath(BVPATH *arg1, long arg2)*/
							copy_arg2to_bvpath(temp_a, 1);  	
						}
						if(temp_b->isValid == 1){
							temp_b->score = psi_prev_tok[iter_P-1][1].score + w_pp + ((float)r[base_index]);
							
							/*void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)*/
							copy_path_arg12(&psi_prev_tok[iter_P-1][1], temp_b);
							
							/*Append 1 at the end of temp_b*/
							/*void copy_arg2to_bvpath(BVPATH *arg1, long arg2)*/
							copy_arg2to_bvpath(temp_b, 1); 
						}
						/* Compare temp_a and temp_b and 
						 * Update psi_current_tok[iter_P][iter_states]
						 */
						if(temp_a->score > temp_b->score){
							psi_current_tok[iter_P][iter_states].score = temp_a->score;
							copy_path_arg12(temp_a,&psi_current_tok[iter_P][iter_states]);
							psi_current_tok[iter_P][iter_states].isValid = temp_a->isValid;
						}
						else{
							psi_current_tok[iter_P][iter_states].score = temp_b->score;
							copy_path_arg12(temp_b,&psi_current_tok[iter_P][iter_states]);
							psi_current_tok[iter_P][iter_states].isValid = temp_b->isValid;
						}						
						/*Free temp_a and temp_b*/
						//free_BVPATH(BVPATH *arg1);
						free_BVPATH(temp_a);
						free_BVPATH(temp_b);
					}
					/****(End) If iter_states == 1*****/
					
					
					/****(Start) If iter_states == 2*****/
					if(iter_states == 2){
						/*Initialize temp_a and temp_b*/
						/*BVPATH *allocate_init_bvpath(long token_ind)*/
						temp_a = allocate_init_bvpath(iter_tok);
						temp_b = allocate_init_bvpath(iter_tok);

						/*Check Validity of temp_a and temp_b*/
						temp_a->isValid = check_validity(iter_tok-1,iter_P,2);
						temp_b->isValid = check_validity(iter_tok-1,iter_P,1);
						
						/*Calculate score of temp_a and temp_b*/
						if(temp_a->isValid == 1){
							temp_a->score = psi_prev_tok[iter_P][2].score + w_nn + ((float)r[base_index+1]);
							
							/*void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)*/
							copy_path_arg12(&psi_prev_tok[iter_P][2], temp_a);
							
							/*Append 2 at the end of temp_a*/
							/*void copy_arg2to_bvpath(BVPATH *arg1, long arg2)*/
							copy_arg2to_bvpath(temp_a, 2);  	
						}
						if(temp_b->isValid == 1){
							temp_b->score = psi_prev_tok[iter_P][1].score + w_pn + ((float)r[base_index+1]);
							
							/*void copy_path_arg12(BVPATH *copy_from, BVPATH *copy_to)*/
							copy_path_arg12(&psi_prev_tok[iter_P][1], temp_b);
							
							/*Append 2 at the end of temp_b*/
							/*void copy_arg2to_bvpath(BVPATH *arg1, long arg2)*/
							copy_arg2to_bvpath(temp_b, 2); 
						}
						/* Compare temp_a and temp_b and 
						 * Update psi_current_tok[iter_P][iter_states]
						 */
						if(temp_a->score > temp_b->score){
							psi_current_tok[iter_P][iter_states].score = temp_a->score;
							copy_path_arg12(temp_a,&psi_current_tok[iter_P][iter_states]);
 						    psi_current_tok[iter_P][iter_states].isValid = temp_a->isValid;
						}
						else{
							psi_current_tok[iter_P][iter_states].score = temp_b->score;
							copy_path_arg12(temp_b,&psi_current_tok[iter_P][iter_states]);
 						    psi_current_tok[iter_P][iter_states].isValid = temp_b->isValid;

						}						
						/*Free temp_a and temp_b*/
						//free_BVPATH(BVPATH *arg1);
						free_BVPATH(temp_a);
						free_BVPATH(temp_b);
					}
					
					/****(End) If iter_states == 2*****/
					
				}
				/***** (End) Continue calculation with this state if isValidPath==1*****/
			}
		}
		/******(End) Normal Balanced Viterbi Algorithm(Main-Loop)*******/
		
		
		/****(Start) Copy psi_current_tok[][] into psi_prev_tok[][] and free psi_current_token****/
		/* Free psi_prev_tok*/
		free_BVPATH_2DArray(psi_prev_tok);
		/* Initialize psi_prev_tok*/
		psi_prev_tok=allocate_init_2darray_bvpath(iter_tok);
		/* Copy psi_current_tok into psi_prev_tok*/
		psi_prev_tok = copy_BVPATH_arg1to_arg2(psi_current_tok,psi_prev_tok);
		/* Free psi_current_tok*/
	    free_BVPATH_2DArray(psi_current_tok);
		/****(End) Copy psi_current_tok[][] into psi_prev_tok[][] and free psi_current_token****/
		
		/*******Update transition indices for next token***********/
		ind_w_pp += 4;
		ind_w_np += 4;
		ind_w_pn += 4;
		ind_w_nn += 4;
		/***************************************************/	
		
	}
	print_BVPATH(psi_prev_tok);
	
	/**Return Best Label**/
	/**Balance Viterbi Final Step**/
	//long best_label_P_ind;/*Balance-Viterbi final step*/
	//long best_label_state_ind;/*Balance-Viterbi final step*/
	//float best_score_BV,temp_score_BV;/*Balance-Viterbi final step*/
	
	/****(Start) Normal Viterbi Label ***********/
	/*
	best_score_BV=-INFINITY;
	for(iter_P=0;iter_P<=lot;iter_P++){
		for(iter_states=1;iter_states<=2;iter_states++){
			if(psi_prev_tok[iter_P][iter_states].isValid == 1){
				temp_score_BV = psi_prev_tok[iter_P][iter_states].score;
				// Compare if best score is beaten
				if(temp_score_BV > best_score_BV){
					best_label_P_ind = iter_P;
					best_label_state_ind = iter_states;
					best_score_BV = temp_score_BV;
				}				
			}
		}
	}
    printf("best_score = %f,psi_previous_tok[%ld][%ld] :\n",best_score_BV,best_label_P_ind,best_label_state_ind);
	*/
	/****(End) Normal Viterbi Label ***********/
	
	/****(Start) Balanced Viterbi Label ***********/
	//long best_label_P_ind;/*Balance-Viterbi final step*/
	//long best_label_state_ind;/*Balance-Viterbi final step*/
	//long min_balanced,max_balanced;/*Balance Constraints*/
	//float beta_BV;/*beta for balanced viterbi*/
	//float best_score_BV,temp_score_BV;/*Balance-Viterbi final step*/
	
	beta_BV = 0.03;
	min_balanced = beta_BV * lot;
	max_balanced = (1-beta_BV) * lot;
	//printf("beta=%f,min_balanced(discarding decimal places)=%ld,max_balanced(discarding decimal places)=%ld\n",beta_BV,min_balanced,max_balanced);
	best_score_BV=-INFINITY;
	for(iter_P=min_balanced;iter_P<=max_balanced;iter_P++){
		for(iter_states=1;iter_states<=2;iter_states++){
			if(psi_prev_tok[iter_P][iter_states].isValid == 1){
				temp_score_BV = psi_prev_tok[iter_P][iter_states].score;
				// Compare if best score is beaten
				if(temp_score_BV > best_score_BV){
					best_label_P_ind = iter_P;
					best_label_state_ind = iter_states;
					best_score_BV = temp_score_BV;
				}				
			}
		}
	}
    //printf("best_score = %f,psi_previous_tok[%ld][%ld] :\n",best_score_BV,best_label_P_ind,best_label_state_ind);
	/****(End) Normal Viterbi Label ***********/
	

		
}












