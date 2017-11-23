/*
 *  Please write your name and net ID below
 *
 *  Last name: Evci
 *  First name: Utku
 *  Net ID: ue225
 *
 */


/*
 * You compile with:
 * 		nvcc -arch sm_35 -rdc=true -o sudokusolver sudokusolver.cu
 */
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
// #define DEBUG

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
/*****************************************************************/

__global__ void fillSudokuSafeAndFork(char* memory,int* stats)
{
  /*
    We have nBlocks many available blocks.
  */
  uint i,current_poss,j,temp,mat_i,mat_j,k;
  char* block_memory = memory+(81*blockIdx.x);

  __shared__ uint row_used_numbers[9];
  __shared__ uint col_used_numbers[9];
  __shared__ uint cell_used_numbers[9];
  __shared__ char progress_flag;
  __shared__ char done_flag;
  __shared__ char error_flag;
  __shared__ int min_forks;
  __shared__ int scheduling_thread;

    // check whether all blocks are idle or not.This shouldn't happen
    if (blockIdx.x==0){ //first block
      if (threadIdx.x==0) {
        progress_flag=0;
      }
      __syncthreads();
      for(i=threadIdx.x;i<gridDim.x;i+=blockDim.x){
        if (stats[i] > 0){
          progress_flag=1;
        }
      }
      __syncthreads();
      if (progress_flag==0 and threadIdx.x == 0){
        //no active block terminate.
        if (threadIdx.x==0) printf("no active blocks...terminating\n");
        stats[gridDim.x]=2;
      }
    }

    //If block is active work on it.
    if(stats[blockIdx.x]==1){
      if (threadIdx.x==0) {
        error_flag = 0;  //set to 1 if the board is wrongs
        done_flag = 0;  //set to 1 if the board is solved
        progress_flag=1; //set to 0 if no reterministic progress can be made.
      }
      __syncthreads();

      while(!error_flag && !done_flag &&progress_flag ){
        __syncthreads();
        // 1st check whether the board is valid and fill X_used_numbers arrays for rows,columns and cells.
        //*************************
        if (threadIdx.x<9){
          // TODO optimize here such that there wraps does row/col/cell.
            row_used_numbers[threadIdx.x] = 0;
            col_used_numbers[threadIdx.x] = 0;
            cell_used_numbers[threadIdx.x] = 0;
            for(i=0;i<9;i++){
                //rows
                temp = block_memory[threadIdx.x*9+i];
                if (temp) { //!=0
                    if ((row_used_numbers[threadIdx.x]>>(temp-1)) & 1){
                        // This is bad, you have the same number in the same row. This solution fails
                        error_flag=10+i;
                    }
                    //set n'th bit to 1.
                    row_used_numbers[threadIdx.x] |= 1<<(temp-1);
                }
                //columns
                temp = block_memory[i*9+threadIdx.x];
                if (temp) { //!=0
                    if ((col_used_numbers[threadIdx.x]>>(temp-1)) & 1){
                        // This is bad, you have the same number in the same column. This solution fails
                        error_flag=20+i;
                    }
                    //set n'th bit to 1.
                    col_used_numbers[threadIdx.x] |= 1<<(temp-1);
                }
            }
            //cells
            for (i=(threadIdx.x/3)*3;i<((threadIdx.x/3+1)*3);i++){
              for (j=(threadIdx.x%3)*3;j<((threadIdx.x%3+1)*3);j++){
                temp = block_memory[i*9+j];
                if (temp) { //!=0
                    if ((cell_used_numbers[threadIdx.x]>>(temp-1)) & 1){
                        // This is bad, you have the same number in the same cell. This solution fails
                        error_flag=30+i;
                    }
                    //set n'th bit to 1.
                    cell_used_numbers[threadIdx.x] |= 1<<(temp-1);
                }
              }
            }

            }
        __syncthreads();
        if (error_flag==0){
            if (threadIdx.x==0) {
              progress_flag = 0;
              done_flag = 1;
            }
            __syncthreads();
           if (threadIdx.x<81){
                // 2nd for each cell calculate available numbers(row_used OR col_used OR cell_used) and if there is one 0
                //*************************
                current_poss = 0;
                mat_i = threadIdx.x/9;
                mat_j = threadIdx.x%9;
                if (block_memory[threadIdx.x] == 0){
                    done_flag = 0;
                    current_poss = (row_used_numbers[mat_i] | col_used_numbers[mat_j] | cell_used_numbers[(mat_i/3)*3+(mat_j/3)]);
                    //printf("thredix=%d,current_poss=%d\n",threadIdx.x,current_poss);
                    temp = 0; // temp for count
                    for (i=0;i<9;i++){
                      if ((current_poss & (1<<i))==0){
                        if (temp){ //if there is a zero found already
                          temp = 10;
                          break;
                        }
                        else{
                          temp = i+1;
                        }
                      }
                    }
                    if (temp==0){
                        #ifdef DEBUG
                          printf("Block:%d,i=%d,j=%d, cannot be filled. Invalidating\n",blockIdx.x,mat_i+1,mat_j+1);
                        #endif
                        error_flag = 1;
                        progress_flag = 1;
                    }
                    else if (temp<=9){
                      #ifdef DEBUG
                      if (blockIdx.x==0)  printf("i=%d,j=%d,val=%d\n",threadIdx.x/9+1,(threadIdx.x%9)+1,temp);
                      #endif
                      block_memory[threadIdx.x] = temp;
                      progress_flag = 1;
                    }
                }
            }
          }
          __syncthreads();
        }
        /*End of while loop, lets see why the while loop is ended.
          error: we make stats[.]=0 to make the block available for future scheduling.
          done: set the extra stat value to 2.
          no_progress: pick the less ambigous cell and fork. For each extra possibility one of the threads find an available block using atomic operations.
          */
        __syncthreads();
        if (done_flag) {
          if (threadIdx.x==0){
            memcpy(memory+gridDim.x*81,block_memory,81);
            stats[gridDim.x]=2;
          }
        }
        else if (error_flag!=0){
          #ifdef DEBUG
            if (threadIdx.x==0)
              printf("There is an error:%d : with this block %d \n",error_flag,blockIdx.x);
          #endif
          if (threadIdx.x==0)
            stats[blockIdx.x]=0;
        }

        else if (progress_flag==0) {
          // Implement scheduling here
          if (threadIdx.x==0){
            min_forks = 9;
            scheduling_thread = blockDim.x; //bigger then and threadID
          }
          __syncthreads();
          //lets find out what is the minimum number of possible digits for any cell.
          temp = 0; // temp for count
          if(current_poss!=0){
            for (i=0;i<9;i++){
              if ((current_poss & (1<<i))==0){
                  temp++;
                }
            }
            atomicMin(&min_forks,temp);
          }
          __syncthreads();
          //then out of most ambigous cells lets choose the one with smallest threadIdx.
          if (temp==min_forks){
            atomicMin(&scheduling_thread,threadIdx.x);
          }
          __syncthreads();
          if (scheduling_thread==threadIdx.x){
            //Find a suitable block to schedule the fork for each extra value.
            k = 1;
            j=0; //to continue wherever we stayed on the previous search.
            for (i=0;i<9;i++){
              if ((current_poss & (1<<i))==0){
                    if (k==1) {
                      // first possibility stays with the current block
                      block_memory[threadIdx.x] = i+1;
                    }
                    else{
                      // look for suitable block
                      for (;j<gridDim.x;j++) {
                        atomicCAS(stats+j,0,gridDim.x*blockIdx.x+threadIdx.x+2); //unique identifier>1
                        if (stats[j] == (gridDim.x*blockIdx.x+threadIdx.x+2)){
                          //succesful scheduling
                          #ifdef DEBUG
                          if (blockIdx.x<5){
                            printf("Error_flag:%d,current_poss=%d,rowi=%d\n",error_flag,current_poss,row_used_numbers[mat_i]);
                             printf("From-Block:%d,Forkedinto: Block:%d,i=%d,j=%d,new_val=%d,k=%d/%d\n",blockIdx.x,j,mat_i+1,mat_j+1,i+1,k,min_forks);
                          }
                           #endif
                          memcpy(memory+j*81,block_memory,81);
                          memory[j*81+threadIdx.x] = i+1;
                          stats[j] = 1;
                          break;
                        }
                      }
                      if (j == gridDim.x){
                         // #ifdef DEBUG
                        printf("From-Block:%d,i=%d,j=%d,k=%d/%d,c_poss=%d,b_mem=%d,chouldnt schedule\n",blockIdx.x,mat_i+1,mat_j+1,k,min_forks,current_poss,block_memory[threadIdx.x]);
                         // #endif
                      }
                    }
                    k++;
                }
            }
          }
          __syncthreads();
        }
    }
  }


__device__ void printSudokuDev(char *arr){
  int i;
  printf("|++++++++++++++++++++");
  for (i = 0;i<81;i++){
    if (i%27==0) printf("|\n|--------------------");
    if (i%9==0) printf("|\n");
    if (i%3==0) printf("|");
    if (arr[i]){
      printf("%d ",arr[i]);
    }
    else{
      printf(". ");
    }
  }
  printf("|\n");
}

__global__ void controller(char* arr_dev,int* block_stat,int nBlocks, int nThreads){
  int  i = 0;
  while (block_stat[nBlocks]!=2 && i<55){//stats[gridDim.x]==2 means, solution is coppied to the last 81 char of memory.
    fillSudokuSafeAndFork<<<nBlocks,nThreads>>>(arr_dev,block_stat);
    cudaDeviceSynchronize();
    #ifdef DEBUG
      printf("Hey\n");
      for (int j=0;j=nBlocks;j++){
        printf("%d,",block_stat[j]);
      }
      printf("\n");
      printSudokuDev(arr_dev);
    #endif
    i++;
  }
}

void  gpu_sudoku_solver(char* arr,int version)
{
  char *arr_dev;
  int *block_stat;

  int nThreads = 96; // wrap_size 32, each thread will have responsible from one cell.
  int nBlocks = 20000; // max available concurent blocks/searches running.
  int memSize = 81*(nBlocks+1); // 0.81 MB for N=9
  //copy array and create a new one temp. last block/stat is for the result
  cudaMalloc((void**) &block_stat,(nBlocks+1)*sizeof(int));
  cudaMemset(block_stat, 0, (nBlocks+1)*sizeof(int));
  cudaMemset(block_stat, 1, 1);
  if( !block_stat )
  {
   fprintf(stderr, " Cannot allocate block_stat array of size %d on the device\n", (nBlocks+1)*sizeof(int));
   exit(1);
  }

  cudaMalloc((void**) &arr_dev,memSize);
  cudaMemcpy(arr_dev,arr,81,cudaMemcpyHostToDevice);

  if( !arr_dev )
  {
   fprintf(stderr, " Cannot allocate arr_dev of size %d on the device\n", memSize);
   exit(1);
  }

  if (version==1){
    printf("Block=%d,threads=%d starting\n",nBlocks,nThreads);
    controller<<<1,1>>>(arr_dev,block_stat,nBlocks,nThreads);
    cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
          printf("Error: %s\n", cudaGetErrorString(err));
  }
  else{
    printf("Invalid version'\n");
    exit(1);
  }
  cudaMemcpy(arr,arr_dev+81*nBlocks,81,cudaMemcpyDeviceToHost);
  cudaFree(arr_dev);
  cudaFree(block_stat);
}

void readNextSudoku(FILE *fp, char *sudoku){
  int i,j;
  for (i=0;i<9;i++){
    fscanf(fp, "%s", sudoku);
    for (j=0;j<9;j++){
      sudoku[j] = sudoku[j]-48; //48 = char '0'
    }
    sudoku = sudoku + 9;
  }
}

void printSudoku(char *arr){
  int i;
  printf("|++++++++++++++++++++");
  for (i = 0;i<81;i++){
    if (i%27==0) printf("|\n|--------------------");
    if (i%9==0) printf("|\n");
    if (i%3==0) printf("|");
    if (arr[i]){
      printf("%d ",arr[i]);
    }
    else{
      printf(". ");
    }
  }
  printf("|\n");
}

void printSudokuToFile(char *arr,FILE* out_fp){
  for (int i = 0;i<81;i++){
    fprintf(out_fp, "%d",arr[i]);
    if (i%9==8){
      fprintf(out_fp, "\n");
    }
  }
  fprintf(out_fp, "\n");
}

int main(int argc, char *argv[])
{
  cudaFree(0);
  cudaSetDevice ( 1 );
  int version=1,flag=1,i;
  char cur_sudoku[81];
  char out_name[200];
  double time_taken;
  clock_t start, end;
  FILE *inp_fp,*out_fp;

  if (argc <=3 && argc>1  )
    {
      memset(out_name, '\0', sizeof(out_name));
      strcpy(out_name, argv[1]);
      i = 0 ;
      while (out_name[i] != '\0' && out_name[i] != '.'){
        i ++;
      }
      if (out_name[i] == '\0'){
        fprintf(stderr, "The input file should be like X.in: %s!\n",out_name[i]);
        exit(1);
      }
      else{
        out_name[i+1] = 's';
        out_name[i+2] = 'o';
        out_name[i+3] = 'l';
        out_name[i+4] = '\0';
      }
      out_fp = fopen(out_name,"w");
      if (out_fp == NULL) {
        fprintf(stderr, "Can't open output file %s!\n",out_name);
        exit(1);
      }

      inp_fp = fopen(argv[1],"r");
      if (inp_fp == NULL) {
        fprintf(stderr, "Can't open input file %s!\n",argv[1]);
        exit(1);
      }
    }
  else
    {
      printf("Usage 'sudokusolver filename [version_no=1]'\n");
      exit(1);
    }

  if (argc == 3 ){
    version = (int) strtol(argv[2], (char **)NULL, 10);
    printf("Version = %d\n", version);
  }
    while (flag!=-1){
      readNextSudoku(inp_fp,cur_sudoku);
      printSudoku(cur_sudoku);
      start = clock();
      gpu_sudoku_solver(cur_sudoku,version);
      end = clock();
      time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
      printf("Time taken for the function_call is %lf\n", time_taken);
      printSudoku(cur_sudoku);
      printf("-----------------------------------------\n");
      printSudokuToFile(cur_sudoku,out_fp);
      fgetc( inp_fp );
      flag=fgetc( inp_fp );
    }
    fclose(inp_fp);
    fclose(out_fp);
}
