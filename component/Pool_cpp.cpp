#include <string>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "mex.h"
#include "matrix.h"
#include "string.h"
#include <stdio.h>
#include <stdlib.h>

#include <limits>
#define OPENMP
#ifdef OPENMP
#include <omp.h>
#endif

mwSize debug = 0;

template <typename T>
void Pool(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], mxClassID classID)
{
    mwSize DATA_DIMS[4];
    mwSize M_DIMS[4];
    
    if (mxGetClassID(mxGetField(prhs[0],0,"size")) != classID)
        mexErrMsgTxt("size must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"stride")) != classID)
        mexErrMsgTxt("stride must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"MapSize")) != classID)
        mexErrMsgTxt("MapSize must be the same real, single/double type");
            
    T *size = (T *)mxGetData(mxGetField(prhs[0],0,"size"));
    T *stride = (T *)mxGetData(mxGetField(prhs[0],0,"stride"));
    T *MapSize = (T *)mxGetData(mxGetField(prhs[0],0,"MapSize"));
    std::string type = mxArrayToString(mxGetField(prhs[0],0,"type"));
    
    const mxArray *ptr_data = prhs[1];
    T *data_point = (T *)mxGetData(prhs[1]);
    const mwSize *tmp = mxGetDimensions(ptr_data);
    DATA_DIMS[0] = tmp[0];
    DATA_DIMS[1] = tmp[1];
    if (mxGetNumberOfDimensions(ptr_data) == 2) {
      DATA_DIMS[2] = 1;
      DATA_DIMS[3] = 1;
    } else if (mxGetNumberOfDimensions(ptr_data) == 3) {
      DATA_DIMS[2] = tmp[2];
      DATA_DIMS[3] = 1;
    } else {
      DATA_DIMS[2] = tmp[2];
      DATA_DIMS[3] = tmp[3];
    }    
    
    M_DIMS[0] = mwSize(int(MapSize[0]));//mwSize(ceil((float(DATA_DIMS[0]) - (size[0])) / (stride[0]))) + 1;
    M_DIMS[1] = mwSize(int(MapSize[1]));//mwSize(ceil((float(DATA_DIMS[1]) - (size[1])) / (stride[1]))) + 1;
    M_DIMS[2] = mwSize(int(MapSize[2]));//DATA_DIMS[2];
    M_DIMS[3] = DATA_DIMS[3];
    
    mwSize MARK_DIMS[6];
    MARK_DIMS[0] = 1;
    MARK_DIMS[1] = 2;
    MARK_DIMS[2] = M_DIMS[0];
    MARK_DIMS[3] = M_DIMS[1];
    MARK_DIMS[4] = DATA_DIMS[2];
    MARK_DIMS[5] = DATA_DIMS[3];
    
    mwSize data_sample_size = DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2];
    mwSize data_channel_size = DATA_DIMS[0] * DATA_DIMS[1];
    
    mwSize mark_sample_size = MARK_DIMS[4] * MARK_DIMS[3] * MARK_DIMS[2] * MARK_DIMS[1] * MARK_DIMS[0];
    mwSize mark_channel_size = MARK_DIMS[3] * MARK_DIMS[2] * MARK_DIMS[1] * MARK_DIMS[0];
    mwSize mark_col_size = MARK_DIMS[2] * MARK_DIMS[1] * MARK_DIMS[0];
    mwSize mark_row_size = MARK_DIMS[1] * MARK_DIMS[0];
    
    mwSize output_sample_size = M_DIMS[0] * M_DIMS[1] * M_DIMS[2];
    mwSize output_channel_size = M_DIMS[0] * M_DIMS[1];
    
    plhs[0] = mxCreateNumericArray(4,M_DIMS,classID,mxREAL);
    mxArray *OutputData = plhs[0];
    T *Output_point = (T *)mxGetData(OutputData);
    
    
    if(type.compare("max") == 0){
        plhs[1] = mxCreateNumericArray(6,MARK_DIMS,classID,mxREAL);
        mxArray *mark = plhs[1];
        T *mark_point = (T *)mxGetData(mark);
        
        for(mwSize i = 0;i < M_DIMS[3];i ++){
        #ifdef OPENMP
        #pragma omp parallel for
        #endif
            for(mwSize j = 0;j < M_DIMS[2];j ++){
                for(mwSize k = 0;k < M_DIMS[1];k ++){
                    for(mwSize l = 0;l < M_DIMS[0];l ++){
//                         if(debug){
//                             mexPrintf("k = %d,l = %d\n",k,l);
//                         }
                        T local_max = - std::numeric_limits<T>::max();;
                        mwSize local_row;
                        mwSize local_col;
                        for(mwSize m = k * stride[1];m < std::min(int(k * stride[1] + size[1]),int(DATA_DIMS[1]));m ++){
                            for(mwSize n = l * stride[0];n < std::min(int(l * stride[0] + size[0]),int(DATA_DIMS[0]));n ++){
                                T data = data_point[i * data_sample_size + j * data_channel_size + m * DATA_DIMS[0] + n];
                                if(local_max < data){
                                    local_max = data;
                                    local_col = m - k * stride[1];//
                                    local_row = n - l * stride[0];//
                                }
//                                 if(debug){
//                                     mexPrintf("%f ",data_point[i * data_sample_size + j * data_channel_size + m * DATA_DIMS[0] + n]);
//                                 }
                            }
//                             if(debug){
//                                 mexPrintf("\n");
//                             }
                        }
                        Output_point[i * output_sample_size + j * output_channel_size + k * M_DIMS[0] + l] = local_max;
                        mark_point[i * mark_sample_size + j * mark_channel_size + k * mark_col_size + l * mark_row_size] = local_row;
                        mark_point[i * mark_sample_size + j * mark_channel_size + k * mark_col_size + l * mark_row_size + 1] = local_col;
//                         if(debug){
//                             mexPrintf("local_max = %f,local_col = %d,local_row = %d\n\n",local_max,local_col,local_row);
//                         }
                    }
                }
            }
        }
    }
    else if(type.compare("average") == 0)
    {        
        for(mwSize i = 0;i < M_DIMS[3];i ++){
        #ifdef OPENMP
        #pragma omp parallel for
        #endif
            for(mwSize j = 0;j < M_DIMS[2];j ++){
                for(mwSize k = 0;k < M_DIMS[1];k ++){
                    for(mwSize l = 0;l < M_DIMS[0];l ++){
                        T local_sum = 0;
                        for(mwSize m = k * stride[1];m < std::min(int(k * stride[1] + size[1]),int(DATA_DIMS[1]));m ++){
                            for(mwSize n = l * stride[0];n < std::min(int(l * stride[0] + size[0]),int(DATA_DIMS[0]));n ++){
                                local_sum = local_sum + data_point[i * data_sample_size + j * data_channel_size + m * DATA_DIMS[0] + n];
                            }
                        }
                        Output_point[i * output_sample_size + j * output_channel_size + k * M_DIMS[0] + l] = local_sum / (size[0] * size[1]);
                    }
                }
            }
        }
    }
    else if(type.compare("abs_max") == 0)
    {
        mexErrMsgTxt("the pooling operation ""abs_max"" is not implemented yet!");
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /***************************************************************************/
  /** Check input ************************************************************/
  /***************************************************************************/
  if(debug)
      mexWarnMsgTxt("Pool operation!");
  
  if (nrhs !=2)
    mexErrMsgTxt("Must have 2 input arguments");

  if (nlhs !=2 && nlhs !=1)
    mexErrMsgTxt("Must have 1/2 output arguments");

//   if (mxIsComplex(prhs[0]) || !(mxIsClass(prhs[0],"single") || mxIsClass(prhs[0],"double")))
//     mexErrMsgTxt("Input data must be real, single/double type");

  if (mxIsComplex(prhs[1]) || !(mxIsClass(prhs[1],"single") || mxIsClass(prhs[1],"double")))
    mexErrMsgTxt("dimensions (rows, cols) must be real, single/double type");

  if (mxGetNumberOfDimensions(prhs[1]) < 2)
    mexErrMsgTxt("Input data must have at least 2-dimensions (rows, cols, nchannels, nsamples) "
        "\nThe last two dimensions will be considered to be 1.");

//   if (mxGetNumberOfDimensions(prhs[1]) != 2)
//     mexErrMsgTxt("Pooling data must have 2-dimensions (prows, pcols)");

  mxClassID classID = mxGetClassID(prhs[1]);

  /** This is mainly to avoid two typenames. Should not be a big usability issue. */
//   if (mxGetClassID(prhs[1]) != classID)
//     mexErrMsgTxt("Input data and pooling need to be of the same type");

  /***************************************************************************/
  /** Switch for the supported data types */
  /***************************************************************************/
  if (classID == mxSINGLE_CLASS) {
      if (debug)
        mexWarnMsgTxt("Executing the single version\n");

      Pool<float>(nlhs, plhs, nrhs, prhs, classID);
  }  else if (classID == mxDOUBLE_CLASS) {
      if (debug)
        mexWarnMsgTxt("Executing the double version\n");

      Pool<double>(nlhs, plhs, nrhs, prhs, classID);
  }
}
