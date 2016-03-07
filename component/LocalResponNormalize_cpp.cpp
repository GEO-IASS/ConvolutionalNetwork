#include <algorithm>
#include <math.h>
#include "mex.h"
#include "matrix.h"

#include <limits>
#define OPENMP
#ifdef OPENMP
#include <omp.h>
#endif

mwSize debug = 0;


template <typename T>
void LocalResponNormalize(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], mxClassID classID)
{
    mwSize DATA_DIMS[4];
    mwSize M_DIMS[4];
    mwSize kernel_size[4];
    
    if (mxGetClassID(mxGetField(prhs[0],0,"k")) != classID)
        mexErrMsgTxt("k must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"alpha")) != classID)
        mexErrMsgTxt("alpha must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"n")) != classID)
        mexErrMsgTxt("n must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"beta")) != classID)
        mexErrMsgTxt("beta must be the same real, single/double type");
            
    T *K = (T *)mxGetData(mxGetField(prhs[0],0,"k"));
    T *alpha = (T *)mxGetData(mxGetField(prhs[0],0,"alpha"));
    T *n = (T *)mxGetData(mxGetField(prhs[0],0,"n"));
    T *beta = (T *)mxGetData(mxGetField(prhs[0],0,"beta"));
    
    const mxArray *ptr_data = prhs[1];
    T *data_point = (T *)mxGetData(ptr_data);
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
    
    mwSize data_sample_size = DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2];
    mwSize data_channel_size = DATA_DIMS[0] * DATA_DIMS[1];
    
    plhs[0] = mxCreateNumericArray(4,DATA_DIMS,classID,mxREAL);
    mxArray *OutputData = plhs[0];
    
    T *Output_point = (T *)mxGetData(OutputData);
    for(mwSize i = 0;i < DATA_DIMS[3];i ++){
        #ifdef OPENMP
        #pragma omp parallel for
        #endif
        for(mwSize k = 0;k < DATA_DIMS[1];k ++){
            for(mwSize l = 0;l < DATA_DIMS[0];l ++){
                for(mwSize j = 0;j < DATA_DIMS[2];j ++){
                    T sum_part = 0;
                    for(mwSize h = std::max(0,int(j - int(n[0]) / 2));h < std::min(int(DATA_DIMS[2]),int(j + int(n[0]) / 2 + 1));h ++){
                        sum_part = sum_part + pow(data_point[i * data_sample_size + h * data_channel_size + \
                                k * DATA_DIMS[0] + l],2);
                    }
                    sum_part = pow(K[0] + alpha[0] * sum_part,beta[0]);
                    Output_point[i * data_sample_size + j * data_channel_size + k * DATA_DIMS[0] + l] =  data_point[i * data_sample_size + \
                            j * data_channel_size + k * DATA_DIMS[0] + l] / sum_part;
                }
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /***************************************************************************/
  /** Check input ************************************************************/
  /***************************************************************************/
  if(debug)
      mexWarnMsgTxt("LocalResponNormalize operation!");
  
  if (nrhs !=2)
    mexErrMsgTxt("Must have 2 input arguments");

  if (nlhs !=1)
    mexErrMsgTxt("Must have 1 output arguments");

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

      LocalResponNormalize<float>(nlhs, plhs, nrhs, prhs, classID);
  }  else if (classID == mxDOUBLE_CLASS) {
      if (debug)
        mexWarnMsgTxt("Executing the double version\n");

      LocalResponNormalize<double>(nlhs, plhs, nrhs, prhs, classID);
  }
}