#include <string>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "mex.h"
#include "matrix.h"
#include "string.h"

#include <limits>
#define OPENMP
#ifdef OPENMP
#include <omp.h>
#endif

mwSize debug = 0;


template <typename T>
void DePool(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], mxClassID classID)
{
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
    
    mwSize SENSI_DIMS[4];
    const mxArray *ptr_sensitivity = prhs[1];
    T *sensitivity_point = (T *)mxGetData(ptr_sensitivity);
    const mwSize *tmp = mxGetDimensions(ptr_sensitivity);
    SENSI_DIMS[0] = tmp[0];
    SENSI_DIMS[1] = tmp[1];
    if (mxGetNumberOfDimensions(ptr_sensitivity) == 2) {
      SENSI_DIMS[2] = 1;
      SENSI_DIMS[3] = 1;
    } else if (mxGetNumberOfDimensions(ptr_sensitivity) == 3) {
      SENSI_DIMS[2] = tmp[2];
      SENSI_DIMS[3] = 1;
    } else {
      SENSI_DIMS[2] = tmp[2];
      SENSI_DIMS[3] = tmp[3];
    }    
    mwSize sensi_sample_size = SENSI_DIMS[0] * SENSI_DIMS[1] * SENSI_DIMS[2];
    mwSize sensi_channel_size = SENSI_DIMS[0] * SENSI_DIMS[1];
    
    mwSize OUPUT_DIMS[4];
    OUPUT_DIMS[0] = stride[0] * (MapSize[0] - 1) + size[0];
    OUPUT_DIMS[1] = stride[1] * (MapSize[1] - 1) + size[1];
    OUPUT_DIMS[2] = SENSI_DIMS[2];
    OUPUT_DIMS[3] = SENSI_DIMS[3];
    plhs[0] = mxCreateNumericArray(4,OUPUT_DIMS,classID,mxREAL);
    mxArray *OutputData = plhs[0];
    T *Output_point = (T *)mxGetData(OutputData);
    mwSize output_sample_size = OUPUT_DIMS[0] * OUPUT_DIMS[1] * OUPUT_DIMS[2];
    mwSize output_channel_size = OUPUT_DIMS[0] * OUPUT_DIMS[1];
    
    
    if(type.compare("max") == 0){
        mwSize MARK_DIMS[6];
        const mxArray *ptr_mark = prhs[2];
        T *mark_point = (T *)mxGetData(ptr_mark);
        const mwSize *tmp_1 = mxGetDimensions(ptr_mark);
        MARK_DIMS[0] = tmp_1[0];
        MARK_DIMS[1] = tmp_1[1];
        MARK_DIMS[2] = tmp_1[2];
        MARK_DIMS[3] = tmp_1[3];
        if (mxGetNumberOfDimensions(ptr_mark) == 4) {
          MARK_DIMS[4] = 1;
          MARK_DIMS[5] = 1;
        } else if (mxGetNumberOfDimensions(ptr_mark) == 5) {
          MARK_DIMS[4] = tmp_1[4];
          MARK_DIMS[5] = 1;
        } else {
          MARK_DIMS[4] = tmp_1[4];
          MARK_DIMS[5] = tmp_1[5];
        }    
        mwSize mark_sample_size = MARK_DIMS[4] * MARK_DIMS[3] * MARK_DIMS[2] * MARK_DIMS[1] * MARK_DIMS[0];
        mwSize mark_channel_size = MARK_DIMS[3] * MARK_DIMS[2] * MARK_DIMS[1] * MARK_DIMS[0];
        mwSize mark_col_size = MARK_DIMS[2] * MARK_DIMS[1] * MARK_DIMS[0];
        mwSize mark_row_size = MARK_DIMS[1] * MARK_DIMS[0];
        
        for(mwSize i = 0;i < SENSI_DIMS[3];i ++){
        #ifdef OPENMP
        #pragma omp parallel for
        #endif
            for(mwSize j = 0;j < SENSI_DIMS[2];j ++){
                for(mwSize k = 0;k < SENSI_DIMS[1];k ++){
                    for(mwSize l = 0;l < SENSI_DIMS[0];l ++){
                        Output_point[i * output_sample_size + j * output_channel_size + mwSize(mark_point[i * mark_sample_size + j * mark_channel_size + \
                                k * mark_col_size + l * mark_row_size + 1] + k * stride[1]) * OUPUT_DIMS[0] + mwSize(mark_point[i * mark_sample_size + j * mark_channel_size + \
                                k * mark_col_size + l * mark_row_size] + l * stride[0])] = \
                                Output_point[i * output_sample_size + j * output_channel_size + mwSize(mark_point[i * mark_sample_size + j * mark_channel_size + \
                                k * mark_col_size + l * mark_row_size + 1] + k * stride[1]) * OUPUT_DIMS[0] + mwSize(mark_point[i * mark_sample_size + j * mark_channel_size + \
                                k * mark_col_size + l * mark_row_size] + l * stride[0])] + sensitivity_point[i * sensi_sample_size + j * sensi_channel_size + \
                                k * SENSI_DIMS[0] + l];
                    }
                }
            }
        }
    }
    else if(type.compare("average") == 0){
        for(mwSize i = 0;i < SENSI_DIMS[3];i ++){
        #ifdef OPENMP
        #pragma omp parallel for
        #endif
            for(mwSize j = 0;j < SENSI_DIMS[2];j ++){
                for(mwSize k = 0;k < SENSI_DIMS[1];k ++){
                    for(mwSize l = 0;l < SENSI_DIMS[0];l ++){
                        for(mwSize m = k * stride[1];m < std::min(int(k * stride[1] + size[1]),int(OUPUT_DIMS[1]));m ++){
                            for(mwSize n = l * stride[0];n < std::min(int(l * stride[0] + size[0]),int(OUPUT_DIMS[0]));n ++){
                                Output_point[i * output_sample_size + j * output_channel_size + m * OUPUT_DIMS[0] + n] = \
                                        Output_point[i * output_sample_size + j * output_channel_size + m * OUPUT_DIMS[0] + n] + \
                                        sensitivity_point[i * sensi_sample_size + j * sensi_channel_size + k * SENSI_DIMS[0] + l] / (size[0] * size[1]);
                            }
                        }
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
      mexWarnMsgTxt("DePool operation!");
  
  if (nrhs !=3 && nrhs !=2)
    mexErrMsgTxt("Must have 2/3 input arguments");

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

      DePool<float>(nlhs, plhs, nrhs, prhs, classID);
  }  else if (classID == mxDOUBLE_CLASS) {
      if (debug)
        mexWarnMsgTxt("Executing the double version\n");

      DePool<double>(nlhs, plhs, nrhs, prhs, classID);
  }
}