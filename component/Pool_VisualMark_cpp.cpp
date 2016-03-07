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
void Pool_VisualMark(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], mxClassID classID)
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
    
    mwSize MARK_DIMS[6];    
    const mxArray *ptr_Mark = prhs[2];
    T *mark_point = (T *)mxGetData(prhs[2]);
    const mwSize *tmp_1 = mxGetDimensions(ptr_Mark);
    MARK_DIMS[0] = tmp_1[0];
    MARK_DIMS[1] = tmp_1[1];
    if (mxGetNumberOfDimensions(ptr_Mark) == 2) {
      MARK_DIMS[2] = 1;
      MARK_DIMS[3] = 1;
      MARK_DIMS[4] = 1;
      MARK_DIMS[5] = 1;
    } else if (mxGetNumberOfDimensions(ptr_Mark) == 3) {
      MARK_DIMS[2] = tmp_1[2];
      MARK_DIMS[3] = 1;
      MARK_DIMS[4] = 1;
      MARK_DIMS[5] = 1;
    } else if (mxGetNumberOfDimensions(ptr_Mark) == 4) {
      MARK_DIMS[2] = tmp_1[2];
      MARK_DIMS[3] = tmp_1[3];
      MARK_DIMS[4] = 1;
      MARK_DIMS[5] = 1;
    } else if (mxGetNumberOfDimensions(ptr_Mark) == 5) {
      MARK_DIMS[2] = tmp_1[2];
      MARK_DIMS[3] = tmp_1[3];
      MARK_DIMS[4] = tmp_1[4];
      MARK_DIMS[5] = 1;
    } else {
      MARK_DIMS[2] = tmp_1[2];
      MARK_DIMS[3] = tmp_1[3];
      MARK_DIMS[4] = tmp_1[4];
      MARK_DIMS[5] = tmp_1[5];
    }
    
    M_DIMS[0] = mwSize(int(MapSize[0]));
    M_DIMS[1] = mwSize(int(MapSize[1]));
    M_DIMS[2] = mwSize(int(MapSize[2]));
    M_DIMS[3] = DATA_DIMS[3];
    
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
    for(mwSize i = 0;i < M_DIMS[3];i ++){
        #ifdef OPENMP
        #pragma omp parallel for
        #endif
        for(mwSize j = 0;j < M_DIMS[2];j ++){
            for(mwSize k = 0;k < M_DIMS[1];k ++){
                for(mwSize l = 0;l < M_DIMS[0];l ++){
                    Output_point[i * output_sample_size + j * output_channel_size + k * M_DIMS[0] + l] = data_point[int(
                            i * data_sample_size + j * data_channel_size + (k * stride[1] + mark_point[i * mark_sample_size + \
                            j * mark_channel_size + k * mark_col_size + l * mark_row_size + 1]) * DATA_DIMS[0] + (l * stride[0] + \
                            mark_point[i * mark_sample_size + j * mark_channel_size + k * mark_col_size + l * mark_row_size]))];
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
      mexWarnMsgTxt("Pool_VisualMark operation!");
  
  if (nrhs !=3)
    mexErrMsgTxt("Must have 3 input arguments");

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

      Pool_VisualMark<float>(nlhs, plhs, nrhs, prhs, classID);
  }  else if (classID == mxDOUBLE_CLASS) {
      if (debug)
        mexWarnMsgTxt("Executing the double version\n");

      Pool_VisualMark<double>(nlhs, plhs, nrhs, prhs, classID);
  }
}