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
void DeConvolution(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], mxClassID classID)
{
    if (mxGetClassID(mxGetField(prhs[0],0,"stride")) != classID)
        mexErrMsgTxt("stride must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"kernel")) != classID)
        mexErrMsgTxt("kernel must be the same real, single/double type");
            
    T *stride = (T *)mxGetData(mxGetField(prhs[0],0,"stride"));
    mxArray *kernel = mxGetField(prhs[0],0,"kernel");
    T *kernel_point = (T *)mxGetData(kernel);
    const mwSize *tmp_1 = mxGetDimensions(kernel);
    mwSize KERNEL_DIMS[4];
    KERNEL_DIMS[0] = tmp_1[0];
    KERNEL_DIMS[1] = tmp_1[1];
    if (mxGetNumberOfDimensions(kernel) == 2) {
      KERNEL_DIMS[2] = 1;
      KERNEL_DIMS[3] = 1;
    } else if (mxGetNumberOfDimensions(kernel) == 3) {
      KERNEL_DIMS[2] = tmp_1[2];
      KERNEL_DIMS[3] = 1;
    } else {
      KERNEL_DIMS[2] = tmp_1[2];
      KERNEL_DIMS[3] = tmp_1[3];
    }    
    
    const mxArray *ptr_sensitivity = prhs[1];
    T *sensitivity_point = (T *)mxGetData(ptr_sensitivity);
    const mwSize *tmp = mxGetDimensions(ptr_sensitivity);
    
    mwSize SENSI_DIMS[4];
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
    
    mwSize OUTPUT_DIMS[4];
    OUTPUT_DIMS[0] = stride[0] * (SENSI_DIMS[0] - 1) + KERNEL_DIMS[0];
    OUTPUT_DIMS[1] = stride[1] * (SENSI_DIMS[1] - 1) + KERNEL_DIMS[1];
    OUTPUT_DIMS[2] = KERNEL_DIMS[2];
    OUTPUT_DIMS[3] = SENSI_DIMS[3];
    
    mwSize sensi_sample_size = SENSI_DIMS[0] * SENSI_DIMS[1] * SENSI_DIMS[2];
    mwSize sensi_channel_size = SENSI_DIMS[0] * SENSI_DIMS[1];
    
    mwSize kernel_sample_size = KERNEL_DIMS[0] * KERNEL_DIMS[1] * KERNEL_DIMS[2];
    mwSize kernel_channel_size = KERNEL_DIMS[0] * KERNEL_DIMS[1];
    
    mwSize output_sample_size = OUTPUT_DIMS[0] * OUTPUT_DIMS[1] * OUTPUT_DIMS[2];
    mwSize output_channel_size = OUTPUT_DIMS[0] * OUTPUT_DIMS[1];
    
    plhs[0] = mxCreateNumericArray(4,OUTPUT_DIMS,classID,mxREAL);
    mxArray *OutputData = plhs[0];
    T *Output_point = (T *)mxGetData(OutputData);
    
    mwSize UP_SENSI_DIMS[2];
    UP_SENSI_DIMS[0] = (SENSI_DIMS[0] - 1) * stride[0] + 2 * KERNEL_DIMS[0] - 1;
    UP_SENSI_DIMS[1] = (SENSI_DIMS[1] - 1) * stride[1] + 2 * KERNEL_DIMS[1] - 1;
    
    mxArray *up_sensitivity = mxCreateNumericArray(2,UP_SENSI_DIMS,classID,mxREAL);
    T *up_sensi_point = (T *)mxGetData(up_sensitivity);
    
    for(mwSize j = 0;j < OUTPUT_DIMS[3];j ++){
        #ifdef OPENMP
        #pragma omp parallel for
        #endif
        for(mwSize m = 0;m < KERNEL_DIMS[3];m ++){
            //amplify
            for(mwSize k = 0;k < SENSI_DIMS[1];k ++){
                for(mwSize l = 0;l < SENSI_DIMS[0];l ++){
                    up_sensi_point[int(( k * stride[1] + KERNEL_DIMS[1] - 1) * UP_SENSI_DIMS[0] + \
                            l * stride[0] + KERNEL_DIMS[0] - 1)] = sensitivity_point[int(j * sensi_sample_size + m * sensi_channel_size \
                            + k * SENSI_DIMS[0] + l)];
                }
            }            
            for(mwSize n = 0;n < OUTPUT_DIMS[2];n ++){
                for(mwSize k = 0;k < OUTPUT_DIMS[1];k ++){
                    for(mwSize l = 0;l < OUTPUT_DIMS[0];l ++){
                        //sum up 
                        T element = 0;
                        for(mwSize h = k;h < std::min(int(k + KERNEL_DIMS[1]),int(UP_SENSI_DIMS[1]));h ++){
                            for(mwSize g = l;g < std::min(int(l + KERNEL_DIMS[0]),int(UP_SENSI_DIMS[0]));g ++){
                                element = element + up_sensi_point[int(h * UP_SENSI_DIMS[0] + g)] * \
                                        kernel_point[int(m * kernel_sample_size + n * kernel_channel_size + (h - k) * KERNEL_DIMS[0] + (g - l))];
                            }
                        }
                        Output_point[j * output_sample_size + n * output_channel_size + k * OUTPUT_DIMS[0] + l] = element + \
                                Output_point[j * output_sample_size + n * output_channel_size + k * OUTPUT_DIMS[0] + l];
                    }
                }              
            }
        }
    }
    mxDestroyArray(up_sensitivity);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /***************************************************************************/
  /** Check input ************************************************************/
  /***************************************************************************/
  if(debug)
      mexWarnMsgTxt("DeConvolution operation!");
  
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

      DeConvolution<float>(nlhs, plhs, nrhs, prhs, classID);
  }  else if (classID == mxDOUBLE_CLASS) {
      if (debug)
        mexWarnMsgTxt("Executing the double version\n");

      DeConvolution<double>(nlhs, plhs, nrhs, prhs, classID);
  }
}