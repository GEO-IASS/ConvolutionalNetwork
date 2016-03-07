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
void Convolution(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], mxClassID classID)
{
    if (mxGetClassID(mxGetField(prhs[0],0,"stride")) != classID)
        mexErrMsgTxt("stride must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"bias")) != classID)
        mexErrMsgTxt("bias must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"MapSize")) != classID)
        mexErrMsgTxt("MapSize must be the same real, single/double type");
    if (mxGetClassID(mxGetField(prhs[0],0,"kernel")) != classID)
        mexErrMsgTxt("kernel must be the same real, single/double type");
            
    T *stride = (T *)mxGetData(mxGetField(prhs[0],0,"stride"));
    T *bias = (T *)mxGetData(mxGetField(prhs[0],0,"bias"));
    T *MapSize = (T *)mxGetData(mxGetField(prhs[0],0,"MapSize"));
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
    
    const mxArray *ptr_data = prhs[1];
    T *data_point = (T *)mxGetData(ptr_data);
    const mwSize *tmp = mxGetDimensions(ptr_data);
    mwSize DATA_DIMS[4];
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
// OutputData = zeros((filter.MapSize(1,1) - 1) * filter.stride(1,1) + 1, ...
//     (filter.MapSize(1,2) - 1) * filter.stride(1,2) + 1,size(filter.kernel,4),size(InputData,4));   
    
    mwSize OUTPUT_DIMS[4];
    OUTPUT_DIMS[0] = mwSize(int(MapSize[0]));
    OUTPUT_DIMS[1] = mwSize(int(MapSize[1]));
    OUTPUT_DIMS[2] = KERNEL_DIMS[3];
    OUTPUT_DIMS[3] = DATA_DIMS[3];
    
    mwSize data_sample_size = DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2];
    mwSize data_channel_size = DATA_DIMS[0] * DATA_DIMS[1];
    
    mwSize kernel_sample_size = KERNEL_DIMS[0] * KERNEL_DIMS[1] * KERNEL_DIMS[2];
    mwSize kernel_channel_size = KERNEL_DIMS[0] * KERNEL_DIMS[1];
    
    mwSize output_sample_size = OUTPUT_DIMS[0] * OUTPUT_DIMS[1] * OUTPUT_DIMS[2];
    mwSize output_channel_size = OUTPUT_DIMS[0] * OUTPUT_DIMS[1];
    
    plhs[0] = mxCreateNumericArray(4,OUTPUT_DIMS,classID,mxREAL);
    mxArray *OutputData = plhs[0];
    
    T *Output_point = (T *)mxGetData(OutputData);
    for(mwSize i = 0;i < OUTPUT_DIMS[3];i ++){
        #ifdef OPENMP
        #pragma omp parallel for
        #endif
        for(mwSize j = 0;j < OUTPUT_DIMS[2];j ++){
            for(mwSize k = 0;k < OUTPUT_DIMS[1];k ++){
                for(mwSize l = 0;l < OUTPUT_DIMS[0];l ++){
                    T element = 0;
                    for(mwSize m = k * stride[1];m < std::min(int(k * stride[1] + KERNEL_DIMS[1]),int(DATA_DIMS[1]));m ++){
                        for(mwSize n = l * stride[0];n < std::min(int(l * stride[0] + KERNEL_DIMS[0]),int(DATA_DIMS[0]));n ++){
                            for(mwSize h = 0;h < DATA_DIMS[2];h ++){
                                element = element + data_point[i * data_sample_size + h * data_channel_size + m * DATA_DIMS[0] + n] * \
                                        kernel_point[int(j * kernel_sample_size + h * kernel_channel_size + kernel_channel_size - 1 - \
                                        ((m - k * stride[1]) * KERNEL_DIMS[0] + (n - l * stride[0])))];
                            }
                        }
                    }
                    Output_point[i * output_sample_size + j * output_channel_size + k * OUTPUT_DIMS[0] + l] = element + bias[j];
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
      mexWarnMsgTxt("Convolution operation!");
      
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

      Convolution<float>(nlhs, plhs, nrhs, prhs, classID);
  }  else if (classID == mxDOUBLE_CLASS) {
      if (debug)
        mexWarnMsgTxt("Executing the double version\n");

      Convolution<double>(nlhs, plhs, nrhs, prhs, classID);
  }
}