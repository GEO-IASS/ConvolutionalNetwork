#include <string>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "mex.h"
#include "matrix.h"
#include "string.h"

mwSize debug = ;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize DATA_DIMS[4];
    mwSize M_DIMS[4];
    
    double *size = mxGetPr(mxGetField(prhs[0],0,"size"));
    double *stride = mxGetPr(mxGetField(prhs[0],0,"stride"));
//     std::string type = mxArrayToString(mxGetField(prhs[0],0,"type"));
    
    const mxArray *ptr_data = prhs[1];
    double *data_point = mxGetPr(prhs[1]);
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
    
    M_DIMS[0] = mwSize(ceil((float(DATA_DIMS[0]) - (size[0])) / (stride[0]))) + 1;
    M_DIMS[1] = mwSize(ceil((float(DATA_DIMS[1]) - (size[1])) / (stride[1]))) + 1;
    if((DATA_DIMS[0] != ((M_DIMS[0] - 1) * stride[0] + size[0])) || (DATA_DIMS[1] != ((M_DIMS[1] - 1) * stride[1] + size[1])))
        mexErrMsgTxt("Convolution Operation error ocurred : Dimension dismatch");
    M_DIMS[2] = DATA_DIMS[2];
    M_DIMS[3] = DATA_DIMS[3];
//     mwSize mark_dims[6];
//     mark_dims[0] = size[0];
//     mark_dims[1] = size[1];
//     mark_dims[2] = M_DIMS[0];
//     mark_dims[3] = M_DIMS[1];
//     mark_dims[4] = DATA_DIMS[2];
//     mark_dims[5] = DATA_DIMS[3];
    
    if(debug)
    {
        mexPrintf("DATA_DIMS[0] = %d\n",DATA_DIMS[0]);
        mexPrintf("DATA_DIMS[1] = %d\n",DATA_DIMS[1]);
        mexPrintf("DATA_DIMS[2] = %d\n",DATA_DIMS[2]);
        mexPrintf("DATA_DIMS[3] = %d\n",DATA_DIMS[3]);
        mexPrintf("size[0] = %d\n",mwSize(size[0]));
        mexPrintf("size[1] = %d\n",mwSize(size[1]));
        mexPrintf("stride[0] = %d\n",mwSize(stride[0]));
        mexPrintf("stride[1] = %d\n",mwSize(stride[1]));
//         for(mwSize i = 0;i < 6;++i)
//         {
//             mexPrintf("%d\n",mark_dims[i]);
//         }
    }
    
    mwSize data_sample_size = DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2];
    mwSize data_channel_size = DATA_DIMS[0] * DATA_DIMS[1];
    
    mwSize mark_sample_size = mark_dims[4] * mark_dims[3] * mark_dims[2] * mark_dims[1] * mark_dims[0];
    mwSize mark_channel_size = mark_dims[3] * mark_dims[2] * mark_dims[1] * mark_dims[0];
    mwSize mark_col_size = mark_dims[2] * mark_dims[1] * mark_dims[0];
    mwSize mark_row_size = mark_dims[1] * mark_dims[0];
    
    mwSize output_sample_size = M_DIMS[0] * M_DIMS[1] * M_DIMS[2];
    mwSize output_channel_size = M_DIMS[0] * M_DIMS[1];
    
    if(debug){
        mexPrintf("data_sample_size = %d   data_channel_size = %d\n",data_sample_size,data_channel_size);
        mexPrintf("mark_sample_size = %d   mark_channel_size = %d\n",mark_sample_size,mark_channel_size);
        mexPrintf("mark_col_size = %d   mark_row_size = %d\n",mark_col_size,mark_row_size);
        mexPrintf("output_sample_size = %d   output_channel_size = %d\n",output_sample_size,output_channel_size);
    }
    plhs[0] = mxCreateNumericArray(4,M_DIMS,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericArray(6,mark_dims,mxDOUBLE_CLASS,mxREAL);
    mxArray *OutputData = plhs[0];
    double *Output_point = mxGetPr(OutputData);
    mxArray *mark = plhs[1];
    double *mark_point = mxGetPr(mark);
    
    if(debug){
        mexPrintf("type.compare(""max"") = %d\n",type.compare("max") != 0);
    }
    
    if(type.compare("max") == 0){
        for(mwSize i = 0;i < M_DIMS[3];i ++){
            for(mwSize j = 0;j < M_DIMS[2];j ++){
                for(mwSize k = 0;k < M_DIMS[1];k ++){
                    for(mwSize l = 0;l < M_DIMS[0];l ++){
                        if(debug){
                            mexPrintf("k = %d,l = %d\n",k,l);
                        }
                        double local_max = - DBL_MAX;
                        mwSize local_row;
                        mwSize local_col;
                        for(mwSize m = k * stride[1];m < mwSize(k * stride[1] + size[1]);m ++){
                            for(mwSize n = l * stride[0];n < mwSize(l * stride[0] + size[0]);n ++){
                                double data = data_point[i * data_sample_size + j * data_channel_size + m * DATA_DIMS[0] + n];
                                if(local_max < data){
                                    local_max = data;
                                    local_col = m - k * stride[1];
                                    local_row = n - l * stride[0];
                                }
                                if(debug){
                                    mexPrintf("%f ",data_point[i * data_sample_size + j * data_channel_size + m * DATA_DIMS[0] + n]);
                                }
                            }
                            if(debug){
                                mexPrintf("\n");
                            }
                        }
                        Output_point[i * output_sample_size + j * output_channel_size + k * M_DIMS[0] + l] = local_max;
                        mark_point[i * mark_sample_size + j * mark_channel_size + k * mark_col_size + l * mark_row_size + local_col * mark_dims[0] + local_row] = 1;
                        if(debug){
                            mexPrintf("local_max = %f,local_col = %d,local_row = %d\n\n",local_max,local_col,local_row);
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
