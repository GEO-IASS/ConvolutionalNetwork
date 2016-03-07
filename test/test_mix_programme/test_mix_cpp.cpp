#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *ptr_data = prhs[0];
    double *data_point = mxGetPr(ptr_data);
    const mwSize *tmp = mxGetDimensions(ptr_data);    
    
    const mxArray *ptr_data_1 = prhs[1];
    double *data_point_1 = mxGetPr(ptr_data_1);
    const mwSize *tmp_1 = mxGetDimensions(ptr_data_1);
    
    if(tmp[0] != tmp_1[0] || tmp[1] != tmp_1[1])
            mexErrMsgTxt("Dimension dismatch!");
    
    mwSize Output_size[2];
    Output_size[0] = tmp[0];
    Output_size[1] = tmp[1];
    plhs[0] = mxCreateNumericArray(2,Output_size,mxSINGLE_CLASS,mxREAL);
    mxArray *OutputData = plhs[0];
    double *Output_point = mxGetPr(OutputData);
    for(mwSize i = 0;i < Output_size[0]; i ++){
        for(mwSize j = 0;j < Output_size[1]; j ++){
            Output_point[j * Output_size[0] + i] =  data_point[j * Output_size[0] + i] / data_point_1[j * Output_size[0] + i];
        }
    }
    
}