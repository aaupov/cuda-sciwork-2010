CUDA_INC=/opt/cuda/sdk/C/common/inc/
CUDA_LIBS=cutil_x86_64
CUDA_PATH=/opt/cuda/sdk/C/lib/

all:
	nvcc main.cu -o prog -I${CUDA_INC} -l${CUDA_LIBS} -L${CUDA_PATH}
