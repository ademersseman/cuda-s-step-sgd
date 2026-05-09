nvcc -o s_step_sgd_on_w1a\s_step_sgd s_step_sgd_on_w1a\s_step_sgd.cpp s_step_sgd_on_w1a\s_step_sgd.cu -lcublas
sys profile --trace=cuda,cublas,nvtx .\s_step_sgd 512 4 1 synthetic_data.txt uniform 8