import numpy as np
import pyopencl as cl
from pyplanknn.layer import Layer


class CLLayer(Layer):
    #TODO: look into http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf
    def __init__(self, *args, **kwargs):
        #TODO: this needs to be adjusted based on the computer
        self.ctx = cl.create_some_context(answers=[0])
        self.queue = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, """
        __kernel void wgt_logit_fwd(const int n_trainees, const int n_inputs,
            const int n_outputs, __global float* input,
            __global float* weights, __global float* result)
        {
            int k;
            int i = get_global_id(0);
            int j = get_global_id(1);
            float tmp = 0.0f;
            for (k=0; k < n_inputs; k++) {
                tmp += input[i * n_inputs + k] * weights[k * n_outputs + j];
            }
            result[i * n_outputs + j] = tmp;
            //result[i * n_outputs + j] = 1 / (1 + exp(-tmp));
        }
        """).build()
        super().__init__(*args, **kwargs)

    def __call__(self, vector):
        # append 1 to beginning of data for bias term
        v_bias = np.hstack([np.ones((vector.shape[0], 1), \
                                    dtype=np.float32), vector])

        self._input = v_bias
        self._f_input = np.empty((vector.shape[0], self.weights.shape[1]), \
                                 dtype=np.float32)

        flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        inbuf = cl.Buffer(self.ctx, flags, hostbuf=v_bias)
        weightsbuf = cl.Buffer(self.ctx, flags, hostbuf=self.weights)
        flags = cl.mem_flags.WRITE_ONLY
        outbuf = cl.Buffer(self.ctx, flags, self._f_input.nbytes)

        args = (np.int32(vector.shape[0]), np.int32(self.weights.shape[0]), \
                np.int32(self.weights.shape[1]), inbuf, weightsbuf, outbuf)
        self.program.wgt_logit_fwd(self.queue, self._f_input.shape, \
                                   None, *args)
        cl.enqueue_copy(self.queue, self._f_input, outbuf)
        self.queue.finish()
        return self.f(self._f_input)
