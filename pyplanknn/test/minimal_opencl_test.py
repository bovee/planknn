"""
OpenCL via Beignet fails on my laptop; this test demonstrates
that copying out of GPU buffers fails.
"""
import numpy as np
import pyopencl as cl

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def copy(array1, device):
    ctx = cl.create_some_context(answers=[device])
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, """
    __kernel void testcopy(__global float *input, __global float *output)
    {
        int i = get_global_id(1);
        //printf((__constant char *)"%i\\n", i);
        //printf((__constant char *)"%f\\n", input[i]);
        //output[i] = input[i];
        output[i] = 1.0f;
    }
    """).build()

    output = np.empty_like(array1, dtype=np.float32)

    flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
    a1buf = cl.Buffer(ctx, flags, hostbuf=array1)

    flags = cl.mem_flags.WRITE_ONLY
    outbuf = cl.Buffer(ctx, flags, size=output.nbytes)

    program.testcopy(queue, output.shape, None, a1buf, outbuf).wait()
    cl.enqueue_read_buffer(queue, outbuf, output).wait()
    queue.finish()

    #i_class = cl.program_info
    #for i_type in dir(i_class):
    #    if i_type.isupper():
    #        print(i_type, program.get_info(ctx.devices[0], \
    #                                       i_class.__dict__[i_type]))

    return output


def test_copy():
    array1 = np.array([[1., 2.]], dtype=np.float32)
    assert copy(array1, 0) == copy(array1, 2)
