# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused__softmax_cat_clamp_min_div_expand_gt_linalg_vector_norm_mul_new_zeros_ones_like_pow_scalar_tensor_sum_unsqueeze_where_0 = async_compile.cpp_pybinding(['float*', 'float*', 'float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto out_ptr5 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr1;
    auto out_ptr2 = in_out_ptr2;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(32LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(64LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 64LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = tmp0 * tmp0;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                out_ptr3[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(64LL); x2+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(64LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + 64LL*x0), static_cast<int64_t>(4));
                                auto tmp1 = tmp0 * tmp0;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    in_out_ptr1[static_cast<int64_t>(x1 + 16LL*x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(64LL); x2+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(64LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + 64LL*x0), static_cast<int64_t>(4));
                                auto tmp1 = tmp0 * tmp0;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    in_out_ptr0[static_cast<int64_t>(x1 + 16LL*x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(64LL); x2+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(64LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + 64LL*x0), static_cast<int64_t>(4));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + 64LL*x1 + 1024LL*x0), static_cast<int64_t>(4));
                                auto tmp3 = out_ptr5[static_cast<int64_t>(x1 + 16LL*x0)];
                                auto tmp9 = out_ptr1[static_cast<int64_t>(x1 + 16LL*x0)];
                                auto tmp15 = out_ptr4[static_cast<int64_t>(x1 + 16LL*x0)];
                                auto tmp20 = out_ptr3[static_cast<int64_t>(x1 + 16LL*x0)];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = std::sqrt(tmp3);
                                auto tmp5 = static_cast<float>(9.99999993922529e-09);
                                auto tmp6 = max_propagate_nan(tmp4, tmp5);
                                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                auto tmp8 = tmp0 / tmp7;
                                auto tmp10 = std::sqrt(tmp9);
                                auto tmp11 = max_propagate_nan(tmp10, tmp5);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp1 / tmp12;
                                auto tmp14 = tmp8 * tmp13;
                                auto tmp16 = std::sqrt(tmp15);
                                auto tmp17 = max_propagate_nan(tmp16, tmp5);
                                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                                auto tmp19 = tmp0 / tmp18;
                                auto tmp21 = std::sqrt(tmp20);
                                auto tmp22 = max_propagate_nan(tmp21, tmp5);
                                auto tmp23 = at::vec::Vectorized<float>(tmp22);
                                auto tmp24 = tmp1 / tmp23;
                                auto tmp25 = tmp19 * tmp24;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                                tmp_acc1_vec = tmp_acc1_vec + tmp14;
                                tmp_acc2_vec = tmp_acc2_vec + tmp25;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<int64_t>(x1 + 16LL*x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    in_out_ptr0[static_cast<int64_t>(x1 + 16LL*x0)] = static_cast<float>(tmp_acc1);
                    out_ptr7[static_cast<int64_t>(x1 + 16LL*x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc2 = tmp_acc2 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc2_vec);
                    in_out_ptr1[static_cast<int64_t>(x1 + 16LL*x0)] = static_cast<float>(tmp_acc2);
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x1 + 16LL*x0), static_cast<int64_t>(4));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<int64_t>(x1 + 16LL*x0), static_cast<int64_t>(4));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 16LL*x0), static_cast<int64_t>(4));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = at::vec::VecMask<float,1>(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp0)::blendv(tmp6, tmp0, tmp3.template cast<float,1>());
                        auto tmp8 = tmp4 / tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp11 = decltype(tmp10)::blendv(tmp2, tmp10, tmp3.template cast<float,1>());
                        auto tmp12 = static_cast<float>(0.1);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 / tmp13;
                        tmp14.store(in_out_ptr2 + static_cast<int64_t>(x1 + 16LL*x0));
                        tmp14.store(out_ptr8 + static_cast<int64_t>(x1 + 17LL*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                {
                    auto tmp0 = static_cast<float>(0.0);
                    out_ptr9[static_cast<int64_t>(17LL*x0)] = tmp0;
                }
            }
        }
    }
    {
        std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(17LL);
        float* local_buffer_data_0 = buf_local_buffer_data_0.get();
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(17LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 17LL*x0), static_cast<int64_t>(4));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(16LL) && x1 < static_cast<int64_t>(17LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 17LL*x0), static_cast<int64_t>(1LL));
                            tmp_acc0_vec = max_masked_reduce(tmp_acc0_vec, tmp0, static_cast<int64_t>(1LL));
                        }
                    }
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr10[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(17LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 17LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr10[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp4 = tmp3.exp();
                            tmp4.store(local_buffer_data_0 + static_cast<int64_t>(x1));
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(16LL) && x1 < static_cast<int64_t>(17LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 17LL*x0), static_cast<int64_t>(1LL));
                            auto tmp1 = out_ptr10[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp4 = tmp3.exp();
                            tmp4.store(local_buffer_data_0 + static_cast<int64_t>(x1), static_cast<int64_t>(1LL));
                            tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp4, static_cast<int64_t>(1LL));
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr11[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(17LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(local_buffer_data_0 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp1 = out_ptr11[static_cast<int64_t>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(out_ptr12 + static_cast<int64_t>(x1 + 17LL*x0));
                    }
                    if(C10_UNLIKELY(x1 >= static_cast<int64_t>(16LL) && x1 < static_cast<int64_t>(17LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(local_buffer_data_0 + static_cast<int64_t>(x1), static_cast<int64_t>(1LL));
                        auto tmp1 = out_ptr11[static_cast<int64_t>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        tmp3.store(out_ptr12 + static_cast<int64_t>(x1 + 17LL*x0), static_cast<int64_t>(1LL));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2, 16, 64), (1024, 64, 1), 'input')
        buf0 = empty_strided_cpu((2, 16), (16, 1), torch.float32)
        buf3 = empty_strided_cpu((2, 16, 1), (16, 1, 32), torch.float32)
        buf6 = empty_strided_cpu((2, 16), (16, 1), torch.float32)
        buf8 = empty_strided_cpu((2, 16, 1), (16, 1, 32), torch.float32)
        assert_size_stride(arg1_1, (2, 64), (64, 1), 'input')
        buf7 = empty_strided_cpu((2, 16, 1), (16, 1, 32), torch.float32)
        buf2 = empty_strided_cpu((2, 16, 1), (16, 1, 32), torch.float32)
        buf1 = empty_strided_cpu((2, 16), (16, 1), torch.float32)
        buf4 = reinterpret_tensor(buf2, (2, 16), (16, 1), 0); del buf2  # reuse
        buf5 = empty_strided_cpu((2, 16), (16, 1), torch.float32)
        buf9 = reinterpret_tensor(buf7, (2, 16), (16, 1), 0); del buf7  # reuse
        buf10 = buf6; del buf6  # reuse
        buf13 = empty_strided_cpu((2, 17), (17, 1), torch.float32)
        buf11 = reinterpret_tensor(buf13, (2, 16), (17, 1), 0)  # alias
        buf12 = reinterpret_tensor(buf13, (2, 1), (17, 1), 16)  # alias
        buf14 = empty_strided_cpu((2, 1), (1, 2), torch.float32)
        buf16 = empty_strided_cpu((2, 1), (1, 2), torch.float32)
        buf17 = empty_strided_cpu((2, 17), (17, 1), torch.float32)
        cpp_fused__softmax_cat_clamp_min_div_expand_gt_linalg_vector_norm_mul_new_zeros_ones_like_pow_scalar_tensor_sum_unsqueeze_where_0(buf4, buf9, buf10, arg0_1, arg1_1, buf13, buf0, buf3, buf8, buf1, buf5, buf11, buf12, buf14, buf16, buf17)
        del arg0_1
        del arg1_1
        return (buf0, buf1, buf4, buf10, buf17, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2, 64), (64, 1), device='cpu', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
