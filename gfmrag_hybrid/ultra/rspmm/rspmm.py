# mypy: ignore-errors
import os
import sys

import torch.backends.openmp
from torch import autograd
from torch.utils import cpp_extension

module = sys.modules[__name__]


class RSPMMAddMulFunction(autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_mul_forward_cuda
        else:
            forward = rspmm.rspmm_add_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(
            edge_index, edge_type, edge_weight, relation, input, output
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_mul_backward_cuda
        else:
            backward = rspmm.rspmm_add_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(
            *ctx.saved_tensors, output_grad
        )
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinMulFunction(autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_mul_forward_cuda
        else:
            forward = rspmm.rspmm_min_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(
            edge_index, edge_type, edge_weight, relation, input, output
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_mul_backward_cuda
        else:
            backward = rspmm.rspmm_min_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(
            *ctx.saved_tensors, output_grad
        )
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxMulFunction(autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_mul_forward_cuda
        else:
            forward = rspmm.rspmm_max_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(
            edge_index, edge_type, edge_weight, relation, input, output
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_mul_backward_cuda
        else:
            backward = rspmm.rspmm_max_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(
            *ctx.saved_tensors, output_grad
        )
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMAddAddFunction(autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_add_forward_cuda
        else:
            forward = rspmm.rspmm_add_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(
            edge_index, edge_type, edge_weight, relation, input, output
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_add_backward_cuda
        else:
            backward = rspmm.rspmm_add_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(
            *ctx.saved_tensors, output_grad
        )
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinAddFunction(autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_add_forward_cuda
        else:
            forward = rspmm.rspmm_min_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(
            edge_index, edge_type, edge_weight, relation, input, output
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_add_backward_cuda
        else:
            backward = rspmm.rspmm_min_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(
            *ctx.saved_tensors, output_grad
        )
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxAddFunction(autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_add_forward_cuda
        else:
            forward = rspmm.rspmm_max_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(
            edge_index, edge_type, edge_weight, relation, input, output
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_add_backward_cuda
        else:
            backward = rspmm.rspmm_max_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(
            *ctx.saved_tensors, output_grad
        )
        return None, None, weight_grad, relation_grad, input_grad


def generalized_rspmm(
    edge_index, edge_type, edge_weight, relation, input, sum="add", mul="mul"
):
    name = f"RSPMM{sum.capitalize()}{mul.capitalize()}Function"
    if not hasattr(module, name):
        raise ValueError(
            f"No generalized rspmm implementation found for summation `{sum}` and multiplication `{mul}`"
        )
    Function = getattr(module, name)

    node_in, node_out = edge_index
    key = node_in * (node_out.max() + 1) + node_out
    order = key.argsort()

    return Function.apply(
        edge_index[:, order], edge_type[order], edge_weight[order], relation, input
    )


def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
    if extra_cflags is None:
        if sys.platform == "win32":
            # MSVC (cl.exe) không hiểu cờ GCC (-Ofast/-fopenmp). Dùng cờ MSVC tương đương.
            # /std:c++17 + /Zc:__cplusplus để pybind11 biên dịch đúng dưới nvcc.
            extra_cflags = ["/O2", "/openmp", "/DAT_PARALLEL_OPENMP", "/std:c++17", "/Zc:__cplusplus"]
        else:
            extra_cflags = ["-Ofast"]
            # PyTorch 2.2.1+ on Apple Silicon is now compiled by default with OpenMP
            # However, installing OpenMP on macs properly and wiring it together to the compiler is tedious
            # So on macs we turn off OpenMP (as the default behavior in all torch < 2.2.1 versions)
            if torch.backends.openmp.is_available() and not sys.platform.startswith(
                "darwin"
            ):
                extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
            else:
                extra_cflags.append("-DAT_PARALLEL_NATIVE")
    if extra_cuda_cflags is None:
        if torch.cuda.is_available():
            extra_cuda_cflags = ["-O3"]
            if sys.platform == "win32":
                # Khớp chuẩn C++17 giữa nvcc và host MSVC để pybind11 không lỗi `operator new`.
                extra_cuda_cflags += [
                    "-std=c++17",
                    "-Xcompiler", "/std:c++17",
                    "-Xcompiler", "/Zc:__cplusplus",
                    "-allow-unsupported-compiler",
                ]
            extra_cflags.append("-DCUDA_OP")
        else:
            new_sources = []
            for source in sources:
                if not cpp_extension._is_cuda_file(source):
                    new_sources.append(source)
            sources = new_sources

    return cpp_extension.load(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)


if sys.platform == "win32":
    # torch.utils.cpp_extension truy cập `distutils._msvccompiler._get_vc_env` để dò MSVC.
    # Trên Python 3.12 (distutils stdlib đã bị bỏ), submodule này không tự import làm attribute,
    # gây `AttributeError: module 'distutils' has no attribute '_msvccompiler'`.
    # Pre-import bản distutils của setuptools để tạo sẵn attribute đó.
    os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
    try:
        import setuptools  # noqa: F401
        import distutils._msvccompiler  # noqa: F401
    except Exception:
        pass

print("Load rspmm extension. This may take a while...")
path = os.path.join(os.path.dirname(__file__), "source")
rspmm = load_extension(
    "rspmm", [os.path.join(path, "rspmm.cpp"), os.path.join(path, "rspmm.cu")]
)
