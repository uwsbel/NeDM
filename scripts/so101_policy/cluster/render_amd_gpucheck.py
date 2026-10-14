"""Quick check that Warp (AMD ROCm port) and torch share one GPU and one HIP runtime in this process.

    python3.12 -P scripts/so101_policy/cluster/render_amd_gpucheck.py

Checks: Warp sees the GPU and runs a kernel; torch runs a matmul and a conv (rocBLAS / MIOpen); a torch tensor goes to
Warp and back without a copy (same device pointer) and a Warp kernel's write is seen by torch; a Warp array
allocated by Warp is read by torch. Prints one JSON line; exit 1 on any failure.
"""
import json
import os
import sys

if os.environ.get("RENDER_IMPORT_ORDER") == "warp_first":     # the import order matters with two HIP runtimes
    import warp as wp
    import torch
else:
    import torch
    import warp as wp


@wp.kernel
def twice(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = 2.0 * a[i] + float(i)


def main():
    wp.init()
    if not wp.is_cuda_available():
        print(json.dumps(dict(all_ok=False, error="Warp sees no GPU", import_order=os.environ.get("RENDER_IMPORT_ORDER",
                                                                                                 "torch_first"),
                              ld_preload=os.environ.get("LD_PRELOAD", ""), torch_cuda=torch.cuda.is_available())))
        return 1
    res = dict(torch=torch.__version__, hip=getattr(torch.version, "hip", None), warp=wp.__version__,
               import_order=os.environ.get("RENDER_IMPORT_ORDER", "torch_first"),
               ld_preload=os.environ.get("LD_PRELOAD", ""))
    d = wp.get_device("cuda:0")
    res["warp_device"], res["warp_arch"] = d.name, str(getattr(d, "arch_str", d.arch))
    res["torch_device"] = torch.cuda.get_device_name(0)
    ok = True
    # torch: matmul + conv
    x = torch.randn(1024, 1024, device="cuda")
    res["torch_matmul_ok"] = bool(torch.allclose((x @ x).cpu(), x.cpu() @ x.cpu(), atol=1e-2, rtol=1e-3))
    c = torch.nn.Conv2d(3, 16, 3).cuda()
    res["torch_conv_ok"] = bool(torch.isfinite(c(torch.randn(8, 3, 64, 64, device="cuda"))).all())
    # torch -> warp (zero copy), warp kernel, warp -> torch
    a = torch.arange(4096, dtype=torch.float32, device="cuda")
    wa = wp.from_torch(a)
    wo = wp.zeros(4096, dtype=float, device="cuda:0")
    wp.launch(twice, dim=4096, inputs=[wa, wo], device="cuda:0")
    wp.synchronize()
    to = wp.to_torch(wo)
    res["zero_copy_in"] = wa.ptr == a.data_ptr()
    res["zero_copy_out"] = to.data_ptr() == wo.ptr
    res["kernel_ok"] = bool(torch.equal(to, 3.0 * a))
    a.mul_(-1.0)                                      # torch writes a[i] = -i, warp reads the same memory
    wp.launch(twice, dim=4096, inputs=[wa, wo], device="cuda:0")
    wp.synchronize()
    res["shared_memory_ok"] = bool(torch.equal(to, a))  # 2 (-i) + i = -i
    # streams: warp on torch's current stream
    s = wp.stream_from_torch(torch.cuda.current_stream())
    a.mul_(2.0)                                       # queued on torch's stream; warp's kernel follows on it
    with wp.ScopedStream(s):
        wp.launch(twice, dim=4096, inputs=[wa, wo])
    torch.cuda.synchronize()
    res["torch_stream_ok"] = bool(torch.equal(to, 1.5 * a))  # a[i] = -2 i: 2 (-2 i) + i = -3 i
    free, total = torch.cuda.mem_get_info()
    res["gpu_total_gb"], res["gpu_used_gb"] = round(total / 2 ** 30, 1), round((total - free) / 2 ** 30, 2)
    for k in ("torch_matmul_ok", "torch_conv_ok", "zero_copy_in", "zero_copy_out", "kernel_ok", "shared_memory_ok",
              "torch_stream_ok"):
        ok &= bool(res[k])
    res["all_ok"] = ok
    print(json.dumps(res), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
