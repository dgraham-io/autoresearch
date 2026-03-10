# Patch Notes

## Problem Summary

The application was not starting for two separate reasons:

1. Inside the Codex sandbox, GPU device nodes are not accessible, so CUDA reports unavailable even though the host machine has a working NVIDIA runtime.
2. Outside the sandbox, the real application failure on this machine was a kernel compatibility issue:
   - The GPU is an NVIDIA GeForce RTX 5090.
   - PyTorch reports compute capability `12.0` (Blackwell).
   - `train.py` selected the `kernels-community/flash-attn3` backend for non-Hopper GPUs.
   - That backend binary does not include `sm_120` kernels, so training failed with:
     - `no kernel image is available for execution on the device`

After switching away from FA3 on Blackwell, the default batch size also caused CUDA OOM with the PyTorch fallback attention path.

## Changes Made

### 1. Added an explicit CUDA startup check

File: [train.py](/home/david/repos/autoresearch/train.py)

Change:
- Added `require_cuda()` and used it at startup before choosing an attention backend.

Reason:
- The original code failed with a low-level CUDA traceback.
- The new check produces a direct, readable error if CUDA is not available or cannot initialize.

Comment:
- This does not make sandboxed GPU access work.
- It only improves failure clarity when the runtime cannot access CUDA.

### 2. Added a Blackwell-safe attention fallback

File: [train.py](/home/david/repos/autoresearch/train.py)

Change:
- Added `attention_backend` selection:
  - `flash_attn3` for pre-Blackwell GPUs
  - `torch_sdpa` for compute capability `>= 12.0`
- Added `build_sliding_window_mask(...)` to preserve the model's sliding-window behavior when using PyTorch SDPA.
- Updated `CausalSelfAttention.forward(...)` to:
  - use FA3 when supported
  - use `torch.nn.functional.scaled_dot_product_attention(...)` on Blackwell

Reason:
- The packaged FA3 binary being loaded on this machine only contains `sm_80` and `sm_90a` cubins.
- It does not contain `sm_120`, so it cannot execute on an RTX 5090.

Comment:
- This is a compatibility fallback, not a performance optimization.
- SDPA is slower and more memory-hungry than the intended FA3 path.

### 3. Reduced batch size for the fallback path

File: [train.py](/home/david/repos/autoresearch/train.py)

Change:
- Changed:
  - `DEVICE_BATCH_SIZE = 128`
- To:
  - `DEVICE_BATCH_SIZE = 16 if attention_backend == "torch_sdpa" else 128`

Reason:
- With the SDPA fallback, the original batch size ran out of memory on the 5090.

Comment:
- This keeps the existing behavior for FA3-capable GPUs.
- On Blackwell, it trades throughput for successful execution.

### 4. Added backend logging

File: [train.py](/home/david/repos/autoresearch/train.py)

Change:
- Added a startup print:
  - `Attention backend: ...`

Reason:
- This makes it obvious which attention implementation is active on a given machine.

## Validation Performed

### Runtime checks

- Confirmed the host driver is healthy outside the sandbox:
  - `nvidia-smi` works
  - `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia-uvm` can be opened
- Confirmed the sandbox cannot access the GPU device nodes
- Confirmed the GPU is:
  - `NVIDIA GeForce RTX 5090`
  - compute capability `(12, 0)`

### Binary inspection

- Inspected the cached `kernels-community/flash-attn3` shared object with `cuobjdump`
- Verified the binary contains `sm_80` and `sm_90a` cubins, but not `sm_120`

### Application verification

- `uv run prepare.py` succeeds
- `python3 -m py_compile train.py` succeeds
- `uv run train.py` outside the sandbox now starts training successfully on the RTX 5090
- A timed verification run reached multiple training steps without the previous kernel-image failure

## Operational Notes

- Running from a normal host shell should now work:

```bash
uv run prepare.py
uv run train.py
```

- Running from this Codex sandbox may still fail to access the GPU because the sandbox itself blocks `/dev/nvidia*`.

## Follow-Up Recommendation

If you want full performance on Blackwell, the proper long-term fix is to use a Flash Attention build that includes `sm_120` support. The current patch is a compatibility path that prioritizes getting the application running today.
