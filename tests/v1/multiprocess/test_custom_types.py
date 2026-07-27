# SPDX-License-Identifier: Apache-2.0
# Standard
from multiprocessing import Queue
import multiprocessing as mp

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheServerKey,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.platform.cuda.ipc_wrapper import (
    CudaIPCWrapper,
    RawCudaIPCWrapper,
)


def test_ipc_cache_engine_key_serialization():
    """Test encoding and decoding of IPCCacheServerKey using msgspec."""
    # Create a sample IPCCacheServerKey
    original_key = IPCCacheServerKey.from_token_ids(
        model_name="test_model",
        world_size=4,
        worker_id=1,
        token_ids=list(range(256)),
        start=0,
        end=256,
        request_id="test_request",
    )

    # Encode the key
    encoded = msgspec.msgpack.encode(original_key)

    # Decode the key
    decoded_key = msgspec.msgpack.decode(encoded, type=IPCCacheServerKey)

    # Verify correctness
    assert original_key == decoded_key, "IPCCacheServerKeys do not match!"


def test_ipc_cache_engine_key_serialization_with_cache_salt():
    """Roundtrip must carry ``cache_salt`` verbatim — it is part of
    cache identity so eq must hold after encode/decode."""
    original_key = IPCCacheServerKey.from_token_ids(
        model_name="test_model",
        world_size=4,
        worker_id=1,
        token_ids=list(range(256)),
        start=0,
        end=256,
        request_id="test_request",
        cache_salt="alice",
    )

    encoded = msgspec.msgpack.encode(original_key)
    decoded_key = msgspec.msgpack.decode(encoded, type=IPCCacheServerKey)

    assert original_key == decoded_key
    assert decoded_key.cache_salt == "alice"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CudaIPCWrapper tests",
)
def test_cudaipc_wrapper_serialization():
    """Test custom encoder/decoder for single CudaIPCWrapper object."""
    encoder = get_customized_encoder(type=CudaIPCWrapper)
    decoder = get_customized_decoder(type=CudaIPCWrapper)

    # Create a sample tensor
    original_tensor = torch.randn(3, 4, device="cuda")
    wrapper = CudaIPCWrapper(original_tensor)

    # Encode the wrapper
    encoded = encoder.encode(wrapper)

    # Decode the wrapper
    decoded_wrapper = decoder.decode(encoded)
    assert isinstance(decoded_wrapper, CudaIPCWrapper), (
        "Decoded object is not of type CudaIPCWrapper"
    )
    assert decoded_wrapper == wrapper, (
        "Decoded CudaIPCWrapper does not match the original"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CudaIPCWrapper tests",
)
def test_cudaipc_wrapper_list_serialization():
    """Test custom encoder/decoder for list of CudaIPCWrapper objects."""
    wrappers = []
    for _ in range(5):
        tensor = torch.randn(2, 2, device="cuda")
        wrapper = CudaIPCWrapper(tensor)
        wrappers.append(wrapper)

    encoder = get_customized_encoder(type=list[CudaIPCWrapper])
    decoder = get_customized_decoder(type=list[CudaIPCWrapper])

    # Encode the list of wrappers
    encoded = encoder.encode(wrappers)

    # Decode the list of wrappers
    decoded_wrappers = decoder.decode(encoded)

    assert len(decoded_wrappers) == len(wrappers), (
        "Decoded list length does not match original"
    )

    for original, decoded in zip(wrappers, decoded_wrappers, strict=False):
        assert original == decoded, "Decoded CudaIPCWrapper does not match the original"


def _worker_process_deserialize_and_reconstruct(
    encoded_data: bytes, result_queue: Queue
):
    """
    Worker function that runs in a separate process.
    Deserializes CudaIPCWrapper list and reconstructs tensors.
    """
    try:
        # Decode the list of wrappers
        torch.cuda.init()
        decoder = get_customized_decoder(type=list[CudaIPCWrapper])
        decoded_wrappers = decoder.decode(encoded_data)

        # Convert each wrapper back to tensor and compute checksum
        checksums = []
        shapes = []
        for wrapper in decoded_wrappers:
            tensor = wrapper.to_tensor()
            # Compute checksum as sum of all elements
            checksum = float(tensor.sum().cpu().item())
            checksums.append(checksum)
            shapes.append(list(tensor.shape))

            # Do add 1 on the tensor to ensure it's writable
            tensor.add_(1)

        result_queue.put(("success", checksums, shapes))
    except Exception as e:
        result_queue.put(("error", str(e), None))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CudaIPCWrapper multiprocessing tests",
)
def test_cudaipc_wrapper_multiprocess_serialization():
    """
    Test CudaIPCWrapper serialization across processes using spawn method.
    This verifies that CUDA IPC handles can be properly shared between processes.
    """
    # Set multiprocessing start method to spawn
    ctx = mp.get_context("spawn")

    # Create test tensors and wrappers in the main process
    num_tensors = 3
    tensors = []
    test_data = []
    wrappers = []

    for i in range(num_tensors):
        # Create a tensor with known values
        tensor = torch.full(
            (2, 3), fill_value=float(i + 1), dtype=torch.float32, device="cuda"
        )
        tensors.append(tensor)
        wrapper = CudaIPCWrapper(tensor)
        wrappers.append(wrapper)

        # Store expected checksum and shape
        expected_checksum = float(tensor.sum().cpu().item())
        expected_shape = list(tensor.shape)
        test_data.append((expected_checksum, expected_shape))

    # Serialize the wrappers
    encoder = get_customized_encoder(type=list[CudaIPCWrapper])
    encoded_data = encoder.encode(wrappers)

    # Create a queue for results
    result_queue = ctx.Queue()

    # Start worker process
    process = ctx.Process(
        target=_worker_process_deserialize_and_reconstruct,
        args=(encoded_data, result_queue),
    )
    process.start()

    # Wait for result with timeout
    process.join(timeout=10)

    # Check if process completed successfully
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Worker process timed out")

    assert process.exitcode == 0, (
        f"Worker process failed with exit code {process.exitcode}"
    )

    # Get result from queue
    assert not result_queue.empty(), "No result received from worker process"
    status, checksums, shapes = result_queue.get()

    assert status == "success", f"Worker process encountered error: {checksums}"
    assert len(checksums) == num_tensors, "Number of checksums does not match"
    assert len(shapes) == num_tensors, "Number of shapes does not match"

    # Verify checksums and shapes match
    for i, (
        (expected_checksum, expected_shape),
        actual_checksum,
        actual_shape,
    ) in enumerate(zip(test_data, checksums, shapes, strict=False)):
        assert actual_shape == expected_shape, (
            f"Tensor {i}: shape mismatch. Expected {expected_shape}, got {actual_shape}"
        )
        assert abs(actual_checksum - expected_checksum) < 1e-5, (
            f"Tensor {i}: checksum mismatch. Expected {expected_checksum}, "
            f"got {actual_checksum}"
        )

    # Verify that the tensors are being modified in the worker process
    for i, (tensor, (expected_checksum, _)) in enumerate(
        zip(tensors, test_data, strict=False)
    ):
        # After adding 1 to each element, the new checksum should be:
        num_elements = tensor.numel()
        new_expected_checksum = expected_checksum + float(num_elements)
        actual_checksum = float(tensor.sum().cpu().item())
        assert abs(actual_checksum - new_expected_checksum) < 1e-5, (
            f"Tensor {i}: post-modification checksum mismatch. "
            f"Expected {new_expected_checksum}, got {actual_checksum}"
        )


def _worker_reconstruct_offset_tensor(encoded_data: bytes, result_queue: Queue):
    """Worker: decode a single CudaIPCWrapper and reconstruct its tensor,
    reporting the layout metadata and a checksum back to the parent."""
    try:
        torch.cuda.init()
        decoder = get_customized_decoder(type=CudaIPCWrapper)
        wrapper = decoder.decode(encoded_data)
        tensor = wrapper.to_tensor()
        result_queue.put(
            (
                "success",
                int(tensor.storage_offset()),
                list(tensor.shape),
                list(tensor.stride()),
                float(tensor.sum().cpu().item()),
            )
        )
    except Exception as e:
        result_queue.put(("error", str(e), None, None, None))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CudaIPCWrapper tests",
)
def test_cudaipc_wrapper_nonzero_storage_offset():
    """CudaIPCWrapper must round-trip a slice/narrow view with
    ``storage_offset > 0`` bit-identically across processes.

    This is the property PR #3853 relies on: with MTP speculative decoding +
    CPU offload, per-layer KV are non-zero-``storage_offset`` slices of a
    unified pool, so ``_validate_dim0_padded_layout`` must accept them. The
    view here is both dim-0-padded (``stride[0] > prod(shape[1:])``) and
    offset-shifted, exactly that shape. ``CudaIPCWrapper`` encodes
    ``storage_offset`` and the receiver rebuilds the view via
    ``set_(storage, storage_offset, shape, stride)``; this verifies the
    reconstructed tensor reads from the correct (offset, strided) region.
    """
    ctx = mp.get_context("spawn")

    # arange so each element's value equals its flat storage index -- the
    # checksum then pins down exactly which storage positions were read.
    base = torch.arange(64, dtype=torch.float32, device="cuda")
    # dim-0-padded view: shape (3, 2, 4), per-block stride 12 > prod(shape[1:])=8
    # (4 elements of padding per block), shifted by storage_offset=8.
    view = base.as_strided((3, 2, 4), (12, 4, 1), storage_offset=8)
    assert view.storage_offset() == 8
    assert not view.is_contiguous()

    wrapper = CudaIPCWrapper(view)
    assert wrapper.storage_offset == view.storage_offset()
    assert wrapper.shape == tuple(view.shape)
    assert wrapper.stride == tuple(view.stride())

    encoder = get_customized_encoder(type=CudaIPCWrapper)
    encoded = encoder.encode(wrapper)

    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_worker_reconstruct_offset_tensor,
        args=(encoded, result_queue),
    )
    process.start()
    process.join(timeout=10)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Worker process timed out")
    assert process.exitcode == 0, (
        f"Worker process failed with exit code {process.exitcode}"
    )
    assert not result_queue.empty(), "No result received from worker process"

    status, offset, shape, stride, checksum = result_queue.get()
    assert status == "success", f"Worker process encountered error: {offset}"
    assert offset == view.storage_offset()
    assert shape == list(view.shape)
    assert stride == list(view.stride())
    assert abs(checksum - float(view.sum().cpu().item())) < 1e-5


def _worker_reconstruct_raw_from_other_device(
    encoded_data: bytes, result_queue: Queue
):
    """Worker: decode a ``RawCudaIPCWrapper`` and reconstruct it while the
    importer's *current* device is ``cuda:0`` (not the exporting device).

    This models the TRT-LLM TP>1 topology: rank 1 exports a KV pool on
    ``cuda:1``, but the MP server thread that reconstructs it may have a
    different device current. ``cudaIpcOpenMemHandle`` maps the buffer for
    peer access from whatever device is *current* at open time, so
    ``to_tensor`` must set the exporting device around the open; otherwise
    the mapping is established for ``cuda:0`` and a later kernel on
    ``cuda:1``'s stream dereferences a pointer invalid on that device.

    The reconstructed tensor reports ``cuda:1`` either way (its device is
    derived from the pointer's UUID, not the open-time current device), so
    device placement is NOT the discriminating signal -- the *readback* is:
    ``tensor.sum()`` launches a kernel on the tensor's own ``cuda:1``
    stream, which faults under the wrong-device open and succeeds under the
    fix. A fault here raises, so the worker reports ``"error"`` (or crashes
    with a nonzero exit code) and the parent's success assertion fails.
    """
    try:
        torch.cuda.init()
        # Pin the importer's current device to 0 -- deliberately NOT the
        # exporting device -- so a wrong-device open maps for cuda:0 while
        # the readback kernel runs on the tensor's own cuda:1 stream.
        torch.cuda.set_device(0)
        decoder = get_customized_decoder(type=RawCudaIPCWrapper)
        wrapper = decoder.decode(encoded_data)
        tensor = wrapper.to_tensor()
        # The discriminating read: a kernel on the reconstructed tensor's
        # own device stream. This is what faulted under the bug.
        checksum = float(tensor.sum().cpu().item())
        result_queue.put(
            (
                "success",
                tensor.device.index,
                list(tensor.shape),
                checksum,
            )
        )
    except Exception as e:
        result_queue.put(("error", str(e), None, None))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="RawCudaIPCWrapper cross-device import test requires >=2 GPUs",
)
def test_rawcudaipc_wrapper_reconstructs_on_exporting_device():
    """``RawCudaIPCWrapper.to_tensor`` must open the IPC handle with the
    *exporting* device current, so a later kernel on the reconstructed
    tensor's own stream can read it.

    Regression test for the TRT-LLM TP>1 crash: each rank exports a KV
    pool on its own physical device (rank 1 -> ``cuda:1``), but
    ``cudaIpcOpenMemHandle`` maps the buffer for peer access from whatever
    device is *current* at open time. Opening under the default ``cuda:0``
    mapped rank 1's pool for the wrong device; the transfer kernel --
    launched on the reconstructed context's own ``cuda:1`` stream -- then
    dereferenced a pointer invalid on that device, surfacing as a CUDA
    "illegal memory access". The fix wraps the open in
    ``with cupy.cuda.Device(index)``.

    The buffer is exported on ``cuda:1`` and reconstructed in a worker
    whose current device is pinned to ``cuda:0``. The reconstructed tensor
    reports ``cuda:1`` in both the broken and fixed cases (its device comes
    from the pointer UUID, not the open-time current device), so the
    discriminator is the readback: the pre-fix open faults when the worker
    reads the tensor on its ``cuda:1`` stream, while the fixed open reads
    the exported ``arange`` pattern back verbatim.
    """
    # Third Party
    import cupy

    try:
        # Third Party
        from cuda.bindings import runtime as cudart
    except ImportError:
        # Third Party
        from cuda import cudart

    exporter_index = 1
    num_elems = 256
    nbytes = num_elems * 4

    # Allocate a raw cudaMalloc'd buffer on cuda:1 -- a genuine allocation
    # base pointer, as cudaIpcGetMemHandle requires, and outside PyTorch's
    # caching allocator (the exact TRT-LLM pool shape RawCudaIPCWrapper
    # targets; CuPy's pool sub-allocates and would not give a valid IPC
    # base). Fill it with arange so each element pins its storage index.
    with cupy.cuda.Device(exporter_index):
        err, raw_ptr = cudart.cudaMalloc(nbytes)
        assert err == cudart.cudaError_t.cudaSuccess, f"cudaMalloc: {err}"
        mem = cupy.cuda.UnownedMemory(raw_ptr, nbytes, owner=None)
        memptr = cupy.cuda.MemoryPointer(mem, 0)
        cp_view = cupy.ndarray((num_elems,), dtype=cupy.float32, memptr=memptr)
        cp_view[...] = cupy.arange(num_elems, dtype=cupy.float32)
        exported = torch.from_dlpack(cp_view)
        assert exported.device.index == exporter_index

        wrapper = RawCudaIPCWrapper(exported)

    expected_checksum = float(num_elems * (num_elems - 1) / 2)

    encoder = get_customized_encoder(type=RawCudaIPCWrapper)
    encoded = encoder.encode(wrapper)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_worker_reconstruct_raw_from_other_device,
        args=(encoded, result_queue),
    )
    process.start()
    process.join(timeout=30)

    try:
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("Worker process timed out")
        assert process.exitcode == 0, (
            f"Worker process failed with exit code {process.exitcode} "
            "(a CUDA illegal-access crash indicates the wrong-device import)"
        )
        assert not result_queue.empty(), "No result received from worker process"

        status, device_index, shape, checksum = result_queue.get()
        # The core assertion: the readback succeeded. Under the pre-fix
        # wrong-device open, tensor.sum() in the worker faults with a CUDA
        # illegal access -> status == "error" (device_index carries the
        # exception string).
        assert status == "success", (
            f"Worker reconstruction/readback failed (wrong-device IPC "
            f"import regression): {device_index}"
        )
        # Sanity: the tensor lands on the exporting device (true both
        # before and after the fix -- device comes from the pointer UUID).
        assert device_index == exporter_index, (
            f"reconstructed on cuda:{device_index}, expected the exporting "
            f"device cuda:{exporter_index}"
        )
        assert shape == [num_elems]
        # The readback read the correct storage region (arange sum).
        assert abs(checksum - expected_checksum) < 1e-3
    finally:
        # Keep the source allocation alive until the child has opened its
        # own mapping (join above), then release it.
        with cupy.cuda.Device(exporter_index):
            cudart.cudaFree(raw_ptr)


def test_block_allocation_record_serialization():
    """Test encoding and decoding of BlockAllocationRecord using msgspec."""
    original = BlockAllocationRecord(
        req_id="req-42",
        new_block_ids=[10, 20, 30],
        new_token_ids=[100, 200, 300, 400],
    )

    encoded = msgspec.msgpack.encode(original)
    decoded = msgspec.msgpack.decode(encoded, type=BlockAllocationRecord)

    assert decoded.req_id == original.req_id
    assert decoded.new_block_ids == original.new_block_ids
    assert decoded.new_token_ids == original.new_token_ids


def test_block_allocation_record_list_serialization():
    """Test encoding and decoding of a list of BlockAllocationRecord."""
    records = [
        BlockAllocationRecord(
            req_id="req-1",
            new_block_ids=[1, 2],
            new_token_ids=[10, 20, 30],
        ),
        BlockAllocationRecord(
            req_id="req-2",
            new_block_ids=[],
            new_token_ids=[40, 50],
        ),
    ]

    encoded = msgspec.msgpack.encode(records)
    decoded = msgspec.msgpack.decode(encoded, type=list[BlockAllocationRecord])

    assert len(decoded) == 2
    assert decoded[0].req_id == "req-1"
    assert decoded[0].new_block_ids == [1, 2]
    assert decoded[1].req_id == "req-2"
    assert decoded[1].new_block_ids == []
    assert decoded[1].new_token_ids == [40, 50]
