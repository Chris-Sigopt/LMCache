#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SGLang + LMCache XPU integration test.

Adapted from sglang/python/sglang/srt/mem_cache/storage/lmcache/unit_test.py
to run on XPU devices. Tests the full store → retrieve → verify roundtrip
using the LMCacheLayerwiseConnector with XPU tensors.

Usage (inside Docker):
    export LMCACHE_USE_EXPERIMENTAL=True
    export LMCACHE_CONFIG_FILE=$(dirname $0)/lmcache_xpu_test_config.yaml
    python tests/sglang_xpu_integration_test.py
"""

import os
import sys

# -- environment must be set BEFORE any lmcache import --
os.environ.setdefault("LMCACHE_USE_EXPERIMENTAL", "True")
_dir = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault(
    "LMCACHE_CONFIG_FILE",
    os.path.join(_dir, "lmcache_xpu_test_config.yaml"),
)

import torch

# Bail out early if no XPU
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    print("SKIP: torch.xpu not available")
    sys.exit(0)

from lmcache.integration.sglang.sglang_adapter import (
    LMCacheLayerwiseConnector,
    LoadMetadata,
    StoreMetadata,
)
from sglang.srt.configs.model_config import ModelConfig


def test_store_and_retrieve_xpu(
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    buffer_size: int = 256,
    input_id_len: int = 16,
    device_id: int = 0,
):
    """Full roundtrip: store KV on XPU → retrieve → verify equality."""
    device = torch.device(f"xpu:{device_id}")
    torch.xpu.set_device(device)

    model_config = ModelConfig(model_path=model_path)
    head_num = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    layer_num = model_config.num_hidden_layers

    print(f"Model: {model_path}")
    print(f"  layers={layer_num}, kv_heads={head_num}, head_dim={head_dim}")
    print(f"  buffer_size={buffer_size}, input_id_len={input_id_len}")
    print(f"  device={device}")

    # --- Create KV buffers on XPU ---
    k_buffer = [
        torch.randn(buffer_size, head_num, head_dim, dtype=torch.bfloat16, device=device)
        for _ in range(layer_num)
    ]
    v_buffer = [
        torch.randn(buffer_size, head_num, head_dim, dtype=torch.bfloat16, device=device)
        for _ in range(layer_num)
    ]

    # --- Init connector ---
    connector = LMCacheLayerwiseConnector(
        model_config, tp_size=1, rank=0, k_pool=k_buffer, v_pool=v_buffer
    )

    # --- Fake token IDs and slot mapping ---
    fake_token_ids = torch.randint(0, model_config.vocab_size, (input_id_len,)).tolist()
    fake_kv_indices = torch.randint(0, buffer_size, (input_id_len,))
    offset = 0

    # --- Ground truth: snapshot the KV values at the chosen indices ---
    gt_k = [k_buffer[i][fake_kv_indices].clone() for i in range(layer_num)]
    gt_v = [v_buffer[i][fake_kv_indices].clone() for i in range(layer_num)]

    # ========== Step 1: Store KV into LMCache ==========
    store_meta = StoreMetadata(
        last_node=None,
        token_ids=fake_token_ids,
        kv_indices=fake_kv_indices,
        offset=offset,
    )
    connector.store_kv(store_meta)
    torch.xpu.synchronize(device)
    print("  Store: OK")

    # ========== Step 2: Zero out KV buffers ==========
    for i in range(layer_num):
        k_buffer[i].zero_()
        v_buffer[i].zero_()

    # ========== Step 3: Retrieve from LMCache ==========
    load_meta = LoadMetadata(
        token_ids=fake_token_ids,
        slot_mapping=fake_kv_indices,
        offset=offset,
    )

    retrieve_count = connector.start_load_kv(load_meta)
    assert retrieve_count == input_id_len, (
        f"Expected {input_id_len} retrieved, got {retrieve_count}"
    )
    print(f"  Retrieve: {retrieve_count} tokens")

    for layer_id in range(layer_num):
        connector.load_kv_layerwise(layer_id)

    torch.xpu.synchronize(device)

    # ========== Step 4: Verify ==========
    for i in range(layer_num):
        k_loaded = k_buffer[i][fake_kv_indices]
        v_loaded = v_buffer[i][fake_kv_indices]
        assert torch.allclose(k_loaded, gt_k[i]), (
            f"K mismatch at layer {i}: max diff={( k_loaded - gt_k[i]).abs().max()}"
        )
        assert torch.allclose(v_loaded, gt_v[i]), (
            f"V mismatch at layer {i}: max diff={(v_loaded - gt_v[i]).abs().max()}"
        )

    print("  Verify: ALL LAYERS MATCH")
    connector.close()
    print("PASSED: test_store_and_retrieve_xpu")


def test_cold_load_returns_zero():
    """First load (before any store) should return 0 tokens."""
    device = torch.device("xpu:0")
    torch.xpu.set_device(device)

    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    model_config = ModelConfig(model_path=model_path)
    head_num = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    layer_num = model_config.num_hidden_layers

    k_buffer = [
        torch.randn(64, head_num, head_dim, dtype=torch.bfloat16, device=device)
        for _ in range(layer_num)
    ]
    v_buffer = [
        torch.randn(64, head_num, head_dim, dtype=torch.bfloat16, device=device)
        for _ in range(layer_num)
    ]

    connector = LMCacheLayerwiseConnector(
        model_config, tp_size=1, rank=0, k_pool=k_buffer, v_pool=v_buffer
    )

    # Random unseen tokens → should return 0
    fake_token_ids = torch.randint(0, model_config.vocab_size, (8,)).tolist()
    load_meta = LoadMetadata(
        token_ids=fake_token_ids,
        slot_mapping=torch.arange(8),
        offset=0,
    )
    count = connector.start_load_kv(load_meta)
    assert count == 0, f"Expected 0 retrieved for unseen tokens, got {count}"
    print("PASSED: test_cold_load_returns_zero")
    connector.close()


if __name__ == "__main__":
    test_cold_load_returns_zero()
    test_store_and_retrieve_xpu()
