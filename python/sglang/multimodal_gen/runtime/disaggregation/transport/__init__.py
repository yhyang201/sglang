# SPDX-License-Identifier: Apache-2.0
"""
Transport layer for disaggregated diffusion pipelines.

Organized by transport mode:
  - rdma/     — RDMA transfer via Mooncake (direct instance-to-instance)

Shared modules:
  - tensor_codec.py      — Zero-copy tensor serialization for ZMQ
  - role_connector.py    — Field manifests + extract/apply helpers
  - transfer_protocol.py — Transfer control message definitions
"""
