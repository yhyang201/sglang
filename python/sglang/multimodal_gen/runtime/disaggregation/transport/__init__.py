# SPDX-License-Identifier: Apache-2.0
"""
Transport layer for disaggregated diffusion pipelines.

Organized by transport mode:
  - rdma/     — P2P transfer via RDMA/Mooncake (direct instance-to-instance)

Shared modules:
  - tensor_codec.py  — Zero-copy tensor serialization for ZMQ
  - role_connector.py — Field manifests + extract/apply helpers
  - p2p_protocol.py  — P2P control message definitions
"""
