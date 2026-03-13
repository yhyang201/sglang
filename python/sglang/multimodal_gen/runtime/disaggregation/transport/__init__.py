# SPDX-License-Identifier: Apache-2.0
"""
Transport layer for disaggregated diffusion pipelines.

Organized by transport mode:
  - relay/    — ZMQ-based tensor serialization (DiffusionServer relays data)
  - rdma/     — P2P transfer via RDMA/Mooncake (direct instance-to-instance)

Shared modules:
  - role_connector.py  — Field manifests + relay pack/unpack helpers
  - p2p_protocol.py    — P2P control message definitions
"""
