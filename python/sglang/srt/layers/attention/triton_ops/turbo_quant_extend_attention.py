# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TurboQuant extend/prefill attention.

For extend (prefill), we dequantize the cached keys and reuse the standard
extend attention kernel. This is acceptable because:
  1. Extend is compute-bound (not memory-bound), so dequantization cost is small.
  2. Extend happens once per prompt, while decode runs at every generation step.
  3. The memory saving from TurboQuant is realized in the KV cache storage,
     not in the temporary dequantized buffer used during extend.

The main benefit of TurboQuant is in the decode phase where the full KV cache
is read every step.
"""

# This module delegates to the standard extend kernel.
# The dequantization is handled by MHATokenToKVPoolTurboQuant._get_key_buffer()
# which returns a dequantized float key buffer on-demand.
#
# No custom Triton kernel needed for extend.
