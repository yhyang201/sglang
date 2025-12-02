# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""


from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class GeneralBeforeDenoisingStage(PipelineStage):
    def __init__(self, processor, text_encoder):
        super().__init__()
        self.processor = processor
        self.text_encoder = text_encoder

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        print(f"GeneralBeforeDenoisingStage forward", flush=True)
        print(f"self.processor={self.processor}", flush=True)
        print(f"self.text_encoder={self.text_encoder}", flush=True)
        batch = server_args.pipeline_config.before_denosing(
            batch, server_args, self.processor, self.text_encoder
        )
        return batch
