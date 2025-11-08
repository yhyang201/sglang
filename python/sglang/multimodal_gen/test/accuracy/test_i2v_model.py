import asyncio
import base64
import subprocess
import tempfile
import time
import unittest
import uuid
import hashlib

from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.test.test_utils import wait_for_port, wait_for_video_completion

class TestT2VModelBase(unittest.TestCase):
    model_name = ""
    md5= ""

    seed = 42
    prompt = "Megatron is converting from his vehicular alternate mode to his Cybertronian robot form."
    timeout = 500
    extra_args = []

    def _create_wait_and_download(
        self, client: OpenAI, prompt: str, size: str
    ) -> bytes:

        image_url = "https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true"
        with downloaded_temp_file(
            image_url, prefix="i2v_input_", suffix=".jpg"
        ) as tmp_path:
            video = client.videos.create(
                prompt=prompt,
                input_reference=tmp_path,
                size=size,
                seconds=10,
                extra_body={"fps": 16, "num_frames": 125},
            )
        # TODO: Some combinations of num_frames and fps may cause errors and need further investigation.
        video_id = video.id
        self.assertEqual(video.status, "queued")

        video = wait_for_video_completion(client, video_id, timeout=self.timeout)
        self.assertEqual(video.status, "completed", "video generate failed")

        response = client.videos.download_content(
            video_id=video_id,
        )
        content = response.read()
        return content

    @classmethod
    def setUpClass(cls):
        cls.base_command = [
            "sglang",
            "serve",
            "--model-path",
            f"{cls.model_name}",
            "--port",
            "30010",
        ]

        process = subprocess.Popen(
            cls.base_command + cls.extra_args,
            text=True,
            bufsize=1,
        )
        cls.pid = process.pid
        wait_for_port(host="127.0.0.1", port=30010)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.pid)

    def test_http_server_basic(self):
        client = OpenAI(
            api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
        )
        content = self._create_wait_and_download(client, self.prompt, "832x480")
        content_md5 = hashlib.md5(content).hexdigest()
        print(f"content_md5: {content_md5}, self.md5: {self.md5}")
        self.assertEqual(content_md5, self.md5)
    
class TestWan2_1T2VModel(TestT2VModelBase):
    model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    md5 = "d41d8cd98f00b204e9800998ecf8427e"

class TestWan2_2T2VModel(TestT2VModelBase):
    model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    md5 = "d41d8cd98f00b204e9800998ecf8427e"

if __name__ == "__main__":
    unittest.main()
