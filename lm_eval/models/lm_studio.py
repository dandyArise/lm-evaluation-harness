import logging
import requests
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences

eval_logger = logging.getLogger(__name__)

@register_model("lm_studio")
class LMStudioLM(TemplateAPI):
    def __init__(
        self,
        base_url="http://localhost:1234",
        api_key="sk-lm-3gwCCdX2:WbSQQH2bebhEvhXM79vs",
        tokenizer_backend=None,
        **kwargs,
    ):
        """
        LM Studio wrapper using native /api/v1 endpoints.
        """
        super().__init__(
            base_url=base_url,
            auth_token=api_key,
            tokenizer_backend=tokenizer_backend,
            **kwargs
        )
        self.api_key_str = api_key
        # Ensure base_url doesn't end with a slash for easier concatenation
        self.base_url = base_url.rstrip("/")
        self.latencies = [] # Store (task, latency) triples

    @property
    def api_key(self):
        return self.api_key_str

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs = gen_kwargs or {}
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            
            # LM Studio native /api/v1/chat uses "input"
            # messages[0] should be the prompt string if not tokenized
            prompt = messages
            if isinstance(messages, list) and len(messages) > 0:
                prompt = messages[0]

            return {
                "model": self.model,
                "input": prompt,
                "context_length": self.max_length,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            # Loglikelihood not supported by native chat API usually
            return {}

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        # LM Studio native response: data["output"][0]["content"][0]["text"]
        # or it might be a list of outputs
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                # Based on user first prompt: data["output"][0]["content"][0]["text"]
                # template_api might pass the full JSON
                if "output" in out:
                    content = out["output"][0]["content"]
                    if isinstance(content, list):
                        res.append(content[0]["text"])
                    else:
                        res.append(content)
                elif "choices" in out: # Fallback to OpenAI style just in case
                    res.append(out["choices"][0]["message"]["content"])
            except Exception as e:
                eval_logger.warning(f"Could not parse generations: {e}. Output was: {out}")
                res.append("")
        return res

    @staticmethod
    def parse_logprobs(outputs, **kwargs):
        raise NotImplementedError("Loglikelihood is not supported for LM Studio native API.")

    def model_call(self, messages, generate=True, gen_kwargs=None, **kwargs):
        # Override to use /api/v1/chat
        url = f"{self.base_url}/api/v1/chat"
        payload = self._create_payload(messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        eval_logger.debug(f"Sending request to {url} with payload: {payload}")
        start_time = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers)
            latency = time.time() - start_time
            if not resp.ok:
                eval_logger.error(f"API request failed with status {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            eval_logger.debug(f"Received response: {data}")
            
            # Store latency (we don't know the task here easily, so we just store the value)
            self.latencies.append(latency)
            
            return data
        except Exception as e:
            eval_logger.error(f"API request failed: {e}")
            return None

    def load_model(self, model_id: str, **kwargs):
        """
        Explicitly load a model in LM Studio.
        Endpoint: POST /api/v1/models/load
        """
        url = f"{self.base_url}/api/v1/models/load"
        payload = {
            "model": model_id,
            "context_length": kwargs.get("context_length", 8000),
            **kwargs
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            eval_logger.info(f"Successfully loaded model: {model_id}")
            return resp.json()
        except Exception as e:
            eval_logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def unload_model(self, model_id: str):
        """
        Explicitly unload a model in LM Studio.
        Endpoint: POST /api/v1/models/unload
        """
        url = f"{self.base_url}/api/v1/models/unload"
        payload = {"instance_id": model_id}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            eval_logger.info(f"Successfully unloaded model: {model_id}")
            return resp.json()
        except Exception as e:
            eval_logger.error(f"Failed to unload model {model_id}: {e}")
            raise
