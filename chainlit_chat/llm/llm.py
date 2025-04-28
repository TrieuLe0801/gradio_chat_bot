import asyncio
import base64
import mimetypes
import os
import re
from abc import ABC, abstractmethod

import litellm
from dotenv import load_dotenv
# from config import *
from loguru import logger

# from openai import AsyncClient
# from PIL import Image


load_dotenv()


class LMMClient(ABC):
    """
    Large MultiModal Model
    """

    def __init__(self, **kwargs):
        self.model_name = os.getenv("LLM_MODEL_NAME")
        self.api_base = os.getenv("LMM_API_BASE")
        self.api_key = os.getenv("LMM_API_KEY")

    @abstractmethod
    async def achat(self, model_name: str, messages: list, **kwargs):
        pass


class LLMClient(LMMClient):
    """Large Language Model"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #    self.loaded_models = {}  # Apply singleton for loading model

    # def chat(self, model_name: str, messages: list = "", **kwargs):
    #     """Handles text-based LLMs (Qwen, GPT, Mistral,..)"""
    #     try:
    #         if model_name not in self.model_hubs:
    #             raise Exception(f"Model {model_name} is not supported")

    #         response = litellm.completion(
    #             model=model_name,
    #             messages=messages,
    #             api_base=self.api_base,
    #             api_key=self.api_key,
    #             **kwargs,
    #         )
    #         return response["choices"][0]["message"]["content"]

    #     except Exception as e:
    #         raise e

    async def achat(self, model_name: str, messages: list, **kwargs):
        """
        Handles text-based LLMs (Qwen, GPT, Mistral,..) asynchronously
        """
        try:
            if model_name != self.model_name:
                logger.warning(
                    f"Model {model_name} is not supported. Use {self.model_name} instead."
                )
                model_name = self.model_name

            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                api_base=self.api_base,
                api_key=self.api_key,
                **kwargs,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise e
