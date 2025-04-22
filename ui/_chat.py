import os
from collections.abc import AsyncIterator
from typing import Any, Dict, Union, cast

import httpx
from dotenv import load_dotenv
import json

from llm import llm_client

# from mindmesh.entity import ID

load_dotenv(".env")

DONE_MSG = "<|DONE|>"


async def chat(session_id: Union[str, int], user_query: str, image: str) -> Union[Dict[str, Any], str]:
    """
    Gửi câu hỏi tới mô hình và nhận kết quả trả về.

    Args:
        session_id (str|int): ID phiên người dùng
        user_query (str): Câu hỏi của người dùng
        image (str): Đường dẫn hoặc base64 ảnh (hiện không dùng)

    Returns:
        Union[Dict[str, Any], str]: Phản hồi từ mô hình
    """
    # try:
    # Gọi client nội bộ (không dùng HTTP)
    response = await llm_client.achat(
        model_name=os.getenv("LLM_MODEL_NAME"),
        messages=[{"role": "user", "content": user_query}],
    )

    # Nếu response là dict, trả về luôn
    if isinstance(response, dict):
        return response

    # Nếu response là httpx.Response
    if hasattr(response, "json"):
        return cast(Dict[str, Any], response.json())
    
    if isinstance(response, str):
        # Nếu response là string, trả về dạng dict
        return {"response": response}

    raise RuntimeError("Unexpected response format from llm_client")

    # except Exception as e:
    #     raise RuntimeError(f"Chat failed: {e}") from e




async def chat_streaming(session_id: Union[str, int], user_query: str, image: str) -> AsyncIterator[str]:
#     """
#     Gọi llm_client.achat để lấy kết quả và yield 1 lần dạng streaming chunk.

#     Args:
#         session_id (str | int): ID phiên người dùng
#         user_query (str): Prompt từ người dùng
#         image (str): ảnh encode dạng base64 (chưa sử dụng)

#     Yields:
#         str: Chuỗi JSON bọc trong prefix "data: " (dùng được với main())
#     """
#     async for chunk in llm_client.stream_chat(
#         model_name=os.getenv("LLM_MODEL_NAME"),
#         messages=[{"role": "user", "content": user_query}]
#     ):
#         yield chunk
    pass
