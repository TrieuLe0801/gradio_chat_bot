import base64
import json
import mimetypes
import uuid
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Switch
from pydantic import BaseModel

from ui._chat import chat, chat_streaming


DONE_MSG = "<|DONE|>"

class ChunkMessage(BaseModel):
    message: str | dict[str, Any]
    index: int


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        encoded_image = f"data:{image_type};base64,{base64_image}"
        return encoded_image


@cl.on_message
async def main(message: cl.Message) -> None:
    encoded_image = ""
    if message.elements:
        # Filter and process images
        images = [file for file in message.elements if "image" in file.mime]
        if images:
            image_path = images[0].path
            encoded_image = encode_image(image_path)

    session_id = str(uuid.uuid4())
    settings = cl.user_session.get("settings")
    streaming = settings.get("Streaming", False)
    is_show_step = settings.get("Show step", False)

    if streaming:
            # Khởi tạo tin nhắn đang stream
            msg = cl.Message(content="")
            await msg.send()

            async for chunk in chat_streaming(
                session_id=session_id,
                user_query=message.content,
                image=encoded_image
            ):
                if chunk == DONE_MSG:
                    break
                if chunk.startswith("data: "):
                    json_data = json.loads(chunk[6:])
                    try:
                        data = ChunkMessage(**json_data)
                        await msg.stream_token(str(data.message))  # type: ignore
                    except Exception as e:
                        await msg.stream_token(f"[Error parsing chunk: {e}]")
                elif chunk.startswith("state: ") and is_show_step:
                    steps = json.loads(chunk[7:])
                    for step in steps:
                        await show_step(step)
            await msg.update()
    else:
        res = await chat(session_id=session_id, user_query=message.content, image=encoded_image)
        if is_show_step:
            for step in res["steps"]:
                await show_step(step)
        await cl.Message(content=res["response"]).send()


@cl.step(language="json", show_input=False)
async def show_step(data: dict[str, Any]) -> None:
    cl.context.current_step.name = "END" if data["kind"] == "end" else data["node"]["node_id"]
    cl.context.current_step.output = json.dumps(data, indent=4, ensure_ascii=False)


@cl.on_chat_start
async def start() -> None:
    settings = await cl.ChatSettings(
        [
            # Select(
            #     id="Model", label="OpenAI - Model", values=["openai:gpt-4o"] * 2, initial_index=1
            # ),
            # Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=False),
            Switch(id="Show step", label="Show step", initial=False),
        ]
    ).send()
    cl.user_session.set("settings", settings)
