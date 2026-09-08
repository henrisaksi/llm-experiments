import os
import httpx
import chainlit as cl
from chainlit.input_widget import Select

LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm_service:8000")


async def fetch_models():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LLM_SERVICE_URL}/models")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("models", []), data.get("current_model")
    except Exception:
        pass
    return [], None


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])

    models, current = await fetch_models()
    if not models:
        models = ["gemma-4-E2B-it"]
        current = "gemma-4-E2B-it"
    elif not current:
        current = models[0]

    cl.user_session.set("selected_model", current)

    # Allow user to switch models via Chainlit chat settings
    await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model Weight Checkpoint",
                values=models,
                initial_value=current,
                description="Select which compatible weight checkpoint to use."
            )
        ]
    ).send()

    await cl.Message(
        content=f"👋 Connected to LLM Service.\nCurrently active model: **`{current}`**.\nYou can change the model anytime in chat settings (gear icon)."
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    new_model = settings.get("model")
    if not new_model:
        return

    old_model = cl.user_session.get("selected_model")
    if new_model == old_model:
        return

    switch_msg = await cl.Message(content=f"Switching model checkpoint to **`{new_model}`**...").send()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{LLM_SERVICE_URL}/models/switch",
                json={"model": new_model}
            )
            if resp.status_code == 200:
                cl.user_session.set("selected_model", new_model)
                switch_msg.content = f"Active model checkpoint changed to **`{new_model}`**."
                await switch_msg.update()
            else:
                switch_msg.content = f"Failed to switch model: {resp.text}"
                await switch_msg.update()
    except Exception as e:
        switch_msg.content = f"Error switching model: {e}"
        await switch_msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    messages = cl.user_session.get("messages", [])
    selected_model = cl.user_session.get("selected_model")
    messages.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    payload = {
        "model": selected_model,
        "messages": messages,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    assistant_reply = ""
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{LLM_SERVICE_URL}/generate/stream",
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    await msg.stream_token(f"Error from LLM backend: {error_text.decode('utf-8', errors='ignore')}")
                    await msg.update()
                    return

                async for chunk in response.aiter_text():
                    assistant_reply += chunk
                    await msg.stream_token(chunk)

        messages.append({"role": "assistant", "content": assistant_reply})
        cl.user_session.set("messages", messages)
        await msg.update()

    except httpx.TimeoutException:
        await msg.stream_token("\n[Request timed out while waiting for model generation. For long responses, consider reducing max_new_tokens.]")
        await msg.update()
    except Exception as e:
        err_msg = str(e) if str(e) else repr(e)
        await msg.stream_token(f"\n[Connection error: {err_msg}]")
        await msg.update()
