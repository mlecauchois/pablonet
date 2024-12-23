import fire
import cv2
import numpy as np
import time
import asyncio
import websockets
import cv2
import numpy as np
import base64
import json

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


import PIL.Image


def load_model(
    base_model_path, acceleration, lora_path=None, lora_scale=1.0, t_index_list=None
):
    pipe = StableDiffusionPipeline.from_pretrained(base_model_path).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    stream = StreamDiffusion(
        pipe,
        t_index_list=t_index_list,
        torch_dtype=torch.float16,
        width=512,
        height=512,
    )

    stream.enable_similar_image_filter(
        0.98,
        10,
    )

    stream.load_lcm_lora()
    stream.fuse_lora()

    if lora_path is not None:
        stream.load_lora(lora_path)
        stream.fuse_lora(lora_scale=lora_scale)
        print(f"Using LoRA: {lora_path}")

    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        device=pipe.device, dtype=pipe.dtype
    )

    if acceleration == "tensorrt":
        stream = accelerate_with_tensorrt(
            stream,
            "engines",
            max_batch_size=2,
        )
    else:
        pipe.enable_xformers_memory_efficient_attention()

    return stream


async def process_image(
    websocket,
    stream,
    prompt,
    num_inference_steps=50,
    preprocessing=None,
    negative_prompt="",
    guidance_scale=1.2,
):
    # Prepare the stream
    stream.prepare(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )

    async for message in websocket:
        try:
            # Check if message is JSON configuration
            message_data = json.loads(message)
            new_prompt = message_data["prompt"]
            new_negative_prompt = message_data["negative_prompt"]
            stream.prepare(
                prompt=new_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=new_negative_prompt,
            )
            continue
        except json.JSONDecodeError:
            # Message is raw image bytes
            pass

        # Convert bytes directly to numpy array
        img = np.frombuffer(message, dtype=np.uint8).reshape(256, 256, 3)

        if preprocessing == "canny":
            img = cv2.Canny(img, 100, 200)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif preprocessing == "canny_blur_shift":
            blur_img = cv2.GaussianBlur(img, (3, 3), 0)
            canny_img = cv2.Canny(blur_img, 100, 200)
            canny_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
            canny_img[np.where((canny_img == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
            img = cv2.addWeighted(img, 0.8, canny_img, 0.2, 0)
        elif preprocessing == "blur":
            img = cv2.GaussianBlur(img, (5, 5), 0)
        elif preprocessing == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif preprocessing == "contrast":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Convert to PIL
        img = PIL.Image.fromarray(img)

        # Process through diffusion model
        x_output = stream(img)
        output = postprocess_image(x_output, output_type="pil")[0]

        # Convert to numpy array and send raw bytes
        output_array = np.array(output)
        await websocket.send(output_array.tobytes())


def run_server(
    base_model_path,
    acceleration,
    prompt,
    host="0.0.0.0",
    port=5678,
    num_inference_steps=50,
    preprocessing=None,
    negative_prompt="",
    guidance_scale=1.2,
    lora_path=None,
    lora_scale=1.0,
    t_index_list=None,
    compression=30,
):
    stream = load_model(
        base_model_path, acceleration, lora_path, lora_scale, t_index_list
    )

    start_server = websockets.serve(
        lambda ws: process_image(
            ws,
            stream,
            prompt,
            num_inference_steps,
            preprocessing,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            compression=compression,
        ),
        host,
        port,
    )

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    fire.Fire(run_server)
