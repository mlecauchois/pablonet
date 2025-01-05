import fire
import cv2
import numpy as np
import asyncio
import websockets
import json
import PIL.Image
import time
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


def load_model(
    base_model_path,
    acceleration,
    lora_path=None,
    lora_scale=1.0,
    t_index_list=None,
    engines_dir="engines",
):
    t_start = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(base_model_path).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    print(f"Pipeline load time: {time.time() - t_start:.4f}s")

    t_start = time.time()
    stream = StreamDiffusion(
        pipe,
        t_index_list=t_index_list,
        torch_dtype=torch.float16,
        width=512,
        height=512,
    )
    print(f"Stream initialization time: {time.time() - t_start:.4f}s")

    t_start = time.time()
    stream.enable_similar_image_filter(0.98, 10)
    stream.load_lcm_lora()
    stream.fuse_lora()
    print(f"LCM LoRA setup time: {time.time() - t_start:.4f}s")

    if lora_path is not None:
        t_start = time.time()
        stream.load_lora(lora_path)
        stream.fuse_lora(lora_scale=lora_scale)
        print(f"Custom LoRA load time: {time.time() - t_start:.4f}s")
        print(f"Using LoRA: {lora_path}")

    t_start = time.time()
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        device=pipe.device, dtype=pipe.dtype
    )
    print(f"VAE load time: {time.time() - t_start:.4f}s")

    t_start = time.time()
    if acceleration == "tensorrt":
        stream = accelerate_with_tensorrt(
            stream,
            engines_dir,
            max_batch_size=2,
        )
    else:
        pipe.enable_xformers_memory_efficient_attention()
    print(f"Acceleration setup time: {time.time() - t_start:.4f}s")

    return stream


async def process_image(
    websocket,
    stream,
    prompt,
    num_inference_steps=50,
    preprocessing=None,
    negative_prompt="",
    guidance_scale=1.2,
    jpeg_quality=90,
):
    # Prepare the stream
    t_start = time.time()
    stream.prepare(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )
    print(f"Initial stream preparation time: {time.time() - t_start:.4f}s")

    frame_count = 0
    async for message in websocket:
        if isinstance(message, str):
            try:
                t_start = time.time()
                message_data = json.loads(message)
                new_prompt = message_data.get("prompt", prompt)
                new_negative_prompt = message_data.get(
                    "negative_prompt", negative_prompt
                )
                stream.prepare(
                    prompt=new_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=new_negative_prompt,
                )
                print(f"Prompt update time: {time.time() - t_start:.4f}s")
                continue
            except json.JSONDecodeError:
                print("Received invalid JSON configuration.")
                continue

        elif isinstance(message, bytes):
            loop_start = time.time()
            try:
                # Decode image
                t_start = time.time()
                decimg = np.frombuffer(message, dtype=np.uint8)
                img = cv2.imdecode(decimg, cv2.IMREAD_COLOR)
                decode_time = time.time() - t_start

                if img is None:
                    print("Failed to decode received JPEG image")
                    continue

                # Preprocessing
                t_start = time.time()
                if preprocessing == "canny":
                    img = cv2.Canny(img, 100, 200)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif preprocessing == "canny_blur_shift":
                    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
                    canny_img = cv2.Canny(blur_img, 100, 200)
                    canny_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
                    canny_img[np.where((canny_img == [0, 0, 0]).all(axis=2))] = [
                        255,
                        0,
                        0,
                    ]
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
                preprocess_time = time.time() - t_start

                # Convert to PIL
                t_start = time.time()
                img_pil = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                to_pil_time = time.time() - t_start

                # Process through diffusion model
                t_start = time.time()
                x_output = stream(img_pil)
                diffusion_time = time.time() - t_start

                # Post-process
                t_start = time.time()
                output = postprocess_image(x_output, output_type="pil")[0]
                output_array = np.array(output)
                postprocess_time = time.time() - t_start

                # Encode output
                t_start = time.time()
                output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                result, encimg = cv2.imencode(".jpg", output_bgr, encode_param)
                if not result:
                    print("Failed to encode output image")
                    continue
                encode_time = time.time() - t_start

                # Send result
                t_start = time.time()
                jpeg_bytes = encimg.tobytes()
                await websocket.send(jpeg_bytes)
                send_time = time.time() - t_start

                total_time = time.time() - loop_start

                # Print timing every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    torch.cuda.synchronize()  # Ensure GPU operations are complete
                    print("\nServer Timing breakdown (seconds):")
                    print(f"Decode input: {decode_time:.4f}")
                    print(f"Preprocessing: {preprocess_time:.4f}")
                    print(f"Convert to PIL: {to_pil_time:.4f}")
                    print(f"Diffusion model: {diffusion_time:.4f}")
                    print(f"Post-processing: {postprocess_time:.4f}")
                    print(f"Encode output: {encode_time:.4f}")
                    print(f"Send result: {send_time:.4f}")
                    print(f"Total processing time: {total_time:.4f}")
                    print(f"Server FPS: {1/total_time:.2f}")
                    print("-" * 40)

            except Exception as e:
                print(f"Error processing image: {e}")
                error_message = json.dumps({"error": str(e)})
                await websocket.send(error_message.encode("utf-8"))

        else:
            print("Received unknown message type.")


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
    jpeg_quality=90,
    engines_dir="engines",
):
    print("Loading model...")
    t_start = time.time()
    stream = load_model(
        base_model_path, acceleration, lora_path, lora_scale, t_index_list, engines_dir
    )
    print(f"Total model load time: {time.time() - t_start:.4f}s")

    start_server = websockets.serve(
        lambda ws: process_image(
            ws,
            stream,
            prompt,
            num_inference_steps,
            preprocessing,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            jpeg_quality=jpeg_quality,
        ),
        host,
        port,
    )

    asyncio.get_event_loop().run_until_complete(start_server)
    print(f"Server started at ws://{host}:{port}")
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    fire.Fire(run_server)
