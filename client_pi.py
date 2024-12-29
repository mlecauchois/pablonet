import asyncio
import websockets
import numpy as np
from PIL import Image
import tkinter as tk
import json
import os
import cv2
from picamera2 import Picamera2
import time

os.environ["DISPLAY"] = ":0"


async def capture_and_send(
    url,
    prompt,
    negative_prompt,
    image_size=256,
    fullscreen=False,
    crop_size=256,
    crop_offset_y=0,
    jpeg_quality=90,
    rotation=0,
    target_fps=30,
):
    uri = url
    async with websockets.connect(uri) as websocket:
        # Initialize picamera2
        t_start = time.time()
        picam2 = Picamera2()
        picam2.configure(
            picam2.create_preview_configuration(main={"size": (1400, 1000)})
        )
        print("Camera init time:", time.time() - t_start)

        print("Starting picamera2...")
        t_start = time.time()
        picam2.start()
        print("Camera start time:", time.time() - t_start)

        # Setup cv2 window
        t_start = time.time()
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                "image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        print("Window setup time:", time.time() - t_start)

        # Shadow tk to get screen size
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        print("Connected to server...")

        # Send prompt to server as json
        await websocket.send(
            json.dumps(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }
            )
        )

        # Initialize frame management variables
        frame_count = 0
        last_send_time = 0
        min_frame_interval = 1 / target_fps
        last_stats_time = time.time()
        frames_since_stats = 0
        last_displayed_frame = None

        while True:
            loop_start = time.time()

            # Capture frame
            t_start = time.time()
            frame = picam2.capture_array()
            capture_time = time.time() - t_start

            # Check if we should process this frame based on target FPS
            current_time = time.time()
            if current_time - last_send_time < min_frame_interval:
                # If we have a previous frame, display it to maintain smooth output
                if last_displayed_frame is not None:
                    cv2.imshow("image", last_displayed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            t_start = time.time()
            h, w, _ = frame.shape
            frame = Image.fromarray(frame)
            frame = frame.convert("RGB")
            pil_convert_time = time.time() - t_start

            if rotation != 0:
                t_start = time.time()
                frame = frame.rotate(rotation)
                rotation_time = time.time() - t_start
            else:
                rotation_time = 0

            t_start = time.time()
            frame = np.array(frame)
            np_convert_time = time.time() - t_start

            # Crop square
            t_start = time.time()
            frame = frame[
                h // 2
                - crop_size // 2
                - crop_offset_y : h // 2
                + crop_size // 2
                - crop_offset_y,
                w // 2 - crop_size // 2 : w // 2 + crop_size // 2,
            ]
            crop_time = time.time() - t_start

            # Reduce size
            t_start = time.time()
            frame = cv2.resize(frame, (image_size, image_size))
            resize_time = time.time() - t_start

            # Encode frame
            t_start = time.time()
            compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            result, buffer = cv2.imencode(".jpg", frame, compression_params)
            encode_time = time.time() - t_start

            if not result:
                print("Failed to encode image")
                continue

            # Send to server
            t_start = time.time()
            jpeg_bytes = buffer.tobytes()
            await websocket.send(jpeg_bytes)
            last_send_time = current_time
            send_time = time.time() - t_start

            # Network receive with timeout
            t_start = time.time()
            try:
                response = await asyncio.wait_for(
                    websocket.recv(), timeout=1 / target_fps
                )
                receive_time = time.time() - t_start

                # Process received data
                t_start = time.time()
                if isinstance(response, bytes):
                    npimg = np.frombuffer(response, dtype=np.uint8)
                    source = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                    if source is None:
                        print("Failed to decode received image")
                        continue

                    source = cv2.flip(source, 1)

                    # Crop to screen aspect ratio and resize
                    aspect_ratio = screen_width / screen_height
                    source_aspect_ratio = source.shape[1] / source.shape[0]

                    if source_aspect_ratio > aspect_ratio:
                        crop_width = int(source.shape[0] * aspect_ratio)
                        crop_offset = (source.shape[1] - crop_width) // 2
                        source = source[:, crop_offset : crop_offset + crop_width]
                    else:
                        crop_height = int(source.shape[1] / aspect_ratio)
                        crop_offset = (source.shape[0] - crop_height) // 2
                        source = source[crop_offset : crop_offset + crop_height, :]

                    source = cv2.resize(
                        source,
                        dsize=(screen_width, screen_height),
                        interpolation=cv2.INTER_CUBIC,
                    )

                    cv2.imshow("image", source)
                    last_displayed_frame = source
                process_display_time = time.time() - t_start

            except asyncio.TimeoutError:
                receive_time = time.time() - t_start
                print("Server response timeout - skipping frame")
                # Display last frame if available
                if last_displayed_frame is not None:
                    cv2.imshow("image", last_displayed_frame)
                continue

            total_loop_time = time.time() - loop_start

            # Update statistics
            frames_since_stats += 1
            current_time = time.time()
            stats_interval = current_time - last_stats_time

            # Print timing every 30 frames or every second, whichever comes first
            if frames_since_stats >= 30 or stats_interval >= 1.0:
                print("\nTiming breakdown (seconds):")
                print(f"Capture: {capture_time:.4f}")
                print(f"PIL convert: {pil_convert_time:.4f}")
                print(f"Rotation: {rotation_time:.4f}")
                print(f"Numpy convert: {np_convert_time:.4f}")
                print(f"Crop: {crop_time:.4f}")
                print(f"Resize: {resize_time:.4f}")
                print(f"Encode: {encode_time:.4f}")
                print(f"Send: {send_time:.4f}")
                print(f"Receive: {receive_time:.4f}")
                print(f"Process & Display: {process_display_time:.4f}")
                print(f"Total loop time: {total_loop_time:.4f}")
                print(f"Actual FPS: {frames_since_stats/stats_interval:.2f}")
                print("-" * 40)

                # Reset statistics
                frames_since_stats = 0
                last_stats_time = current_time

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.001)  # Small sleep to prevent CPU overload

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", type=str, help="URL of server", default="ws://localhost:5678"
    )
    parser.add_argument(
        "--prompt", type=str, help="Prompt to send to server", required=True
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        help="Negative prompt to send to server",
        default="low quality",
    )
    parser.add_argument("--image_size", type=int, help="Image size", default=256)
    parser.add_argument(
        "--fullscreen", action="store_true", help="Display window in fullscreen mode"
    )
    parser.add_argument(
        "--crop_size", type=int, default=256, help="Crop size of the image"
    )
    parser.add_argument(
        "--crop_offset_y",
        type=int,
        default=0,
        help="Offset of the crop from the top of the image",
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=90, help="JPEG compression quality"
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="Rotation of the camera image in degrees",
    )
    parser.add_argument(
        "--target_fps", type=int, default=30, help="Target FPS for frame capture"
    )

    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(
        capture_and_send(
            args.url,
            args.prompt,
            args.negative_prompt,
            args.image_size,
            args.fullscreen,
            args.crop_size,
            args.crop_offset_y,
            args.jpeg_quality,
            args.rotation,
            args.target_fps,
        )
    )
