import asyncio
import websockets
import numpy as np
from PIL import Image
import tkinter as tk
import json
import io
import cv2
from picamera2 import Picamera2

# Removed base64 import as it's no longer needed


async def capture_and_send(
    url,
    prompt,
    negative_prompt,
    image_size=256,
    fullscreen=False,
    crop_size=256,
    crop_offset_y=0,
    compression=90,
    rotation=0,
):
    uri = url
    async with websockets.connect(uri) as websocket:
        # Initialize picamera2
        picam2 = Picamera2()
        # Full resolution preview
        picam2.configure(
            picam2.create_preview_configuration(main={"size": (1400, 1000)})
        )
        print("Starting picamera2...")
        picam2.start()

        # Setup cv2 window
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                "image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

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

        while True:
            # Capture frame
            frame = picam2.capture_array()
            h, w, _ = frame.shape

            frame = Image.fromarray(frame)
            frame = frame.convert("RGB")

            if rotation != 0:
                frame = frame.rotate(rotation)

            frame = np.array(frame)

            # Crop square of crop_size in the middle with offset
            frame = frame[
                h // 2
                - crop_size // 2
                - crop_offset_y : h // 2
                + crop_size // 2
                - crop_offset_y,
                w // 2 - crop_size // 2 : w // 2 + crop_size // 2,
            ]

            # Reduce size
            frame = cv2.resize(frame, (image_size, image_size))

            # Encode frame
            compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), compression]
            result, buffer = cv2.imencode(".jpg", frame, compression_params)
            if not result:
                print("Failed to encode image")
                continue

            # Convert to bytes (Removed Base64 encoding)
            jpeg_bytes = buffer.tobytes()

            # Send to server (Send raw bytes instead of Base64 string)
            await websocket.send(jpeg_bytes)

            # Receive and display image
            response = await websocket.recv()

            # Process received data as bytes (Removed Base64 decoding)
            if isinstance(response, bytes):
                npimg = np.frombuffer(response, dtype=np.uint8)
                source = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                if source is None:
                    print("Failed to decode received image")
                    continue

                # Flip horizontally
                source = cv2.flip(source, 1)

                # Crop to screen aspect ratio and resize
                aspect_ratio = screen_width / screen_height
                source_aspect_ratio = source.shape[1] / source.shape[0]

                if source_aspect_ratio > aspect_ratio:
                    # Crop width
                    crop_width = int(source.shape[0] * aspect_ratio)
                    crop_offset = (source.shape[1] - crop_width) // 2
                    source = source[:, crop_offset : crop_offset + crop_width]
                else:
                    # Crop height
                    crop_height = int(source.shape[1] / aspect_ratio)
                    crop_offset = (source.shape[0] - crop_height) // 2
                    source = source[crop_offset : crop_offset + crop_height, :]
                source = cv2.resize(
                    source,
                    dsize=(screen_width, screen_height),
                    interpolation=cv2.INTER_CUBIC,
                )

                cv2.imshow("image", source)
            else:
                print("Received non-bytes message from server")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Minimal sleep to yield control
            await asyncio.sleep(0.0001)

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
        "--compression", type=int, default=90, help="JPEG compression quality"
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="Rotation of the camera image in degrees",
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
            args.compression,
            args.rotation,
        )
    )
