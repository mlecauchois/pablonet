import asyncio
import websockets
import cv2
import base64
import numpy as np
import PIL.Image
import json


async def capture_and_send(
    url, prompt, negative_prompt, image_size=256, rotate=0, fullscreen=False
):
    uri = url
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)

        if fullscreen:
            cv2.setWindowProperty(
                "image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

        print("Connected to server...")

        # Send prompt configuration as JSON
        await websocket.send(
            json.dumps(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }
            )
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop frame as square
            h, w, _ = frame.shape
            min_side = min(h, w)
            frame = frame[
                h // 2 - min_side // 2 : h // 2 + min_side // 2,
                w // 2 - min_side // 2 : w // 2 + min_side // 2,
            ]

            # Resize and convert to RGB bytes
            frame = cv2.resize(frame, (image_size, image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Send raw bytes directly
            await websocket.send(frame.tobytes())

            # Receive raw image bytes
            response = await websocket.recv()

            # Convert bytes directly to numpy array
            source = np.frombuffer(response, dtype=np.uint8).reshape(512, 512, 3)
            source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)

            # Flip horizontally
            source = cv2.flip(source, 1)

            # Resize for display
            source = cv2.resize(source, (1400, 1400), interpolation=cv2.INTER_CUBIC)

            # Rotate if needed
            if rotate != 0:
                (h, w) = source.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, rotate, 1.0)
                source = cv2.warpAffine(source, M, (w, h))

            cv2.imshow("image", source)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.0001)

        cap.release()
        cv2.destroyAllWindows()


# Command args cli

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="URL of server", default="")
    parser.add_argument("--prompt", type=str, help="Prompt to send to server")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        help="Negative prompt to send to server",
        default="low quality",
    )
    parser.add_argument("--image_size", type=int, help="Image size", default=256)
    parser.add_argument(
        "--rotate", type=float, default=0, help="Rotate the image by specified degrees"
    )
    parser.add_argument(
        "--fullscreen", action="store_true", help="Display window in fullscreen mode"
    )
    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(
        capture_and_send(
            args.url,
            args.prompt,
            args.negative_prompt,
            args.image_size,
            args.rotate,
            args.fullscreen,
        )
    )
