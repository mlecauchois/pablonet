import asyncio
import websockets
import cv2
import numpy as np
import json


async def capture_and_send(
    url,
    prompt,
    negative_prompt,
    image_size=256,
    rotate=0,
    fullscreen=False,
    jpeg_quality=90,
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

            # Resize and convert to RGB
            frame = cv2.resize(frame, (image_size, image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Encode frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            result, encimg = cv2.imencode(".jpg", frame, encode_param)
            if not result:
                print("Failed to encode image")
                continue
            jpeg_bytes = encimg.tobytes()

            # Send JPEG bytes
            await websocket.send(jpeg_bytes)

            # Receive JPEG bytes from server
            response = await websocket.recv()

            # Decode JPEG bytes to image
            if isinstance(response, bytes):
                decimg = np.frombuffer(response, dtype=np.uint8)
                frame_decoded = cv2.imdecode(decimg, cv2.IMREAD_COLOR)
                if frame_decoded is None:
                    print("Failed to decode received image")
                    continue

                # Flip horizontally
                frame_decoded = cv2.flip(frame_decoded, 1)

                # Resize for display
                frame_decoded = cv2.resize(
                    frame_decoded, (1400, 1400), interpolation=cv2.INTER_CUBIC
                )

                # Rotate if needed
                if rotate != 0:
                    (h_dec, w_dec) = frame_decoded.shape[:2]
                    center = (w_dec / 2, h_dec / 2)
                    M = cv2.getRotationMatrix2D(center, rotate, 1.0)
                    frame_decoded = cv2.warpAffine(frame_decoded, M, (w_dec, h_dec))

                cv2.imshow("image", frame_decoded)

            else:
                print("Received non-bytes message from server")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.0001)

        cap.release()
        cv2.destroyAllWindows()


# Command args cli

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
        "--rotate", type=float, default=0, help="Rotate the image by specified degrees"
    )
    parser.add_argument(
        "--fullscreen", action="store_true", help="Display window in fullscreen mode"
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=90, help="JPEG compression quality (1-100)"
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
            args.jpeg_quality,
        )
    )
