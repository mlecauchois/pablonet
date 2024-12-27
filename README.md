# PabloNet


## Overview

<div class="image-container" style="flex: 1; text-align: left">
    <img src="frame.png"  width="200">
</div>

PabloNet is a real-time AI art frame consisting of a display with an integrated camera that performs diffusion on captured images through a remote GPU server. For purchasing information and technical insights, please visit our [blogpost](https://mlecauchois.github.io/posts/pablonet/).

## Quick Start Guide

Complete setup should take under 20 minutes.

### Device SSH access

Upon receiving your device:
-  The device comes pre-configured with an access point named `pablonet`.
- Connect to this AP using password `pablonet`.
- SSH into the device:
```bash
ssh pablonet@raspberrypi.local
```
- Connect the device to your local WiFi network to enable internet access:
```bash
sudo nmcli device wifi connect "Your_Internet_SSID" password "Your_Password" ifname wlan1
```
- From now on, you can SSH into the device both from the device's AP or your WiFi network.
- Test internet access on the device:
```bash
ping google.com
```
- Note that even if you are connected to the AP, you will have internet access on your computer through the device. However, it will be slower than connecting directly to your WiFi network.

### GPU server setup

Currently, the device is not able to do inference. We use a non-distilled Stable Diffusion under the hood which would not be able to run on a Pi. Until we distill the model and add a Jetson to future devices, it is necessary to setup a remote GPU server.

**Custom Deployment**

We released a Docker image containing all the required packages and code: [matthieulc/pablonet:latest](https://hub.docker.com/r/matthieulc/pablonet). You can use the container serving service of your choice.

**RunPod Deployment**

Here, we detail the steps to setup a server on RunPod specifically, which should be doable in under 10 minutes.

- Create a persistent storage volume in order to cache the TensorRT engine, which is GPU-specific.
- Create a new pod with:
    - GPU: RTX 4090 (recommended).
    - Volume: Mount your persistent storage at `/workspace/tensorrt_engines`.
    - Docker Image: [matthieulc/pablonet:latest](https://hub.docker.com/r/matthieulc/pablonet).
    - Exposed Ports: 22 for SSH and 6000 for the backend.
    - Add your SSH public key to access the server.
- Access the pod:
```bash
ssh -p <pod_ssh_port> root@ssh.runpod.io
```
- Activate the virtual environment:
```bash
source venv/bin/activate
```
- Launch the server with the default parameters, which should then be accessible at `ws://IP:6000`:
```bash
python pablonet/server.py --base_model_path "Lykon/DreamShaper" \
                          --acceleration "tensorrt" \
                          --prompt "" \
                          --num_inference_steps 30 \
                          --guidance_scale 1.0 \
                          --t_index_list "[14,18]" \
                          --preprocessing canny_blur_shift \
                          --jpeg_quality 80 \
                          --port 6000
```

Note: The container's entrypoint script handles SSH setup and environment configuration automatically.

### Device setup

SSH into your device after following the internet access steps above:
```bash
ssh pablonet@raspberrypi.local
```

Launch the client wth the default parameters:
```bash
python pablonet/client_pi.py --prompt "painting in the style of pablo picasso, cubism, sharp high quality painting, oil painting, mute colors red yellow orange, background of green, color explosion, abstract surrealism" \
                             --image_size 150 \
                             --url ws://URL \
                             --fullscreen \
                             --crop_size 900 \
                             --crop_offset_y 40 \
                             --jpeg_quality 50 \
                             --rotation 270 \
                             --target_fps 10
```

### Performance

With an RTX 4090 GPU and a good internet connection, these are the FPS ranges you should get:
- Server side FPS: 15-25
- Device FPS: 2-9

Below these FPS, the quality of the experience will be degraded and you should investigate potential network latency issues, or improper GPU setup. Typically, you should make sure that the TensorRT engine is properly re-compiled for your GPU.

### Debugging Client

You can setup a client on your computer using your webcam:

```bash
python pablonet/client.py --prompt "painting in the style of pablo picasso, cubism, sharp high quality painting, oil painting, mute colors red yellow orange, background of green, color explosion, abstract surrealism" \
                          --image_size 150 \
                          --url ws://URL \
                          --jpeg_quality 70 \
                          --target_fps 10
```

## Building Your Own Device

Since the current device does not run inference, it is very minimal. The main thing is to make it clean enough to look like a normal frame.

### Part list

Purchased parts:
- [Raspberry Pi Zero 2 W]()
- [10.1" Pi screen](https://www.amazon.fr/HMTECH-Raspberry-Moniteur-portable-Raspbian/dp/B098762GVK)
- [Black frame with enough depth to fit the electronics](https://www.leroymerlin.fr/produits/decoration-eclairage/decoration-murale/cadre-photo/cadre-noir/cadre-milo-21-x-29-7-cm-noir-inspire-71670942.html)
- [Infrared Pi camera](https://www.raspberrypi.com/products/pi-noir-camera-v2/)
- [Infrared light for in-the-dark visuals](https://www.amazon.fr/dp/B0BG5HM2Q8?ref=ppx_yo2ov_dt_b_fed_asin_title)

Custom parts:
- Top mount 3D printed. STEP file located at `hardware/top_mount.step`
- Bottom mount 3D printed. STEP file located at `hardware/bottom_mount.step`
- Back panel laser cut in acrylic. DXF file located at `hardware/back_panel.dxf`

Tools:
- [Large-depth cardboard puncher for the camera hole]()

### Assembly

### Setting up a new Pi

Burn a new SD card with raspbian bookworm, set raspberry local, set your wifi, activate SSH and hostname in the settings.

```bash
wlr-randr
```
Note the name 


Create or edit the autostart configuration file:

```bash
sudo mkdir -p /etc/xdg/autostart
sudo nano /etc/xdg/autostart/screen-rotation.desktop
```

Add these lines to the file:

```bash
iniCopy[Desktop Entry]
Type=Application
Name=Screen Rotation
Exec=wlr-randr --output HDMI-A-1 --transform 90
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
```

Make the file executable:

```bash
sudo chmod +x /etc/xdg/autostart/screen-rotation.desktop
```

```bash
sudo echo "export DISPLAY=:0" >> /etc/profile
```


No solutions worked for the newer labwc so did the dirty but reversible thing:


```bash
# Backup the original cursors
sudo mv /usr/share/icons/PiXflat/cursors/left_ptr /usr/share/icons/PiXflat/cursors/left_ptr.bak
sudo mv /usr/share/icons/Adwaita/cursors/left_ptr /usr/share/icons/Adwaita/cursors/left_ptr.bak

# Download an invisible cursor
mkdir -p ~/.icons/inviscursor-theme/cursors
wget -O ~/.icons/inviscursor-theme/cursors/left_ptr https://raw.githubusercontent.com/gysi/ubuntu-invis-cursor-theme/main/inviscursor-theme/cursors/left_ptr

# Move them to default cursor location
sudo cp ~/.icons/inviscursor-theme/cursors/left_ptr /usr/share/icons/PiXflat/cursors/left_ptr
sudo cp ~/.icons/inviscursor-theme/cursors/left_ptr /usr/share/icons/PiXflat/cursors/left_ptr

sudo reboot
```

### Setting up the Pi as an Access Point

```bash
sudo apt update
sudo apt install network-manager

nmcli device status

# Set up the AP
sudo nmcli connection add type wifi ifname wlan0 con-name pablonet-2 autoconnect yes ssid "pablonet-2"
sudo nmcli connection modify pablonet-2 802-11-wireless.mode ap
sudo nmcli connection modify pablonet-2 802-11-wireless.band bg
sudo nmcli connection modify pablonet-2 802-11-wireless.channel 6
sudo nmcli connection modify pablonet-2 802-11-wireless-security.key-mgmt wpa-psk
sudo nmcli connection modify pablonet-2 802-11-wireless-security.proto rsn
sudo nmcli connection modify pablonet-2 802-11-wireless-security.psk "pablonet-2"
sudo nmcli connection modify pablonet-2 ipv4.method shared
sudo nmcli connection up pablonet-2

# Verify the connection is active
nmcli connection show --active

# Increase priority
sudo nmcli connection modify pablonet-2 connection.autoconnect-priority 100
```

### Setting up the client

```bash
sudo apt update
sudo apt upgrade

sudo apt install python3-pip python3-dev python3-tk python3-opencv python3-picamera2 libatlas-base-dev tk-dev python3-websockets python3-numpy python3-pillow

git clone https://github.com/mlecauchois/pablonet.git
```