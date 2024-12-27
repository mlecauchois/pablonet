# Pablonet

<div class="image-container" style="flex: 1; text-align: left">
    <img src="frame.png"  width="200">
</div>

Blogpost: https://mlecauchois.github.io/posts/pablonet/

# Setting up a pre-configured device



To use it over wifi do the following commands...


Once connected to wifi better and more bandwidth to remove AP and ssh via your wifi

### Server

4090 good


Server CLI:
```
python server.py --base_model_path "Lykon/DreamShaper" --acceleration "tensorrt" --prompt "" --num_inference_steps 30 --guidance_scale 1.0 --t_index_list "[14,18]" --preprocessing canny_blur_shift --jpeg_quality 80 --port 6000
```

### Pi client

Connect with the dongle to your WiFi

```bash
sudo nmcli device wifi connect "Your_Internet_SSID" password "Your_Password" ifname wlan1
```

```bash
ssh pablonet2@raspberrypi.local
```

Raspberry Pi client CLI:
```
DISPLAY=:0 python client_pi.py --prompt "painting in the style of pablo picasso, cubism, sharp high quality painting, oil painting, mute colors red yellow orange, background of green, color explosion, abstract surrealism" --image_size 150 --url ws://URL --fullscreen --crop_size 900 --crop_offset_y 40 --compression 50 --rotation 270 --target_fps 10
```

### Normal client

```
python client.py --prompt "painting in the style of pablo picasso, cubism, sharp high quality painting, oil painting, mute colors red yellow orange, background of green, color explosion, abstract surrealism" \
--image_size 150 \
--url ws://URL --jpeg_quality 70 --target_fps 10
```


Sever side FPS should be 15-25
Pi FPS should be 2-9



# Building your own device

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
