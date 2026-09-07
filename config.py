"""SafeRide Sentinel — perception subsystem configuration.

All tunable constants live here. Edit values to match your hardware.
"""

# Serial port to the ESP32.
# Windows: 'COM3', 'COM4', ...  (Device Manager → Ports (COM & LPT))
# macOS:   '/dev/tty.usbserial-XXXX' or '/dev/tty.SLAB_USBtoUART'
# Linux:   '/dev/ttyUSB0' or '/dev/ttyACM0'
SERIAL_PORT = 'COM3'

BAUD = 115200

CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480

MODEL_PATH = 'models/best.pt'
CONF_THRESH = 0.5

TARGET_CLASSES = ('helmet', 'no_helmet')

SMOOTH_WINDOW = 5
SEND_INTERVAL = 0.15

LOG_CSV = 'logs/events.csv'
SNAPSHOT_DIR = 'logs/snapshots'
