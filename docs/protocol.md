# SafeRide Sentinel — perception ↔ controller protocol

This is the contract between the laptop perception code (this repo) and
the ESP32 controller firmware (teammate's repo). Both sides MUST follow
this exactly.

## Transport

| Property      | Value                          |
| ------------- | ------------------------------ |
| Physical      | USB serial (CDC over USB)      |
| Baud          | `115200`                       |
| Framing       | `8N1`                          |
| Encoding      | UTF-8                          |
| Message shape | One JSON object per line, terminated by `\n` |
| Direction     | Bidirectional, full-duplex     |

The ESP32 auto-resets when the host opens the serial port (DTR toggle).
The host MUST sleep for 2 seconds after `open()` before sending anything,
otherwise the first messages are eaten by the bootloader.

## Perception → Controller (detection)

Sent every `SEND_INTERVAL` seconds (default 150ms) when a smoothed
detection is stable. The controller uses this to decide whether to
track, hold the barrier, or actuate.

```json
{"ts": 1733512345.421, "class": "no_helmet", "conf": 0.87, "bbox": [120, 95, 245, 260], "frame_w": 640, "frame_h": 480}
```

| Field      | Type           | Notes                                                          |
| ---------- | -------------- | -------------------------------------------------------------- |
| `ts`       | float seconds  | Host wall-clock time (Unix epoch). Diagnostic only.            |
| `class`    | string         | One of `helmet`, `no_helmet`. Smoothed (majority vote of 5).   |
| `conf`     | float [0, 1]   | Confidence of the single best detection in this frame.         |
| `bbox`     | int[4]         | `[x1, y1, x2, y2]` in pixels, top-left origin.                 |
| `frame_w`  | int            | Frame width in pixels (use to normalise bbox if you want).     |
| `frame_h`  | int            | Frame height in pixels.                                        |

Only one object per line. Always terminated by `\n`. Perception will
NOT send a message if no class clears `CONF_THRESH` for this frame.

## Controller → Perception (heartbeat / state)

The controller may send back human-readable lines OR JSON state at any
time. Perception drains incoming bytes every iteration and prints them.
The expected shape is:

```json
{"state": "ACTUATE", "pan": 92, "tilt": 48, "barrier": "DOWN"}
```

| Field     | Type    | Notes                                                       |
| --------- | ------- | ----------------------------------------------------------- |
| `state`   | string  | `IDLE`, `TRACK`, `ACTUATE`, `RESET`, or similar FSM label.  |
| `pan`     | int     | Pan servo angle in degrees, 0–180.                          |
| `tilt`    | int     | Tilt servo angle in degrees, 0–180.                         |
| `barrier` | string  | `UP` or `DOWN`.                                             |

Perception treats incoming lines as informational — it logs them but
does not change behaviour based on them. (Closed-loop coordination is
out of scope for the v1 demo.)

## Versioning

Any breaking change to either direction must bump a `proto_version`
field in the message and be agreed by both teams.
