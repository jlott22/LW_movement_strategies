# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===========================================================
# Pololu 3pi+ 2040 OLED — Hardcoded Sweep Search
# ===========================================================
# Runs on the Pololu 3pi+ 2040 OLED using MicroPython.
# Communication uses simple text frames over UART; an attached ESP32 relays
# those frames to MQTT topics.
#
# Behavior overview:
#   * Each robot follows a predetermined lawn‑mower sweep path across its
#     assigned columns of the grid.
#   * Clues are still detected and broadcast but do not alter the robot's
#     path. Robots therefore continue their hardcoded sweep regardless of
#     incoming clue data.
#   * The next intended cell is published so peers can yield and avoid
#     collisions.
#   * Bump sensors detect the object; on a bump both robots halt and report.
#   * A clue is any intersection where the centered line sensor reads white.
#
# Threads:
#   * A background movement thread follows lines while main thread processes UART and executes predetermined paths.
#   * The main thread plans paths and moves the robot, always stopping the
#     motors if the program exits unexpectedly.
#
# Tuning hints:
#   * Set UART pins and baud rate to match the hardware.
#   * Calibrate line sensors and adjust cfg.MIDDLE_WHITE_THRESH accordingly.
#   * Tune yaw timings (cfg.YAW_90_MS / cfg.YAW_180_MS) for your platform.
# ===========================================================

import time
import _thread
import sys
import gc
from array import array
from machine import UART, Pin
from pololu_3pi_2040_robot import robot
from pololu_3pi_2040_robot.extras import editions
from pololu_3pi_2040_robot.buzzer import Buzzer

# -----------------------------
# Robot identity & start pose
# -----------------------------
ROBOT_ID = "00" # set to "00", "01", "02", or "03" at deployment
GRID_SIZE = 10
GRID_CENTER = (GRID_SIZE - 1) / 2

DEBUG_LOG_FILE = "debug-log-00.txt"

METRICS_LOG_FILE = "metrics-log-00.txt"
BOOT_TIME_MS = time.ticks_ms()
METRIC_START_TIME_MS = None  # set after first post-calibration intersection
start_signal = False  # set when hub command received
intersection_visits = {}
reported_robot_ids = set()  # robots that have shared a post-calibration location
intersection_count = 0          # steps taken by this robot
repeat_intersection_count = 0   # this robot's revisits
yield_count = 0                 # times this robot yielded an intended move
FIRST_CLUE_TIME_MS = None       # ms from start to first clue
object_location = None  # set when object is found

buzzer = None  # replaced after hardware initialization

# Energy/Time metrics
motor_time_ms = 0              # cumulative ms motors were commanded non-zero
_motor_start_ms = None         # internal tracker for motor activity

def finalize_motor_time(now_ticks=None):
    """Ensure motor_time_ms captures any active span before sampling metrics."""
    global _motor_start_ms, motor_time_ms
    if _motor_start_ms is not None:
        if now_ticks is None:
            now_ticks = time.ticks_ms()
        motor_time_ms += time.ticks_diff(now_ticks, _motor_start_ms)
        _motor_start_ms = None


# Communication counters
position_msgs_sent = 0
visited_msgs_sent = 0
clue_msgs_sent = 0
object_msgs_sent = 0
position_msgs_received = 0
visited_msgs_received = 0
clue_msgs_received = 0
object_msgs_received = 0

# D Algorithm: Minimal compute cycles (predetermined paths)
# Simple energy constants (mA)
MOTOR_POWER_MA = 800
CPU_POWER_MA = 100


def log_error(message):
    """Log errors with a timestamp and play a low buzzer tone."""
    elapsed_ms = time.ticks_diff(time.ticks_ms(), BOOT_TIME_MS)
    try:
        with open(DEBUG_LOG_FILE, "a") as _fp:
            _fp.write(f"{elapsed_ms} ERROR: {message}\n")
    except (OSError, MemoryError):
        pass
    try:
        if buzzer is not None:
            buzzer.play("O2c16")
    except Exception:
        pass


def safe_assert(condition, message):
    if not condition:
        log_error(message)
        raise AssertionError(message)


def record_intersection(x, y):
    """Track intersection visits and repeated counts."""
    safe_assert(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE, "intersection out of range")
    global intersection_count, repeat_intersection_count
    intersection_count += 1
    if first_clue_seen:
        global steps_after_first_clue
        steps_after_first_clue += 1
    key = (x, y)
    if key in intersection_visits:
        repeat_intersection_count += 1
        intersection_visits[key] += 1
    else:
        intersection_visits[key] = 1


def simple_energy_metrics(elapsed_ms):
    """Calculate time metrics only - power can be computed offline if needed."""
    # Compute time = everything except motor time (minimal for D algorithm)
    compute_time_ms = max(0, elapsed_ms - motor_time_ms)

    # Message totals (D algorithm: no intent messages)
    total_msgs_sent = position_msgs_sent + visited_msgs_sent + clue_msgs_sent + object_msgs_sent
    total_msgs_received = position_msgs_received + visited_msgs_received + clue_msgs_received + object_msgs_received

    return {
        'motor_time_ms': motor_time_ms,
        'compute_time_ms': compute_time_ms,
        'msgs_sent': total_msgs_sent,
        'msgs_received': total_msgs_received,
    }

def metrics_log():
    """Write summary metrics for the search run and return them."""
    start = METRIC_START_TIME_MS if METRIC_START_TIME_MS is not None else BOOT_TIME_MS
    now = time.ticks_ms()
    finalize_motor_time(now)
    elapsed_ms = time.ticks_diff(now, start)
    unique_cells = len(intersection_visits)
    path_eff = (
        unique_cells / intersection_count if intersection_count else 0.0
    )
    compute_time_ms = max(0, elapsed_ms - motor_time_ms)
    if object_location is not None and intersection_count:
        optimal_steps = abs(object_location[0] - START_POS[0]) + abs(object_location[1] - START_POS[1])
        obj_path_eff = optimal_steps / intersection_count
    else:
        obj_path_eff = -1.0

    metrics = {
        "elapsed_ms": elapsed_ms,
        "compute_time_ms": compute_time_ms,
        "motor_time_ms": motor_time_ms,
        "first_clue_time_ms": FIRST_CLUE_TIME_MS if FIRST_CLUE_TIME_MS is not None else -1,
        "unique_cells": unique_cells,
        "steps": intersection_count,
        "individual_revisits": repeat_intersection_count,
        "yields": yield_count,
        "path_eff": round(path_eff, 2),
        "obj_path_eff": round(obj_path_eff, 2),
        "object": object_location,
        "clues": clues,
    }

    fieldnames = [
        "elapsed_ms",
        "compute_time_ms",
        "motor_time_ms",
        "first_clue_time_ms",
        "unique_cells",
        "steps",
        "individual_revisits",
        "yields",
        "path_eff",
        "obj_path_eff",
        "object",
        "clues",
    ]

    try:
        try:
            with open(METRICS_LOG_FILE) as _fp:
                write_header = _fp.read(1) == ""
        except OSError:
            write_header = True
        with open(METRICS_LOG_FILE, "a") as _fp:
            if write_header:
                _fp.write(",".join(fieldnames) + "\n")
            _fp.write(",".join(str(metrics[f]) for f in fieldnames) + "\n")
    except OSError:
        pass
    return metrics


try:
    open(DEBUG_LOG_FILE, "a").close()
except OSError:
    pass

try:
    open(METRICS_LOG_FILE, "a").close()
except OSError:
    pass

# Starting position & heading (grid coordinates, cardinal heading)
# pos = (x, y)    heading = (dx, dy) where (0,1)=N, (1,0)=E, (0,-1)=S, (-1,0)=W
START_CONFIG = {
    "00": ((0, 0), (0, 1)),                       # SW corner, facing north
    "01": ((GRID_SIZE - 1, GRID_SIZE - 1), (0, -1)),  # NE corner, facing south
    "02": ((0, GRID_SIZE - 1), (1, 0)),           # NW corner, facing east
    "03": ((GRID_SIZE - 1, 0), (-1, 0)),          # SE corner, facing west
}
try:
    START_POS, START_HEADING = START_CONFIG[ROBOT_ID]
except KeyError as e:
    raise ValueError("ROBOT_ID must be one of '00', '01', '02', or '03'") from e
safe_assert(0 <= START_POS[0] < GRID_SIZE and 0 <= START_POS[1] < GRID_SIZE,
            "start position out of bounds")

def register_robot_position(robot_id):
    """Record that ``robot_id`` has shared a calibrated location update."""

    if robot_id in START_CONFIG:
        reported_robot_ids.add(robot_id)


def validate_orthogonal_path(path):
    """Verify that all moves in path are orthogonal (no diagonals)."""
    DIRS4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W

    if len(path) < 2:
        return True

    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        move_vec = (curr[0] - prev[0], curr[1] - prev[1])
        if move_vec not in DIRS4:
            return False
    return True


def test_complete_coverage():
    """Test that all 4 robots together cover every cell in the grid."""
    all_covered = set()
    for robot_id in ["00", "01", "02", "03"]:
        # Generate path for each robot using the sectored sweep
        path = generate_sectored_sweep(GRID_SIZE, robot_id)
        robot_cells = set(path)
        all_covered.update(robot_cells)

    # Generate all cells in grid
    all_grid_cells = set()
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            all_grid_cells.add((x, y))

    missing_cells = all_grid_cells - all_covered
    return len(missing_cells) == 0


def generate_sectored_sweep(grid_size, robot_id):
    """Generate simple quadrant sectors with orthogonal paths starting from robot positions.
    Compatible with D-sim.py interface.

    Robot starting positions and their quadrants:
    - Robot 00: (0,0) SW corner -> Southwest quadrant
    - Robot 01: (grid_size-1, grid_size-1) NE corner -> Northeast quadrant
    - Robot 02: (0, grid_size-1) NW corner -> Northwest quadrant
    - Robot 03: (grid_size-1, 0) SE corner -> Southeast quadrant
    """
    mid = grid_size // 2
    path = []

    if robot_id == "00":  # Southwest quadrant - start at (0,0)
        # Cover bottom-left: (0,0) to (mid-1, mid-1)
        for y in range(mid):
            if y % 2 == 0:  # Even rows: left to right
                for x in range(mid):
                    path.append((x, y))
            else:  # Odd rows: right to left
                for x in range(mid - 1, -1, -1):
                    path.append((x, y))

    elif robot_id == "01":  # Northeast quadrant - start at (grid_size-1, grid_size-1)
        # Cover top-right: (mid, mid) to (grid_size-1, grid_size-1)
        for y in range(grid_size - 1, mid - 1, -1):
            if (grid_size - 1 - y) % 2 == 0:  # Even rows from top: right to left
                for x in range(grid_size - 1, mid - 1, -1):
                    path.append((x, y))
            else:  # Odd rows from top: left to right
                for x in range(mid, grid_size):
                    path.append((x, y))

    elif robot_id == "02":  # Northwest quadrant - start at (0, grid_size-1)
        # Cover top-left: (0, mid) to (mid-1, grid_size-1)
        for y in range(grid_size - 1, mid - 1, -1):
            if (grid_size - 1 - y) % 2 == 0:  # Even rows from top: left to right
                for x in range(mid):
                    path.append((x, y))
            else:  # Odd rows from top: right to left
                for x in range(mid - 1, -1, -1):
                    path.append((x, y))

    elif robot_id == "03":  # Southeast quadrant - start at (grid_size-1, 0)
        # Cover bottom-right: (mid, 0) to (grid_size-1, mid-1)
        for y in range(mid):
            if y % 2 == 0:  # Even rows: right to left
                for x in range(grid_size - 1, mid - 1, -1):
                    path.append((x, y))
            else:  # Odd rows: left to right
                for x in range(mid, grid_size):
                    path.append((x, y))

    else:
        return []

    # Validate orthogonal path before returning
    if not validate_orthogonal_path(path):
        raise ValueError(f"Generated sectored path for robot {robot_id} contains non-orthogonal moves")

    return path



def generate_sweep_path(active_ids):
    """Return the predetermined sweep path for this robot.

    ``active_ids`` is the collection of robots that broadcast their position
    after calibration but before the start signal.

    Supports 1, 2, or 4 robots on a 10×10 grid. For one robot the entire
    grid is serpentine scanned. With two robots the grid is split into left
    and right halves. For four robots the grid is partitioned into four
    triangles that shrink toward the centre, ensuring every cell is assigned
    to exactly one robot so no collisions occur.
    """

    if ROBOT_ID not in active_ids:
        raise ValueError("This robot is not present in the active robot list")

    total = len(active_ids)

    if total == 1:
        columns = range(GRID_SIZE)
        path = []
        for y in range(GRID_SIZE):
            cols = columns if y % 2 == 0 else reversed(list(columns))
            for x in cols:
                path.append((x, y))
        start_index = path.index(START_POS)
        final_path = path[start_index:] + path[:start_index]

        # Validate orthogonal path before returning
        if not validate_orthogonal_path(final_path):
            raise ValueError(f"Generated path for robot {ROBOT_ID} contains non-orthogonal moves")

        return final_path

    if total == 2:
        left_ids = [rid for rid in active_ids if rid in ("00", "02")]
        right_ids = [rid for rid in active_ids if rid in ("01", "03")]
        safe_assert(len(left_ids) == 1 and len(right_ids) == 1,
                    "Two-robot sweep requires one left and one right robot")

        width = GRID_SIZE // 2
        path = []

        if ROBOT_ID in left_ids:  # Left half
            # Cover left half: (0,0) to (width-1, GRID_SIZE-1) starting from bottom-left
            for y in range(GRID_SIZE):
                if y % 2 == 0:  # Even rows: left to right
                    for x in range(width):
                        path.append((x, y))
                else:  # Odd rows: right to left
                    for x in range(width - 1, -1, -1):
                        path.append((x, y))

        else:  # Right half
            # Cover right half: (width, 0) to (GRID_SIZE-1, GRID_SIZE-1) starting from top-right corner
            for y in range(GRID_SIZE - 1, -1, -1):  # Start from top, go down
                if (GRID_SIZE - 1 - y) % 2 == 0:  # Even rows from top: right to left
                    for x in range(GRID_SIZE - 1, width - 1, -1):
                        path.append((x, y))
                else:  # Odd rows from top: left to right
                    for x in range(width, GRID_SIZE):
                        path.append((x, y))

        # Validate orthogonal path before returning
        if not validate_orthogonal_path(path):
            raise ValueError(f"Generated path for robot {ROBOT_ID} contains non-orthogonal moves")

        return path

    if total == 4:
        safe_assert(set(active_ids) == {"00", "01", "02", "03"},
                    "Four-robot sweep requires robots 00, 01, 02, and 03")

        # Simple quadrant assignment matching D-sim.py
        mid = GRID_SIZE // 2
        path = []

        if ROBOT_ID == "00":  # Southwest quadrant - start at (0,0)
            # Cover bottom-left: (0,0) to (mid-1, mid-1)
            for y in range(mid):
                if y % 2 == 0:  # Even rows: left to right
                    for x in range(mid):
                        path.append((x, y))
                else:  # Odd rows: right to left
                    for x in range(mid - 1, -1, -1):
                        path.append((x, y))

        elif ROBOT_ID == "01":  # Northeast quadrant - start at (GRID_SIZE-1, GRID_SIZE-1)
            # Cover top-right: (mid, mid) to (GRID_SIZE-1, GRID_SIZE-1)
            for y in range(GRID_SIZE - 1, mid - 1, -1):
                if (GRID_SIZE - 1 - y) % 2 == 0:  # Even rows from top: right to left
                    for x in range(GRID_SIZE - 1, mid - 1, -1):
                        path.append((x, y))
                else:  # Odd rows from top: left to right
                    for x in range(mid, GRID_SIZE):
                        path.append((x, y))

        elif ROBOT_ID == "02":  # Northwest quadrant - start at (0, GRID_SIZE-1)
            # Cover top-left: (0, mid) to (mid-1, GRID_SIZE-1)
            for y in range(GRID_SIZE - 1, mid - 1, -1):
                if (GRID_SIZE - 1 - y) % 2 == 0:  # Even rows from top: left to right
                    for x in range(mid):
                        path.append((x, y))
                else:  # Odd rows from top: right to left
                    for x in range(mid - 1, -1, -1):
                        path.append((x, y))

        elif ROBOT_ID == "03":  # Southeast quadrant - start at (GRID_SIZE-1, 0)
            # Cover bottom-right: (mid, 0) to (GRID_SIZE-1, mid-1)
            for y in range(mid):
                if y % 2 == 0:  # Even rows: right to left
                    for x in range(GRID_SIZE - 1, mid - 1, -1):
                        path.append((x, y))
                else:  # Odd rows: left to right
                    for x in range(mid, GRID_SIZE):
                        path.append((x, y))

        # Validate orthogonal path before returning
        if not validate_orthogonal_path(path):
            raise ValueError(f"Generated path for robot {ROBOT_ID} contains non-orthogonal moves")

        return path

    raise ValueError("Supports only 1, 2, or 4 robots")

# UART0 for ESP32 communication (TX=GP28, RX=GP29)
uart = UART(0, baudrate=115200, tx=28, rx=29)

# -----------------------------
# Grid / Maps / Shared State
# -----------------------------
# Grid cell states
CELL_UNSEARCHED = 0
CELL_OBSTACLE   = 1  # object or peer reservation
CELL_SEARCHED   = 2

grid = bytearray(GRID_SIZE * GRID_SIZE)
prob_map = array('f', [1 / (GRID_SIZE * GRID_SIZE)] * (GRID_SIZE * GRID_SIZE))
REWARD_FACTOR = 5
clues = []                            # list of (x, y) clue cells

# D Algorithm: No A* planning arrays needed for predetermined paths


def idx(x, y):
    """Convert Cartesian (x, y) to linear index in map arrays."""
    safe_assert(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE, "idx out of range")
    return (GRID_SIZE - 1 - y) * GRID_SIZE + x


pos = [START_POS[0], START_POS[1]]    # current grid position
heading = (START_HEADING[0], START_HEADING[1])

# Flags used by threads for clean exits
running = True                         # global run flag
found_object = False                   # set True on bump or peer alert
first_clue_seen = False                # once True, disable lawn‑mower bias
move_forward_flag = False

# Peer tracking removed - D algorithm requires no inter-robot coordination
peer_intent = {}  # Empty dict for compatibility with existing code
peer_pos = {}     # Empty dict for compatibility with existing code
last_visited_from_sender = {}  # sender_id -> (x, y) to detect duplicate visited messages

# D Algorithm: No cost shaping needed for predetermined sweep paths

# -----------------------------
# Motion configuration
# -----------------------------
class MotionConfig:
    def __init__(self):
        self.MIDDLE_WHITE_THRESH = 200  # center sensor threshold for "white" (tune by calibration)
        # D Algorithm: No visited step penalty needed for predetermined paths
        self.KP = 0.5                # proportional gain around LINE_CENTER
        self.CALIBRATE_SPEED = 1130  # speed to rotate when calibrating
        self.BASE_SPEED = 800        # nominal wheel speed
        self.MIN_SPD = 400           # clamp low (avoid stall)
        self.MAX_SPD = 1200          # clamp high
        self.LINE_CENTER = 2000      # weighted position target (0..4000)
        self.BLACK_THRESH = 600      # calibrated "black" threshold (0..1000)
        self.STRAIGHT_CREEP = 900    # forward speed while "locked" straight
        self.START_LOCK_MS = 300     # hold straight this long after function starts
        self.TURN_SPEED = 1000
        self.YAW_90_MS = 0.3
        self.YAW_180_MS = 0.6

cfg = MotionConfig()

# D Algorithm: No intent penalty needed for collision-free predetermined paths

#UART handling globals
# ---------- ring buffer ----------
RB_SIZE = 1024
buf = bytearray(RB_SIZE)
head = 0
tail = 0
DELIM = ord('-')

# ---------- message builder ----------
MSG_BUF_SIZE = 256
msg_buf = bytearray(MSG_BUF_SIZE)
msg_len = 0

# ---------- outbound buffer ----------
TX_BUF_SIZE = 64
tx_buf = bytearray(TX_BUF_SIZE)

def _write_int(buf, idx, val):
    """Write an integer as ASCII into buf starting at idx.

    Returns the new index after writing."""
    if val < 0:
        buf[idx] = ord('-')
        idx += 1
        val = -val
    if val == 0:
        buf[idx] = ord('0')
        return idx + 1
    # Determine number of digits
    tmp = val
    digits = 0
    while tmp:
        tmp //= 10
        digits += 1
    end = idx + digits
    for _ in range(digits):
        buf[end - 1] = ord('0') + (val % 10)
        val //= 10
        end -= 1
    return idx + digits

# -----------------------------
# Hardware interfaces
# -----------------------------
motors = robot.Motors()
line_sensors = robot.LineSensors()
bump = robot.BumpSensors()
rgb_leds = robot.RGBLEDs()
rgb_leds.set_brightness(10)
buzzer = Buzzer()

# ===========================================================
# Utility: Motors & Stop Control
# ===========================================================

RED   = (230, 0, 0)
GREEN = (0, 230, 0)
BLUE = (0, 0, 230)
OFF   = (0, 0, 0)

def flash_LEDS(color, n):
    for _ in range(n):
        for led in range(6):
            rgb_leds.set(led, color)  # reuses same tuple, no new allocation
        rgb_leds.show()
        time.sleep_ms(100)
        for led in range(6):
            rgb_leds.set(led, OFF)
        rgb_leds.show()
        time.sleep_ms(100)
        
def buzz(event):
    """
    Play short chirps for turn, intersection, clue,
    and a longer sequence for object.
    """
    if event == "turn":
        buzzer.play("O5c16")            # short high chirp
    elif event == "intersection":
        buzzer.play("O4g16")            # short mid chirp
    elif event == "clue":
        buzzer.play("O6e16")            # short very high chirp
    elif event == "object":
        buzzer.play("O4c8e8g8c5")       # longer sequence, rising melody


        
flash_LEDS(GREEN,1)

def set_speeds(left, right):
    """Wrapper to track motor active time before delegating to hardware."""
    global _motor_start_ms
    if left != 0 or right != 0:
        if _motor_start_ms is None:
            _motor_start_ms = time.ticks_ms()
    else:
        finalize_motor_time()
    motors.set_speeds(left, right)


def motors_off():
    """Hard stop both wheels (safety: call in finally/stop paths)."""
    set_speeds(0, 0)

def stop_all():
    """
    Idempotent global stop:
      - Set flags so all loops/threads exit
      - Ensure motors are off
      - Set a green LED to indicate finished
    """
    global running
    running = False
    motors_off()
    summary = metrics_log()
    publish_result(summary)

def stop_and_alert_object():
    """
    Called when THIS robot detects the object via bump.
    Publishes alert and performs a global stop.

    The robot may bump into the object before reaching the next
    intersection, leaving ``pos`` pointing to the last intersection it
    successfully crossed.  Report the object at the *next* intersection in
    the current heading direction so external consumers know where it is.
    """
    global object_location, found_object, intersection_count, steps_after_first_clue
    global intersection_visits, system_visits, system_revisits
    next_x = pos[0] + heading[0]
    next_y = pos[1] + heading[1]
    object_location = (next_x, next_y)
    key = (next_x, next_y)
    if key in system_visits:
        system_revisits += 1
        system_visits[key] += 1
    else:
        system_visits[key] = 1
    if key in intersection_visits:
        intersection_visits[key] += 1
    else:
        intersection_visits[key] = 1
    publish_object(next_x, next_y)
    buzz('object')
    found_object = True
    intersection_count += 1
    steps_after_first_clue += 1
    stop_all()
    flash_LEDS(BLUE, 1)

flash_LEDS(GREEN,1)
# ===========================================================
# UART Messaging
# Format: "<topic#>:<payload>\n"
# position = 1, visited = 2, clue = 3, alert = 4, result = 6
# D algorithm doesn't use intent (topic 5) - paths are collision-free by design
# Examples:
#   001.3,4-  robot 00 position (x,y only)
#   003.5,2-  robot 00 clue at (5,2)
#   004.6,1-  robot 00 object at (6,1)
# ===========================================================
def uart_send(topic, payload_len):
    """Send the prepared message in tx_buf with topic and payload_len."""
    tx_buf[0] = ord(topic)
    tx_buf[1] = ord('.')
    tx_buf[payload_len + 2] = ord('-')
    uart.write(tx_buf[:payload_len + 3])

def publish_position():
    """Publish current pose (for UI/diagnostics)."""
    global position_msgs_sent
    position_msgs_sent += 1
    if not start_signal and ROBOT_ID not in reported_robot_ids:
        register_robot_position(ROBOT_ID)
    i = 2
    i = _write_int(tx_buf, i, pos[0])
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, pos[1])
    uart_send('1', i - 2)

def publish_visited(x, y):
    """Publish that we visited cell (x,y)."""
    global visited_msgs_sent
    visited_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('2', i - 2)

def publish_clue(x, y):
    """Publish a clue at (x,y)."""
    global clue_msgs_sent
    clue_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('3', i - 2)

def publish_object(x, y):
    """Publish that we found the object at (x,y)."""
    global object_msgs_sent
    object_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('4', i - 2)

def publish_intent(x, y):
    """Compatibility stub - D algorithm doesn't use intent messages.

    Kept for code structure consistency with other algorithms.
    D uses predetermined collision-free paths, so no coordination needed.
    """
    pass

# Intent publishing removed - D algorithm uses predetermined collision-free paths

def publish_result(msg):
    """Publish final search metrics or result to the hub."""
    # ``msg`` can be numeric; ensure it is converted to string before
    # concatenation to avoid ``TypeError: can't convert 'int' object to str``.
    uart.write("6." + str(msg) + "-")

def handle_msg(line):
    """
    Parse and apply incoming messages from the other robot or hub.

    Accepts:
    011.3,4-       # topic 1: position (x,y only)
    002.3,4-       # topic 2: visited
    003.5,2-       # topic 3: clue
    004.6,1-       # topic 4: object/alert
    005.7,2-       # topic 5: intent
    996.1-         # topic 6: hub command

    Ignores:
      - other status fields we don't currently need
    """
    global object_location, start_signal, found_object, FIRST_CLUE_TIME_MS

    # Minimal parsing: "<sender>/<topic>:<payload>"
    try:
        left, payload = line.split(".", 1)
        if len(left) < 3:
            return
        sender = left[0:2]
        topic  = left[2]
    except ValueError:
        return

    if topic == "2":  #visited
        try:
            x, y = map(int, payload.split(","))
        except ValueError:
            return
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            # Check for duplicate message from same sender (heartbeat spam filter)
            if last_visited_from_sender.get(sender) == (x, y):
                # Same cell from same sender - update grid but don't process further
                i = idx(x, y)
                grid[i] = CELL_SEARCHED
                return

            # New cell from this sender - update tracking
            last_visited_from_sender[sender] = (x, y)

            # Process normally
            i = idx(x, y)
            grid[i] = CELL_SEARCHED

    elif topic == "3":   #clue
        try:
            x, y = map(int, payload.split(","))
        except ValueError:
            return
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            clue = (x, y)
            if clue not in clues:
                clues.append(clue)
                if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
                    FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)

    elif topic == "4": #object
        # Peer found the object → stop immediately
        try:
            x, y = map(int, payload.split(","))
            object_location = (x, y)
        except ValueError:
            object_location = None
        found_object = True
        stop_all()

    elif topic == "1": #position - only used for robot registration before start
        if not start_signal and sender not in reported_robot_ids:
            register_robot_position(sender)

    # Intent handling removed - D algorithm uses collision-free predetermined paths
    elif topic == "6":  # hub command
        if payload.strip() == "1":
            start_signal = True

# ---------- ring buffer helpers ----------
def rb_put_byte(b):
    """Push one byte into the ring buffer."""
    global tail, head
    buf[tail] = b
    nxt = (tail + 1) % RB_SIZE
    if nxt == head:                # buffer full, drop oldest
        head = (head + 1) % RB_SIZE
    tail = nxt

def rb_pull_into_msg():
    """Pull bytes into message buffer until '-' is found."""
    global head, tail, msg_len
    if head == tail:
        return None
    while head != tail:
        b = buf[head]
        head = (head + 1) % RB_SIZE
        if b == DELIM:  # complete frame
            s = msg_buf[:msg_len].decode('utf-8', 'ignore').strip()
            msg_len = 0
            return s
        if msg_len < MSG_BUF_SIZE:
            msg_buf[msg_len] = b
            msg_len += 1
    return None

# ---------- UART service ----------
def uart_service():
    """Read and parse any complete messages from UART."""
    data = uart.read()     # returns None or bytes object
    if not data:
        return
    for b in data:         # iterate over bytes
        rb_put_byte(b)
    while True:
        msg = rb_pull_into_msg()
        if msg is None:
            break
        handle_msg(msg)

# ===========================================================
# Sensing & Motion
# ===========================================================
flash_LEDS(GREEN,1)
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def move_forward_one_cell():
    """
    Drive forward following the line until an intersection is detected:
      - T or + intersections: trigger if either outer sensor is black.
      - Require 3 consecutive qualifying reads (debounce).
      - On first candidate, lock steering straight (no P-correction)
        until intersection is confirmed → avoids grabbing side lines.
      - Also hold a 0.5 s straight "roll-through" at start to clear
        the cross you’re sitting on before re-engaging P-control.
    Returns:
      True  -> reached an intersection (no bump)
      False -> stopped due to bump or external stop condition
    """
    global move_forward_flag
    first_loop = False
    lock_release_time = time.ticks_ms() #flag to reset start lock time
    #outter infinite loop to keep thread check for activation
    while running:
        
        while move_forward_flag:
            # 1) Safety/object check
            if first_loop:
                # Initial lock to roll straight for half a second
                lock_release_time = time.ticks_add(time.ticks_ms(), cfg.START_LOCK_MS)
                first_loop = False

            # 3) During initial lock window, always drive straight
            if time.ticks_diff(time.ticks_ms(), lock_release_time) < 0:
                set_speeds(cfg.STRAIGHT_CREEP, cfg.STRAIGHT_CREEP)
                continue
            
            # 2) Read sensors
            readings = line_sensors.read_calibrated()
            
            if readings[0] >= cfg.BLACK_THRESH or readings[4] >= cfg.BLACK_THRESH:
                motors_off()
                flash_LEDS(GREEN,1)
                move_forward_flag = False
                first_loop = True
                break
            
            bump.read()
            if bump.left_is_pressed() or bump.right_is_pressed():
                stop_and_alert_object()
                motors_off()
                move_forward_flag = False
                break    

            # 6) Normal P-control when not locked
            total = readings[0] + readings[1] + readings[2] + readings[3] + readings[4]
            if total == 0:
                set_speeds(cfg.STRAIGHT_CREEP, cfg.STRAIGHT_CREEP)
                continue
            # weights: 0, 1000, 2000, 3000, 4000
            pos = (0*readings[0] + 1000*readings[1] + 2000*readings[2] + 3000*readings[3] + 4000*readings[4]) // total
            error = pos - cfg.LINE_CENTER
            correction = int(cfg.KP * error)

            left  = _clamp(cfg.BASE_SPEED + correction, cfg.MIN_SPD, cfg.MAX_SPD)
            right = _clamp(cfg.BASE_SPEED - correction, cfg.MIN_SPD, cfg.MAX_SPD)
            set_speeds(left, right)

        # Shorter sleep to allow rapid response when move_forward_flag is set
        time.sleep_ms(20)

def calibrate():
    """Calibrate line sensors then advance to the first intersection.

    The robot spins in place while repeatedly sampling the line sensors to
    establish min/max values.  The robot should be placed one cell behind its
    intended starting position; after calibration it drives forward to the
    first intersection and updates the global ``pos`` to ``START_POS`` so the
    caller sees that intersection as the starting point of the search. The
    metric timer begins once this intersection is reached.
    """
    global pos, move_forward_flag, METRIC_START_TIME_MS

    # 1) Spin in place to expose sensors to both edges of the line.
    #    A single full rotation is enough, so spin in one direction while
    #    repeatedly sampling the sensors.  The Pololu library recommends
    #    speeds of 920/-920 with ~10 ms pauses for calibration.
    for _ in range(50):
        if not running:
            motors_off()
            return

        set_speeds(cfg.CALIBRATE_SPEED, -cfg.CALIBRATE_SPEED)
        line_sensors.calibrate()
        time.sleep_ms(5)
        
    motors_off()
    bump.calibrate()
    time.sleep_ms(5)


    # 2) Move forward until an intersection is detected.  After the forward
    #    move the robot is sitting on our true starting cell (defined by
    #    ``START_POS`` at the top of the file) so overwrite any temporary
    #    position with that constant and mark the cell visited.
    move_forward_flag = True
    while move_forward_flag:
        uart_service()
        time.sleep_ms(1)
    pos[0], pos[1] = START_POS
    if 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE:
        grid[idx(pos[0], pos[1])] = CELL_SEARCHED
    publish_position()
    publish_visited(pos[0], pos[1])

    motors_off()
    METRIC_START_TIME_MS = time.ticks_ms()
    gc.collect()
    

def at_intersection_and_white():
    """
    Detect a 'clue':
      - Center line sensor reads white ( < cfg.MIDDLE_WHITE_THRESH )
    Returns bool.
    """
    r = line_sensors.read_calibrated()      # [0]..[4], center is [2]
    if r[2] < cfg.MIDDLE_WHITE_THRESH:
        buzz('clue')
        return True
    else:
        return False


def check_current_cell_for_clue(stage="start"):
    """Check the current cell for a clue without moving off of it."""
    global first_clue_seen, FIRST_CLUE_TIME_MS
    if not running or found_object:
        return
    if at_intersection_and_white():
        clue = (pos[0], pos[1])
        is_new = clue not in clues
        if is_new:
            clues.append(clue)
        first_clue_seen = True
        if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
            FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
        print(f"[INFO] {stage}: clue detected at {clue}")
        publish_clue(pos[0], pos[1])


flash_LEDS(GREEN,1)
# ===========================================================
# Heading / Turning (cardinal NSEW)
# ===========================================================
def rotate_degrees(deg):
    """
    Rotate in place by a signed multiple of 90°.
    deg ∈ {-180, -90, 0, 90, 180}
    Obeys 'running' flag and always cuts motors at the end.
    """
    
    if deg == 0 or not running:
        motors_off()
        return
    
    #inch forward to make clean turn
    set_speeds(cfg.BASE_SPEED, cfg.BASE_SPEED)
    time.sleep(.2)
    motors_off()

    if deg == 180 or deg == -180:
        buzz('turn')
        set_speeds(cfg.TURN_SPEED, -cfg.TURN_SPEED)
        if running: time.sleep(cfg.YAW_180_MS)

    elif deg == 90:
        buzz('turn')
        set_speeds(cfg.TURN_SPEED, -cfg.TURN_SPEED)
        if running: time.sleep(cfg.YAW_90_MS)

    elif deg == -90:
        buzz('turn')
        set_speeds(-cfg.TURN_SPEED, cfg.TURN_SPEED)
        if running: time.sleep(cfg.YAW_90_MS)

    motors_off()

def turn_towards(cur, nxt):
    """
    Turn from current heading to face the neighbor cell `nxt`.
    - cur: (x,y) current cell
    - nxt: (x,y) next cell (must be a 4-neighbor of cur)
    Updates the global 'heading'.
    """
    global heading
    dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]
    target = (dx, dy)

    dirs = [(0,1),(1,0),(0,-1),(-1,0)]   # N,E,S,W (clockwise)
    i = dirs.index(heading)
    j = dirs.index(target)
    delta = (j - i) % 4

    # Map delta to minimal signed degrees
    if delta == 0:   deg = 0
    elif delta == 1: deg = 90
    elif delta == 2: deg = 180
    elif delta == 3: deg = -90

    rotate_degrees(deg)
    heading = target
flash_LEDS(GREEN,1)
# ===========================================================
# D Algorithm: No reward model or serpentine bias needed
# ===========================================================
# (Complex cost/reward functions removed for predetermined sweep paths)

# Yielding logic removed - D algorithm paths are guaranteed collision-free

# D Algorithm: No dynamic goal selection needed - uses predetermined sweep paths

# D Algorithm: No dynamic goal selection needed - uses predetermined sweep paths
flash_LEDS(GREEN,1)
# ===========================================================
# D Algorithm: No A* planner needed - uses predetermined sweep paths
# ===========================================================


flash_LEDS(GREEN,1)
# ===========================================================
# Main Search Loop
# ===========================================================
def search_loop():
    """Main mission loop running the hardcoded sweep.

    Robots wait for a start signal then march through the precomputed sweep
    path. Clues are reported but do not influence the chosen path.
    """
    global move_forward_flag, start_signal, METRIC_START_TIME_MS, pos, yield_count, FIRST_CLUE_TIME_MS

    sweep_path = []
    path_index = 1  # starting cell already occupied

    try:
        calibrate()

        publish_position()
        publish_visited(pos[0], pos[1])
        last_pose_publish = time.ticks_ms()

        # wait for hub start command, periodically sharing start position
        while not start_signal:
            uart_service()
            now = time.ticks_ms()
            if time.ticks_diff(now, last_pose_publish) >= 3000:
                publish_position()
                publish_visited(pos[0], pos[1])
                last_pose_publish = now
            time.sleep_ms(10)

        # Freeze the set of active robots at the moment the start signal arrives
        register_robot_position(ROBOT_ID)
        active_ids = sorted(reported_robot_ids)
        safe_assert(active_ids, "No active robots reported positions before start")
        sweep_path = generate_sweep_path(active_ids)
        path_index = 1
        METRIC_START_TIME_MS = time.ticks_ms()
        check_current_cell_for_clue("start_signal")

        while running and not found_object and path_index < len(sweep_path):
            gc.collect()
            nxt = sweep_path[path_index]
            path_index += 1

            # Validate orthogonal movement (no diagonals)
            DIRS4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
            move_vec = (nxt[0] - pos[0], nxt[1] - pos[1])
            if move_vec not in DIRS4:
                flash_LEDS(RED, 3)
                safe_assert(False, f"Non-orthogonal move attempted: {tuple(pos)} -> {nxt}")

            # No intent publishing or yielding needed - paths are collision-free by design
            uart_service()  # Still process incoming messages (clues, object alerts, etc.)

            turn_towards(tuple(pos), nxt)
            if not running or found_object:
                break

            move_forward_flag = True
            while move_forward_flag:
                uart_service()
                time.sleep_ms(1)

            pos[0], pos[1] = nxt[0], nxt[1]
            record_intersection(pos[0], pos[1])
            grid[idx(pos[0], pos[1])] = CELL_SEARCHED
            publish_position()
            publish_visited(pos[0], pos[1])

            if at_intersection_and_white():
                buzz('clue')
                clue = (pos[0], pos[1])
                if clue not in clues:
                    clues.append(clue)
                    if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
                        FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
                    publish_clue(pos[0], pos[1])

        # Check if we completed our sweep path without finding the object
        if path_index >= len(sweep_path) and not found_object:
            flash_LEDS(BLUE, 2)  # Signal path completion
            print(f"Robot {ROBOT_ID}: Completed predetermined sweep path")

    finally:
        motors_off()
flash_LEDS(GREEN,1)
# ===========================================================
# Entry Point
# ===========================================================

flash_LEDS(RED,1)
# Start the single UART RX thread (clean exit when 'running' goes False)
_thread.start_new_thread(move_forward_one_cell, ())

# Kick off the mission
try:
    search_loop()
finally:
    # Ensure absolutely everything is stopped
    running = False
    flash_LEDS(RED,5)
    time.sleep_ms(200)  # give RX thread time to fall out cleanly
