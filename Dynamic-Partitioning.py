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
# Pololu 3pi+ 2040 OLED — Coordinated Search (UART → ESP32 → MQTT)
# ===========================================================
# Runs on the Pololu 3pi+ 2040 OLED using MicroPython.
# Communication uses simple text frames over UART; an attached ESP32 relays
# those frames to MQTT topics.
#
# Behavior overview:
#   * Before any clue is found the robot sweeps its half of the grid in a
#     lawn‑mower pattern, nudged outward by a small centre‑ward cost.
#   * After a clue appears, A* planning pursues cells with the highest
#     probability scores.
#   * The next intended cell is published so peers can yield and avoid
#     collisions.
#   * Bump sensors detect the target; on a bump both robots halt and report.
#   * A clue is any intersection where the centered line sensor reads white.
#
# Threads:
#   * A background movement thread handles forward motion while the main thread processes UART and coordinates movement.
#   * The main thread plans paths and moves the robot, always stopping the
#     motors if the program exits unexpectedly.
#
# Tuning hints:
#   * Set UART pins and baud rate to match the hardware.
#   * Calibrate line sensors and adjust cfg.MIDDLE_WHITE_THRESH accordingly.
#   * Tune yaw timings (cfg.YAW_90_MS / cfg.YAW_180_MS) for your platform.

# ===========================================================
import random
import time
import _thread
import heapq
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
ROBOT_ID = "03"  # set to "00", "01", "02", or "03" at deployment
GRID_SIZE = 19
GRID_CENTER = (GRID_SIZE - 1) / 2

DEBUG_LOG_FILE = "debug-log.txt"

METRICS_LOG_FILE = "metrics-log-DP.txt"
BOOT_TIME_MS = time.ticks_ms()
METRIC_START_TIME_MS = None  # set after first post-calibration intersection
start_signal = False  # set when hub command received
intersection_visits = {}
unique_cells_count = 0       # cells first visited by this robot for the entire team
system_visits = {}              # all cells visited by any robot (for tracking system_revisits)
intersection_count = 0          # steps taken by this robot
system_revisits = 0             # this robot's revisits to cells visited by ANY robot
yield_count = 0                 # times this robot yielded an intended move
path_replan_count = 0           # times we replanned due to collision avoidance
goal_replan_count = 0           # times our goal changed and we replanned
FIRST_CLUE_TIME_MS = None       # ms from start to first clue (by any robot)
FIRST_CLUE_POSITION = None      # this robot's position when first clue found (by any robot)
target_location = None          # set when target is found
system_clues_found = 0          # total unique clues found by all robots
steps_after_first_clue = 0      # steps taken after first clue was found
clue_misses = 0                 # simulated clue detections that failed the POD check
clue_POD = .7  # probability of detection at the clue cell

_metrics_logged = False
_metrics_cache = None

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


def busy_timer_reset():
    """Start a fresh busy-time measurement for the current control loop."""
    global _busy_start_us, _busy_accum_us
    _busy_accum_us = 0
    _busy_start_us = time.ticks_us()


def busy_timer_pause():
    """Accumulate elapsed busy time and pause the timer."""
    global _busy_start_us, _busy_accum_us
    if _busy_start_us is not None:
        now_us = time.ticks_us()
        _busy_accum_us += time.ticks_diff(now_us, _busy_start_us)
        _busy_start_us = None


def busy_timer_resume():
    """Resume the busy-time timer after a pause."""
    global _busy_start_us
    _busy_start_us = time.ticks_us()


def busy_timer_value_ms():
    """Return the current busy time in milliseconds, pausing measurement."""
    global _busy_start_us, _busy_accum_us
    if _busy_start_us is not None:
        now_us = time.ticks_us()
        _busy_accum_us += time.ticks_diff(now_us, _busy_start_us)
        _busy_start_us = None
    return _busy_accum_us // 1000


def update_mem_headroom():
    """Refresh current free heap measurement and track the lowest observed value."""
    global mem_free_min
    current = gc.mem_free()
    if current < mem_free_min:
        mem_free_min = current
    return current


# Simple energy tracking - message counters only
position_msgs_sent = 0
clue_msgs_sent = 0
target_msgs_sent = 0
position_msgs_received = 0
clue_msgs_received = 0
target_msgs_received = 0
bytes_sent = 0                 # raw UART bytes sent
bytes_received = 0             # raw UART bytes received
# Time metrics
busy_ms = 0                 # cumulative compute time spent outside motion/sleeps (ms)
mem_free_min = gc.mem_free()  # lowest observed free heap bytes

_busy_start_us = None       # internal timer start (microseconds)
_busy_accum_us = 0          # accumulated busy time (microseconds)


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
    """Track intersection visits and system revisit counts."""
    safe_assert(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE, "intersection out of range")
    global intersection_count, system_revisits, system_visits, unique_cells_count
    intersection_count += 1
    if first_clue_seen:
        global steps_after_first_clue
        steps_after_first_clue += 1
    key = (x, y)

    # Track system-wide revisits (ANY robot visited before)
    first_visit = key not in system_visits
    if not first_visit:
        system_revisits += 1

    # Update system-wide visits (this robot's contribution)
    system_visits[key] = system_visits.get(key, 0) + 1
    if first_visit:
        unique_cells_count += 1

    # Update individual visit tracking
    if key in intersection_visits:
        intersection_visits[key] += 1
    else:
        intersection_visits[key] = 1


def simple_energy_metrics(elapsed_ms):
    """Calculate time metrics only - power can be computed offline if needed."""
    # Compute time = everything except motor time
    compute_time_ms = max(0, elapsed_ms - motor_time_ms)

    # Message totals
    total_msgs_sent = position_msgs_sent + clue_msgs_sent + target_msgs_sent
    total_msgs_received = position_msgs_received + clue_msgs_received + target_msgs_received

    return {
        'motor_time_ms': motor_time_ms,
        'compute_time_ms': compute_time_ms,
        'msgs_sent': total_msgs_sent,
        'msgs_received': total_msgs_received,
    }

def manhatt_dist_metric(clue_location,bot_location):
    """Calculate Manhattan distance between first clue and bot location when first clue discovered."""
    return abs(clue_location[0] - bot_location[0]) + abs(clue_location[1] - bot_location[1])

def metrics_log():
    """Write summary metrics for the search run and return them."""
    global unique_cells_count, busy_ms, mem_free_min, _metrics_logged, _metrics_cache
    if _metrics_logged and _metrics_cache is not None:
        return _metrics_cache
    start = METRIC_START_TIME_MS if METRIC_START_TIME_MS is not None else BOOT_TIME_MS
    now = time.ticks_ms()
    finalize_motor_time(now)
    elapsed_ms = time.ticks_diff(now, start)
    unique_cells = unique_cells_count
    compute_time_ms = max(0, elapsed_ms - motor_time_ms)
    dist_from_first_clue = manhatt_dist_metric(clues[0],FIRST_CLUE_POSITION) if FIRST_CLUE_POSITION is not None and len(clues)>0 else -1
    steps_before_first_clue = max(0, intersection_count - steps_after_first_clue)
    time_before_first_clue = FIRST_CLUE_TIME_MS if FIRST_CLUE_TIME_MS is not None else -1
    time_after_first_clue = elapsed_ms - FIRST_CLUE_TIME_MS if FIRST_CLUE_TIME_MS is not None else -1
    mem_total = gc.mem_alloc() + gc.mem_free()
    mem_used_peak = mem_total - mem_free_min
    cpu_util_pct = (busy_ms * 100) // elapsed_ms if elapsed_ms > 0 else 0
    bandwidth_bytes = bytes_sent + bytes_received

    # Calculate time metrics
    energy = simple_energy_metrics(elapsed_ms)

    metrics = {
        "robot_id": ROBOT_ID,
        "target_location": target_location,
        "clue_locations": clues,
        "first_clue_position": FIRST_CLUE_POSITION,
        "elapsed_ms": elapsed_ms,
        "motor_time_ms": motor_time_ms,
        "compute_time_ms": compute_time_ms,
        "busy_ms": busy_ms,
        "cpu_util_pct": cpu_util_pct,
        "mem_used_peak": mem_used_peak,
        "mem_free_min": mem_free_min,
        "steps": intersection_count,
        "steps_before_first_clue": steps_before_first_clue,
        "first_clue_time_ms": time_before_first_clue,
        "time_after_first_clue_ms": time_after_first_clue,
        "dist_from_1st_clue": dist_from_first_clue,
        "steps_after_first_clue": steps_after_first_clue,
        "system_clues_found": system_clues_found,
        "clues_found": len(clues),
        "clues_missed": clue_misses,
        "system_revisits": system_revisits,
        "unique_cells": unique_cells,
        "yields": yield_count,
        "goal_replans": goal_replan_count,
        "msgs_sent": energy['msgs_sent'],
        "msgs_received": energy['msgs_received'],
        "bytes_sent": bytes_sent,
        "bytes_received": bytes_received,
        "bandwidth_bytes": bandwidth_bytes,
        "path_replans": path_replan_count,
    }

    fieldnames = [
        "robot_id",
        "target_location",
        "clue_locations",
        "first_clue_position",
        "elapsed_ms",
        "motor_time_ms",
        "compute_time_ms",
        "busy_ms",
        "cpu_util_pct",
        "mem_used_peak",
        "mem_free_min",
        "steps",
        "steps_before_first_clue",
        "first_clue_time_ms",
        "time_after_first_clue_ms",
        "dist_from_1st_clue",
        "steps_after_first_clue",
        "clues_found",
        "clues_missed",
        "system_clues_found",
        "system_revisits",
        "unique_cells",
        "yields",
        "goal_replans",
        "msgs_sent",
        "msgs_received",
        "bytes_sent",
        "bytes_received",
        "bandwidth_bytes",
        "path_replans",
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
    _metrics_cache = metrics
    _metrics_logged = True
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
    "00": ((0, 0), (1, 0)),                       # west edge, evenly spaced facing east
    "01": ((0, 5), (1, 0)),
    "02": ((0, 10), (1, 0)),
    "03": ((0, 15), (1, 0)),
}
DIRS4 = ((0, 1), (1, 0), (0, -1), (-1, 0))

try:
    START_POS, START_HEADING = START_CONFIG[ROBOT_ID]
except KeyError as e:
    raise ValueError("ROBOT_ID must be one of '00', '01', '02', or '03'") from e
safe_assert(0 <= START_POS[0] < GRID_SIZE and 0 <= START_POS[1] < GRID_SIZE,
            "start position out of bounds")

# UART0 for ESP32 communication (TX=GP28, RX=GP29)
uart = UART(0, baudrate=115200, tx=28, rx=29)

# -----------------------------
# Grid / Maps / Shared State
# -----------------------------
# Grid cell states
CELL_UNSEARCHED = 0
CELL_OBSTACLE   = 1  # target or peer reservation
CELL_SEARCHED   = 2

grid = bytearray(GRID_SIZE * GRID_SIZE)
prob_map = array('f', [1 / (GRID_SIZE * GRID_SIZE)] * (GRID_SIZE * GRID_SIZE))
REWARD_FACTOR = 5
clues = []                            # list of (x, y) clue cells

# --- Target and Clue Belief Maps ---
# P_target[i]: belief target is at cell i
# P_clue[i]:   belief there is an undiscovered clue at cell i
target_p = array('f', [1 / (GRID_SIZE * GRID_SIZE)] * (GRID_SIZE * GRID_SIZE))
clue_p   = array('f', [1 / (GRID_SIZE * GRID_SIZE)] * (GRID_SIZE * GRID_SIZE))

# --- Decay exponents (tunable) ---
# Higher exponent -> stronger / narrower decay
# Lower exponent  -> wider / softer decay
TARGET_DECAY_EXP = 1.0   # target correlation around clues (tighter)
CLUE_DECAY_EXP   = 0.5   # future-clue correlation around clues (wider)


# Preallocated arrays for A* planning
# ----------------------------------
# Parent indices and path costs for each cell are stored here. Reusing these
# arrays each planning cycle avoids repeated allocations, which are expensive
# on MicroPython.
came_from = array('i', [-1] * (GRID_SIZE * GRID_SIZE))
cost_so_far = array('f', [0.0] * (GRID_SIZE * GRID_SIZE))
frontier = []


def idx(x, y):
    """Convert Cartesian (x, y) to linear index in map arrays."""
    safe_assert(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE, "idx out of range")
    return (GRID_SIZE - 1 - y) * GRID_SIZE + x

def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def renorm(arr):
    """Normalize an array of floats in-place so it sums to 1 (if possible)."""
    total = 0.0
    for v in arr:
        total += v
    if total <= 0.0:
        # fallback: uniform over all cells
        n = GRID_SIZE * GRID_SIZE
        val = 1.0 / n
        for i in range(n):
            arr[i] = val
        return
    inv = 1.0 / total
    for i in range(len(arr)):
        arr[i] *= inv


def recompute_value_map():
    """
    Combine target and clue beliefs into the unified value map used by
    goal selection and A*.
      V(i) = P_target(i) + P_clue(i) * clue_POD
    """
    n = GRID_SIZE * GRID_SIZE
    for i in range(n):
        prob_map[i] = target_p[i] + (clue_p[i] * clue_POD)
    # Optional: keep it normalized, not strictly required but nice
    renorm(prob_map)


pos = [START_POS[0], START_POS[1]]    # current grid position
heading = (START_HEADING[0], START_HEADING[1])
current_goal = None

# Flags used by threads for clean exits
running = True                         # global run flag
found_target = False                   # set True on bump or peer alert
first_clue_seen = False                # once True, disable lawn‑mower bias
move_forward_flag = False

# --- Dynamic partitioning & sync state ---

# Team membership (robot IDs as strings matching ROBOT_ID)
TEAM_IDS = ["00", "01", "02", "03"]
NUM_ROBOTS = len(TEAM_IDS)

# Partitions: list of regions [(xmin, xmax, ymin, ymax), ...]
partitions = None
my_region_index = None   # which region this robot currently owns

# Sync barrier after a clue
sync_active = False      # True when we are synchronizing after a clue
sync_positions = {}      # robot_id -> (x, y) paused positions for current sync
have_sent_sync = False   # whether we have already published our sync position this round
peer_pos = {}     # peer_id -> (x, y) last reported position


# -----------------------------
# Cost shaping for early sweeping pattern
# A small cost per step toward the center keeps robots sweeping their region
# before clues are discovered. It must exceed the TURN_COST (1.0).
TURN_COST = 1.0
CENTER_STEP = 0.4

# -----------------------------
# Motion configuration
# -----------------------------
class MotionConfig:
    def __init__(self):
        self.MIDDLE_WHITE_THRESH = 200  # center sensor threshold for "white" (tune by calibration)
        self.VISITED_STEP_PENALTY = 4
        self.KP = 0.7                # proportional gain around LINE_CENTER
        self.CALIBRATE_SPEED = 1150  # speed to rotate when calibrating
        self.BASE_SPEED = 650        # nominal wheel speed
        self.MIN_SPD = 350           # clamp low (avoid stall)
        self.MAX_SPD = 1100          # clamp high
        self.LINE_CENTER = 2000      # weighted position target (0..4000)
        self.BLACK_THRESH = 600      # calibrated "black" threshold (0..1000)
        self.STRAIGHT_CREEP = 650    # forward speed while "locked" straight
        self.START_LOCK_MS = 250     # hold straight this long after function starts
        self.TURN_SPEED = 1000
        self.YAW_90_MS = 0.31
        self.YAW_180_MS = 0.61

cfg = MotionConfig()

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
    and a longer sequence for target.
    """
    if event == "turn":
        buzzer.play("O5c16")            # short high chirp
    elif event == "intersection":
        buzzer.play("O4g16")            # short mid chirp
    elif event == "clue":
        buzzer.play("O6e16")            # short very high chirp
    elif event == "target":
        buzzer.play("O4c8e8g8c5")       # longer sequence, rising melody
    elif event == "missed_clue":
        buzzer.play("O4g16e16")



        
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

def stop_and_alert_target():
    """
    Called when THIS robot detects the target via bump.
    Publishes alert and performs a global stop.

    The robot may bump into the target before reaching the next
    intersection, leaving ``pos`` pointing to the last intersection it
    successfully crossed.  Report the target at the *next* intersection in
    the current heading direction so external consumers know where it is.
    """
    global target_location, found_target, intersection_count, steps_after_first_clue
    global intersection_visits, system_visits, system_revisits, unique_cells_count
    next_x = pos[0] + heading[0]
    next_y = pos[1] + heading[1]
    target_location = (next_x, next_y)
    key = (next_x, next_y)
    first_visit = key not in system_visits
    if not first_visit:
        system_revisits += 1
        system_visits[key] += 1
    else:
        system_visits[key] = 1
        unique_cells_count += 1
    if key in intersection_visits:
        intersection_visits[key] += 1
    else:
        intersection_visits[key] = 1
    publish_target(next_x, next_y)
    buzz('target')
    found_target = True
    intersection_count += 1
    steps_after_first_clue += 1
    stop_all()
    flash_LEDS(BLUE, 1)

flash_LEDS(GREEN,1)
# ===========================================================
# UART Messaging
# Format: "<topic#>:<payload>\n"
# position = 1, intent (not used) = 2, goal(not used) = 3, clue = 4, alert = 5, sync state = 6, hub command = 7
# Examples:
#   001.3,4-  robot 00 position (x,y only)
#   004.5,2-  robot 00 clue at (5,2)
#   005.9,1-  robot 00 target/alert at (9,1)
#   006.7,8-  robot 00 sync pause at (7,8)
# ===========================================================
def uart_send(topic, payload_len):
    """Send the prepared message in tx_buf with topic and payload_len."""
    global bytes_sent
    tx_buf[0] = ord(topic)
    tx_buf[1] = ord('.')
    tx_buf[payload_len + 2] = ord('-')
    uart.write(tx_buf[:payload_len + 3])
    bytes_sent += payload_len + 3

def publish_position():
    """Publish current pose (for UI/diagnostics)."""
    global position_msgs_sent
    position_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, pos[0])
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, pos[1])
    uart_send('1', i - 2)

def publish_clue(x, y):
    """Publish a clue at (x,y)."""
    global clue_msgs_sent
    clue_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('4', i - 2)

def publish_sync_state():
    """
    Publish that we are paused at an intersection for the current sync.
    Topic 8, payload: x_pause,y_pause
    """
    global have_sent_sync
    if have_sent_sync:
        return
    i = 2
    i = _write_int(tx_buf, i, pos[0])
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, pos[1])
    uart_send('6', i - 2)

    # Record locally that we've contributed our sync position
    have_sent_sync = True
    sync_positions[ROBOT_ID] = (pos[0], pos[1])


def publish_target(x, y):
    """Publish that we found the target at (x,y)."""
    global target_msgs_sent
    target_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('5', i - 2)

def handle_msg(line):
    """
    Parse and apply incoming messages from the other robot or hub.

    Accepts:
    011.3,4-       # topic 1: position (x,y only) - previous pos treated as visited
    004.7,8-       # topic 4: clue at (x,y)
    008.9,1-       # topic 8: sync state (paused at x,y)
    005.6,1-       # topic 5: target/alert
    996.1-         # topic 7: hub command

    Ignores:
      - other status fields we don't currently need
    """
    global peer_pos, current_goal, first_clue_seen, target_location, start_signal, found_target, FIRST_CLUE_TIME_MS, goal_replan_count, sync_active, have_sent_sync, sync_positions

    # Minimal parsing: "<sender>/<topic>:<payload>"
    try:
        left, payload = line.split(".", 1)
        if len(left) < 3:
            return
        sender = left[0:2]
        topic  = left[2]
    except ValueError:
        return

    if topic == "4":   #clue
        global clue_msgs_received, system_clues_found, FIRST_CLUE_POSITION
        clue_msgs_received += 1
        try:
            x, y = map(int, payload.split(","))
        except ValueError:
            return
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            clue = (x, y)
            if clue not in clues:
                clues.append(clue)
                system_clues_found += 1
                first_clue_seen = True
                if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
                    FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
                    FIRST_CLUE_POSITION = (pos[0], pos[1])
                # Use this clue to update both fields
                clue_probability_field(clue[0], clue[1])
                i = idx(clue[0], clue[1])
                grid[i] = CELL_SEARCHED
                update_prob_map()
                gc.collect()
                sync_active = True
                have_sent_sync = False


    elif topic == "5": #target
        # Peer found the target → stop immediately
        global target_msgs_received
        target_msgs_received += 1
        try:
            x, y = map(int, payload.split(","))
            target_location = (x, y)
        except ValueError:
            target_location = None
        found_target = True
        stop_all()

    elif topic == "1": #position
        global position_msgs_received
        position_msgs_received += 1
        try:
            ox, oy = map(int, payload.split(","))
        except ValueError:
            return
        if not (0 <= ox < GRID_SIZE and 0 <= oy < GRID_SIZE):
            return
        prev = peer_pos.get(sender)
        if prev and prev != (ox, oy):
            px, py = prev
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                i_prev = idx(px, py)
                grid[i_prev] = CELL_SEARCHED
                # Peer searched this cell and did not report a clue/target
                system_visits[(px, py)] = system_visits.get((px, py), 0) + 1
                update_clue_on_miss(i_prev)
                update_target_on_miss(i_prev)
                if current_goal == (px, py) and not (pos[0] == px and pos[1] == py):
                    current_goal = None
        peer_pos[sender] = (ox, oy)
        grid[idx(ox, oy)] = CELL_SEARCHED

    elif topic == "6":  # sync state: peer paused at (x,y)
        ox, oy = map(int, payload.split(","))
        sync_positions[sender] = (ox, oy)

    elif topic == "7":  # hub command
        if payload.strip() == "1":
            start_signal = True
            sync_active = True


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
    global bytes_received
    data = uart.read()     # returns None or bytes target
    if not data:
        return
    bytes_received += len(data)
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
            # 1) Safety/target check
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
                stop_and_alert_target()
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
        update_target_on_miss(idx(pos[0], pos[1]))  # start cell is a searched/target-miss cell
    update_prob_map()
    publish_position()

    motors_off()
    METRIC_START_TIME_MS = time.ticks_ms()
    gc.collect()
    

def at_intersection_and_white():
    """
    Detect a 'clue' and simulates POD:
      - Center line sensor reads white ( < cfg.MIDDLE_WHITE_THRESH )
    Returns bool.
    """
    global clue_misses
    r = line_sensors.read_calibrated()      # [0]..[4], center is [2]
    if r[2] < cfg.MIDDLE_WHITE_THRESH:
        if random.random() <= clue_POD: # simulate POD
            flash_LEDS(BLUE,1)
            buzz('clue')
            return True
        else:
            clue_misses += 1
            buzz('missed_clue')
            flash_LEDS(RED,1)

            return False
    else:
        return False


def check_current_cell_for_clue(stage="start"):
    """Check the current cell for a clue without moving off of it."""
    global first_clue_seen, FIRST_CLUE_TIME_MS, system_clues_found, FIRST_CLUE_POSITION
    if not running or found_target:
        return
    if at_intersection_and_white():
        clue = (pos[0], pos[1])
        is_new = clue not in clues
        if is_new:
            clues.append(clue)
            system_clues_found += 1
        first_clue_seen = True
        if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
            FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
            FIRST_CLUE_POSITION = (pos[0], pos[1])
        publish_clue(pos[0], pos[1])
        if is_new:
            clue_probability_field(clue[0], clue[1])
            update_prob_map()
            gc.collect()
    else:
        # Checked current cell and did not detect a clue
        if 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE:
            update_clue_on_miss(idx(pos[0], pos[1]))


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
    time.sleep(.3)
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

def quarter_turns(from_dir, to_dir):
    if from_dir == to_dir:
        return 0
    if from_dir is None:
        return 1
    try:
        fi = DIRS4.index(from_dir)
        ti = DIRS4.index(to_dir)
    except ValueError:
        return 1
    delta = (ti - fi) % 4
    if delta == 2:
        return 2
    return 1

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

    i = DIRS4.index(heading)
    j = DIRS4.index(target)
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
# Reward Model (clues) & Pre-Clue Serpentine Bias
# ===========================================================
def update_prob_map():
    """
    Recompute target_p from all known clues (if any), then update prob_map
    as the unified value map:
        V(i) = P_target(i) + P_clue(i) * clue_POD

    Pre-clue (no clues yet):
        - We leave target_p as-is (typically uniform) and just recompute value.
        - clue_p is maintained incrementally via misses and clue_probability_field.
    Post-clue:
        - target_p[i] ∝ sum_k 1 / (1 + d(i, clue_k))**TARGET_DECAY_EXP
          for unsearched cells; 0 for visited.
        - clue_p has already been widened around clue locations.
    """
    has_clues = len(clues) > 0

    if has_clues:
        # Rebuild target_p from the clues with tunable target decay
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                i = idx(x, y)
                if grid[i] == CELL_SEARCHED:  # visited: target cannot be here (POD_target=1)
                    target_p[i] = 0.0
                    continue
                s = 0.0
                for (cx, cy) in clues:
                    d = manhattan(x, y, cx, cy)
                    s += 1.0 / ((1.0 + d) ** TARGET_DECAY_EXP)
                target_p[i] = s
        renorm(target_p)

    # Whether or not we have clues, recompute the unified value map
    recompute_value_map()


def update_clue_on_miss(i):
    """
    We searched cell i for a clue and did NOT detect one.
    Update clue_p[i] using a simple Bayes miss step with POD_clue.
    """
    p_i = clue_p[i]
    if p_i <= 0.0:
        return
    # Simple shrink then renormalize (approximate Bayes):
    clue_p[i] = p_i * (1.0 - clue_POD)
    renorm(clue_p)
    recompute_value_map()

def update_target_on_miss(i):
    """
    We have effectively searched cell i for the target (POD_target = 1)
    and did NOT find it. Set P_target(i) = 0 and renormalize.
    """
    if target_p[i] <= 0.0:
        return
    target_p[i] = 0.0
    renorm(target_p)
    recompute_value_map()


def clue_probability_field(cx, cy):
    """
    When we detect a clue at (cx, cy), that location joins 'clues' for the
    target correlation, AND we update clue_p to reflect that future clues
    are more likely in this general area, but with a WIDER decay.
    """
    n = GRID_SIZE * GRID_SIZE
    # Add a wide bump around (cx, cy)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            i = idx(x, y)
            if grid[i] == CELL_OBSTACLE:
                clue_p[i] = 0.0
                continue
            d = manhattan(x, y, cx, cy)
            # Wider decay: exponent CLUE_DECAY_EXP (usually < TARGET_DECAY_EXP)
            bump = 1.0 / ((1.0 + d) ** CLUE_DECAY_EXP)
            clue_p[i] += bump
    # We've already found the clue at (cx, cy); don't look for another there
    clue_p[idx(cx, cy)] = 0.0
    renorm(clue_p)
    recompute_value_map()

def pick_goal():
    """
    Select a search target inside our assigned region.
    Tier 1: best unsearched cell (highest prob_map, then closest).
    Tier 2: if everything is searched, pick the highest-probability cell
            in-region even if it is already marked searched.
    If no partitions are defined yet (no clue / sync), search the whole grid.
    """
    # Determine our search bounds
    if partitions is not None and my_region_index is not None and 0 <= my_region_index < len(partitions):
        xmin, xmax, ymin, ymax = partitions[my_region_index]
    else:
        xmin, xmax, ymin, ymax = 0, GRID_SIZE - 1, 0, GRID_SIZE - 1

    def select_candidate(allow_searched: bool):
        best = None
        best_prob = -1.0
        best_dist = None

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                cell_state = grid[idx(x, y)]
                if not allow_searched and cell_state != CELL_UNSEARCHED:
                    continue
                if cell_state == CELL_OBSTACLE:
                    continue
                p = prob_map[idx(x, y)]
                if p <= 0:
                    continue
                d = abs(x - pos[0]) + abs(y - pos[1])
                if p > best_prob:
                    best_prob = p
                    best = (x, y)
                    best_dist = d
                elif p == best_prob and best is not None and d < best_dist:
                    best = (x, y)
                    best_dist = d
        return best

    # First try unsearched cells; fall back to best overall in-region.
    goal = select_candidate(allow_searched=False)
    if goal is None:
        goal = select_candidate(allow_searched=True)
    return goal

def compute_partitions():
    """
    Compute 4 rectangular partitions of the GRID_SIZE x GRID_SIZE grid.
    Goal: balance (1) probability mass and (2) cell count.
    Method: guillotine split into 2, then split each half again.
    Returns:
        List of 4 regions: [(xmin, xmax, ymin, ymax), ...]
    """
    # ---- Helper: compute POA + cell count for a region ----
    def region_stats(xmin, xmax, ymin, ymax):
        total_poa = 0.0
        total_cells = 0
        weighted_x = 0.0
        weighted_y = 0.0
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                p = prob_map[idx(x, y)]
                total_poa += p
                total_cells += 1
                weighted_x += p * x
                weighted_y += p * y
        if total_poa > 0.0:
            cmx = weighted_x / total_poa
            cmy = weighted_y / total_poa
        else:
            cmx = (xmin + xmax) / 2
            cmy = (ymin + ymax) / 2
        return total_poa, total_cells, cmx, cmy

    # ---- Helper: cost function for splits ----
    # minimizes imbalance of POA + cell count + skinny region penalty
    def split_cost(left_poa, right_poa, left_cells, right_cells,
                   left_cmx, left_cmy, right_cmx, right_cmy,
                   aspect_penalty):
        poa_total = left_poa + right_poa
        cell_total = left_cells + right_cells

        # Normalize by totals so scale does not dominate
        poa_err  = abs(left_poa / poa_total - 0.5) if poa_total > 0 else 0.5
        cell_err = abs(left_cells / cell_total - 0.5) if cell_total > 0 else 0.5

        center_x = GRID_CENTER
        center_y = GRID_CENTER
        dist_left = (abs(left_cmx - center_x) + abs(left_cmy - center_y)) / GRID_SIZE
        dist_right = (abs(right_cmx - center_x) + abs(right_cmy - center_y)) / GRID_SIZE
        centroid_err = (dist_left + dist_right) / 2

        return (
            0.50 * poa_err
            + 0.20 * cell_err
            + 0.20 * centroid_err
            + 0.10 * aspect_penalty
        )

    # ---- Helper: compute aspect ratio penalty ----
    def aspect_ratio_penalty(width, height):
        if width == 0 or height == 0:
            return 10.0
        r = max(width / height, height / width)
        return r - 1.0  # 0 if square, grows as it becomes skinny

    # --------------------------------------------------------
    # 1) First split the full grid into TWO regions
    # --------------------------------------------------------

    best_first_split = None
    best_cost = 1e9

    # Try vertical splits: x = k
    for k in range(1, GRID_SIZE - 1):
        # Left = [0..k-1], Right = [k..end]
        left_poa, left_cells, left_cmx, left_cmy = region_stats(0, k - 1, 0, GRID_SIZE - 1)
        right_poa, right_cells, right_cmx, right_cmy = region_stats(k, GRID_SIZE - 1, 0, GRID_SIZE - 1)

        asp_L = aspect_ratio_penalty(k, GRID_SIZE)
        asp_R = aspect_ratio_penalty(GRID_SIZE - k, GRID_SIZE)
        asp = (asp_L + asp_R) / 2

        cost = split_cost(left_poa, right_poa, left_cells, right_cells,
                          left_cmx, left_cmy, right_cmx, right_cmy,
                          asp)
        if cost < best_cost:
            best_cost = cost
            best_first_split = ("V", k, (0, k - 1, 0, GRID_SIZE - 1), (k, GRID_SIZE - 1, 0, GRID_SIZE - 1))

    # Try horizontal splits: y = m
    for m in range(1, GRID_SIZE - 1):
        # Top = [0..m-1], Bottom = [m..end]
        top_poa, top_cells, top_cmx, top_cmy = region_stats(0, GRID_SIZE - 1, 0, m - 1)
        bottom_poa, bottom_cells, bottom_cmx, bottom_cmy = region_stats(0, GRID_SIZE - 1, m, GRID_SIZE - 1)

        asp_T = aspect_ratio_penalty(GRID_SIZE, m)
        asp_B = aspect_ratio_penalty(GRID_SIZE, GRID_SIZE - m)
        asp = (asp_T + asp_B) / 2

        cost = split_cost(top_poa, bottom_poa, top_cells, bottom_cells,
                          top_cmx, top_cmy, bottom_cmx, bottom_cmy,
                          asp)
        if cost < best_cost:
            best_cost = cost
            best_first_split = ("H", m, (0, GRID_SIZE - 1, 0, m - 1), (0, GRID_SIZE - 1, m, GRID_SIZE - 1))

    # Now we have the best first region split
    _, _, region_A, region_B = best_first_split

    # --------------------------------------------------------
    # 2) Split each of these into two again (same method)
    # --------------------------------------------------------
    final_regions = []

    for (xmin, xmax, ymin, ymax) in (region_A, region_B):

        best_sub_cost = 1e9
        best_subregion_split = None
        W = xmax - xmin + 1
        H = ymax - ymin + 1

        # Vertical splits inside this region
        for k in range(xmin + 1, xmax):
            left_poa, left_cells, left_cmx, left_cmy = region_stats(xmin, k - 1, ymin, ymax)
            right_poa, right_cells, right_cmx, right_cmy = region_stats(k, xmax, ymin, ymax)

            asp_L = aspect_ratio_penalty(k - xmin, H)
            asp_R = aspect_ratio_penalty(xmax - k + 1, H)
            asp = (asp_L + asp_R) / 2

            cost = split_cost(left_poa, right_poa, left_cells, right_cells,
                              left_cmx, left_cmy, right_cmx, right_cmy,
                              asp)
            if cost < best_sub_cost:
                best_sub_cost = cost
                best_subregion_split = [
                    (xmin, k - 1, ymin, ymax),
                    (k, xmax, ymin, ymax)
                ]

        # Horizontal splits in this region
        for m in range(ymin + 1, ymax):
            top_poa, top_cells, top_cmx, top_cmy = region_stats(xmin, xmax, ymin, m - 1)
            bottom_poa, bottom_cells, bottom_cmx, bottom_cmy = region_stats(xmin, xmax, m, ymax)

            asp_T = aspect_ratio_penalty(W, m - ymin)
            asp_B = aspect_ratio_penalty(W, ymax - m + 1)
            asp = (asp_T + asp_B) / 2

            cost = split_cost(top_poa, bottom_poa, top_cells, bottom_cells,
                              top_cmx, top_cmy, bottom_cmx, bottom_cmy,
                              asp)
            if cost < best_sub_cost:
                best_sub_cost = cost
                best_subregion_split = [
                    (xmin, xmax, ymin, m - 1),
                    (xmin, xmax, m, ymax)
                ]

        # Add the chosen split (2 regions)
        final_regions.extend(best_subregion_split)

    # final_regions is now 4 rectangles
    return final_regions



def assign_regions(regions, paused_positions):
    """
    Assign each region to a unique robot, using nearest-robot (Manhattan distance) greedy matching.
    paused_positions: dict robot_id -> (x, y)
    Returns: dict robot_id -> region_index
    """
    # Representative point for each region: its center
    centers = []
    for (xmin, xmax, ymin, ymax) in regions:
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        centers.append((cx, cy))

    remaining_regions = list(range(len(regions)))
    remaining_robots = sorted(paused_positions.keys())
    assignment = {}

    while remaining_regions and remaining_robots:
        best_pair = None
        best_dist = 10**9
        for rid in remaining_robots:
            rx, ry = paused_positions[rid]
            for j in remaining_regions:
                cx, cy = centers[j]
                d = abs(rx - cx) + abs(ry - cy)
                if (
                    d < best_dist
                    or (d == best_dist and (best_pair is None or j < best_pair[1] or (j == best_pair[1] and rid < best_pair[0])))
                ):
                    best_dist = d
                    best_pair = (rid, j)
        if best_pair is None:
            break
        rid, j = best_pair
        assignment[rid] = j
        remaining_robots.remove(rid)
        remaining_regions.remove(j)

    return assignment

def recompute_partitions_and_assign():
    """
    Called when we have sync_positions from all robots.
    Recompute partitions and update my_region_index.
    """
    global partitions, my_region_index

    partitions = compute_partitions()
    assignment = assign_regions(partitions, sync_positions)
    my_region_index = assignment.get(ROBOT_ID, None)

flash_LEDS(GREEN,1)
# ===========================================================
# A* Planner (4-neighbor grid, cardinal)
# ===========================================================
def a_star(start, goal):
    """
    A* over the 4-neighbor grid, with costs:
      +1 per step
      + TURN_COST per 90-degree heading change
      + cfg.VISITED_STEP_PENALTY if stepping onto a visited cell (grid==2)
    The reward from prob_map is applied as a bonus in the node priority.
    Returns a path as a list: [start, ..., goal], or [] if failure.
    """
    # Simple energy tracking - no function call counting needed
    frontier.clear()
    for i in range(GRID_SIZE * GRID_SIZE):
        came_from[i] = -1
        cost_so_far[i] = 1e30

    start_idx = idx(start[0], start[1])
    goal_idx = idx(goal[0], goal[1])
    heapq.heappush(frontier, (0, start_idx, heading))
    came_from[start_idx] = start_idx
    cost_so_far[start_idx] = 0.0
    turn_cost_per_turn = TURN_COST if not first_clue_seen else TURN_COST * 0.5

    while frontier and running and not found_target:
        _, current_idx, cur_dir = heapq.heappop(frontier)
        if current_idx == goal_idx:
            break

        cx = current_idx % GRID_SIZE
        cy = GRID_SIZE - 1 - (current_idx // GRID_SIZE)
        for dx, dy in DIRS4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue

            # Restrict planning to our assigned region, if defined:
            # allow paths outside only until we have entered our region.
            if partitions is not None and my_region_index is not None:
                xmin, xmax, ymin, ymax = partitions[my_region_index]
                inside_current = xmin <= cx <= xmax and ymin <= cy <= ymax
                inside_next = xmin <= nx <= xmax and ymin <= ny <= ymax
                if inside_current and not inside_next:
                    continue

            i = idx(nx, ny)
            if grid[i] == CELL_OBSTACLE:  # obstacle/reserved
                continue
            if peer_pos and (nx, ny) in peer_pos.values():
                continue

            move_cost = 1.0
            turns = quarter_turns(cur_dir, (dx, dy))
            turn_cost = turn_cost_per_turn * turns
            visited_pen = cfg.VISITED_STEP_PENALTY if grid[i] == CELL_SEARCHED else 0.0
            base_cost = move_cost + turn_cost + visited_pen

            reward_bonus = prob_map[i] * REWARD_FACTOR
            max_bonus = base_cost - 0.01
            if max_bonus < 0.0:
                max_bonus = 0.0
            if reward_bonus > max_bonus:
                reward_bonus = max_bonus

            step_cost = base_cost - reward_bonus
            if step_cost < 0.01:
                step_cost = 0.01

            new_cost = cost_so_far[current_idx] + step_cost

            if new_cost < cost_so_far[i]:
                cost_so_far[i] = new_cost
                priority = (
                    new_cost
                    + abs(goal[0] - nx)
                    + abs(goal[1] - ny)
                )
                heapq.heappush(frontier, (priority, i, (dx, dy)))
                came_from[i] = current_idx

    if came_from[goal_idx] == -1:
        return []

    # Reconstruct path
    path = []
    cur_idx = goal_idx
    while cur_idx != start_idx:
        x = cur_idx % GRID_SIZE
        y = GRID_SIZE - 1 - (cur_idx // GRID_SIZE)
        path.append((x, y))
        cur_idx = came_from[cur_idx]
    path.reverse()
    return [start] + path

def is_next_step_blocked_by_peer(cell):
    """Return True if cell is within Manhattan-1 of any known peer position."""
    cx, cy = cell
    for px, py in peer_pos.values():
        if abs(px - cx) + abs(py - cy) <= 1:
            return True
    return False

flash_LEDS(GREEN,1)
# ===========================================================
# Main Search Loop
# ===========================================================
def search_loop():
    """Main mission loop.

    1. Update the probability map.
    2. Choose a goal: sweep bias before clues, reward chasing after.
    3. Plan with A* using turn, center, and reward costs.
    4. Turn and advance one cell (abort on bump).
    5. Mark the cell, report status, and check for clues.
    6. Repeat until the target is found or no goals remain.

    Motors are always stopped in a ``finally`` block.
    """
    global first_clue_seen, move_forward_flag, start_signal, METRIC_START_TIME_MS, pos, yield_count, path_replan_count, goal_replan_count, FIRST_CLUE_TIME_MS, current_goal, system_clues_found, FIRST_CLUE_POSITION, system_visits, busy_ms, mem_free_min, sync_active, have_sent_sync

    try:
        calibrate()
        update_prob_map()

        # wait for hub start command, periodically sharing start position
        last_pose_publish = time.ticks_ms()
        while not start_signal:
            uart_service()
            now = time.ticks_ms()
            if time.ticks_diff(now, last_pose_publish) >= 3000:
                publish_position()
                last_pose_publish = now
            time.sleep_ms(10)
        if sync_active:
            if not have_sent_sync:
                publish_sync_state()

            # Wait here until we have sync positions from all robots
            while sync_active and running and not found_target:
                uart_service()
                # Do we have sync info from all robots yet?
                if len(sync_positions) >= NUM_ROBOTS:
                    # Recompute partitions and assignments
                    recompute_partitions_and_assign()
                    # Reset sync state and continue
                    sync_active = False
                    have_sent_sync = False
                    sync_positions.clear()
                    break
                time.sleep_ms(5)
        METRIC_START_TIME_MS = time.ticks_ms()
        check_current_cell_for_clue("start_signal")

        while running and not found_target:
            busy_timer_reset()
            # free any unused memory from previous iteration to avoid
            # MicroPython allocation failures during long searches
            gc.collect()
            update_mem_headroom()

            blocked_retry_cells = set()
            try:
                prev_goal = current_goal
                goal = pick_goal()
                if goal is None:
                    current_goal = None
                    break

                if goal != prev_goal:
                    if first_clue_seen:
                        goal_replan_count += 1
                    current_goal = goal

                while True:
                    # Temporarily treat any blocked retry cells as obstacles for planning
                    _block_backup = []
                    for bx, by in blocked_retry_cells:
                        ci = idx(bx, by)
                        _block_backup.append((ci, grid[ci]))
                        grid[ci] = CELL_OBSTACLE
                    try:
                        path = a_star(tuple(pos), goal)
                    finally:
                        for ci, prev_state in _block_backup:
                            grid[ci] = prev_state

                    if len(path) < 2:
                        break

                    nxt = path[1]
                    if is_next_step_blocked_by_peer(nxt):
                        path_replan_count += 1
                        blocked_retry_cells.add(nxt)
                        continue
                    # Safe to proceed
                    break

                update_mem_headroom()
                # Maintain low memory usage between planning iterations
                gc.collect()
                if len(path) < 2:
                    break

                nxt = path[1]

                # Face the neighbor and try to move one cell
                busy_timer_pause()
                turn_towards(tuple(pos), nxt)
                if not running or found_target:
                    break

                move_forward_flag = True
                while move_forward_flag:
                    uart_service()
                    time.sleep_ms(1)
                busy_timer_resume()

                # Arrived + update state & publish
                pos[0], pos[1] = nxt[0], nxt[1]
                record_intersection(pos[0], pos[1])
                cell_i = idx(pos[0], pos[1])
                grid[cell_i] = CELL_SEARCHED
                publish_position()
                update_target_on_miss(cell_i)

                # If we are in a sync round, publish our sync position once and wait
                if sync_active:
                    if not have_sent_sync:
                        publish_sync_state()

                    # Wait here until we have sync positions from all robots
                    while sync_active and running and not found_target:
                        uart_service()
                        # Do we have sync info from all robots yet?
                        if len(sync_positions) >= NUM_ROBOTS:
                            # Recompute partitions and assignments
                            recompute_partitions_and_assign()
                            # Reset sync state and continue
                            sync_active = False
                            have_sent_sync = False
                            sync_positions.clear()
                            break
                        time.sleep_ms(5)

                # Clue detection: centered + white center sensor
                detected = at_intersection_and_white()
                if detected:
                    clue = (pos[0], pos[1])
                    if clue not in clues:
                        clues.append(clue)
                        system_clues_found += 1
                        first_clue_seen = True
                        if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
                            FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
                            FIRST_CLUE_POSITION = (pos[0], pos[1])
                        publish_clue(pos[0], pos[1])

                        # Update both target and clue beliefs around the new clue
                        clue_probability_field(clue[0], clue[1])
                        update_prob_map()      # rebuild target_p from all clues, recompute value
                        update_mem_headroom()
                        gc.collect()

                        # Start/continue a sync round locally as well
                        sync_active = True
                        have_sent_sync = False
                        sync_positions.clear()
                else:
                    # We searched this cell and did NOT detect a clue
                    update_clue_on_miss(cell_i)


            finally:
                busy_ms += busy_timer_value_ms()
                update_mem_headroom()

    finally:
        motors_off()   # safety: ensure motors are cut even on exceptions
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
    metrics_log()
    flash_LEDS(RED,5)
    time.sleep_ms(200)  # give RX thread time to fall out cleanly



