#!/usr/bin/env python3
"""A single-window live tuning studio for an OpenCV webcam."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np

WINDOW_NAME = "Camera Studio"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 860
UVC_HELPER = Path(__file__).resolve().parent / ".camera-tools/camtint/bin/uvcctl"

BACKGROUND = (18, 20, 26)
CARD = (29, 32, 41)
CARD_LIGHT = (37, 41, 52)
BORDER = (54, 59, 73)
TEXT = (235, 238, 244)
MUTED = (151, 158, 176)
ACCENT = (237, 151, 68)
GREEN = (101, 211, 137)
YELLOW = (82, 191, 244)
RED = (106, 106, 245)


@dataclass(frozen=True)
class ControlSpec:
    name: str
    group: str
    property_id: int
    minimum: float
    maximum: float
    step: float
    default: float
    toggle: bool = False
    uvc_name: str = ""
    options: tuple[float, ...] = ()

    @property
    def positions(self) -> int:
        if self.options:
            return len(self.options) - 1
        return round((self.maximum - self.minimum) / self.step)

    def decode(self, position: int) -> float:
        if self.options:
            return self.options[min(max(position, 0), len(self.options) - 1)]
        return self.minimum + position * self.step

    def encode(self, value: float) -> int:
        if self.options:
            return min(range(len(self.options)), key=lambda index: abs(self.options[index] - value))
        value = min(max(value, self.minimum), self.maximum)
        return round((value - self.minimum) / self.step)


@dataclass
class ControlState:
    spec: ControlSpec
    startup: float
    requested: float
    actual: float
    status: str
    touched: bool = False
    last_apply: float = 0.0


@dataclass
class InterfaceState:
    slider_rects: dict[str, tuple[int, int, int, int]]
    toggle_rects: dict[str, tuple[int, int, int, int]]
    button_rects: dict[str, tuple[int, int, int, int]]
    active_slider: str | None = None
    pending_action: str | None = None
    toast: str = ""
    toast_frames: int = 0


class UVCController:
    """Drive real USB Video Class controls through macOS IOKit."""

    def __init__(self, executable: Path, device: str | None) -> None:
        self.executable = executable
        self.device = device

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        command = [str(self.executable), *arguments]
        if self.device:
            command.extend(("-d", self.device))
        return subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)

    def capabilities(self) -> dict[str, Any]:
        result = self._run("caps")
        if result.returncode != 0:
            detail = result.stderr.strip() or "the camera did not return its UVC controls"
            raise RuntimeError(detail)
        return json.loads(result.stdout)

    def set(self, name: str, value: float) -> tuple[bool, float, str]:
        result = self._run("set", name, str(round(value)))
        if result.returncode != 0:
            return False, value, result.stderr.strip() or "UVC write failed"
        try:
            actual = float(result.stdout.strip())
        except ValueError:
            actual = value
        return True, actual, ""

    def get(self, name: str) -> float | None:
        result = self._run("get", name)
        if result.returncode != 0:
            return None
        try:
            return float(result.stdout.strip())
        except ValueError:
            return None

    def reset(self) -> tuple[bool, str]:
        result = self._run("reset")
        return result.returncode == 0, result.stderr.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", nargs="?", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--device",
        help="UVC USB device ID as vendor:product hex, for example 090c:b371",
    )
    return parser.parse_args()


def control_specs() -> list[ControlSpec]:
    return [
        ControlSpec("Autofocus", "FOCUS", cv2.CAP_PROP_AUTOFOCUS, 0, 1, 1, 1, True),
        ControlSpec("Focus", "FOCUS", cv2.CAP_PROP_FOCUS, 0, 255, 1, 128),
        ControlSpec(
            "Auto exposure",
            "EXPOSURE",
            cv2.CAP_PROP_AUTO_EXPOSURE,
            0.25,
            0.75,
            0.5,
            0.75,
            True,
        ),
        ControlSpec("Exposure", "EXPOSURE", cv2.CAP_PROP_EXPOSURE, -20, 1, 0.25, -6),
        ControlSpec("Gain", "EXPOSURE", cv2.CAP_PROP_GAIN, 0, 255, 1, 0),
        ControlSpec(
            "Auto white balance", "COLOR", cv2.CAP_PROP_AUTO_WB, 0, 1, 1, 1, True
        ),
        ControlSpec(
            "White balance", "COLOR", cv2.CAP_PROP_WB_TEMPERATURE, 2000, 10000, 50, 4500
        ),
        ControlSpec("Brightness", "COLOR", cv2.CAP_PROP_BRIGHTNESS, 0, 255, 1, 128),
        ControlSpec("Contrast", "COLOR", cv2.CAP_PROP_CONTRAST, 0, 255, 1, 128),
        ControlSpec("Saturation", "COLOR", cv2.CAP_PROP_SATURATION, 0, 255, 1, 128),
        ControlSpec("Sharpness", "IMAGE", cv2.CAP_PROP_SHARPNESS, 0, 255, 1, 128),
        ControlSpec("Gamma", "IMAGE", cv2.CAP_PROP_GAMMA, 1, 500, 1, 100),
        ControlSpec("Zoom", "IMAGE", cv2.CAP_PROP_ZOOM, 0, 400, 1, 100),
    ]


def controls_from_uvc(payload: dict[str, Any]) -> dict[str, ControlState]:
    states: dict[str, ControlState] = {}
    for item in payload.get("controls", []):
        if not item.get("writable", False):
            continue

        control_name = str(item["name"])
        kind = str(item.get("kind", "range"))
        options: tuple[float, ...] = ()
        if control_name == "exposure_auto":
            mask = int(item.get("step", 0))
            options = tuple(float(value) for value in (1, 2, 4, 8) if not mask or mask & value)
        elif control_name == "power_line_frequency":
            options = (0.0, 1.0, 2.0)

        minimum = float(item.get("min", 0))
        maximum = float(item.get("max", 1))
        step = max(1.0, float(item.get("step", 1)))
        value = float(item.get("value", item.get("default", minimum)))
        spec = ControlSpec(
            name=str(item.get("label", control_name.replace("_", " ").title())),
            group=str(item.get("group", "image")).upper(),
            property_id=-1,
            minimum=minimum,
            maximum=maximum,
            step=step,
            default=float(item.get("default", value)),
            toggle=kind == "bool",
            uvc_name=control_name,
            options=options,
        )
        states[control_name] = ControlState(spec, value, value, value, "ready")
    return states


def finite_or(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback


def current_or_default(camera: cv2.VideoCapture, spec: ControlSpec) -> float:
    value = finite_or(camera.get(spec.property_id), spec.default)
    if value < spec.minimum or value > spec.maximum:
        return spec.default
    return value


def format_value(value: float) -> str:
    if not math.isfinite(value) or value == -1:
        return "N/A"
    if abs(value - round(value)) < 0.001:
        return str(round(value))
    return f"{value:.2f}"


def display_value(state: ControlState) -> str:
    if state.spec.toggle:
        return "ON" if state.requested > state.spec.minimum else "OFF"
    if state.spec.uvc_name == "exposure_auto":
        return {1: "MANUAL", 2: "AUTO", 4: "SHUTTER", 8: "APERTURE"}.get(
            round(state.requested),
            format_value(state.requested),
        )
    if state.spec.uvc_name == "power_line_frequency":
        return {0: "OFF", 1: "50 HZ", 2: "60 HZ"}.get(
            round(state.requested),
            format_value(state.requested),
        )
    return format_value(state.requested)


def focus_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    return float(cv2.Laplacian(center, cv2.CV_64F).var())


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_card(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int] = CARD,
) -> None:
    left, top, right, bottom = rect
    cv2.rectangle(image, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (left, top), (right, bottom), BORDER, 1, cv2.LINE_AA)


def fit_preview(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    resized_width = max(1, round(frame_width * scale))
    resized_height = max(1, round(frame_height * scale))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def status_color(status: str) -> tuple[int, int, int]:
    if status == "applied":
        return GREEN
    if "rejected" in status or "unsupported" in status:
        return RED
    if "another" in status:
        return YELLOW
    return MUTED


def draw_header(canvas: np.ndarray, camera: cv2.VideoCapture, camera_index: int) -> None:
    draw_text(canvas, "CAMERA STUDIO", (24, 37), 0.74, TEXT, 2)
    draw_text(canvas, "Live image tuning", (24, 59), 0.44, MUTED)
    cv2.circle(canvas, (1054, 37), 5, GREEN, -1, cv2.LINE_AA)
    draw_text(canvas, f"Camera {camera_index} connected", (1068, 42), 0.48, TEXT)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    draw_text(canvas, f"{width} x {height}  |  {fps:.1f} FPS", (1068, 61), 0.38, MUTED)
    cv2.line(canvas, (24, 73), (1416, 73), BORDER, 1, cv2.LINE_AA)


def draw_preview(canvas: np.ndarray, frame: np.ndarray, score: float) -> None:
    rect = (24, 92, 968, 624)
    draw_card(canvas, rect, (10, 11, 15))
    left, top, right, bottom = rect
    available_width = right - left - 4
    available_height = bottom - top - 4
    preview = fit_preview(frame, available_width, available_height)
    preview_height, preview_width = preview.shape[:2]
    x = left + 2 + (available_width - preview_width) // 2
    y = top + 2 + (available_height - preview_height) // 2
    canvas[y : y + preview_height, x : x + preview_width] = preview

    center_x = x + preview_width // 2
    center_y = y + preview_height // 2
    guide_width = preview_width // 3
    guide_height = preview_height // 3
    cv2.rectangle(
        canvas,
        (center_x - guide_width // 2, center_y - guide_height // 2),
        (center_x + guide_width // 2, center_y + guide_height // 2),
        (100, 210, 150),
        1,
        cv2.LINE_AA,
    )

    badge = (left + 18, top + 18, left + 260, top + 56)
    cv2.rectangle(canvas, badge[:2], badge[2:], (20, 24, 30), -1, cv2.LINE_AA)
    cv2.circle(canvas, (left + 34, top + 37), 5, GREEN, -1, cv2.LINE_AA)
    draw_text(canvas, f"CENTER SHARPNESS  {score:.1f}", (left + 48, top + 43), 0.43, TEXT)


def draw_info_cards(canvas: np.ndarray) -> None:
    cards = [
        (
            (24, 642, 322, 738),
            "FOCUS",
            "Disable autofocus, then adjust",
            "Focus for maximum sharpness.",
        ),
        (
            (336, 642, 648, 738),
            "EXPOSURE",
            "Disable auto exposure first.",
            "Balance exposure and gain.",
        ),
        (
            (662, 642, 968, 738),
            "COLOR",
            "Disable auto white balance.",
            "Tune temperature and saturation.",
        ),
    ]
    for rect, title, line_one, line_two in cards:
        draw_card(canvas, rect)
        left, top, _, _ = rect
        draw_text(canvas, title, (left + 14, top + 24), 0.42, ACCENT, 1)
        draw_text(canvas, line_one, (left + 14, top + 53), 0.42, TEXT)
        draw_text(canvas, line_two, (left + 14, top + 76), 0.42, MUTED)


def draw_controls(
    canvas: np.ndarray,
    states: dict[str, ControlState],
    interface: InterfaceState,
) -> None:
    panel = (992, 92, 1416, 738)
    draw_card(canvas, panel)
    draw_text(canvas, "CONTROLS", (1012, 120), 0.55, TEXT, 2)
    draw_text(canvas, "Changes are applied live", (1012, 140), 0.40, MUTED)

    interface.slider_rects.clear()
    interface.toggle_rects.clear()
    y = 154
    previous_group = ""
    for name, state in states.items():
        spec = state.spec
        if spec.group != previous_group:
            if previous_group:
                y += 3
            draw_text(canvas, spec.group, (1012, y), 0.35, ACCENT, 1)
            y += 17
            previous_group = spec.group

        value_text = display_value(state)
        if "rejected" in state.status:
            value_text = "REJECTED"
        elif "unsupported" in state.status:
            value_text = "N/A"
        draw_text(canvas, name, (1012, y + 9), 0.40, TEXT)
        draw_text(canvas, value_text, (1340, y + 9), 0.38, status_color(state.status))
        cv2.circle(canvas, (1400, y + 4), 3, status_color(state.status), -1, cv2.LINE_AA)

        if spec.toggle:
            rect = (1280, y - 5, 1325, y + 13)
            interface.toggle_rects[name] = rect
            enabled = state.requested > spec.minimum
            track_color = ACCENT if enabled else BORDER
            cv2.rectangle(canvas, rect[:2], rect[2:], track_color, -1, cv2.LINE_AA)
            knob_x = rect[2] - 10 if enabled else rect[0] + 10
            cv2.circle(canvas, (knob_x, (rect[1] + rect[3]) // 2), 7, TEXT, -1, cv2.LINE_AA)
            y += 28
        else:
            slider = (1012, y + 14, 1388, y + 24)
            interface.slider_rects[name] = slider
            line_y = (slider[1] + slider[3]) // 2
            cv2.line(canvas, (slider[0], line_y), (slider[2], line_y), BORDER, 4, cv2.LINE_AA)
            ratio = state.spec.encode(state.requested) / max(1, state.spec.positions)
            knob_x = round(slider[0] + ratio * (slider[2] - slider[0]))
            cv2.line(canvas, (slider[0], line_y), (knob_x, line_y), ACCENT, 4, cv2.LINE_AA)
            cv2.circle(canvas, (knob_x, line_y), 7, TEXT, -1, cv2.LINE_AA)
            cv2.circle(canvas, (knob_x, line_y), 8, ACCENT, 1, cv2.LINE_AA)
            y += 34

    draw_text(
        canvas,
        "Status: gray ready  |  green applied  |  red unavailable",
        (1012, 723),
        0.31,
        MUTED,
    )


def draw_buttons(canvas: np.ndarray, interface: InterfaceState) -> None:
    buttons = [
        ("save", "SAVE SETTINGS", (992, 758, 1125, 808), ACCENT),
        ("capture", "CAPTURE FRAME", (1137, 758, 1270, 808), CARD_LIGHT),
        ("reset", "RESET", (1282, 758, 1416, 808), CARD_LIGHT),
    ]
    interface.button_rects.clear()
    for action, label, rect, color in buttons:
        interface.button_rects[action] = rect
        cv2.rectangle(canvas, rect[:2], rect[2:], color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, rect[:2], rect[2:], BORDER, 1, cv2.LINE_AA)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        x = rect[0] + (rect[2] - rect[0] - text_size[0]) // 2
        y = rect[1] + (rect[3] - rect[1] + text_size[1]) // 2
        draw_text(canvas, label, (x, y), 0.35, TEXT, 1)

    draw_text(
        canvas,
        "Shortcuts: S save  |  P capture  |  R reset  |  Q quit",
        (24, 786),
        0.43,
        MUTED,
    )
    draw_text(
        canvas,
        "Tip: compare sharpness scores only while the scene and lighting stay unchanged.",
        (24, 816),
        0.43,
        MUTED,
    )


def draw_toast(canvas: np.ndarray, interface: InterfaceState) -> None:
    if not interface.toast or interface.toast_frames <= 0:
        return
    text_size = cv2.getTextSize(interface.toast, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    width = text_size[0] + 36
    rect = (24, 830, 24 + width, 854)
    cv2.rectangle(canvas, rect[:2], rect[2:], (44, 62, 52), -1, cv2.LINE_AA)
    draw_text(canvas, interface.toast, (42, 848), 0.48, GREEN)
    interface.toast_frames -= 1


def save_settings(
    path: Path,
    args: argparse.Namespace,
    camera: cv2.VideoCapture,
    states: dict[str, ControlState],
) -> None:
    payload: dict[str, Any] = {
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_index": args.index,
        "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": camera.get(cv2.CAP_PROP_FPS),
        "controls": {
            name: {
                "requested": state.requested,
                "readback": state.actual,
                "status": state.status,
            }
            for name, state in states.items()
            if state.touched
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_control(controller: UVCController, state: ControlState, requested: float) -> None:
    accepted, actual, _detail = controller.set(state.spec.uvc_name, requested)
    state.requested = requested
    state.actual = actual
    state.touched = True
    state.last_apply = monotonic()
    tolerance = max(state.spec.step * 1.5, abs(requested) * 0.05)
    if not accepted:
        state.status = "rejected by backend"
    elif actual == -1:
        state.status = "accepted; no readback"
    elif abs(actual - requested) <= tolerance:
        state.status = "applied"
    else:
        state.status = "camera chose another value"


def update_slider_from_mouse(
    controller: UVCController,
    state: ControlState,
    rect: tuple[int, int, int, int],
    mouse_x: int,
    *,
    force: bool = False,
) -> None:
    ratio = (mouse_x - rect[0]) / max(1, rect[2] - rect[0])
    ratio = min(max(ratio, 0.0), 1.0)
    position = round(ratio * state.spec.positions)
    requested = state.spec.decode(position)
    state.requested = requested
    if force or monotonic() - state.last_apply >= 0.08:
        apply_control(controller, state, requested)


def build_mouse_handler(
    controller: UVCController,
    states: dict[str, ControlState],
    interface: InterfaceState,
):
    def handle(event: int, x: int, y: int, flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, rect in interface.button_rects.items():
                if point_in_rect(x, y, rect):
                    interface.pending_action = action
                    return
            for name, rect in interface.toggle_rects.items():
                if point_in_rect(x, y, rect):
                    state = states[name]
                    new_value = (
                        state.spec.minimum
                        if state.requested > state.spec.minimum
                        else state.spec.maximum
                    )
                    apply_control(controller, state, new_value)
                    return
            for name, rect in interface.slider_rects.items():
                expanded = (rect[0], rect[1] - 8, rect[2], rect[3] + 8)
                if point_in_rect(x, y, expanded):
                    interface.active_slider = name
                    update_slider_from_mouse(controller, states[name], rect, x, force=True)
                    return
        elif event == cv2.EVENT_MOUSEMOVE and interface.active_slider is not None:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                name = interface.active_slider
                update_slider_from_mouse(
                    controller,
                    states[name],
                    interface.slider_rects[name],
                    x,
                )
        elif event == cv2.EVENT_LBUTTONUP:
            if interface.active_slider is not None:
                state = states[interface.active_slider]
                apply_control(controller, state, state.requested)
            interface.active_slider = None

    return handle


def reset_controls(controller: UVCController, states: dict[str, ControlState]) -> None:
    accepted, _detail = controller.reset()
    for state in states.values():
        actual = controller.get(state.spec.uvc_name) if accepted else None
        if actual is None:
            state.status = "reset failed"
            continue
        state.startup = actual
        state.requested = actual
        state.actual = actual
        state.touched = False
        state.status = "restored"


def main() -> int:
    args = parse_args()
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.index, backend)
    if not camera.isOpened():
        print(f"Could not open camera index {args.index}.", file=sys.stderr)
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)

    if not UVC_HELPER.is_file():
        camera.release()
        print(f"Native UVC helper not found: {UVC_HELPER}", file=sys.stderr)
        return 1

    controller = UVCController(UVC_HELPER, args.device)
    try:
        capabilities = controller.capabilities()
    except (RuntimeError, json.JSONDecodeError) as exc:
        camera.release()
        print(f"Could not query native UVC controls: {exc}", file=sys.stderr)
        print("Reconnect the USB camera and verify the --device ID.", file=sys.stderr)
        return 1

    states = controls_from_uvc(capabilities)
    if not states:
        camera.release()
        print("This camera did not advertise any writable UVC controls.", file=sys.stderr)
        return 1

    interface = InterfaceState({}, {}, {})
    settings_path = Path(f"camera_{args.index}_settings.json").resolve()
    frame_path = Path(f"camera_{args.index}_tuned.png").resolve()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, build_mouse_handler(controller, states, interface))
    camera_name = capabilities.get("camera", {}).get("name", "USB camera")
    print(
        f"Camera Studio opened for {camera_name}. "
        "Use the on-screen controls or S/P/R/Q shortcuts."
    )

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("The camera stopped returning frames.", file=sys.stderr)
                return 1

            canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), BACKGROUND, dtype=np.uint8)
            draw_header(canvas, camera, args.index)
            draw_preview(canvas, frame, focus_score(frame))
            draw_info_cards(canvas)
            draw_controls(canvas, states, interface)
            draw_buttons(canvas, interface)
            draw_toast(canvas, interface)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s"):
                interface.pending_action = "save"
            elif key == ord("p"):
                interface.pending_action = "capture"
            elif key == ord("r"):
                interface.pending_action = "reset"

            if interface.pending_action == "save":
                save_settings(settings_path, args, camera, states)
                interface.toast = f"Settings saved: {settings_path.name}"
                interface.toast_frames = 90
            elif interface.pending_action == "capture":
                if cv2.imwrite(str(frame_path), frame):
                    interface.toast = f"Frame saved: {frame_path.name}"
                    interface.toast_frames = 90
            elif interface.pending_action == "reset":
                reset_controls(controller, states)
                interface.toast = "Controls restored to startup values"
                interface.toast_frames = 90
            interface.pending_action = None

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
