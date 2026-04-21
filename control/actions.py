"""Low-level OS action primitives via pynput and pyautogui.

Each function maps to exactly one user-visible action (scroll, click,
hotkey press).  All calls are guarded with ``pyautogui.FAILSAFE = True``
so that moving the mouse to the top-left corner aborts execution — a
critical safety measure for accessibility software.
"""

from __future__ import annotations

import pyautogui
from pynput.mouse import Button, Controller as MouseController

# Safety: moving mouse to (0, 0) aborts any pyautogui action
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0  # no artificial inter-action delay

_mouse = MouseController()


def scroll_up(amount: int = 5) -> None:
    """Scroll the active window upward."""
    pyautogui.scroll(amount)


def scroll_down(amount: int = 5) -> None:
    """Scroll the active window downward."""
    pyautogui.scroll(-amount)


def scroll_left(amount: int = 5) -> None:
    """Scroll the active window to the left (horizontal scroll)."""
    pyautogui.hscroll(-amount)


def scroll_right(amount: int = 5) -> None:
    """Scroll the active window to the right (horizontal scroll)."""
    pyautogui.hscroll(amount)


def left_click() -> None:
    """Perform a single left-click at the current cursor position."""
    _mouse.click(Button.left, 1)


def right_click() -> None:
    """Perform a single right-click at the current cursor position."""
    _mouse.click(Button.right, 1)


def app_launcher(hotkey: str = "win") -> None:
    """Open the OS application launcher.

    Parameters
    ----------
    hotkey:
        Key name recognised by ``pyautogui.press()``.  Defaults to the
        Windows key.
    """
    pyautogui.press(hotkey)


def no_action() -> None:
    """Intentional no-op for the *neutral* gesture class."""
