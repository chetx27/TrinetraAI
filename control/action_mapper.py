"""Maps gesture/head-pose labels to OS actions.

The mapping is driven entirely by ``gesture_action_map`` in
``settings.yaml``.  Each value must correspond to a function name
exported by :mod:`control.actions`.
"""

from __future__ import annotations

from typing import Callable

from control import actions as act
from config.loader import ActionsConfig

# Registry of all known action names → callables
_ACTION_REGISTRY: dict[str, Callable[..., None]] = {
    "scroll_up": act.scroll_up,
    "scroll_down": act.scroll_down,
    "scroll_left": act.scroll_left,
    "scroll_right": act.scroll_right,
    "left_click": act.left_click,
    "right_click": act.right_click,
    "app_launcher": act.app_launcher,
    "no_action": act.no_action,
}


class ActionMapper:
    """Resolve a gesture label to an OS action and execute it.

    Parameters
    ----------
    gesture_action_map:
        ``{gesture_label: action_name}`` dict from the config file.
    actions_config:
        Action-specific parameters (scroll amount, hotkey, …).
    """

    def __init__(
        self,
        gesture_action_map: dict[str, str],
        actions_config: ActionsConfig,
    ) -> None:
        self._map: dict[str, str] = gesture_action_map
        self._actions_cfg: ActionsConfig = actions_config

        # Validate at init time
        for label, action_name in self._map.items():
            if action_name not in _ACTION_REGISTRY:
                raise ValueError(
                    f"Unknown action '{action_name}' for gesture '{label}'.  "
                    f"Available: {sorted(_ACTION_REGISTRY)}"
                )

    def execute(self, label: str) -> None:
        """Look up and run the action bound to *label*.

        Parameters
        ----------
        label:
            Gesture or head-pose label string (e.g. ``"open_palm"``).
            Unknown labels are silently ignored (treated as neutral).
        """
        action_name = self._map.get(label)
        if action_name is None or action_name == "no_action":
            return

        fn = _ACTION_REGISTRY[action_name]

        # Inject config-driven kwargs where applicable
        if action_name in {"scroll_up", "scroll_down", "scroll_left", "scroll_right"}:
            fn(amount=self._actions_cfg.scroll_amount)
        elif action_name == "app_launcher":
            fn(hotkey=self._actions_cfg.app_launcher_hotkey)
        else:
            fn()
