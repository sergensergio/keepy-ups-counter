"""Keepy-ups counter driven by the ball tracker's metric state.

Algorithm: a 3-state machine on the ball's vertical velocity (in m/s,
image y-axis pointing down, so positive vy means *falling*).

  WAITING --(vy > V_FALL)-->  FALLING --(vy < V_RISE)--> RISING
                                                            |
                                              (vy > V_FALL) |
                                                            v
                                                         FALLING

A `FALLING -> RISING` transition is a *flick candidate*: the ball
suddenly reversed direction. To decide if it counts, the candidate is
classified by proximity to the player's pose keypoints:

  - Valid contact parts: head, shoulders, knees, feet
  - Explicitly excluded: arms (elbows, wrists, hands)

If the closest valid keypoint is within `contact_distance_m` of the
ball, the count is incremented and the contact part is recorded for the
HUD. Otherwise the reversal is treated as something we don't want to
count — most commonly a ground bounce (no body part nearby) or a hand
touch (arm keypoints are not in the candidate set).

A brief frame cooldown after each count prevents Kalman vy noise from
double-counting the same flick.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .types import Pose


class Phase(Enum):
    """Vertical-motion phase of the tracked ball.

    The flick detector counts a keepy-up on the `FALLING -> RISING`
    transition (sudden vy reversal). `WAITING` is the cold-start state
    used until the ball first exceeds the falling-velocity threshold,
    and the state we fall back to whenever the ball is fully untracked.
    """

    WAITING = auto()
    FALLING = auto()
    RISING = auto()


# BlazePose 33-keypoint indices grouped by the body parts the rules allow
# for a keepy-up. The `nose` neighbourhood (0..10) covers the whole head:
# any nearby head keypoint will pass the distance check, so heading the
# ball is robust to which one happens to be most visible.
_KEEPY_UP_KEYPOINT_GROUPS: Dict[str, Tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "shoulder": (11, 12),
    "knee": (25, 26),
    "foot": (27, 28, 29, 30, 31, 32),
}


@dataclass
class _CounterState:
    phase: Phase = field(default=Phase.WAITING)
    cooldown: int = 0


class KeepyUpsCounter:
    """Counts keepy-ups by detecting ball-velocity reversals near valid body parts."""

    def __init__(
        self,
        contact_distance_m: float = 0.3,
        head_extra_distance_m: float = 0.12,
        fall_threshold_m_s: float = 0.5,
        rise_threshold_m_s: float = -0.5,
        keypoint_conf_threshold: float = 0.3,
        cooldown_frames: int = 5,
    ):
        # Max distance in metres between the ball centroid and a valid
        # body-part keypoint at the moment of velocity reversal for the
        # flick to count. ~ball radius + body-part radius is a reasonable
        # default for a 22 cm football.
        self.contact_distance_m = contact_distance_m
        # Extra slack on top of `contact_distance_m` for the head group only.
        # BlazePose head keypoints (nose / eyes / ears) sit at the face plane,
        # but a header actually lands on the forehead or crown — roughly half
        # a head-height further from those keypoints than the foot/knee/
        # shoulder cases (where the keypoint sits at the joint with contact
        # happening near it).
        self.head_extra_distance_m = head_extra_distance_m
        # vy thresholds in m/s. `fall_threshold_m_s` is positive (image
        # y-axis points down); `rise_threshold_m_s` is negative.
        self.fall_threshold_m_s = fall_threshold_m_s
        self.rise_threshold_m_s = rise_threshold_m_s
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.cooldown_frames = cooldown_frames

        self.count: int = 0
        self.last_contact_part: Optional[str] = None
        self._state = _CounterState()

    @property
    def phase(self) -> Phase:
        return self._state.phase

    @property
    def cooldown(self) -> int:
        return self._state.cooldown

    def reset(self) -> None:
        self.count = 0
        self.last_contact_part = None
        self._state = _CounterState()

    def update(
        self,
        ball_pos_m: Optional[Tuple[float, float]],
        ball_vy_m_s: Optional[float],
        poses: List[Pose],
        px_per_m: float,
    ) -> None:
        """Step the state machine for one frame.

        `ball_pos_m` / `ball_vy_m_s` are `None` when no track is active
        (the ball is fully untracked). In that case the state machine is
        reset to WAITING so a stale FALLING state can't trigger a spurious
        flick when tracking resumes.
        """
        if self._state.cooldown > 0:
            self._state.cooldown -= 1

        if ball_pos_m is None or ball_vy_m_s is None:
            self._state.phase = Phase.WAITING
            return

        phase = self._state.phase
        vy = ball_vy_m_s

        if phase in (Phase.WAITING, Phase.RISING):
            if vy > self.fall_threshold_m_s:
                self._state.phase = Phase.FALLING
        elif phase == Phase.FALLING:
            if vy < self.rise_threshold_m_s:
                # Flick candidate — see if a valid body part is in contact.
                self._state.phase = Phase.RISING
                if self._state.cooldown == 0:
                    contact = _classify_contact(
                        ball_pos_m,
                        poses,
                        px_per_m,
                        self.contact_distance_m,
                        self.head_extra_distance_m,
                        self.keypoint_conf_threshold,
                    )
                    if contact is not None:
                        self.count += 1
                        self.last_contact_part = contact
                        self._state.cooldown = self.cooldown_frames


def _classify_contact(
    ball_pos_m: Tuple[float, float],
    poses: List[Pose],
    px_per_m: float,
    contact_distance_m: float,
    head_extra_distance_m: float,
    keypoint_conf_threshold: float,
) -> Optional[str]:
    """Return the name of the closest valid body part within its allowed
    contact radius of the ball, or `None` if none qualifies.

    Each group has its own radius: `contact_distance_m` for foot/knee/
    shoulder (joint keypoint ≈ contact surface), `contact_distance_m +
    head_extra_distance_m` for head (face keypoints are interior to the
    head; a header lands on the forehead/crown). Among keypoints that
    pass their group's radius, the absolute closest still wins so that a
    nearby foot beats a barely-in-range head.

    Returning `None` is what rejects ground bounces (no body part near
    the ball) and arm touches (arm keypoints aren't in the candidate set
    at all, so the closest *valid* keypoint is too far — head/torso).
    """
    if not poses or px_per_m <= 0:
        return None
    kps = poses[0].keypoints
    ball_x_m, ball_y_m = ball_pos_m
    inv_scale = 1.0 / px_per_m

    base_d2 = contact_distance_m * contact_distance_m
    head_radius = contact_distance_m + head_extra_distance_m
    head_d2 = head_radius * head_radius
    group_d2_max = {
        "head": head_d2,
        "shoulder": base_d2,
        "knee": base_d2,
        "foot": base_d2,
    }

    best_part: Optional[str] = None
    best_d2 = float("inf")
    for part, indices in _KEEPY_UP_KEYPOINT_GROUPS.items():
        max_d2 = group_d2_max[part]
        for i in indices:
            if i >= len(kps):
                continue
            kp = kps[i]
            if kp.confidence < keypoint_conf_threshold:
                continue
            dx = ball_x_m - kp.x * inv_scale
            dy = ball_y_m - kp.y * inv_scale
            d2 = dx * dx + dy * dy
            if d2 <= max_d2 and d2 < best_d2:
                best_d2 = d2
                best_part = part
    return best_part
