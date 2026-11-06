"""
Project: Intelligent Baby Monitor Simulator (Fuzzy Edge AI)
Module: Real-time "sleeping baby" simulation

Description:
This module is a time-driven demonstration of the fuzzy wake-up risk
estimator from fuzzy_logic.py. A virtual baby sleeps in a cot. Every tick
(1 simulated minute) the simulated sensors change: audio and movement follow
a smoothed random walk around a calm baseline, random events ("cry", "stir")
disturb the baby, and the time since the last feeding grows. The fuzzy system
turns the three sensor readings into a wake-up risk index on every tick.

The simulation is a closed loop. When the risk exceeds a high threshold, the
virtual parent starts soothing the baby (rocking / white noise). While
soothing is on, the sensor values return to the calm baseline faster and
the sensor noise is reduced, so the risk comes back down. Soothing is turned
off again once the risk falls below a lower threshold (hysteresis).

The window shows the baby drawn with matplotlib patches (eyes, mouth and
background colour depend on the risk), a scrolling plot of audio, movement
and risk, the current values and the soothing state. Buttons let the user
pause the simulation, feed the baby, make a noise and reset, and a slider
sets the initial time since the last feeding.

The simulation logic (BabySimulator) is independent of the GUI, so it can be
tested without opening a window.

Input variables:
- Audio: Change in RMS audio level, measured in dB/s (range: -100 to 100).
- Movement: Percentage of changed pixels between video frames (0 to 100).
- Feeding: Time elapsed since the last feeding, in minutes (0 to 240).

Output variable:
- Risk: Expert-based wake-up risk index ranging from 0 to 100.
  This value is not a statistically validated probability.

Authors:
- Jakub Kolad
- Igor Czeczot

Environment setup (run all commands from the directory containing
simulation.py, i.e. baby_monitor/):
1. Create a virtual environment:
   python -m venv venv

2. Activate the environment:

   Windows:
   venv\\Scripts\\activate

   Linux/macOS:
   source venv/bin/activate

3. Install required packages:
   pip install -r requirements.txt

4. Run the simulation:
   python simulation.py

Command line options (all optional):
- --seed N       Seed of the random number generator (default: 42).
                 The same seed always gives the same run.
- --feeding MIN  Initial time since the last feeding, in minutes
                 (default: 0, range 0 to 240).
- --interval MS  Delay between animation frames in milliseconds
                 (default: 150).
- --save PATH    Do not open a window. Render the animation to an animated
                 GIF at PATH instead (uses the Pillow writer).
- --frames N     Number of frames rendered with --save (default: 200).

Example (record a short GIF without a display):
   python simulation.py --save simulation_demo.gif --frames 120 --seed 42

Documentation (generated from docstrings with pdoc):
- Generate static HTML into the docs/ directory:
  python -m pdoc -o docs fuzzy_logic main simulation

Note:
Importing this module does not create any figure or window. The GUI is only
built by run(). All simulation parameters and membership function parameters
are experimental design assumptions, not medical recommendations.
"""

import argparse
import contextlib
import io
import itertools
import random
import warnings
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, Slider

from fuzzy_logic import check_rule_coverage, create_fuzzy_system, evaluate_risk


# =============================================================================
# 1. TUNING CONSTANTS - SIMULATION
# =============================================================================

# Valid ranges of the fuzzy system inputs (membership functions are only
# defined inside these ranges, so all values are clamped to them)
AUDIO_MIN = -100.0
AUDIO_MAX = 100.0
MOVEMENT_MIN = 0.0
MOVEMENT_MAX = 100.0
FEEDING_MIN = 0.0
FEEDING_MAX = 240.0
RISK_MIN = 0.0
RISK_MAX = 100.0

# Simulated minutes that pass during one tick
MINUTES_PER_TICK = 1.0

# Calm baseline that the sensors are pulled back to
AUDIO_BASELINE = 0.0        # dB/s, a sleeping baby has a stable sound level
MOVEMENT_BASELINE = 2.0     # %, almost no movement

# Fraction of the distance to the baseline that is removed in one tick
# (mean reversion strength). Larger value = faster return to calm.
AUDIO_PULL = 0.10
MOVEMENT_PULL = 0.06
SOOTHING_AUDIO_PULL = 0.25      # stronger pull while soothing is on
SOOTHING_MOVEMENT_PULL = 0.18

# Standard deviation of the Gaussian noise added in every tick
AUDIO_NOISE_STD = 2.5       # dB/s
MOVEMENT_NOISE_STD = 1.5    # %
SOOTHING_NOISE_FACTOR = 0.4  # noise is multiplied by this while soothing

# Probability of a random event in one tick. A hungry baby is restless, so
# the probability grows linearly with the time since the last feeding:
# 3% right after feeding up to 3% + 2% = 5% after 240 minutes.
EVENT_PROBABILITY_BASE = 0.03
EVENT_PROBABILITY_HUNGER_GAIN = 0.02
CRY_SHARE = 0.45            # fraction of events that are cries (rest: stirs)

# Size of the kick that an event adds to the sensors (uniform random range)
CRY_AUDIO_KICK = (55.0, 95.0)       # dB/s, loud sudden increase
CRY_MOVEMENT_KICK = (35.0, 65.0)    # %, flailing
STIR_AUDIO_KICK = (-30.0, 30.0)     # dB/s, may also be a decrease
STIR_MOVEMENT_KICK = (20.0, 45.0)   # %, turning over

# Closed loop with hysteresis: soothing starts when the previous risk reaches
# SOOTHING_ON_RISK and stops when it drops to SOOTHING_OFF_RISK or below
SOOTHING_ON_RISK = 65.0
SOOTHING_OFF_RISK = 35.0

# Names of the random events
EVENT_CRY = "cry"
EVENT_STIR = "stir"

# Default random seed (reproducible runs)
DEFAULT_SEED = 42


# =============================================================================
# 2. TUNING CONSTANTS - GUI
# =============================================================================

# Risk thresholds of the visual states (relative to the Risk fuzzy sets:
# low ends at 45, medium peaks at 50, high starts at 55)
VISUAL_LOW_RISK = 35.0      # below: sleeping soundly (green)
VISUAL_HIGH_RISK = 65.0     # at or above: about to wake up (red)

LEVEL_LOW = "low"
LEVEL_MEDIUM = "medium"
LEVEL_HIGH = "high"

# Background colours: green -> yellow -> red
COLOR_LOW = "#bfe8bf"
COLOR_MEDIUM = "#fff1a8"
COLOR_HIGH = "#f4a3a3"

# Baby drawing colours
COLOR_SKIN = "#ffd9b8"
COLOR_OUTLINE = "#5a3b24"
COLOR_BLANKET = "#8fb8e8"
COLOR_BLANKET_STRIPE = "#c9defa"
COLOR_WOOD = "#d9b38c"
COLOR_WOOD_DARK = "#a67c52"
COLOR_MATTRESS = "#f6f1e7"
COLOR_PILLOW = "#ffffff"

# Time-series plot
HISTORY_LENGTH = 100        # number of ticks visible in the plot
PLOT_Y_MIN = -100.0
PLOT_Y_MAX = 100.0

# Simulated wall clock: the night starts at 22:00
CLOCK_START_MINUTE = 22 * 60

# Animation
DEFAULT_INTERVAL_MS = 150
DEFAULT_SAVE_FRAMES = 200
FIGURE_SIZE = (11.0, 5.8)   # inches
SAVE_DPI = 70               # resolution of the exported GIF

# Baby drawing geometry (data coordinates of the baby axes, 10 x 8)
HEAD_CENTER = (2.8, 3.5)
HEAD_RADIUS = 1.15
EYE_Y = 3.68
EYE_X_OFFSET = 0.38         # distance of each eye from the head centre line
MOUTH_Y = 3.05

# Eye shapes as (width of the white, height of the white, pupil radius)
EYE_HALF_OPEN = (0.40, 0.20, 0.08)
EYE_OPEN = (0.46, 0.46, 0.14)


# =============================================================================
# 3. SIMULATION LOGIC (NO GUI)
# =============================================================================

def clamp(value, low, high):
    """
    Limit a value to the closed interval [low, high].

    Args:
        value (float): The value to limit.
        low (float): The lower bound.
        high (float): The upper bound.

    Returns:
        float: low if value < low, high if value > high, otherwise value.
    """
    return max(low, min(high, value))


def risk_level(risk):
    """
    Classify a risk value into one of three visual states.

    Args:
        risk (float): Wake-up risk index (range: 0 to 100).

    Returns:
        str: LEVEL_LOW if risk < VISUAL_LOW_RISK, LEVEL_HIGH if
             risk >= VISUAL_HIGH_RISK, otherwise LEVEL_MEDIUM.
    """
    if risk < VISUAL_LOW_RISK:
        return LEVEL_LOW
    if risk >= VISUAL_HIGH_RISK:
        return LEVEL_HIGH
    return LEVEL_MEDIUM


class BabySimulator:
    """
    Time-driven simulation of a sleeping baby watched by a fuzzy monitor.

    The simulator owns the sensor state (audio, movement, feeding), a private
    random number generator and the soothing flag. Each call of step()
    advances the simulation by one tick (one simulated minute) and evaluates
    the fuzzy wake-up risk. It does not use matplotlib.

    Attributes:
        seed (int): Seed used to initialise the random number generator.
        rng (random.Random): The private random number generator.
        audio (float): Audio change in dB/s (-100 to 100).
        movement (float): Movement in percent (0 to 100).
        feeding (float): Minutes since the last feeding (0 to 240).
        tick (int): Number of simulated minutes since the last reset.
        soothing (bool): True while the parent is rocking the baby.
        risk (float): Latest wake-up risk index (0 to 100).
        last_event (str | None): Event that happened in the latest tick
            (EVENT_CRY or EVENT_STIR), or None.
        last_error (str | None): Message of the latest inference error, or
            None if the latest inference succeeded.
        uncovered_ticks (int): Number of evaluations for which no fuzzy rule
            fired (the previous risk value was kept in these cases).
    """

    def __init__(self, seed=DEFAULT_SEED, initial_feeding=0.0,
                 fuzzy_system=None):
        """
        Create a simulator in its initial calm state.

        Args:
            seed (int): Seed of the random number generator.
            initial_feeding (float): Initial minutes since the last feeding.
                Values outside 0 to 240 are clamped.
            fuzzy_system (FuzzySystem): Configured fuzzy system. If None, a
                new one is created with create_fuzzy_system().
        """
        self.seed = seed
        if fuzzy_system is None:
            fuzzy_system = create_fuzzy_system()
        self.fuzzy_system = fuzzy_system
        self.reset(initial_feeding)

    def reset(self, initial_feeding=0.0):
        """
        Restart the simulation from a calm state.

        The random number generator is re-seeded with the original seed, so a
        reset run with the same initial feeding time is identical to the
        first one.

        Args:
            initial_feeding (float): Initial minutes since the last feeding.
                Values outside 0 to 240 are clamped.
        """
        self.rng = random.Random(self.seed)
        self.audio = AUDIO_BASELINE
        self.movement = MOVEMENT_BASELINE
        self.feeding = clamp(float(initial_feeding), FEEDING_MIN, FEEDING_MAX)
        self.tick = 0
        self.soothing = False
        self.risk = RISK_MIN
        self.last_event = None
        self.last_error = None
        self.uncovered_ticks = 0
        self._update_risk()

    def event_probability(self):
        """
        Compute the probability of a random event in the next tick.

        Returns:
            float: Probability between EVENT_PROBABILITY_BASE (just fed) and
                   EVENT_PROBABILITY_BASE + EVENT_PROBABILITY_HUNGER_GAIN
                   (not fed for 240 minutes).
        """
        hunger = (self.feeding - FEEDING_MIN) / (FEEDING_MAX - FEEDING_MIN)
        return EVENT_PROBABILITY_BASE + EVENT_PROBABILITY_HUNGER_GAIN * hunger

    def step(self):
        """
        Advance the simulation by one tick (one simulated minute).

        Order of operations: time passes, the closed-loop controller updates
        the soothing flag from the previous risk, audio and movement take a
        mean-reverting random step, a random event may happen, values are
        clamped and finally the fuzzy risk is evaluated.

        Returns:
            float: The wake-up risk index after this tick.
        """
        self.tick += 1
        self.feeding = min(self.feeding + MINUTES_PER_TICK, FEEDING_MAX)

        self._update_soothing()

        if self.soothing:
            audio_pull = SOOTHING_AUDIO_PULL
            movement_pull = SOOTHING_MOVEMENT_PULL
            noise_factor = SOOTHING_NOISE_FACTOR
        else:
            audio_pull = AUDIO_PULL
            movement_pull = MOVEMENT_PULL
            noise_factor = 1.0

        self.audio += (
            audio_pull * (AUDIO_BASELINE - self.audio)
            + self.rng.gauss(0.0, AUDIO_NOISE_STD * noise_factor)
        )
        self.movement += (
            movement_pull * (MOVEMENT_BASELINE - self.movement)
            + self.rng.gauss(0.0, MOVEMENT_NOISE_STD * noise_factor)
        )

        # Random event: the kick is added on top of the smoothed walk and
        # then decays through the mean reversion of the following ticks
        self.last_event = None
        if self.rng.random() < self.event_probability():
            if self.rng.random() < CRY_SHARE:
                self._apply_event(EVENT_CRY)
            else:
                self._apply_event(EVENT_STIR)

        self._clamp_state()
        self._update_risk()
        return self.risk

    def feed(self):
        """
        Feed the baby: reset the time since the last feeding to zero.

        The risk is re-evaluated immediately, so the effect is visible even
        when the simulation is paused.
        """
        self.feeding = FEEDING_MIN
        self._update_risk()

    def trigger_noise(self):
        """
        Force a loud event (a cry) immediately.

        The kick is added to the current audio and movement values, and the
        risk is re-evaluated immediately. The kick then decays during the
        following ticks like any other event.
        """
        self._apply_event(EVENT_CRY)
        self._clamp_state()
        self._update_risk()

    def _apply_event(self, kind):
        """
        Add the kick of an event to the audio and movement values.

        Args:
            kind (str): EVENT_CRY (loud and strong movement) or EVENT_STIR
                (mild sound change in either direction and some movement).
        """
        if kind == EVENT_CRY:
            audio_kick = self.rng.uniform(*CRY_AUDIO_KICK)
            movement_kick = self.rng.uniform(*CRY_MOVEMENT_KICK)
        else:
            audio_kick = self.rng.uniform(*STIR_AUDIO_KICK)
            movement_kick = self.rng.uniform(*STIR_MOVEMENT_KICK)
        self.audio += audio_kick
        self.movement += movement_kick
        self.last_event = kind

    def _clamp_state(self):
        """
        Clamp audio, movement and feeding to their valid input ranges.
        """
        self.audio = clamp(self.audio, AUDIO_MIN, AUDIO_MAX)
        self.movement = clamp(self.movement, MOVEMENT_MIN, MOVEMENT_MAX)
        self.feeding = clamp(self.feeding, FEEDING_MIN, FEEDING_MAX)

    def _update_soothing(self):
        """
        Closed-loop controller: switch soothing on or off from the last risk.

        Two different thresholds (hysteresis) prevent the flag from
        flickering when the risk hovers around a single threshold.
        """
        if not self.soothing and self.risk >= SOOTHING_ON_RISK:
            self.soothing = True
        elif self.soothing and self.risk <= SOOTHING_OFF_RISK:
            self.soothing = False

    def _update_risk(self):
        """
        Evaluate the fuzzy system for the current state and store the risk.

        Simpful reports uncovered input combinations by printing a warning
        and returning 0, which would show up as a false "perfectly calm"
        reading. In that case the previous risk value is kept and
        uncovered_ticks is incremented. If inference raises an exception,
        the previous risk is kept as well and the message is stored in
        last_error, so one bad tick never stops the simulation.
        """
        try:
            # Suppress Simpful warnings about zero firing strengths. Simpful
            # uses print() for them, so stdout is redirected as well.
            with warnings.catch_warnings(), \
                    contextlib.redirect_stdout(io.StringIO()):
                warnings.filterwarnings(
                    "ignore", message=".*sum of rules' firing.*"
                )
                result = evaluate_risk(
                    self.fuzzy_system, self.audio, self.movement, self.feeding
                )
                risk = float(result["Risk"])
                if risk == 0.0:
                    covered, _ = check_rule_coverage(
                        self.fuzzy_system,
                        self.audio,
                        self.movement,
                        self.feeding,
                    )
                    if not covered:
                        self.uncovered_ticks += 1
                        self.last_error = None
                        return
            self.risk = clamp(risk, RISK_MIN, RISK_MAX)
            self.last_error = None
        except Exception as error:
            self.last_error = str(error)


# =============================================================================
# 4. GUI HELPERS
# =============================================================================

# Colour map of the background: green below VISUAL_LOW_RISK, yellow in the
# middle and red from VISUAL_HIGH_RISK, with smooth transitions in between
RISK_COLORMAP = LinearSegmentedColormap.from_list(
    "risk",
    [
        (0.0, COLOR_LOW),
        (VISUAL_LOW_RISK / RISK_MAX, COLOR_LOW),
        ((VISUAL_LOW_RISK + VISUAL_HIGH_RISK) / 2 / RISK_MAX, COLOR_MEDIUM),
        (VISUAL_HIGH_RISK / RISK_MAX, COLOR_HIGH),
        (1.0, COLOR_HIGH),
    ],
)


def risk_color(risk):
    """
    Map a risk value to a background colour.

    Args:
        risk (float): Wake-up risk index (range: 0 to 100).

    Returns:
        tuple: RGBA colour from RISK_COLORMAP.
    """
    return RISK_COLORMAP(clamp(risk, RISK_MIN, RISK_MAX) / RISK_MAX)


def format_clock(tick):
    """
    Convert a tick number to a simulated wall clock time.

    Args:
        tick (int): Simulated minutes since the start of the night.

    Returns:
        str: Time of day as "HH:MM", starting at CLOCK_START_MINUTE.
    """
    minutes = (CLOCK_START_MINUTE + int(tick)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# =============================================================================
# 5. GUI
# =============================================================================

class SimulationUI:
    """
    Matplotlib window that animates and controls a BabySimulator.

    The window contains the drawn baby (left), a scrolling time-series plot
    (right), text with the current values, and the controls (Start/Pause,
    Feed, Noise, Reset and the "Initial feeding time" slider). The figure is
    only created when the object is constructed, never at import time.

    Attributes:
        simulator (BabySimulator): The simulation that is displayed.
        running (bool): True while the animation advances the simulation.
        history (dict): Deques with the last HISTORY_LENGTH values of
            tick, audio, movement and risk.
        fig (matplotlib.figure.Figure): The figure of the window.
        animation (FuncAnimation): The animation. This reference must be kept
            alive, otherwise the animation is garbage-collected and stops.
    """

    def __init__(self, simulator, interval=DEFAULT_INTERVAL_MS, frames=None):
        """
        Build the figure, the widgets and the animation.

        Args:
            simulator (BabySimulator): The simulation to display.
            interval (int): Delay between animation frames in milliseconds.
            frames (int | None): Number of frames to animate. None means an
                endless animation (interactive use).
        """
        self.simulator = simulator
        self.running = False
        self.history = {
            "tick": deque(maxlen=HISTORY_LENGTH),
            "audio": deque(maxlen=HISTORY_LENGTH),
            "movement": deque(maxlen=HISTORY_LENGTH),
            "risk": deque(maxlen=HISTORY_LENGTH),
        }

        self.fig = plt.figure(figsize=FIGURE_SIZE)
        self._build_baby()
        self._build_plot()
        self._build_texts()
        self._build_controls()

        self._record()
        self.refresh()

        self.animation = FuncAnimation(
            self.fig,
            self.on_frame,
            frames=frames if frames is not None else itertools.count(),
            interval=interval,
            blit=False,
            cache_frame_data=False,
        )

    # -- construction ---------------------------------------------------------

    def _build_baby(self):
        """
        Draw the cot, the baby and the risk-dependent parts with patches.

        Static parts (cot, pillow, head, blanket) are drawn once. The parts
        that change with the risk (background, eyes, mouth, "Zzz", "!") are
        stored as attributes and updated by _apply_visuals().
        """
        ax = self.fig.add_axes([0.02, 0.30, 0.46, 0.58])
        self.ax_baby = ax
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Background that is tinted by the risk
        self.background = Rectangle((0, 0), 10, 8, zorder=0)
        ax.add_patch(self.background)

        # Cot: back panel, bars, top rail, legs and mattress
        ax.add_patch(Rectangle((0.5, 1.0), 9.0, 5.4, fc=COLOR_WOOD,
                               ec=COLOR_OUTLINE, lw=2, zorder=1))
        bar_count = 9
        for index in range(bar_count):
            x = 1.2 + index * (7.6 / (bar_count - 1))
            ax.add_patch(Rectangle((x - 0.07, 2.0), 0.14, 4.4,
                                   fc=COLOR_WOOD_DARK, ec="none", zorder=2))
        ax.add_patch(Rectangle((0.4, 6.4), 9.2, 0.4, fc=COLOR_WOOD_DARK,
                               ec=COLOR_OUTLINE, lw=2, zorder=2))
        for x in (0.6, 8.8):
            ax.add_patch(Rectangle((x, 0.3), 0.3, 0.7, fc=COLOR_WOOD_DARK,
                                   ec=COLOR_OUTLINE, lw=1.5, zorder=1))
        ax.add_patch(Rectangle((0.9, 1.0), 8.2, 1.0, fc=COLOR_MATTRESS,
                               ec=COLOR_OUTLINE, lw=1.5, zorder=3))

        # Pillow, head, hair curl and cheeks
        ax.add_patch(Ellipse((HEAD_CENTER[0], 2.45), 3.6, 0.9,
                             fc=COLOR_PILLOW, ec=COLOR_OUTLINE, lw=1.5,
                             zorder=4))
        ax.add_patch(Circle(HEAD_CENTER, HEAD_RADIUS, fc=COLOR_SKIN,
                            ec=COLOR_OUTLINE, lw=2, zorder=5))
        ax.add_patch(Arc((HEAD_CENTER[0], HEAD_CENTER[1] + HEAD_RADIUS),
                         0.6, 0.5, theta1=20, theta2=160,
                         ec=COLOR_OUTLINE, lw=2, zorder=6))
        for sign in (-1, 1):
            ax.add_patch(Circle((HEAD_CENTER[0] + sign * 0.62, MOUTH_Y + 0.15),
                                0.17, fc="#ff9aa2", ec="none", alpha=0.5,
                                zorder=6))

        self._build_face(ax)

        # Blanket (drawn over the baby's body, to the right of the head)
        ax.add_patch(FancyBboxPatch(
            (3.75, 1.95), 5.1, 2.25,
            boxstyle="round,pad=0.05,rounding_size=0.5",
            fc=COLOR_BLANKET, ec=COLOR_OUTLINE, lw=2, zorder=9))
        for x in (5.6, 6.7):
            ax.add_patch(Rectangle((x, 2.0), 0.3, 2.15,
                                   fc=COLOR_BLANKET_STRIPE, ec="none",
                                   zorder=10))

        # Text labels: state caption, "Zzz", "!" and soothing hint
        label_box = dict(boxstyle="round", fc="white", ec="none", alpha=0.75)
        self.caption_text = ax.text(5.0, 7.4, "", ha="center", va="center",
                                    fontsize=15, fontweight="bold", zorder=11)
        self.zzz_text = ax.text(5.6, 5.3, "Z z z", ha="center", va="center",
                                fontsize=22, fontstyle="italic",
                                fontweight="bold", color="#35508a",
                                bbox=label_box, zorder=11)
        self.alert_text = ax.text(HEAD_CENTER[0], 5.95, "!", ha="center",
                                  va="center", fontsize=26,
                                  fontweight="bold", color="#c0392b",
                                  bbox=label_box, zorder=11)
        self.soothing_hint = ax.text(7.2, 0.55, "", ha="center", va="center",
                                     fontsize=12, fontstyle="italic",
                                     color="#2e6b3a", zorder=11)

    def _build_face(self, ax):
        """
        Create the eyes and mouth patches whose look depends on the risk.

        Each eye consists of a closed lid (arc), the white of the eye and a
        pupil. The mouth is a smile (arc) or an open mouth (ellipse).

        Args:
            ax (matplotlib.axes.Axes): The axes of the baby drawing.
        """
        self.eye_lids = []
        self.eye_whites = []
        self.eye_pupils = []
        for sign in (-1, 1):
            x = HEAD_CENTER[0] + sign * EYE_X_OFFSET
            lid = Arc((x, EYE_Y), 0.45, 0.3, theta1=200, theta2=340,
                      ec=COLOR_OUTLINE, lw=2.5, zorder=7)
            white = Ellipse((x, EYE_Y), EYE_OPEN[0], EYE_OPEN[1], fc="white",
                            ec=COLOR_OUTLINE, lw=1.5, zorder=7)
            pupil = Circle((x, EYE_Y), EYE_OPEN[2], fc=COLOR_OUTLINE,
                           ec="none", zorder=8)
            for patch in (lid, white, pupil):
                ax.add_patch(patch)
            self.eye_lids.append(lid)
            self.eye_whites.append(white)
            self.eye_pupils.append(pupil)

        self.mouth_smile = Arc((HEAD_CENTER[0], MOUTH_Y + 0.08), 0.5, 0.3,
                               theta1=200, theta2=340, ec=COLOR_OUTLINE,
                               lw=2, zorder=7)
        self.mouth_open = Ellipse((HEAD_CENTER[0], MOUTH_Y), 0.34, 0.42,
                                  fc="#c0392b", ec=COLOR_OUTLINE, lw=1.5,
                                  zorder=7)
        ax.add_patch(self.mouth_smile)
        ax.add_patch(self.mouth_open)

    def _build_plot(self):
        """
        Create the scrolling time-series plot of audio, movement and risk.
        """
        ax = self.fig.add_axes([0.58, 0.40, 0.39, 0.45])
        self.ax_plot = ax
        (self.line_audio,) = ax.plot([], [], color="tab:blue", lw=1.3,
                                     label="Audio change (dB/s)")
        (self.line_movement,) = ax.plot([], [], color="tab:orange", lw=1.3,
                                        label="Movement (%)")
        (self.line_risk,) = ax.plot([], [], color="tab:red", lw=2.5,
                                    label="Wake-up risk (0-100)")
        for level in (VISUAL_LOW_RISK, VISUAL_HIGH_RISK):
            ax.axhline(level, color="gray", ls="--", lw=0.8)
        ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
        ax.set_xlim(0, HISTORY_LENGTH)
        ax.set_xlabel("Simulated time (minutes since start)")
        ax.set_ylabel("Sensor value / risk index")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                  fontsize=8, frameon=False, columnspacing=1.0,
                  handlelength=1.5)

    def _build_texts(self):
        """
        Create the risk headline, the details text and the soothing label.
        """
        self.risk_text = self.fig.text(0.5, 0.955, "", ha="center",
                                       va="center", fontsize=18,
                                       fontweight="bold")
        self.details_text = self.fig.text(0.58, 0.315, "", ha="left",
                                          va="top", fontsize=9.5,
                                          family="monospace")
        self.soothing_text = self.fig.text(0.05, 0.255, "", ha="left",
                                           va="center", fontsize=12,
                                           fontweight="bold")

    def _build_controls(self):
        """
        Create the buttons and the initial feeding slider.

        The widgets are stored as attributes so they are not garbage-collected
        (matplotlib widgets stop responding when no reference is kept).
        """
        ax_slider = self.fig.add_axes([0.28, 0.115, 0.62, 0.03])
        self.slider_feeding = Slider(
            ax_slider,
            "Initial feeding time (min)",
            FEEDING_MIN,
            FEEDING_MAX,
            valinit=self.simulator.feeding,
        )
        self.slider_feeding.on_changed(self.on_slider)

        specs = [
            ("button_start", "Start", 0.08, self.on_start_pause),
            ("button_feed", "Feed", 0.30, self.on_feed),
            ("button_noise", "Noise", 0.52, self.on_noise),
            ("button_reset", "Reset", 0.74, self.on_reset),
        ]
        for name, label, x, callback in specs:
            button = Button(self.fig.add_axes([x, 0.025, 0.18, 0.065]), label)
            button.on_clicked(callback)
            setattr(self, name, button)

    # -- state -> display -----------------------------------------------------

    def _record(self):
        """
        Append the current simulator state to the plot history.
        """
        sim = self.simulator
        self.history["tick"].append(sim.tick)
        self.history["audio"].append(sim.audio)
        self.history["movement"].append(sim.movement)
        self.history["risk"].append(sim.risk)

    def _clear_history(self):
        """
        Remove all points from the plot history.
        """
        for values in self.history.values():
            values.clear()

    def _update_last_point(self):
        """
        Overwrite the newest history point with the current simulator state.

        Feed and Noise change the state between two ticks, so the plot shows
        their effect without adding an extra point on the time axis.
        """
        for values in self.history.values():
            values.pop()
        self._record()

    def _apply_visuals(self, risk):
        """
        Update the baby drawing for the given risk.

        Low risk: green background, closed eyes, smile and "Zzz".
        Medium risk: yellow background and half-open eyes.
        High risk: red background, wide-open eyes, open mouth and "!".

        Args:
            risk (float): Wake-up risk index (range: 0 to 100).
        """
        level = risk_level(risk)
        self.background.set_facecolor(risk_color(risk))

        for lid, white, pupil in zip(self.eye_lids, self.eye_whites,
                                     self.eye_pupils):
            lid.set_visible(level == LEVEL_LOW)
            white.set_visible(level != LEVEL_LOW)
            pupil.set_visible(level != LEVEL_LOW)
            if level == LEVEL_MEDIUM:
                white.set_width(EYE_HALF_OPEN[0])
                white.set_height(EYE_HALF_OPEN[1])
                pupil.set_radius(EYE_HALF_OPEN[2])
            elif level == LEVEL_HIGH:
                white.set_width(EYE_OPEN[0])
                white.set_height(EYE_OPEN[1])
                pupil.set_radius(EYE_OPEN[2])

        self.mouth_smile.set_visible(level != LEVEL_HIGH)
        self.mouth_open.set_visible(level == LEVEL_HIGH)
        self.zzz_text.set_visible(level == LEVEL_LOW)
        self.alert_text.set_visible(level == LEVEL_HIGH)
        self.caption_text.set_text({
            LEVEL_LOW: "Sleeping soundly",
            LEVEL_MEDIUM: "Restless",
            LEVEL_HIGH: "Waking up!",
        }[level])

    def refresh(self):
        """
        Redraw all dynamic elements from the simulator and the history.

        Updates the plot lines and the scrolling x range, the headline, the
        details, the soothing label and the baby drawing. If the latest
        inference failed, an error message is shown instead of the risk,
        like in main.py.
        """
        sim = self.simulator
        ticks = list(self.history["tick"])
        self.line_audio.set_data(ticks, list(self.history["audio"]))
        self.line_movement.set_data(ticks, list(self.history["movement"]))
        self.line_risk.set_data(ticks, list(self.history["risk"]))
        right = max(sim.tick, HISTORY_LENGTH)
        self.ax_plot.set_xlim(right - HISTORY_LENGTH, right)

        if sim.last_error is not None:
            # The error replaces the event line to keep the text compact
            self.risk_text.set_text("Unable to calculate risk")
            last_line = f"Error: {sim.last_error[:40]}"
        else:
            self.risk_text.set_text(
                f"Wake-up Risk Index: {sim.risk:.1f} / 100"
            )
            event = sim.last_event if sim.last_event is not None else "-"
            last_line = f"Event this tick:   {event}"

        self.details_text.set_text(
            f"Clock (simulated): {format_clock(sim.tick)}"
            f"  (tick {sim.tick})\n"
            f"Audio change:      {sim.audio:7.1f} dB/s\n"
            f"Movement:          {sim.movement:7.1f} %\n"
            f"Since last feed:   {sim.feeding:7.0f} min\n"
            f"{last_line}"
        )

        if sim.soothing:
            self.soothing_text.set_text("Soothing: ON")
            self.soothing_text.set_color("#2e6b3a")
            self.soothing_hint.set_text("~ rocking ~")
        else:
            self.soothing_text.set_text("Soothing: OFF")
            self.soothing_text.set_color("#555555")
            self.soothing_hint.set_text("")

        self._apply_visuals(sim.risk)

    # -- animation and widget callbacks ---------------------------------------

    def on_frame(self, _frame):
        """
        Animation callback: advance the simulation by one tick if running.

        Args:
            _frame: Frame number supplied by FuncAnimation (unused).
        """
        if self.running:
            self.simulator.step()
            self._record()
        self.refresh()

    def on_start_pause(self, _event):
        """
        Toggle between running and paused and update the button label.

        Args:
            _event: Matplotlib button event (unused).
        """
        self.running = not self.running
        self.button_start.label.set_text("Pause" if self.running else "Start")
        self.fig.canvas.draw_idle()

    def on_feed(self, _event):
        """
        Feed the baby and show the effect immediately.

        Args:
            _event: Matplotlib button event (unused).
        """
        self.simulator.feed()
        self._update_last_point()
        self.refresh()
        self.fig.canvas.draw_idle()

    def on_noise(self, _event):
        """
        Make a loud noise and show the effect immediately.

        Args:
            _event: Matplotlib button event (unused).
        """
        self.simulator.trigger_noise()
        self._update_last_point()
        self.refresh()
        self.fig.canvas.draw_idle()

    def on_reset(self, _event):
        """
        Reset the simulation using the slider value as initial feeding time.

        The simulation is paused after the reset.

        Args:
            _event: Matplotlib button event (unused).
        """
        self.simulator.reset(self.slider_feeding.val)
        self._clear_history()
        self._record()
        self.running = False
        self.button_start.label.set_text("Start")
        self.refresh()
        self.fig.canvas.draw_idle()

    def on_slider(self, value):
        """
        Apply the initial feeding time if the simulation has not started yet.

        Once the simulation has advanced, the slider value is only used by
        the next reset.

        Args:
            value (float): New slider value in minutes.
        """
        if self.simulator.tick == 0:
            self.simulator.reset(value)
            self._clear_history()
            self._record()
            self.refresh()
            self.fig.canvas.draw_idle()


# =============================================================================
# 6. ENTRY POINT
# =============================================================================

def run(seed=DEFAULT_SEED, initial_feeding=0.0,
        interval=DEFAULT_INTERVAL_MS, save_path=None,
        frames=DEFAULT_SAVE_FRAMES, show=True):
    """
    Build the simulation window and either show it or record it to a GIF.

    No figure exists before this function is called.

    Args:
        seed (int): Seed of the random number generator.
        initial_feeding (float): Initial minutes since the last feeding.
        interval (int): Delay between animation frames in milliseconds.
        save_path (str | None): If given, the animation is rendered with the
            Pillow writer to this GIF file and no window is shown.
        frames (int): Number of frames rendered when save_path is given.
        show (bool): If False and save_path is None, the figure is built but
            plt.show() is not called (used for tests under the Agg backend).

    Returns:
        SimulationUI: The user interface object. It holds the reference to
        the animation, which must stay alive while the animation runs.
    """
    simulator = BabySimulator(seed=seed, initial_feeding=initial_feeding)
    ui = SimulationUI(
        simulator,
        interval=interval,
        frames=frames if save_path else None,
    )

    if save_path:
        # Start running immediately so that the recording shows the baby
        ui.running = True
        ui.button_start.label.set_text("Pause")
        fps = max(1, round(1000 / interval))
        ui.animation.save(save_path, writer="pillow", fps=fps, dpi=SAVE_DPI)
        plt.close(ui.fig)
        print(f"Saved {frames} frames to {save_path}")
    elif show:
        plt.show()

    return ui


def parse_args(argv=None):
    """
    Parse the command line options.

    Args:
        argv (list[str] | None): Arguments to parse. None means sys.argv.

    Returns:
        argparse.Namespace: Parsed options seed, feeding, interval, save and
        frames.
    """
    parser = argparse.ArgumentParser(
        description="Real-time sleeping baby simulation driven by the "
                    "fuzzy wake-up risk estimator."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed for reproducible runs "
                             "(default: %(default)s)")
    parser.add_argument("--feeding", type=float, default=0.0,
                        help="initial time since last feeding in minutes "
                             "(default: %(default)s)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_MS,
                        help="delay between frames in ms "
                             "(default: %(default)s)")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="render the animation to a GIF file instead of "
                             "opening a window")
    parser.add_argument("--frames", type=int, default=DEFAULT_SAVE_FRAMES,
                        help="number of frames rendered with --save "
                             "(default: %(default)s)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    ui = run(
        seed=args.seed,
        initial_feeding=args.feeding,
        interval=args.interval,
        save_path=args.save,
        frames=args.frames,
    )
