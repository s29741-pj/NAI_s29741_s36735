"""
Project: Intelligent Baby Monitor Simulator (Fuzzy Edge AI)

Description:
This project uses fuzzy logic to estimate the wake-up risk
of an infant based on changes in audio intensity, movement
activity, and time elapsed since the last feeding.

Input variables:
- Audio: Change in RMS audio level, measured in dB/s.
- Movement: Percentage of changed pixels between video frames.
- Feeding: Time elapsed since the last feeding, in minutes.

Output variable:
- Risk: Expert-based wake-up risk index ranging from 0 to 100.
  This value is not a statistically validated probability.

Authors:
- Jakub Kolad
- Igor Czeczot

Environment setup (run all commands from the repository root,
i.e. the directory containing main.py):
1. Create a virtual environment:
   python -m venv venv

2. Activate the environment:

   Windows:
   venv\\Scripts\\activate

   Linux/macOS:
   source venv/bin/activate

3. Install required packages:
   pip install -r requirements.txt

4. Run the application:
   python main.py

Documentation (generated from docstrings with pdoc):
- Generate static HTML into the docs/ directory:
  python -m pdoc -o docs fuzzy_logic main

- Or browse it live in the web browser:
  python -m pdoc fuzzy_logic main

Note:
The FuzzySystem is initialized with show_banner=False
to disable the Simpful welcome message.

All membership function parameters are experimental
design assumptions, not medical recommendations.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings
from fuzzy_logic import create_fuzzy_system, evaluate_risk


# Create the fuzzy system from the shared configuration
fs = create_fuzzy_system()

fig, ax = plt.subplots(figsize=(10, 6))

plt.subplots_adjust(
    left=0.15,
    bottom=0.35,
    top=0.9
)

ax.axis("off")

result_text = ax.text(
    0.5,
    0.65,
    "",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=17,
    transform=ax.transAxes
)

details_text = ax.text(
    0.5,
    0.35,
    "",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=11,
    transform=ax.transAxes
)

# AUDIO SLIDER
ax_audio = plt.axes([0.15, 0.24, 0.7, 0.03])
slider_audio = Slider(
    ax_audio,
    "Audio Change (dB/s)",
    -100,
    100,
    valinit=0
)

# MOVEMENT SLIDER
ax_movement = plt.axes([0.15, 0.17, 0.7, 0.03])
slider_movement = Slider(
    ax_movement,
    "Movement (%)",
    0,
    100,
    valinit=0
)

# FEEDING SLIDER
ax_feeding = plt.axes([0.15, 0.10, 0.7, 0.03])
slider_feeding = Slider(
    ax_feeding,
    "Time Since Last Feeding (min)",
    0,
    240,
    valinit=0
)


def update(val):
    """
    Update the risk display when slider values change.
    
    This function is called whenever any slider is moved. It retrieves
    the current slider values, performs fuzzy inference, and updates
    the displayed risk index and input details.
    
    Args:
        val: The slider value (not used, but required by Matplotlib).
    """
    audio_value = slider_audio.val
    movement_value = slider_movement.val
    feeding_value = slider_feeding.val
    
    try:
        # Suppress Simpful warnings about zero firing strengths in the GUI
        # These are expected for uncovered input combinations and don't
        # represent errors in the application logic
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*sum of rules' firing.*")
            result = evaluate_risk(fs, audio_value, movement_value, feeding_value)
        
        risk_value = result["Risk"]
        
        result_text.set_text(
            f"Wake-up Risk Index: {risk_value:.2f} / 100"
        )
        
        details_text.set_text(
            f"Audio change: {audio_value:.1f} dB/s\n"
            f"Movement: {movement_value:.1f}%\n"
            f"Time since last feeding: {feeding_value:.1f} min"
        )
    
    except Exception as error:
        result_text.set_text(
            "Unable to calculate risk"
        )
        
        details_text.set_text(str(error))
    
    fig.canvas.draw_idle()


# slider event handlers
slider_audio.on_changed(update)
slider_movement.on_changed(update)
slider_feeding.on_changed(update)


if __name__ == "__main__":
    # Perform the initial inference
    update(None)
    
    # Display the GUI
    plt.show()
