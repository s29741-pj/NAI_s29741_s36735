"""
Fuzzy Logic Module for Intelligent Baby Monitor Simulator

This module defines and configures the fuzzy inference system used to estimate
the wake-up risk of an infant based on sensor inputs.

The fuzzy system is shared by the interactive slider GUI (main.py) and the
time-driven simulation (simulation.py), ensuring that both use exactly the
same membership functions and rules.

Input variables:
- Audio: Change in RMS audio level, measured in dB/s (range: -100 to 100)
- Movement: Percentage of changed pixels between video frames (range: 0 to 100)
- Feeding: Time elapsed since the last feeding, in minutes (range: 0 to 240)

Output variable:
- Risk: Expert-based wake-up risk index (range: 0 to 100)
  This value is not a statistically validated probability.

All membership function parameters are experimental design assumptions,
not medical recommendations.

Authors:
- Jakub Kolad
- Igor Czeczot
"""

from simpful import (
    FuzzySystem,
    FuzzySet,
    LinguisticVariable,
    Triangular_MF,
    Trapezoidal_MF,
    Gaussian_MF
)


def create_fuzzy_system(save_figure=False):
    """
    Create and configure the fuzzy inference system for baby monitoring.
    
    Args:
        save_figure (bool): If True, plot the membership functions of all
                            linguistic variables and save them to
                            'inputs.png' in the current working directory.
                            Default is False, so creating the system has no
                            side effects on the file system.
    
    Returns:
        FuzzySystem: A configured Simpful FuzzySystem instance with all
                     linguistic variables and fuzzy rules defined.
    """
    
    fs = FuzzySystem(show_banner=False)

    # Audio variable
    audio_decrease = FuzzySet(
        function=Trapezoidal_MF(-100, -100, -80, 0),
        term="sudden_decrease"
    )
    
    audio_stable = FuzzySet(
        function=Triangular_MF(-25, 0, 25),
        term="stable"
    )
    
    audio_slight = FuzzySet(
        function=Triangular_MF(0, 25, 50),
        term="slight_increase"
    )
    
    audio_sudden = FuzzySet(
        function=Trapezoidal_MF(25, 70, 100, 100),
        term="sudden_increase"
    )
    
    fs.add_linguistic_variable(
        "Audio",
        LinguisticVariable(
            [
                audio_decrease,
                audio_stable,
                audio_slight,
                audio_sudden
            ],
            universe_of_discourse=[-100, 100]
        )
    )
    
    # Movement variable
    movement_none = FuzzySet(
        function=Trapezoidal_MF(0, 0, 3, 10),
        term="no_movement"
    )
    
    movement_low = FuzzySet(
        function=Triangular_MF(5, 15, 30),
        term="low"
    )
    
    movement_moderate = FuzzySet(
        function=Triangular_MF(20, 35, 55),
        term="moderate"
    )
    
    movement_high = FuzzySet(
        function=Triangular_MF(45, 60, 80),
        term="high"
    )
    
    movement_intense = FuzzySet(
        function=Trapezoidal_MF(70, 85, 100, 100),
        term="intense"
    )
    
    fs.add_linguistic_variable(
        "Movement",
        LinguisticVariable(
            [
                movement_none,
                movement_low,
                movement_moderate,
                movement_high,
                movement_intense
            ],
            universe_of_discourse=[0, 100]
        )
    )
    
    # Feeding variable
    feeding_recent = FuzzySet(
        function=Gaussian_MF(mu=0, sigma=80), 
        term="recent"
    )
    
    feeding_long_time = FuzzySet(
        function=Gaussian_MF(mu=240, sigma=60),
        term="long_time"
    )
    
    fs.add_linguistic_variable(
        "Feeding",
        LinguisticVariable(
            [
                feeding_recent,
                feeding_long_time
            ],
            universe_of_discourse=[0, 240]
        )
    )
    
    # Risk output variable
    risk_low = FuzzySet(
        function=Trapezoidal_MF(0, 0, 20, 45),
        term="low"
    )
    
    risk_medium = FuzzySet(
        function=Triangular_MF(30, 50, 70),
        term="medium"
    )
    
    risk_high = FuzzySet(
        function=Trapezoidal_MF(55, 75, 100, 100),
        term="high"
    )
    
    fs.add_linguistic_variable(
        "Risk",
        LinguisticVariable(
            [
                risk_low,
                risk_medium,
                risk_high
            ],
            universe_of_discourse=[0, 100]
        )
    )
    
    rules = [
        "IF (Audio IS sudden_increase) THEN (Risk IS high)",
        "IF (Movement IS intense) THEN (Risk IS high)",
        "IF (Audio IS slight_increase) AND (Movement IS high) THEN (Risk IS high)",
        "IF (Audio IS sudden_decrease) AND (Movement IS high) THEN (Risk IS high)",
        "IF (Audio IS sudden_decrease) AND (Movement IS intense) THEN (Risk IS high)",
        
        # Umiarkowany ruch (bez płaczu)
        "IF (Audio IS stable) AND (Movement IS moderate) AND (Feeding IS long_time) THEN (Risk IS medium)",
        "IF (Audio IS stable) AND (Movement IS moderate) AND (Feeding IS recent) THEN (Risk IS low)",
        
        # Lekka zmiana dzwięku)
        "IF (Audio IS slight_increase) AND (Movement IS low) AND (Feeding IS long_time) THEN (Risk IS medium)",
        "IF (Audio IS slight_increase) AND (Movement IS low) AND (Feeding IS recent) THEN (Risk IS low)",

        # Silniejszy ruch, ale jeszcze nie panika
        "IF (Movement IS high) AND (Feeding IS long_time) THEN (Risk IS high)",
        "IF (Movement IS high) AND (Feeding IS recent) THEN (Risk IS medium)",

        # Głęboki sen
        "IF (Audio IS stable) AND (Movement IS no_movement) THEN (Risk IS low)",
        "IF (Audio IS stable) AND (Movement IS low) THEN (Risk IS low)",
        
        # Brak ruchu przy spadku hałasu (głęboki sen) -> Low Risk
        "IF (Audio IS sudden_decrease) AND (Movement IS no_movement) THEN (Risk IS low)",
        
        # Słaby ruch przy spadku hałasu (tie-breaker z jedzeniem)
        "IF (Audio IS sudden_decrease) AND (Movement IS low) AND (Feeding IS recent) THEN (Risk IS low)",
        "IF (Audio IS sudden_decrease) AND (Movement IS low) AND (Feeding IS long_time) THEN (Risk IS medium)",
        
        # Umiarkowany ruch przy spadku hałasu (tie-breaker z jedzeniem)
        "IF (Audio IS sudden_decrease) AND (Movement IS moderate) AND (Feeding IS recent) THEN (Risk IS medium)",
        "IF (Audio IS sudden_decrease) AND (Movement IS moderate) AND (Feeding IS long_time) THEN (Risk IS high)",
    ]

    fs.add_rules(rules)

    if save_figure:
        fs.produce_figure(outputfile="inputs.png")
    
    return fs


def evaluate_risk(fuzzy_system, audio, movement, feeding):
    """
    Evaluate the wake-up risk for given input values using pure fuzzy logic.
    
    Args:
        fuzzy_system (FuzzySystem): The configured fuzzy system instance.
        audio (float): Audio change in dB/s (range: -100 to 100).
        movement (float): Movement percentage (range: 0 to 100).
        feeding (float): Time since last feeding in minutes (range: 0 to 240).
    
    Returns:
        dict: Inference result containing the Risk value.
              Example: {'Risk': 45.2}
    
    Raises:
        Exception: If the fuzzy inference encounters an error.
    """
    fuzzy_system.set_variable("Audio", audio)
    fuzzy_system.set_variable("Movement", movement)
    fuzzy_system.set_variable("Feeding", feeding)
    
    return fuzzy_system.inference()


def check_rule_coverage(fuzzy_system, audio, movement, feeding, threshold=0.0):
    """
    Check if at least one fuzzy rule is activated for the given inputs.
    
    This function determines rule coverage by inspecting the firing strengths
    of all rules after setting the input values. A rule is considered activated
    if its firing strength exceeds the specified threshold.
    
    Args:
        fuzzy_system (FuzzySystem): The configured fuzzy system instance.
        audio (float): Audio change in dB/s (range: -100 to 100).
        movement (float): Movement percentage (range: 0 to 100).
        feeding (float): Time since last feeding in minutes (range: 0 to 240).
        threshold (float): Minimum firing strength to consider a rule activated.
                          Default is 0.0 (any non-zero activation counts).
    
    Returns:
        tuple: (is_covered, max_firing_strength)
               - is_covered (bool): True if at least one rule is activated.
               - max_firing_strength (float): The maximum firing strength among all rules.
    """
    fuzzy_system.set_variable("Audio", audio)
    fuzzy_system.set_variable("Movement", movement)
    fuzzy_system.set_variable("Feeding", feeding)
    
    # Get firing strengths for all rules
    firing_strengths = fuzzy_system.get_firing_strengths()
    
    # Check if any rule is activated above the threshold
    max_firing = max(firing_strengths) if firing_strengths else 0.0
    is_covered = max_firing > threshold
    
    return is_covered, max_firing
