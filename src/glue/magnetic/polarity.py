"""
Magnetic polarity definitions for the GLUE framework.

This module defines the MagneticPolarity enum which is used to specify
the polarity of magnetic flows between teams.
"""

from enum import Enum

class MagneticPolarity(str, Enum):
    """
    Types of magnetic polarities that control team interactions.
    
    ATTRACT: Teams are drawn to work together, facilitating smooth information flow
    REPEL: Teams operate independently with conditional interactions
    """
    
    ATTRACT = "attract"  # Teams work together smoothly
    REPEL = "repel"      # Teams operate independently with conditional interactions