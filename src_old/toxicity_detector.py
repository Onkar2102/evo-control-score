"""
toxicity_detector.py
--------------------
This module defines an abstract base class for toxicity detection.
All toxicity detector implementations should inherit from this class
and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any


class ToxicityDetector(ABC):
    """
    Abstract base class for toxicity detectors.

    Any class that implements a toxicity detection mechanism should
    inherit from this class and override the following methods:
    
    - get_toxicity(text: str) -> float: Analyze the provided text and
      return a toxicity score between 0 (not toxic) and 1 (extremely toxic).

    - configure(**kwargs: Any) -> None: Accept arbitrary configuration
      parameters (like API keys, model configurations, etc.) to set up the detector.
    """

    @abstractmethod
    def get_toxicity(self, text: str) -> float:
        """
        Analyze the provided text and return a toxicity score.

        Parameters:
            text (str): The text to analyze.

        Returns:
            float: A toxicity score between 0 (not toxic) and 1 (extremely toxic).
        """
        pass

    @abstractmethod
    def configure(self, **kwargs: Any) -> None:
        """
        Configure the toxicity detector with necessary parameters. This
        might include API keys, model variants, or any other settings.

        Parameters:
            **kwargs: Arbitrary keyword arguments used to configure the detector.
        """
        pass


# Optionally, you could add a factory function to help instantiate detectors.
def create_detector(detector_type: str, **kwargs: Any) -> ToxicityDetector:
    """
    Factory method to create a toxicity detector instance based on a given type.

    Parameters:
        detector_type (str): The type of detector to create. For example,
                             "perspective" or "detoxify".
        **kwargs: Additional keyword arguments for configuration.

    Returns:
        ToxicityDetector: An instance of a subclass of ToxicityDetector.

    Raises:
        ValueError: If the detector type is unsupported.
    """
    detector_type = detector_type.lower()
    
    if detector_type == "perspective":
        from perspective_detector import PerspectiveDetector
        detector = PerspectiveDetector(api_key=kwargs.get("api_key", ""))
    
    elif detector_type == "detoxify":
        from detoxify_detector import DetoxifyDetector
        detector = DetoxifyDetector(model_variant=kwargs.get("model_variant", "original"))
    
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    detector.configure(**kwargs)
    return detector