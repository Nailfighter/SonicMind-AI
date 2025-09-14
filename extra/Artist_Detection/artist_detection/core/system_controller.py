import time

class SystemController:
    """
    Manages the system's mute state for both mic and speakers, including a cooldown period.
    This class provides a simple API to control and query the mute status.
    """
    def __init__(self, mute_duration=5):
        """
        Initializes the SystemController.

        Args:
            mute_duration (int): The duration in seconds for the mute cooldown (default: 5).
        """
        self.mute_duration = mute_duration
        self._mute_end_time = 0

    def trigger_mute(self):
        """
        Activates the mute state for both mic and speakers and sets the cooldown timer.
        Both mic and speakers will be muted for the configured duration.
        """
        self._mute_end_time = time.time() + self.mute_duration
        print(f"API: Dual mute triggered. Both mic and speakers will be muted for {self.mute_duration} seconds.")

    def is_muted(self):
        """
        Checks if the system is currently in a muted state.

        Returns:
            bool: True if the system is muted, False otherwise.
        """
        return time.time() < self._mute_end_time

    def get_status(self):
        """
        Returns the current status of the system.

        Returns:
            dict: A dictionary containing the mute status and remaining time.
        """
        muted = self.is_muted()
        remaining_time = max(0, self._mute_end_time - time.time())
        
        return {
            "muted": muted,
            "status_text": "MUTED" if muted else "CLEAR",
            "cooldown_remaining": round(remaining_time, 2)
        }
