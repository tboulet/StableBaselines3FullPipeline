"""Scheduler functions for learning rate, epsilon, etc."""

def exponential_decay(percentage : float) -> float:
    """Exponential decay function.

    Args:
        percentage (float): Percentage of the total number of steps.
    
    Returns:
        float: The decayed value.
    """
    return 0.99 ** (int(percentage * 100))
    
def linear_decay(percentage : float) -> float:
    """Linear decay function.

    Args:
        percentage (float): Percentage of the total number of steps.
    
    Returns:
        float: The decayed value.
    """
    return max(1 - 0.9 * percentage * 4, 0.1)