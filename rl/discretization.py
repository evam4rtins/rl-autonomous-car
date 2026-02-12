import numpy as np

def create_discretizer(y_min, y_max, psi_min, psi_max, n_bins_y, n_bins_psi):
    """
    Factory function to create a state discretizer for the environments.
        Parameters
    ----------
    y_min, y_max : float
        Minimum and maximum values for lateral error (e_y).
    psi_min, psi_max : float
        Minimum and maximum values for heading error (e_psi).
    n_bins_y, n_bins_psi : int
        Number of bins to discretize the lateral and heading errors.
    """

    y_bins = np.linspace(y_min, y_max, n_bins_y)
    psi_bins = np.linspace(psi_min, psi_max, n_bins_psi)

    def discretize_state(state):
        e_y, e_psi = state

        y_idx = np.digitize(e_y, y_bins) - 1
        psi_idx = np.digitize(e_psi, psi_bins) - 1

        y_idx = np.clip(y_idx, 0, n_bins_y - 1)
        psi_idx = np.clip(psi_idx, 0, n_bins_psi - 1)

        return (y_idx, psi_idx)

    return discretize_state
