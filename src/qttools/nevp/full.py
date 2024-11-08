import numpy as np

from qttools import NDArray, xp
from qttools.nevp.nevp import NEVP
from qttools.utils.gpu_utils import get_device, get_host


class Full(NEVP):
    """An NEVP solver based on linearization.

    References
    ----------
    [1] S. Brück, Ab-initio Quantum Transport Simulations for
    Nanoelectronic Devices, ETH Zurich, 2017.

    """

    def __call__(self, a_xx: tuple[NDArray, ...]) -> tuple[NDArray, NDArray]:

        # Allow for batched input.
        if a_xx[0].ndim == 2:
            a_xx = tuple(a_x[xp.newaxis, :, :] for a_x in a_xx)

        inverse = xp.linalg.inv(sum(a_xx))

        # NOTE: CuPy does not expose a `block` function.
        row = xp.concatenate(
            [inverse @ sum(a_xx[:i]) for i in range(1, len(a_xx) - 1)]
            + [inverse @ -a_xx[-1]],
            axis=-1,
        )
        A = xp.concatenate([row] * (len(a_xx) - 1), axis=-2)
        B = xp.kron(xp.tri(len(a_xx) - 2).T, xp.eye(a_xx[0].shape[-1]))
        A[:, : B.shape[0], : B.shape[1]] -= B

        w, v = np.linalg.eig(get_host(A))
        w, v = get_device(w), get_device(v)

        # Recover the original eigenvalues from the spectral transform.
        w = xp.where((xp.abs(w) == 0.0), -1.0, w)
        w = 1 / w + 1
        v = v[:, : a_xx[0].shape[-1]]

        return w, v
