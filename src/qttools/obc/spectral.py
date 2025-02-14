# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, xp
from qttools.datastructures.dsbsparse import _block_view
from qttools.nevp import NEVP
from qttools.obc.obc import OBCSolver


class Spectral(OBCSolver):
    """Spectral open-boundary condition solver.

    This technique of obtaining the surface Green's function is based on
    the solution of a non-linear eigenvalue problem (NEVP), defined via
    the system-matrix blocks in the semi-infinite contacts.

    Those eigenvalues corresponding to reflected modes are filtered out,
    so that only the ones that correspond to modes that propagate into
    the leads or those that decay away from the system are retained.

    The surface Green's function is then calculated from these filtered
    eigenvalues and eigenvectors.

    Parameters
    ----------
    nevp : NEVP
        The non-linear eigenvalue problem solver to use.
    block_sections : int, optional
        The number of sections to split the periodic matrix layer into.
    min_decay : float, optional
        The decay threshold after which modes are considered to be
        evanescent.
    max_decay : float, optional
        The maximum decay to consider for evanescent modes. If not
        provided, the maximum decay is set to the logarithm of the outer
        radius of the contour annulus if applicable. Otherwise, it is
        set to log(10).
    num_ref_iterations : int, optional
        The number of refinement iterations to perform on the surface
        Green's function.
    x_ii_formula : str, optional
        The formula to use for the calculation of the surface Green's
        function. The default is via the boundary "self-energy". The
        other option is "direct". The "self-energy" formula corresponds
        to Equation (13.1) in the paper [^1] and the "direct" formula
        corresponds to Equation (15).
    two_sided : bool, optional
        Whether to solve the NEVP for both left and right eigenvectors,
        and construct the surface Green's function from both.
    treat_pairwise : bool, optional
        Whether to match complex conjugate modes and treat them in pairs
        during the determining of reflected modes.
    pairing_threshold : float, optional
        The threshold for which two modes are considered to be a mode
        pair.
    min_propagation : float, optional
        The minimum ratio between the real and imaginary part of the
        group velocity of a mode. This ratio is used to determine how
        clearly a mode propagates.
    residual_tolerance : float, optional
        The tolerance for the residual of the NEVP.
    residual_normalization_formula : str, optional
        The formula to use for the normalization of the residual. The
        default is the "operator_norm" formula. The other options are
        "abs_eigenvalue" and "no_normalization".The "operator_norm"
        formula corresponds to normalization by the frobenius norm of
        the operator, the "abs_eigenvalue" formula corresponds to
        normalization by the absolute of the eigenvalues, and
        "no_normalization" results in no normalization.

        [^1]: S. Brück, et al., Efficient algorithms for large-scale
        quantum transport calculations, The Journal of Chemical Physics,
        2017.

    """

    def __init__(
        self,
        nevp: NEVP,
        block_sections: int = 1,
        min_decay: float = 1e-3,
        max_decay: float | None = None,
        num_ref_iterations: int = 2,
        x_ii_formula: str = "self-energy",
        two_sided: bool = False,
        treat_pairwise: bool = True,
        pairing_threshold: float = 0.25,
        min_propagation: float = 0.01,
        residual_tolerance: float = 1e-3,
        residual_normalization_formula: str = "operator_norm",
    ) -> None:
        """Initializes the spectral OBC solver."""
        self.nevp = nevp

        self.min_decay = min_decay
        if max_decay is None:
            max_decay = xp.log(getattr(nevp, "r_o", 1000.0))
        self.max_decay = max_decay

        self.num_ref_iterations = num_ref_iterations
        self.block_sections = block_sections
        self.x_ii_formula = x_ii_formula

        self.two_sided = two_sided
        self.treat_pairwise = treat_pairwise
        self.pairing_threshold = pairing_threshold
        self.min_propagation = min_propagation
        self.residual_tolerance = residual_tolerance
        self.residual_normalization_formula = residual_normalization_formula

    def _extract_subblocks(
        self,
        a_ji: NDArray,
        a_ii: NDArray,
        a_ij: NDArray,
    ) -> tuple[NDArray, ...]:
        """Extracts the coefficient blocks from the periodic matrix.

        Parameters
        ----------
        a_ji : NDArray
            The subdiagonal block of the periodic matrix.
        a_ii : NDArray
            The diagonal block of the periodic matrix.
        a_ij : NDArray
            The superdiagonal block of the periodic matrix.

        Returns
        -------
        blocks : tuple[NDArray, ...]
            The non-zero blocks making up the matrix layer.

        """
        # Construct layer of periodic matrix in semi-infinite lead.
        layer = (a_ji, a_ii, a_ij)
        if self.block_sections == 1:
            return layer

        # Get a nested block view of the layer.
        view = _block_view(xp.concatenate(layer, axis=-1), -1, 3 * self.block_sections)
        view = _block_view(view, -2, self.block_sections)

        # Make sure that the reduction leads to periodic sublayers.
        # NOTE: I'm not 100% sure that this is really necessary.
        for i in range(self.block_sections):
            if not xp.allclose(view[0, :], xp.roll(view[i, :], -i, axis=0)):
                raise ValueError("Requested block sectioning is not periodic.")

        # Select relevant blocks and remove empty ones.
        blocks = view[0, : -self.block_sections + 1]
        return tuple(block for block in blocks if xp.any(block))

    def _find_pairwise_propagating(
        self,
        dEk_dk: NDArray,
        ks: NDArray,
    ):
        """Filter propagating modes that are opposite.

        Parameters
        ----------
        dEk_dk : NDArray
            The group velocity of the modes.
        ks : NDArray
            The wavevector of the modes.

        Returns
        -------
        mask_pairwise_propagating : NDArray
            A boolean mask indicating which eigenvalues correspond to
            matched modes that propagate.

        """

        # match modes to the most opposite ones
        diff = xp.abs(dEk_dk[:, :, xp.newaxis] + dEk_dk[:, xp.newaxis, :])
        match_indices = xp.argmin(diff, axis=-1)
        ks_match = xp.array(
            [batch[indices] for batch, indices in zip(ks, match_indices)]
        )
        dEk_dk_match = xp.array(
            [batch[indices] for batch, indices in zip(dEk_dk, match_indices)]
        )

        # pair of modes decay slowly
        mask_pairwise_propagating = (
            xp.abs(ks_match.imag) + xp.abs(ks.imag)
        ) / 2 < self.min_decay

        # modes opposite enough (0 would be perfect opposite)
        eta = xp.finfo(dEk_dk.dtype).eps
        mask_pairwise_propagating &= (
            xp.abs(dEk_dk + dEk_dk_match) / (xp.abs(dEk_dk) + eta)
            < self.pairing_threshold
        )
        mask_pairwise_propagating &= (
            xp.abs(ks + ks_match) / (xp.abs(ks) + eta) < self.pairing_threshold
        )

        return mask_pairwise_propagating

    def _find_reflected_modes(
        self, ws: NDArray, vrs: NDArray, a_xx: list[NDArray], vls: NDArray | None = None
    ) -> NDArray:
        """Determines which eigenvalues correspond to reflected modes.

        For the computation of the surface Green's function, only the
        eigenvalues corresponding to modes that propagate or decay into
        the leads are retained.

        Parameters
        ----------
        ws : NDArray
            The eigenvalues of the NEVP.
        vrs : NDArray
            The right eigenvectors of the NEVP.
        a_xx : tuple[NDArray, ...]
            The blocks of the periodic matrix.
        vls : NDArray, optional
            The left eigenvectors of the NEVP. Required for two-sided

        Returns
        -------
        mask : NDArray
            A boolean mask indicating which eigenvalues correspond to
            reflected modes.

        """
        if self.two_sided and vls is None:
            raise ValueError("Two-sided calculation requires left eigenvectors.")

        batchsize = a_xx[0].shape[0]
        b = len(a_xx) // 2

        # Calculate the residual
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            # NOTE: This consumes a lot of memory since
            # the operators are explicitly calculated.
            operators = sum(
                a_x[:, xp.newaxis, :, :]
                * ws[..., xp.newaxis, xp.newaxis] ** (i - len(a_xx) // 2)
                for i, a_x in enumerate(a_xx)
            )
            products = operators @ vrs.swapaxes(-1, -2)[..., xp.newaxis]

            residuals = xp.linalg.norm(products, axis=(-1, -2))

            # eigenvectors are not necessarily normalized
            eigenvector_norm = xp.linalg.norm(vrs, axis=-2)
            residuals /= eigenvector_norm

            if self.residual_normalization_formula == "operator_norm":
                operator_norm = xp.linalg.norm(operators, axis=(-1, -2))
                residuals /= operator_norm
            elif self.residual_normalization_formula == "abs_eigenvalue":
                residuals /= xp.abs(ws)
            elif self.residual_normalization_formula == "no_normalization":
                pass
            else:
                raise ValueError(
                    f"Unknown formula: {self.residual_normalization_formula}"
                    "Choose 'operator_norm', 'abs_eigenvalue', or 'no_normalization'."
                )

            if xp.any(residuals > self.residual_tolerance):
                print("Warning: Residuals are larger than the tolerance.")

        # Calculate the group velocity to select propagation direction.
        # The formula can be derived by taking the derivative of the
        # polynomial eigenvalue equation with respect to k.
        # NOTE: This is actually only correct if we have no overlap.
        dEk_dk = xp.zeros_like(ws)
        with warnings.catch_warnings(
            action="ignore", category=RuntimeWarning
        ):  # Ignore division by zero.
            # TODO: Replace this for loop with a faster implementation.
            for i in range(batchsize):
                for j, w in enumerate(ws[i]):
                    a = -sum(
                        (1j * n) * w**n * a_xn[i]
                        for a_xn, n in zip(a_xx, range(-b, b + 1))
                    )

                    if self.two_sided:
                        phi_right = vrs[i, :, j]
                        phi_left = vls[i, :, j]
                    else:
                        phi_right = vrs[i, :, j]
                        phi_left = vrs[i, :, j]

                    dEk_dk[i, j] = (phi_left.conj().T @ a @ phi_right) / (
                        phi_left.conj().T @ phi_right
                    )

            ks = -1j * xp.log(ws)

        # replace nan and infs with 0 due to zero eigenvalues
        dEk_dk = xp.nan_to_num(dEk_dk, nan=0, posinf=0, neginf=0)
        ks = xp.nan_to_num(ks, nan=0, posinf=0, neginf=0)

        # Find eigenvalues that correspond to reflected modes. These are
        # modes that either propagate into the leads or decay away from
        # the system.
        # Determine (matched) modes that decay slow enough to be
        # considered propagating.
        if self.treat_pairwise:
            mask_propagating = self._find_pairwise_propagating(dEk_dk, ks)
            mask_decaying = ~mask_propagating
        else:
            mask_propagating = xp.abs(ks.imag) < self.min_decay
            mask_decaying = xp.ones_like(dEk_dk, dtype=bool)

        # Make sure decaying modes decay fast enough.
        mask_decaying &= ks.imag < -self.min_decay

        # fast enough propagation (group velocity)
        eta = xp.finfo(dEk_dk.dtype).eps
        mask_propagating &= self.min_propagation < abs(dEk_dk.real) / (
            abs(dEk_dk.imag) + eta
        )
        # propgation direction
        mask_propagating &= dEk_dk.real < 0

        # ingore modes that decay incredibly fast
        mask_decaying &= ks.imag > -self.max_decay

        return (mask_propagating | mask_decaying) & (
            residuals < self.residual_tolerance
        )

    def _upscale_eigenmodes(
        self,
        ws: NDArray,
        vs: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Upscales the eigenvectors to the full periodic matrix layer.

        The extraction of subblocks and hence the solution of a higher-
        ordere, but smaller, NEVP leads to eigenvectors that are only
        defined on the reduced matrix layer. This function upscales the
        eigenvectors back to the full periodic matrix layer.

        Parameters
        ----------
        ws : NDArray
            The eigenvalues of the NEVP.
        vs : NDArray
            The eigenvectors of the (potentially) higher order NEVP.

        Returns
        -------
        ws : NDArray
            The upscaled eigenvalues.
        vs : NDArray
            The upscaled eigenvectors.

        """
        if self.block_sections == 1:
            return ws, vs

        batchsize, subblock_size, num_modes = vs.shape
        block_size = subblock_size * self.block_sections

        vs_upscaled = xp.zeros((batchsize, block_size, num_modes), dtype=vs.dtype)
        for i in range(batchsize):
            for j, w in enumerate(ws[i]):
                vs_upscaled[i, :, j] = xp.kron(
                    xp.array([w**n for n in range(self.block_sections)]), vs[i, :, j]
                )
                with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
                    vs_upscaled[i, :, j] /= xp.linalg.norm(vs_upscaled[i, :, j])

        return ws**self.block_sections, vs_upscaled

    def _compute_x_ii(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        ws: NDArray,
        vrs: NDArray,
        mask: NDArray,
        vls: NDArray | None = None,
    ) -> NDArray:
        """Computes the surface Green's function.

        Parameters
        ----------
        a_ii : NDArray
            The diagonal block of the periodic matrix.
        a_ij : NDArray
            The superdiagonal block of the periodic matrix.
        a_ji : NDArray
            The subdiagonal block of the periodic matrix.
        ws : NDArray
            The eigenvalues of the NEVP.
        vrs : NDArray
            The right eigenvectors of the NEVP.
        mask : NDArray
            A boolean mask indicating which eigenvalues correspond to
            reflected modes.
        vls : NDArray, optional
            The left eigenvectors of the NEVP. Required for two-sided

        Returns
        -------
        x_ii : NDArray
            The surface Green's function.

        """
        if self.two_sided and vls is None:
            raise ValueError("Two-sided calculation requires left eigenvectors.")

        if self.x_ii_formula == "self-energy":
            # Equation (13.1).
            x_ii_a_ij = xp.zeros((mask.shape[0], *a_ij.shape[-2:]), dtype=a_ij.dtype)
            for i, m in enumerate(mask):
                vr = vrs[i][:, m]
                if self.two_sided:
                    vl = vls[i][:, m]
                w = ws[i, m]
                # Moore-Penrose pseudoinverse.
                if self.two_sided:
                    v_inv = xp.linalg.inv(vl.conj().T @ vr) @ vl.conj().T
                else:
                    v_inv = xp.linalg.inv(vr.conj().T @ vr) @ vr.conj().T
                x_ii_a_ij[i] = vr / w @ v_inv

            # Calculate the surface Green's function.
            return xp.linalg.inv(a_ii + a_ji @ x_ii_a_ij)

        if self.x_ii_formula == "direct":
            # Equation (15).
            x_ii = xp.zeros((mask.shape[0], *a_ij.shape[-2:]), dtype=a_ij.dtype)
            for i, m in enumerate(mask):
                vr = vrs[i][:, m]
                w = ws[i, m]
                # "More stable" computation of the surface Green's function.
                inverse = xp.linalg.inv(
                    vr.conj().T @ a_ii[i] @ vr + vr.conj().T @ a_ji[i] @ vr / w
                )
                x_ii[i] = vr @ inverse @ vr.conj().T

            return x_ii

        raise ValueError(
            f"Unknown formula: {self.x_ii_formula}" "Choose 'self-energy' or 'direct'."
        )

    def _match_eigenmodes(
        self,
        wrs: NDArray,
        vrs: NDArray,
        wls: NDArray,
        vls: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Matches the left and right eigenvalues to reorder the eigenvectors.

        Parameters
        ----------
        wrs : NDArray
            The right eigenvalues of the NEVP.
        vrs : NDArray
            The right eigenvectors of the NEVP.
        wls : NDArray
            The left eigenvalues of the NEVP.
        vls : NDArray
            The left eigenvectors of the NEVP.

        Returns
        -------
        vls : NDArray
            The matched left eigenvectors.

        """

        # the left and right eigenvalues are not sorted
        diff = xp.abs(wrs[..., xp.newaxis] - wls[:, xp.newaxis, :])

        # Find the indices to reorder the left problem
        match_indices = xp.argmin(diff, axis=-1)

        vls = xp.array(
            [batch[:, indices] for batch, indices in zip(vls, match_indices)]
        )

        # TODO: test that the matching is correct

        return vls

    def __call__(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Returns the surface Green's function.

        Parameters
        ----------
        a_ii : NDArray
            Diagonal boundary block of a system matrix.
        a_ij : NDArray
            Superdiagonal boundary block of a system matrix.
        a_ji : NDArray
            Subdiagonal boundary block of a system matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x_ii : NDArray
            The system's surface Green's function.

        """
        if a_ii.ndim == 2:
            a_ii = a_ii[xp.newaxis, :, :]
            a_ij = a_ij[xp.newaxis, :, :]
            a_ji = a_ji[xp.newaxis, :, :]

        blocks = self._extract_subblocks(a_ji, a_ii, a_ij)
        if self.two_sided:
            wrs, vrs, wls, vls = self.nevp(blocks, left=True)
            vls = self._match_eigenmodes(wrs, vrs, wls, vls)
        else:
            wrs, vrs = self.nevp(blocks, left=False)
            vls = None

        mask = self._find_reflected_modes(wrs, vrs, blocks, vls=vls)

        wrs, vrs = self._upscale_eigenmodes(wrs, vrs)

        if self.two_sided:
            wls, vls = self._upscale_eigenmodes(wrs, vls)

        x_ii = self._compute_x_ii(a_ii, a_ij, a_ji, wrs, vrs, mask, vls=vls)

        # Perform a number of refinement iterations.
        for __ in range(self.num_ref_iterations):
            x_ii = xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij)

        # Return the surface Green's function.
        if out is not None:
            out[...] = x_ii
            return

        return x_ii
