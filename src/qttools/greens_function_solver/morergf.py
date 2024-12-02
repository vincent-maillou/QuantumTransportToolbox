# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from qttools import xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver
from qttools.utils.solvers_utils import get_batches


class moreRGF(GFSolver):
    def selected_inv(
        self, a: DSBSparse, out=None, max_batch_size: int = 1
    ) -> None | DSBSparse:
        """
        Perform the selected inversion of a general block sparse matrix.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        out : _type_, optional
            Output matrix, by default None.
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1.

        Returns
        -------
        None | DBSparse
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object.
        """
        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], max_batch_size)

        if out is not None:
            x = out
        else:
            x = a.__class__.zeros_like(a)

        x.return_None = True
        a.return_None = True

        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            x_ = x.stack[stack_slice]

            # working memory
            w = [[None for j in range(a.num_blocks)] for i in range(a.num_blocks)]

            w[0][0] = xp.linalg.inv(a_.blocks[0, 0])

            # Forwards sweep.
            for i in range(a.num_blocks - 1):

                for j in range(i + 1, a.num_blocks):
                    a_ji = a_.blocks[j, i]

                    if a_ji is not None:
                        if w[j][i] is None:
                            w[j][i] = a_ji  # cache into working memory
                        for k in range(i + 1, a.num_blocks):
                            a_ik = a_.blocks[i, k]
                            a_jk = a_.blocks[j, k]
                            if (a_ik is not None) and (a_jk is not None):
                                if w[i][k] is None:
                                    w[i][k] = a_ik  # cache into working memory
                                if w[j][k] is None:
                                    w[j][k] = a_jk
                                w[j][k] -= w[j][i] @ w[i][i] @ w[i][k]

                j = i + 1
                w[j][j] = xp.linalg.inv(w[j][j])

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):

                dx_ii = xp.zeros_like(w[i][i])

                # temporary memory to avoid overwrite a_ji
                x_ki = [None for j in range(a.num_blocks)]
                x_ik = [None for j in range(a.num_blocks)]

                # off-diagonal blocks of invert
                for k in range(i + 1, a.num_blocks):

                    if (x_.blocks[k,i] is not None):

                        for j in range(i + 1, a.num_blocks):

                            if (w[j][i] is not None) and (w[k][j] is not None):
                                if x_ki[k] is None:
                                    x_ki[k] = -w[k][j] @ w[j][i] @ w[i][i]
                                else:
                                    x_ki[k] -= w[k][j] @ w[j][i] @ w[i][i]

                    if (w[i][k] is not None) and (x_ki[k] is not None):
                        dx_ii += w[i][i] @ w[i][k] @ x_ki[k]

                    if (x_.blocks[i,k] is not None):

                        for j in range(i + 1, a.num_blocks):

                            if (w[i][j] is not None) and (w[j][k] is not None):
                                if x_ik[k] is None:
                                    x_ik[k] = -w[i][i] @ w[i][j] @ w[j][k]
                                else:
                                    x_ik[k] -= w[i][i] @ w[i][j] @ w[j][k]

                for k in range(i + 1, a.num_blocks):
                    if (x_ki[k] is not None) and (x_.blocks[k,i] is not None):
                        w[k][i] = x_ki[k]

                    if (x_ik[k] is not None) and (x_.blocks[i,k] is not None):
                        w[i][k] = x_ik[k]

                # diagonal blocks of invert
                w[i][i] += -dx_ii

            # copy the result from working memory to x
            for i in range(x.num_blocks):
                for j in range(x.num_blocks):
                    if (x_.blocks[i, j] is not None) and (w[i][j] is not None):
                        x_.blocks[i, j] = w[i][j]

        if out is None:
            return x

    def selected_solve(
        self,
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        out: tuple | None = None,
        return_retarded: bool = False,
        max_batch_size: int = 1,
    ) -> None | tuple:
        """Perform a selected-solve of the congruence matrix equation: A * X * A^T = B.

        Note
        ----
        If the diagonal blocks of the input matrix ```a: DSBSparse``` are not dense,
        the selected-solve will not be performed correctly. This happen because during
        the forward sweep, only the elements of the inverse that match the sparsity
        pattern of the input diagonal blocks will be stored. Leading to incomplete
        matrix-multiplications down the line.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        sigma_lesser : DBSparse
            Lesser matrix. This matrix is expected to be skewed-hermitian.
        sigma_greater : DBSparse
            Greater matrix. This matrix is expected to be skewed-hermitian.
        out : tuple | None, optional
            Output matrix, by default None
        return_retarded : bool, optional
            Weither the retarded Green's functioln should be returned, by default False
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1


        Returns
        -------
        None | tuple
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object. If `return_retarded` is True, returns a tuple with
            the retarded Green's function as the last element.
        """
        raise NotImplementedError(
            "Selected solve with general sparse matrices is not implemented."
        )
