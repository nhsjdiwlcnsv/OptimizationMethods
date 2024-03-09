import numpy as np


def simplex_dual(c: np.ndarray, A: np.ndarray, b: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Dual simplex method implementation."""
    m, n = A.shape

    while True:
        # Step 1
        A_B: np.ndarray = A[:, B]
        A_B_inv: np.ndarray = np.linalg.inv(A_B)

        # Step 2
        c_B: np.ndarray = c[B]

        # Step 3
        y: np.ndarray = c_B @ A_B_inv

        # Step 4
        kk_B: np.ndarray = A_B_inv @ b
        kk = np.array([kk_B[np.where(B == i)][0] if i in B else 0 for i in range(n)])

        # Step 5
        if np.all(kk >= 0):
            return kk

        # Step 6
        j_k: int = np.where(kk < 0)[0][-1]

        # Step 7
        k: int = np.where(B == j_k)[0][0]
        delta_y: np.ndarray = A_B_inv[k]
        mu = np.array([delta_y @ A[:, j] if j in np.setdiff1d(np.arange(n), B) else 0 for j in range(n)])

        # Step 8
        if np.all(mu[np.setdiff1d(np.arange(n), B)] >= 0):
            raise ValueError("Unable to find a solution!")

        # Step 9
        sigma = np.array([(c[j] - A[:, j] @ y) / mu[j] for j in np.setdiff1d(np.arange(n), B) if mu[j] < 0])

        # Step 10
        j_0: np.intp = np.argmin(sigma)
        B[k] = j_0


if __name__ == "__main__":
    c = np.array([-4, -3, -7, 0, 0])
    A = np.array([
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1],
    ])
    b = np.array([-1, -3. / 2])
    B = np.array([3, 4])

    x: np.ndarray = simplex_dual(c=c, A=A, b=b, B=B)

    print(f"x = {x}")
