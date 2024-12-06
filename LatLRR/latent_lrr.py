import numpy as np


def latent_lrr(X, lambda_):
    """
    Latent Low-Rank Representation for Subspace Segmentation and Feature Extraction
    Guangcan Liu, Shuicheng Yan. ICCV 2011.
    Problem:
        min_Z,L,E ||Z||_* + ||L||_* + lambda||E||_1,
                s.t. X = XZ + LX + E.
    Solving problem by Inexact ALM
    """
    # 原代码中A就是X，这里不再重复定义A
    tol = 1e-6
    rho = 1.1
    max_mu = 1e6
    mu = 1e-6
    maxIter = int(1e6)
    d, n = X.shape
    m = X.shape[1]
    atx = X.T @ X
    inv_a = np.linalg.inv(X.T @ X + np.eye(m))
    inv_b = np.linalg.inv(X @ X.T + np.eye(d))

    # Initializing optimization variables
    J = np.zeros((m, n))
    Z = np.zeros((m, n))
    L = np.zeros((d, d))
    S = np.zeros((d, d))
    E = np.zeros((d, n))
    Y1 = np.zeros((d, n))
    Y2 = np.zeros((m, n))
    Y3 = np.zeros((d, d))

    iter = 0
    print('initial')
    while iter < maxIter:
        iter += 1
        # updating J by the Singular Value Thresholding(SVT) operator
        temp_J = Z + Y2 / mu
        U_J, sigma_J, V_J = np.linalg.svd(temp_J, full_matrices=False)
        sigma_J = np.diag(sigma_J)
        svp_J = np.sum(sigma_J > 1 / mu)
        if svp_J >= 1:
            # 只取第一个元素构建对角矩阵，确保维度正确
            sigma_J = np.diag(sigma_J[0:svp_J] - 1 / mu)[0]
            J = U_J[:, 0:svp_J] @ np.diag(sigma_J) @ V_J[:, 0:svp_J].T
        else:
            svp_J = 1
            sigma_J = np.array([0])
            J = np.zeros((U_J.shape[0], V_J.shape[1]))
        print(f"Updated sigma_J shape: {sigma_J.shape}")
        print("________________________________")
        J = U_J[:, 0:svp_J] @ np.diag(sigma_J) @ V_J[:, 0:svp_J].T

        # updating S by the Singular Value Thresholding(SVT) operator
        temp_S = L + Y3 / mu
        U_S, sigma_S, V_S = np.linalg.svd(temp_S, full_matrices=False)
        sigma_S = np.diag(sigma_S)
        svp_S = np.sum(sigma_S > 1 / mu)
        if svp_S >= 1:
            sigma_S = sigma_S[0:svp_S] - 1 / mu
        else:
            svp_S = 1
            sigma_S = np.array([0])
        S = U_S[:, 0:svp_S] @ np.diag(sigma_S) @ V_S[:, 0:svp_S].T

        # update Z
        Z = inv_a @ (atx - X.T @ L @ X - X.T @ E + J + (X.T @ Y1 - Y2) / mu)

        # update L
        L = ((X - X @ Z - E) @ X.T + S + (Y1 @ X.T - Y3) / mu) @ inv_b

        # update E
        xmaz = X - X @ Z - L @ X
        temp = xmaz + Y1 / mu
        E = np.maximum(0, temp - lambda_ / mu) + np.minimum(0, temp + lambda_ / mu)

        leq1 = xmaz - E
        leq2 = Z - J
        leq3 = L - S
        max_l1 = np.max(np.abs(leq1))
        max_l2 = np.max(np.abs(leq2))
        max_l3 = np.max(np.abs(leq3))

        stopC1 = max(max_l1, max_l2)
        stopC = max(stopC1, max_l3)
        if stopC < tol:
            print('LRR done.')
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            Y3 = Y3 + mu * leq3
            mu = min(max_mu, mu * rho)

    return Z, L, E
