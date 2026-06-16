def value_iteration(P, R, gamma, tol=1e-6, max_iters=1000):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    S = len(P)
    A = len(P[0])

    V = [0.00] * S

    for _ in range(max_iters):
        V_new = bellman_optimality_backup(P, R, gamma, V)

        diff = max(abs(V_new[s] - V[s]) for s in range(S))
        V = V_new
        
        if diff < tol:
            break

    policy = []

    for s in range(S):
        best_a = 0
        best_q = float('-inf')

        for a in range(A):
            q = 0.0

            for s_next in range(S):
                q += P[s][a][s_next] * (R[s][a][s_next] + gamma * V[s_next])

            if q > best_q:
                best_q = q
                best_a = a

        policy.append(best_a)
    V = [round(v, 4) for v in V]
    return V, policy


def bellman_optimality_backup(P, R, gamma, V):
    """
    Apply one Bellman optimality backup to V.

    Args:
        P: 3D nested list of transition probabilities, shape (S, A, S)
        R: 3D nested list of expected rewards, shape (S, A, S)
        gamma: discount factor in [0, 1]
        V: list of current value estimates, length S

    Returns:
        list of length S, V_new[s] rounded to 4 decimals
    """
    S = len(V)
    A = len(P[0])
    V_new = [0.0] * S
    for s in range(S):
        best = float('-inf')
        for a in range(A):
            q = 0.0
            for sp in range(S):
                q += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
            if q > best:
                best = q
        V_new[s] = best
    return V_new
