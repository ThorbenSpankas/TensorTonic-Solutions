def policy_iteration(P, R, gamma, eval_tol=1e-8, max_iters=200):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """

    S = len(P)
    A = len(P[0])
    V = [0.0] * S
    policy = [0] * S

    # step 1: evaluate policy
    for _ in range(max_iters):
        while True:
            V_new = [0.0]*S

            for s in range(S):
                a = policy[s]

                V_new[s] = sum(P[s][a][sp] * (R[s][a][sp] + gamma * V[sp]) for sp in range(S))
                print(f'V_new {V_new}')
            delta = max(abs(V_new[s] - V[s]) for s in range(S))
            V = V_new

            if delta < eval_tol:
                break

    # step 2: policy improvement
        policy_stable = True
    
        for s in range(S):
            old_a = policy[s]
    
            best_q = float('-inf')
            best_a = 0
    
            for a in range(A):
                q = sum(P[s][a][sp] * (R[s][a][sp] + gamma * V[sp]) for sp in range(S))
                print(f'Q {q}')
                if q > best_q + 1e-12:
                    best_q = q
                    best_a = a
    
            policy[s] = best_a
    
            if best_a != old_a:
                policy_stable = False

        if policy_stable:
            break

    V_rounded = [round(v,4) for v in V]
    return (V_rounded, policy)

        