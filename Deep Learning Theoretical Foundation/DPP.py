import numpy as np

def dpp_sample(kerner_size):

    # eigendecomposition
    eignval, eignvec = np.linalg.eigh(kerner_size)
    J = set()

    N = kerner_size.shape[1]
    eignval = np.real(eignval)
    eignval = np.maximum(eignval, 0)

    for n in range(N):
        prob = eignval[n] / (eignval[n] + 1)
        if np.random.binomial(n=1, p=prob) == 1:
            J.add(n)
    
    if not J:
        return []
    
    V = eignvec[:, list(J)]

    Y = set()

    while V.shape[1] > 0:

        Y_complement = list(set(range(N)) - Y)

        if not Y_complement:
            break
        
        probs = np.zeros(len(Y_complement))

        for idx, i in enumerate(Y_complement):
            e_i = np.zeros(N)
            e_i[i] = 1
            probs[idx] = np.sum((v.T @ e_i)**2 for v in V.T)
        
        if np.sum(probs) < 1e-10:
            break

        probs = probs / V.shape[1]

        probs = np.maximum(probs, 0)

        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        
        else:
            break

        try:
            i = Y_complement[np.random.choice(len(Y_complement), p=probs)]

            Y.add(i)
        
        except ValueError as e:
            print(e)
            break

        if V.shape[1] > 1:
            e_i = np.zeros(N)
            e_i[i] = 1

            V_proj = V - np.outer(e_i, e_i @ V)

            Q, R = np.linalg.qr(V_proj)

            mask = np.sqrt(np.sum(Q * Q, axis=0)) > 1e-10

            if not np.any(mask):
                break

        else:
            break

    return sorted(list(Y))



if __name__ == "__main__":
    # Generate a test kernel matrix
    N = 5
    X = np.random.randn(N, 2)
    L = np.exp(-0.5 * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    
    # Sample from DPP
    Y = dpp_sample(L)
    print("Selected indices:", Y)
    
    # Print kernel matrix for verification
    print("\nKernel matrix:")
    print(L)







