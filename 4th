import numpy as np

# --- Create Augmented Matrix (A|B) ---
def create_augmented_matrix():
    n = int(input("Enter number of equations (and unknowns): "))
    A = []
    print("\nEnter coefficients of each equation:")
    for i in range(n):
        row = []
        for j in range(n):
            val = float(input(f"  Coefficient a{i+1}{j+1}: "))
            row.append(val)
        A.append(row)
    A = np.array(A, float)

    print("\nEnter constants (right side values):")
    B = np.array([float(input(f"  b{i+1}: ")) for i in range(n)], float)
    AB = np.column_stack((A, B))  # Augmented matrix
    return AB

# --- Perform Gauss Elimination ---
def gauss_elimination(AB):
    n = len(AB)
    for i in range(n):
        if AB[i, i] == 0:  # Avoid division by zero
            for k in range(i+1, n):
                if AB[k, i] != 0:
                    AB[[i, k]] = AB[[k, i]]
                    break
        AB[i] = AB[i] / AB[i, i]  # Make diagonal element 1
        for j in range(i+1, n):   # Eliminate below pivot
            AB[j] = AB[j] - AB[i] * AB[j, i]
    return AB

# --- Back Substitution ---
def back_substitution(AB):
    n = len(AB)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = AB[i, -1] - np.sum(AB[i, i+1:n] * x[i+1:n])
    return x

# --- Main Program ---
print("Solve Linear Equations using Gauss Elimination\n")
AB = create_augmented_matrix()
print("\nAugmented Matrix [A|B]:\n", AB)

E = gauss_elimination(AB.copy())
print("\nUpper Triangular (Echelon) Form:\n", np.round(E, 3))

solution = back_substitution(E)
print("\nSolution of the system:")
for i, val in enumerate(solution, 1):
    print(f"  x{i} = {val:.3f}")
