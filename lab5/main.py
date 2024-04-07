from collections import Counter

import numpy as np


def balance_supply_and_demand(supply: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    :param supply:
    :param demand:
    :return:
    """
    total_supply: int = np.sum(supply)
    total_demand: int = np.sum(demand)

    if total_supply > total_demand:
        demand = np.hstack((demand, np.array([total_supply - total_demand])))
    elif total_supply < total_demand:
        supply = np.hstack((supply, np.array([total_demand - total_supply])))

    return supply, demand


def solve_transportation_problem(cost_matrix: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """

    :param cost_matrix:
    :param supply:
    :param demand:
    :return:
    """
    m, n = cost_matrix.shape

    # Phase 1 – Find the first basis plan
    basis_plan = np.zeros_like(cost_matrix)
    basis_indices: list[tuple] = []
    supply_copy: np.ndarray = supply.copy()
    demand_copy: np.ndarray = demand.copy()

    i, j = 0, 0
    while i < m and j < n:
        basis_indices += [(i, j)]

        if supply_copy[i] <= demand_copy[j]:
            basis_plan[i, j] = supply_copy[i]
            demand_copy[j] -= supply_copy[i]
            supply_copy[i] = 0
            i += 1
        else:
            basis_plan[i, j] = demand_copy[j]
            supply_copy[i] -= demand_copy[j]
            demand_copy[j] = 0
            j += 1

    # Phase 2 – Find optimal basis plan
    while True:
        A: np.ndarray = np.zeros(shape=(m + n, m + n))
        b: np.ndarray = np.zeros(shape=(m + n,))

        for b_i, (i, j) in enumerate(basis_indices):
            A[b_i, i] = 1
            A[b_i, m + j] = 1
            b[b_i] = cost_matrix[i, j]

        A[-1, 0] = 1

        solution: np.ndarray = np.linalg.solve(A, b)
        supply_potentials: np.ndarray = solution[:m]
        demand_potentials: np.ndarray = solution[m:]

        non_basis_pos = tuple()
        basis_plan_is_optimal: bool = True

        for i in range(m):
            for j in range(n):
                if supply_potentials[i] + demand_potentials[j] > cost_matrix[i, j]:
                    basis_plan_is_optimal = False
                    non_basis_pos = (i, j)
                    break
            if non_basis_pos:
                basis_indices += [non_basis_pos]
                break

        if basis_plan_is_optimal:
            return basis_plan

        basis_indices_copy = basis_indices.copy()

        while True:
            i_counter = Counter([i for (i, _) in basis_indices_copy])
            j_counter = Counter([j for (_, j) in basis_indices_copy])

            i_to_rm: list = [i for i in i_counter if i_counter[i] == 1]
            j_to_rm: list = [j for j in j_counter if j_counter[j] == 1]

            if not i_to_rm and not j_to_rm:
                break

            basis_indices_copy = [(i, j) for (i, j) in basis_indices_copy
                                  if i not in i_to_rm and j not in j_to_rm]

        plus, minus = [], []
        plus += [basis_indices_copy.pop()]

        while len(basis_indices_copy):
            if len(plus) > len(minus):
                for index, (i, j) in enumerate(basis_indices_copy):
                    if plus[-1][0] == i or plus[-1][1] == j:
                        minus.append(basis_indices_copy.pop(index))
                        break
            else:
                for index, (i, j) in enumerate(basis_indices_copy):
                    if minus[-1][0] == i or minus[-1][1] == j:
                        plus.append(basis_indices_copy.pop(index))
                        break

        theta: float = min(basis_plan[i][j] for i, j in minus)

        for i, j in plus:
            basis_plan[i][j] += theta
        for i, j in minus:
            basis_plan[i][j] -= theta

        for i, j in minus:
            if basis_plan[i][j] == 0:
                basis_indices.remove((i, j))
                break


# Example usage:
if __name__ == "__main__":
    np.random.seed(42)

    supply = np.array([100, 300, 300])
    demand = np.array([300, 200, 200])

    supply, demand = balance_supply_and_demand(supply, demand)
    cost_matrix = np.array([[8, 4, 1],
                            [8, 4, 3],
                            [9, 7, 5]])

    print(f"* Supply line: {supply}")
    print(f"* Demand line: {demand}")
    print(f"* Total supply: {np.sum(supply)}, total demand: {np.sum(demand)}")
    print(f"* Cost matrix: \n{cost_matrix}")

    optimal_allocation = solve_transportation_problem(cost_matrix, supply, demand)

    print(f"* Optimal allocation matrix: \n{optimal_allocation}")
