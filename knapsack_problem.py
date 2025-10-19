import numpy as np
import time
from simpleai.search import SearchProblem, simulated_annealing
import random

# -----------------------------
# 1️⃣ Định nghĩa bài toán Knapsack
# -----------------------------
class KnapsackProblem(SearchProblem):
    def __init__(self, weights, values, capacity):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        initial_state = tuple([0 for _ in range(len(weights))])
        super().__init__(initial_state)

    def actions(self, state):
        return list(range(len(state)))

    def result(self, state, action):
        new_state = list(state)
        new_state[action] = 1 - new_state[action]
        return tuple(new_state)

    def value(self, state):
        total_weight = sum(w * s for w, s in zip(self.weights, state))
        total_value = sum(v * s for v, s in zip(self.values, state))
        if total_weight > self.capacity:
            total_value -= (total_weight - self.capacity) * 10
        return total_value


# -----------------------------
# 2️⃣ Simulated Annealing (SA)
# -----------------------------
def run_SA(weights, values, capacity):
    problem = KnapsackProblem(weights, values, capacity)
    start_time = time.time()
    result = simulated_annealing(problem, iterations_limit=5000)
    elapsed = time.time() - start_time

    best_state = result.state
    total_weight = sum(w * s for w, s in zip(weights, best_state))
    total_value = sum(v * s for v, s in zip(values, best_state))
    return best_state, total_value, total_weight, elapsed, 5000


# -----------------------------
# 3️⃣ Bee Colony Optimization (BCO)
# -----------------------------
def run_BCO(weights, values, capacity, num_bees=30, num_iterations=200):
    start_time = time.time()
    n = len(weights)
    population = np.random.randint(0, 2, (num_bees, n))

    def fitness(state):
        w = np.sum(np.array(weights) * state)
        v = np.sum(np.array(values) * state)
        if w > capacity:
            v -= (w - capacity) * 10
        return v

    best_solution = None
    best_fitness = float("-inf")

    for _ in range(num_iterations):
        fitness_values = np.array([fitness(s) for s in population])
        probs = (fitness_values - fitness_values.min() + 1e-6)
        probs = probs / probs.sum()

        new_population = []
        for _ in range(num_bees):
            j = np.random.choice(range(num_bees), p=probs)
            candidate = population[j].copy()
            flip = np.random.randint(0, n)
            candidate[flip] = 1 - candidate[flip]
            new_population.append(candidate)
        population = np.array(new_population)

        for s in population:
            f = fitness(s)
            if f > best_fitness:
                best_fitness = f
                best_solution = s.copy()

    elapsed = time.time() - start_time
    total_weight = np.sum(np.array(weights) * best_solution)
    return best_solution, best_fitness, total_weight, elapsed, num_bees * num_iterations


# -----------------------------
# 4️⃣ Genetic Algorithm (GA)
# -----------------------------
def run_GA(weights, values, capacity, pop_size=30, generations=100, mutation_rate=0.1):
    start_time = time.time()
    n = len(weights)
    population = np.random.randint(0, 2, (pop_size, n))

    def fitness(state):
        w = np.sum(np.array(weights) * state)
        v = np.sum(np.array(values) * state)
        if w > capacity:
            v -= (w - capacity) * 10
        return v

    best_solution = None
    best_fitness = float("-inf")

    for _ in range(generations):
        fitness_values = np.array([fitness(s) for s in population])
        parents_idx = np.argsort(fitness_values)[-pop_size // 2:]
        parents = population[parents_idx]

        children = []
        for _ in range(pop_size // 2):
            p1, p2 = parents[np.random.randint(0, len(parents), 2)]
            point = np.random.randint(1, n - 1)
            child = np.concatenate([p1[:point], p2[point:]])
            if np.random.rand() < mutation_rate:
                flip = np.random.randint(0, n)
                child[flip] = 1 - child[flip]
            children.append(child)

        population = np.vstack((parents, children))

        for s in population:
            f = fitness(s)
            if f > best_fitness:
                best_fitness = f
                best_solution = s.copy()

    elapsed = time.time() - start_time
    total_weight = np.sum(np.array(weights) * best_solution)
    return best_solution, best_fitness, total_weight, elapsed, generations * pop_size


# -----------------------------
# 5️⃣ Sinh dữ liệu ngẫu nhiên
# -----------------------------
def random_dataset():
    n = random.randint(5, 12)  # số vật phẩm ngẫu nhiên
    weights = [random.randint(5, 40) for _ in range(n)]
    values = [random.randint(10, 100) for _ in range(n)]
    capacity = random.randint(sum(weights)//3, sum(weights)//2)
    return weights, values, capacity


# -----------------------------
# 6️⃣ Giao diện Console
# -----------------------------
def main():
    print("🎒 BÀI TOÁN KNAPSACK - TỐI ƯU HÓA METAHEURISTIC")
    print("=" * 65)

    weights = []
    values = []
    capacity = 0

    while True:
        print("\n🔹 Chọn chế độ hoạt động:")
        print("1. Nhập dữ liệu thủ công")
        print("2. Tạo dữ liệu ngẫu nhiên")
        print("3. Simulated Annealing (SA)")
        print("4. Bee Colony Optimization (BCO)")
        print("5. Genetic Algorithm (GA)")
        print("6. So sánh tất cả")
        print("0. Thoát")
        choice = input("👉 Nhập lựa chọn của bạn: ")

        # 0️⃣ Thoát
        if choice == "0":
            print("👋 Kết thúc chương trình.")
            break

        # 1️⃣ Nhập dữ liệu thủ công
        elif choice == "1":
            n_items = int(input("Nhập số lượng vật phẩm: "))
            weights = []
            values = []
            for i in range(n_items):
                w = float(input(f"  ➤ Trọng lượng vật {i + 1}: "))
                v = float(input(f"  ➤ Giá trị vật {i + 1}: "))
                weights.append(w)
                values.append(v)
            capacity = float(input("🎯 Nhập sức chứa tối đa của ba lô: "))
            print("✅ Dữ liệu đã được nhập thành công!")

        # 2️⃣ Dữ liệu ngẫu nhiên
        elif choice == "2":
            weights, values, capacity = random_dataset()
            print("\n🎲 DỮ LIỆU NGẪU NHIÊN:")
            print(f"Số vật phẩm: {len(weights)}")
            print(f"Trọng lượng: {weights}")
            print(f"Giá trị: {values}")
            print(f"Sức chứa ba lô: {capacity}")

        # 3️⃣ SA
        elif choice == "3":
            s, v, w, t, c = run_SA(weights, values, capacity)
            print(f"\n🎯 SA: Giá trị = {v:.2f}, Trọng lượng = {w:.2f}")
            print(f"   Trạng thái: {s}\n⏱️  Thời gian: {t:.4f}s | Độ phức tạp: {c}")

        # 4️⃣ BCO
        elif choice == "4":
            s, v, w, t, c = run_BCO(weights, values, capacity)
            print(f"\n🐝 BCO: Giá trị = {v:.2f}, Trọng lượng = {w:.2f}")
            print(f"   Trạng thái: {s}\n⏱️  Thời gian: {t:.4f}s | Độ phức tạp: {c}")

        # 5️⃣ GA
        elif choice == "5":
            s, v, w, t, c = run_GA(weights, values, capacity)
            print(f"\n🧬 GA: Giá trị = {v:.2f}, Trọng lượng = {w:.2f}")
            print(f"   Trạng thái: {s}\n⏱️  Thời gian: {t:.4f}s | Độ phức tạp: {c}")

        # 6️⃣ So sánh tất cả
        elif choice == "6":
            print("\n🚀 Đang chạy tất cả các thuật toán, vui lòng chờ...\n")
            sa = run_SA(weights, values, capacity)
            bco = run_BCO(weights, values, capacity)
            ga = run_GA(weights, values, capacity)

            print("\n📊 BẢNG SO SÁNH KẾT QUẢ")
            print("=" * 90)
            print(f"{'Thuật toán':<25}{'Giá trị':<12}{'Trọng lượng':<15}{'Thời gian (s)':<15}{'Độ phức tạp':<15}")
            print("-" * 90)
            print(f"{'Simulated Annealing':<25}{sa[1]:<12.2f}{sa[2]:<15.2f}{sa[3]:<15.4f}{sa[4]:<15}")
            print(f"{'Bee Colony Optimization':<25}{bco[1]:<12.2f}{bco[2]:<15.2f}{bco[3]:<15.4f}{bco[4]:<15}")
            print(f"{'Genetic Algorithm':<25}{ga[1]:<12.2f}{ga[2]:<15.2f}{ga[3]:<15.4f}{ga[4]:<15}")
            print("=" * 90)

        else:
            print("⚠️ Lựa chọn không hợp lệ. Vui lòng thử lại.")


if __name__ == "__main__":
    main()
