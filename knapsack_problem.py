from simpleai.search import SearchProblem, simulated_annealing

class KnapsackProblem(SearchProblem):
    def __init__(self, weights, values, capacity):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        initial_state = tuple([0 for _ in range(len(weights))])
        super().__init__(initial_state)

    def actions(self, state):
        # có thể lật 1 vật (bật/tắt)
        return list(range(len(state)))

    def result(self, state, action):
        new_state = list(state)
        new_state[action] = 1 - new_state[action]
        return tuple(new_state)

    def value(self, state):
        total_weight = sum(w * s for w, s in zip(self.weights, state))
        total_value = sum(v * s for v, s in zip(self.values, state))
        # nếu vượt quá sức chứa thì phạt
        if total_weight > self.capacity:
            total_value -= (total_weight - self.capacity) * 10
        return total_value

# ---- dữ liệu ví dụ ----
weights = [12, 7, 11, 8, 9]
values = [24, 13, 23, 15, 16]
capacity = 42

problem = KnapsackProblem(weights, values, capacity)

# ---- chạy SA ----
result = simulated_annealing(problem, iterations_limit=10000)

# ---- in kết quả ----
best_state = result.state
total_weight = sum(w * s for w, s in zip(weights, best_state))
total_value = sum(v * s for v, s in zip(values, best_state))

print("🎒 Kết quả Simulated Annealing (simpleAI):")
print("Trạng thái:", best_state)
print("Tổng trọng lượng:", total_weight)
print("Tổng giá trị:", total_value)
print("Các vật được chọn:", [i for i, s in enumerate(best_state) if s == 1])
