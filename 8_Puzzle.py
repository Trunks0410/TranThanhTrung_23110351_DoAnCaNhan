import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import copy
import heapq
import time
import random
import math

class PuzzleState:
    def __init__(self, board, moves=0, previous=None, strategy=None):
        self.board = board
        self.moves = moves
        self.previous = previous
        self.strategy = strategy
        self.h = self._calculate_heuristic() if strategy in [
            "Greedy", "A*", "IDA*", "Simple Hill Climbing", "Steepest-Hill Climbing",
            "Stochastic Hill Climbing", "Simulated Annealing"] else 0

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(str(self.board))

    def __lt__(self, other):
        h1, h2 = self._calculate_heuristic(), other._calculate_heuristic()
        if self.strategy == "UCS":
            return self.moves < other.moves
        elif self.strategy == "Greedy":
            return h1 < h2
        return (self.moves + h1) < (other.moves + h2)

    def _calculate_heuristic(self):
        goal_pos = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1),
                    6: (1, 2), 7: (2, 0), 8: (2, 1), 0: (2, 2)}
        return sum(abs(i - goal_pos[val][0]) + abs(j - goal_pos[val][1])
                   for i in range(3) for j in range(3) if (val := self.board[i][j]) != 0)

def find_blank(board):
    return next((i, j) for i in range(3) for j in range(3) if board[i][j] == 0)

def is_valid(x, y):
    return 0 <= x < 3 and 0 <= y < 3

def get_new_state(board, old_x, old_y, new_x, new_y):
    new_board = copy.deepcopy(board)
    new_board[old_x][old_y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[old_x][old_y]
    return new_board

def get_possible_moves(board):
    i, j = find_blank(board)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    possible_states = []
    for dx, dy in directions:
        new_i, new_j = i + dx, j + dy
        if is_valid(new_i, new_j):
            new_board = get_new_state(board, i, j, new_i, new_j)
            possible_states.append(new_board)
    return possible_states

def get_hash(board):
    return ''.join(str(num) for row in board for num in row)

def manhattan_distance(board, goal):
    goal_pos = {val: (i, j) for i, row in enumerate(goal)
                for j, val in enumerate(row) if val != 0}
    return sum(abs(i - goal_pos[val][0]) + abs(j - goal_pos[val][1])
               for i in range(3) for j in range(3) if (val := board[i][j]) in goal_pos)

def is_solvable(start_board, goal_board):
    def get_inversions(board):
        flat = [num for row in board for num in row if num != 0]
        inversions = sum(1 for i in range(len(flat))
                         for j in range(i + 1, len(flat)) if flat[i] > flat[j])
        return inversions
    return (get_inversions(start_board) % 2) == (get_inversions(goal_board) % 2)

def genetic_algorithm_solve(start, goal, population_size=200, generations=1000, mutation_rate=0.2):
    def generate_individual():
        individual = copy.deepcopy(start)
        current = PuzzleState(individual)
        for _ in range(random.randint(5, 15)):
            moves = get_possible_moves(current.board)
            if moves:
                new_board = random.choice(moves)
                current = PuzzleState(new_board, current.moves + 1, current)
        return current

    def fitness(state):
        distance = manhattan_distance(state.board, goal)
        penalty = sum(10 * (abs(i - 1) + abs(j - 1)) if state.board[i][j] == 5 else 0
                      for i in range(3) for j in range(3))
        return distance + penalty

    def crossover(parent1, parent2):
        child_board = [[0]*3 for _ in range(3)]
        used = set()
        better_parent = parent1 if fitness(
            parent1) < fitness(parent2) else parent2
        for i in range(3):
            for j in range(3):
                if better_parent.board[i][j] == 5:
                    child_board[i][j] = 5
                    used.add(5)
        for i in range(3):
            for j in range(3):
                if child_board[i][j] == 0:
                    val = parent1.board[i][j] if random.random(
                    ) < 0.5 else parent2.board[i][j]
                    if val not in used:
                        child_board[i][j] = val
                        used.add(val)
        remaining = [x for x in range(9) if x not in used]
        random.shuffle(remaining)
        for i in range(3):
            for j in range(3):
                if child_board[i][j] == 0:
                    child_board[i][j] = remaining.pop()
        return PuzzleState(child_board, parent1.moves + 1, parent1)

    def mutate(state):
        moves = get_possible_moves(state.board)
        if moves and random.random() < mutation_rate:
            return PuzzleState(random.choice(moves), state.moves + 1, state)
        return state

    population = [generate_individual() for _ in range(population_size)]
    best_state = population[0]
    best_fitness = fitness(best_state)

    for _ in range(generations):
        current_best = min(population, key=fitness)
        current_fitness = fitness(current_best)
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_state = current_best
        if current_best.board == goal:
            return current_best
        selected = sorted(population, key=fitness)[:population_size // 2]
        new_population = selected[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    return best_state

def and_or_graph_search(start_state, goal_state):
    def or_search(state, visited, path_cost, bound):
        if state.board == goal_state:
            return state, True
        state_hash = get_hash(state.board)
        if state_hash in visited and visited[state_hash] <= path_cost:
            return None, False
        h = manhattan_distance(state.board, goal_state)
        if path_cost + h > bound:
            return None, False
        visited[state_hash] = path_cost
        blank_x, blank_y = find_blank(state.board)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        best_solution = None
        min_moves = float('inf')
        for dx, dy in directions:
            new_x, new_y = blank_x + dx, blank_y + dy
            if is_valid(new_x, new_y):
                new_board = get_new_state(
                    state.board, blank_x, blank_y, new_x, new_y)
                new_state = PuzzleState(
                    new_board, state.moves + 1, state, "AND-OR Graph Search")
                if any(new_state.board == p.board for p in get_path(state)):
                    continue
                result, success = and_search(
                    new_state, visited.copy(), path_cost + 1, bound)
                if success and result.moves < min_moves:
                    min_moves = result.moves
                    best_solution = result
        return best_solution, best_solution is not None

    def and_search(state, visited, path_cost, bound):
        return or_search(state, visited, path_cost, bound)

    def get_path(state):
        path = []
        while state:
            path.append(state)
            state = state.previous
        return path[::-1]

    if not is_solvable(start_state, goal_state):
        return None
    start = PuzzleState(start_state, strategy="AND-OR Graph Search")
    bound = manhattan_distance(start_state, goal_state)
    while True:
        visited = {}
        result, success = and_search(start, visited, 0, bound)
        if success:
            return result
        if not result:
            return None
        bound += 2

def solve_puzzle(start_state, goal_state, strategy="BFS"):
    start_time = time.time()
    if not is_solvable(start_state, goal_state):
        return None, time.time() - start_time

    if strategy == "BFS":
        queue = deque([PuzzleState(start_state)])
        visited = set()
        while queue:
            current = queue.popleft()
            if current.board == goal_state:
                return current, time.time() - start_time
            if current in visited:
                continue
            visited.add(current)
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy)
                if new_state not in visited:
                    queue.append(new_state)
        return None, time.time() - start_time

    elif strategy == "DFS":
        stack = [PuzzleState(start_state)]
        visited = set()
        while stack:
            current = stack.pop()
            if current.board == goal_state:
                return current, time.time() - start_time
            if current in visited:
                continue
            visited.add(current)
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy)
                if new_state not in visited:
                    stack.append(new_state)
        return None, time.time() - start_time

    elif strategy == "UCS":
        queue = []
        heapq.heappush(queue, PuzzleState(start_state, strategy=strategy))
        visited = set()
        while queue:
            current = heapq.heappop(queue)
            if current.board == goal_state:
                return current, time.time() - start_time
            if current in visited:
                continue
            visited.add(current)
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy)
                if new_state not in visited:
                    heapq.heappush(queue, new_state)
        return None, time.time() - start_time

    elif strategy in ["Greedy", "A*"]:
        queue = []
        heapq.heappush(queue, PuzzleState(start_state, strategy=strategy))
        visited = set()
        while queue:
            current = heapq.heappop(queue)
            if current.board == goal_state:
                return current, time.time() - start_time
            if current in visited:
                continue
            visited.add(current)
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy)
                if new_state not in visited:
                    heapq.heappush(queue, new_state)
        return None, time.time() - start_time

    elif strategy == "IDA*":
        def search(state, cost_limit, visited):
            f = state.moves + state.h
            if f > cost_limit:
                return None, f
            if state.board == goal_state:
                return state, cost_limit
            min_exceeded_cost = float('inf')
            visited.add(state)
            for new_board in get_possible_moves(state.board):
                new_state = PuzzleState(
                    new_board, state.moves + 1, state, strategy)
                if new_state not in visited:
                    result, new_cost = search(new_state, cost_limit, visited)
                    if result:
                        return result, cost_limit
                    min_exceeded_cost = min(min_exceeded_cost, new_cost)
            visited.remove(state)
            return None, min_exceeded_cost

        start = PuzzleState(start_state, strategy=strategy)
        cost_limit = start.h
        while True:
            visited = set()
            result, new_limit = search(start, cost_limit, visited)
            if result:
                return result, time.time() - start_time
            if new_limit == float('inf'):
                return None, time.time() - start_time
            cost_limit = new_limit

    elif strategy == "IDS":
        def dfs_limited(state, depth_limit, visited):
            if state.board == goal_state:
                return state
            if state.moves >= depth_limit:
                return None
            visited.add(state)
            for new_board in get_possible_moves(state.board):
                new_state = PuzzleState(
                    new_board, state.moves + 1, state, strategy)
                if new_state not in visited:
                    result = dfs_limited(new_state, depth_limit, visited)
                    if result:
                        return result
            return None

        start = PuzzleState(start_state, strategy=strategy)
        depth = 0
        while depth < 100:
            visited = set()
            result = dfs_limited(start, depth, visited)
            if result:
                return result, time.time() - start_time
            depth += 1
        return None, time.time() - start_time

    elif strategy == "Simple Hill Climbing":
        current = PuzzleState(start_state, strategy=strategy)
        visited = set()
        while current.board != goal_state:
            visited.add(current)
            best_neighbor = None
            best_heuristic = current.h
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy)
                if new_state not in visited and new_state.h < best_heuristic:
                    best_heuristic = new_state.h
                    best_neighbor = new_state
            if not best_neighbor:
                return None, time.time() - start_time
            current = best_neighbor
        return current, time.time() - start_time

    elif strategy == "Steepest-Hill Climbing":
        current = PuzzleState(start_state, strategy=strategy)
        visited = set()
        while current.board != goal_state:
            visited.add(current)
            best_neighbor = None
            best_heuristic = float('inf')
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy)
                if new_state not in visited and new_state.h < best_heuristic:
                    best_heuristic = new_state.h
                    best_neighbor = new_state
            if not best_neighbor or best_heuristic >= current.h:
                return None, time.time() - start_time
            current = best_neighbor
        return current, time.time() - start_time

    elif strategy == "Stochastic Hill Climbing":
        current = PuzzleState(start_state, strategy=strategy)
        visited = set()
        max_restarts = 100
        restarts = 0
        while current.board != goal_state and restarts < max_restarts:
            visited.add(current)
            next_states = [PuzzleState(new_board, current.moves + 1, current, strategy)
                           for new_board in get_possible_moves(current.board) if PuzzleState(new_board) not in visited]
            if not next_states:
                current = PuzzleState(start_state, strategy=strategy)
                restarts += 1
                visited.clear()
                continue
            current = random.choice(next_states)
        return current if current.board == goal_state else None, time.time() - start_time

    elif strategy == "Simulated Annealing":
        current = PuzzleState(start_state, strategy=strategy)
        best_state = current
        best_heuristic = current.h
        temperature = 1000.0
        cooling_rate = 0.995
        stagnant_steps = 0
        max_stagnant = 200
        max_iterations = 2000

        for iteration in range(max_iterations):
            if current.board == goal_state:
                return current, time.time() - start_time

            next_states = [PuzzleState(new_board, current.moves + 1, current, strategy)
                           for new_board in get_possible_moves(current.board)]
            if not next_states:
                break

            next_state = random.choice(next_states)
            delta_e = next_state.h - current.h

            acceptance_prob = math.exp(-delta_e / max(temperature, 1e-6))
            if delta_e < 0 or random.random() < acceptance_prob:
                current = next_state
                if current.h < best_heuristic:
                    best_heuristic = current.h
                    best_state = current
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1
            else:
                stagnant_steps += 1

            if stagnant_steps >= max_stagnant:
                if random.random() < 0.5:
                    current = PuzzleState(start_state, strategy=strategy)
                    stagnant_steps = 0
                else:
                    current = best_state
                    stagnant_steps = 0

            temperature *= cooling_rate

            if temperature < 0.1:
                temperature = 50.0

        if best_state.board == goal_state:
            return best_state, time.time() - start_time

        if best_state.h < current.h:
            return best_state, time.time() - start_time

        return current if current.h < float('inf') else None, time.time() - start_time

    elif strategy == "Beam Search":
        beam_width = 4
        queue = [PuzzleState(start_state, strategy=strategy)]
        visited = set()
        while queue:
            queue.sort(key=lambda state: state.h)
            queue = queue[:beam_width]
            next_queue = []
            for current in queue:
                if current.board == goal_state:
                    return current, time.time() - start_time
                visited.add(current)
                for new_board in get_possible_moves(current.board):
                    new_state = PuzzleState(
                        new_board, current.moves + 1, current, strategy)
                    if new_state not in visited:
                        next_queue.append(new_state)
            queue = next_queue
        return None, time.time() - start_time

    elif strategy == "Genetic Algorithm":
        solution = genetic_algorithm_solve(start_state, goal_state)
        return solution, time.time() - start_time

    elif strategy == "AND-OR Graph Search":
        return and_or_graph_search(start_state, goal_state), time.time() - start_time

    return None, time.time() - start_time

def get_solution_path(solution):
    if not solution:
        return []
    path = []
    current = solution
    while current:
        path.append((current.board, current.moves))
        current = current.previous
    return path[::-1]

def find_moved_tile(prev_board, current_board):
    return next((prev_board[i][j] for i in range(3) for j in range(3)
                 if prev_board[i][j] != 0 and prev_board[i][j] != current_board[i][j]), None)

def get_blank_position(board):
    return next((i, j) for i in range(3) for j in range(3) if board[i][j] == 0)


def is_valid_input(board):
    return sorted(num for row in board for num in row) == list(range(9))

class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trò Chơi 8-Puzzle")
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", lambda event: self.root.attributes(
            "-fullscreen", False))
        self.solving_time = 0.0
        self.speed_delay = 300
        self.current_state_cells = [[None]*3 for _ in range(3)]
        self.animation_ids = []  # Track active animation IDs
        self.setup_ui()
        self.current_board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.solution_path = []
        self.step_index = 0
        self.is_solving = False

    def setup_ui(self):
        self.root.configure(bg="#f0f2f5")
        main_frame = tk.Frame(self.root, bg="#ffffff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        left_frame = tk.Frame(main_frame, bg="#2c3e50", width=600)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        right_frame = tk.Frame(main_frame, bg="#ffffff")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_left(left_frame)
        self.setup_right(right_frame)

    def setup_left(self, parent):
        tk.Label(parent, text="Trò Chơi 8-Puzzle", font=("Helvetica", 24,
                 "bold"), fg="#ffffff", bg="#2c3e50", pady=20).pack(fill=tk.X)

        boards_frame = tk.Frame(parent, bg="#2c3e50")
        boards_frame.pack(pady=20, padx=20)

        start_frame = tk.Frame(boards_frame, bg="#3498db")
        start_frame.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(start_frame, text="Trạng Thái Ban Đầu", font=(
            "Helvetica", 14, "bold"), bg="#3498db", fg="#ffffff").pack(pady=5)
        self.start_cells = [[None]*3 for _ in range(3)]
        self.start_entries = [[None]*3 for _ in range(3)]
        self.setup_grid(start_frame, self.start_entries,
                        self.start_cells, "#3498db")

        goal_frame = tk.Frame(boards_frame, bg="#2ecc71")
        goal_frame.pack(side=tk.LEFT, padx=(10, 0))
        tk.Label(goal_frame, text="Trạng Thái Mục Tiêu", font=(
            "Helvetica", 14, "bold"), bg="#2ecc71", fg="#ffffff").pack(pady=5)
        self.goal_entries = [[None]*3 for _ in range(3)]
        self.setup_grid(goal_frame, self.goal_entries, bg_color="#2ecc71")

        control_frame = tk.Frame(parent, bg="#2c3e50")
        control_frame.pack(fill=tk.X, pady=20, padx=20)

        tk.Label(control_frame, text="Thuật Toán:", font=("Helvetica", 14,
                 "bold"), bg="#2c3e50", fg="#ecf0f1").pack(anchor="w", pady=(10, 5))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Modern.TCombobox",
                        fieldbackground="#1abc9c",
                        background="#1abc9c",
                        foreground="#ffffff",
                        arrowcolor="#ffffff",
                        font=("Helvetica", 16, "bold"),
                        padding=10,
                        bordercolor="#16a085",
                        lightcolor="#16a085",
                        darkcolor="#16a085")
        style.map("Modern.TCombobox",
                  fieldbackground=[("active", "#16a085"),
                                   ("readonly", "#1abc9c")],
                  background=[("active", "#16a085"), ("readonly", "#1abc9c")],
                  foreground=[("active", "#ffffff"), ("readonly", "#ffffff")],
                  arrowcolor=[("active", "#ecf0f1"), ("readonly", "#ffffff")])
        style.configure("Modern.TCombobox*Listbox",
                        background="#2c3e50",
                        foreground="#ffffff",
                        font=("Helvetica", 12),
                        borderwidth=0)
        style.configure("Modern.TCombobox*Listbox*Frame",
                        background="#2c3e50")

        self.algorithm_var = tk.StringVar(value="BFS")
        algo_menu = ttk.Combobox(control_frame, textvariable=self.algorithm_var,
                                 values=["BFS", "UCS", "DFS", "IDS", "Greedy", "A*", "IDA*", "Simple Hill Climbing",
                                         "Steepest-Hill Climbing", "Stochastic Hill Climbing", "Simulated Annealing",
                                         "Beam Search", "Genetic Algorithm", "AND-OR Graph Search"],
                                 style="Modern.TCombobox", state="readonly", width=25)
        algo_menu.pack(fill=tk.X, pady=10)

        button_frame = tk.Frame(control_frame, bg="#2c3e50")
        button_frame.pack(pady=15)

        self.start_button = tk.Button(button_frame, text="Bắt Đầu", command=self.start_solving,
                                      bg="#4682b4", fg="#ffffff", font=("Segoe UI", 12, "bold"))
        self.start_button.pack(pady=5)

        middle_frame = tk.Frame(button_frame, bg="#2c3e50")
        middle_frame.pack(pady=10)
        self.solve_button = tk.Button(middle_frame, text="Giải", command=self.auto_solve,
                                      bg="#66cdaa", fg="#ffffff", font=("Segoe UI", 12, "bold"), state=tk.DISABLED)
        self.solve_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(middle_frame, text="Dừng", command=self.stop_solving,
                                     bg="#ff6347", fg="#ffffff", font=("Segoe UI", 12, "bold"), state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        bottom_frame = tk.Frame(button_frame, bg="#2c3e50")
        bottom_frame.pack(pady=10)
        self.back_button = tk.Button(bottom_frame, text="Quay Lại", command=self.step_back,
                                     bg="#ffa07a", fg="#ffffff", font=("Segoe UI", 12, "bold"), state=tk.DISABLED)
        self.back_button.pack(side=tk.LEFT, padx=5)
        self.forward_button = tk.Button(bottom_frame, text="Tiến Tới", command=self.step_forward,
                                        bg="#9370db", fg="#ffffff", font=("Segoe UI", 12, "bold"), state=tk.DISABLED)
        self.forward_button.pack(side=tk.LEFT, padx=5)
        self.reset_button = tk.Button(bottom_frame, text="Đặt Lại", command=self.reset_puzzle,
                                      bg="#a9a9a9", fg="#ffffff", font=("Segoe UI", 12, "bold"), state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.bind_input_navigation()

    def setup_grid(self, parent, entries, cells=None, bg_color="#ffffff"):
        grid = tk.Frame(parent, bg=bg_color)
        grid.pack(pady=10, padx=10)
        for i in range(3):
            for j in range(3):
                frame = tk.Frame(grid, width=70, height=70, bg="#ffffff",
                                 highlightbackground="#ffffff", highlightthickness=2, bd=4, relief="ridge")
                frame.grid(row=i, column=j, padx=2, pady=2)
                frame.grid_propagate(False)
                entry = tk.Entry(frame, width=2, font=(
                    "Helvetica", 18), justify="center", bg="#ffffff", fg="#2c3e50", borderwidth=0)
                entry.place(relx=0.5, rely=0.5, anchor="center")
                entries[i][j] = entry
                if cells:
                    cells[i][j] = tk.Label(frame, text="", font=(
                        "Helvetica", 18), bg="#ffffff", fg="#2c3e50")

    def setup_right(self, parent):
        tk.Label(parent, text="Quá Trình Giải", font=("Helvetica", 24,
                 "bold"), bg="#ffffff", fg="#2c3e50").pack(pady=(20, 10))

        self.info_frame = tk.Frame(parent, bg="#ecf0f1")
        self.info_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.time_label = tk.Label(self.info_frame, text="Thời Gian: 0.000 giây", font=(
            "Helvetica", 14), bg="#ecf0f1", fg="#2c3e50")
        self.time_label.pack(anchor="w", padx=10, pady=5)

        self.result_container = tk.Frame(parent, bg="#ffffff")
        self.result_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=0)

        self.state_view_frame = tk.Frame(
            self.result_container, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.state_view_frame.pack(fill=tk.X, pady=(0, 15), ipady=10)
        tk.Label(self.state_view_frame, text="Các Bước Di Chuyển", font=(
            "Helvetica", 16, "bold"), bg="#ffffff", fg="#2c3e50").pack(pady=5)

        self.canvas_frame = tk.Frame(self.state_view_frame, bg="#ecf0f1")
        self.canvas_frame.pack(fill=tk.X, padx=10, pady=5)
        self.state_frame = tk.Frame(self.canvas_frame, bg="#ecf0f1")
        self.state_frame.pack(anchor="center")
        self.setup_solution_grid()

        self.summary_frame = tk.Frame(
            self.result_container, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.summary_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))
        tk.Label(self.summary_frame, text="Tóm Tắt Lời Giải", font=(
            "Helvetica", 16, "bold"), bg="#ffffff", fg="#2c3e50").pack(pady=5)
        self.result_text = tk.Text(self.summary_frame, height=10, font=(
            "Helvetica", 12), bg="#ecf0f1", fg="#2c3e50", bd=0)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.result_text.config(state=tk.DISABLED)

    def setup_solution_grid(self):
        self.grid_frame = tk.Frame(
            self.state_frame, bg="#ecf0f1", width=240, height=240)
        self.grid_frame.pack(pady=15, padx=15)
        self.grid_frame.pack_propagate(False)

    def bind_input_navigation(self):
        for i in range(3):
            for j in range(3):
                start_entry = self.start_entries[i][j]
                goal_entry = self.goal_entries[i][j]
                for entry, is_goal in [(start_entry, False), (goal_entry, True)]:
                    entry.bind("<Up>", lambda e, x=i, y=j,
                               g=is_goal: self.move_focus(x, y, -1, 0, g))
                    entry.bind("<Down>", lambda e, x=i, y=j,
                               g=is_goal: self.move_focus(x, y, 1, 0, g))
                    entry.bind("<Left>", lambda e, x=i, y=j,
                               g=is_goal: self.move_focus(x, y, 0, -1, g))
                    entry.bind("<Right>", lambda e, x=i, y=j,
                               g=is_goal: self.move_focus(x, y, 0, 1, g))

    def move_focus(self, x, y, dx, dy, is_goal=False):
        new_x = max(0, min(2, x + dx))
        new_y = max(0, min(2, y + dy))
        entries = self.goal_entries if is_goal else self.start_entries
        entries[new_x][new_y].focus_set()

    def update_time_display(self):
        self.time_label.config(text=f"Thời Gian: {self.solving_time:.3f} giây")

    def draw_solution_info(self):
        for widget in self.info_frame.winfo_children()[1:]:
            widget.destroy()
        if self.solution_path:
            total_steps = len(self.solution_path) - 1
            total_cost = self.solution_path[-1][1]
            tk.Label(self.info_frame, text=f"Tổng Số Bước: {total_steps}", font=(
                "Helvetica", 14), bg="#ecf0f1", fg="#2c3e50").pack(anchor="w", padx=10)
            tk.Label(self.info_frame, text=f"Tổng Chi Phí: {total_cost}", font=(
                "Helvetica", 14), bg="#ecf0f1", fg="#2c3e50").pack(anchor="w", padx=10)
            tk.Label(self.info_frame, text=f"Tổng Số Trạng Thái: {len(self.solution_path)}", font=(
                "Helvetica", 14), bg="#ecf0f1", fg="#2c3e50").pack(anchor="w", padx=10)

    def draw_solution_summary(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        if not self.solution_path:
            self.result_text.insert(tk.END, "")
        else:
            summary = f"Thuật Toán: {self.algorithm_var.get()}\nCác bước di chuyển:\n"
            for i in range(1, len(self.solution_path)):
                prev_board, curr_board = self.solution_path[i -
                                                            1][0], self.solution_path[i][0]
                moved_tile = find_moved_tile(prev_board, curr_board)
                if moved_tile:
                    prev_blank, curr_blank = get_blank_position(
                        prev_board), get_blank_position(curr_board)
                    direction = "đi xuống" if prev_blank[0] > curr_blank[0] else "đi lên" if prev_blank[
                        0] < curr_blank[0] else "qua phải" if prev_blank[1] > curr_blank[1] else "qua trái"
                    summary += f"Bước {i}: Di chuyển ô {moved_tile} {direction}\n"
            self.result_text.insert(tk.END, summary)
        self.result_text.config(state=tk.DISABLED)

    def cancel_animations(self):
        """Cancel all pending animations."""
        for anim_id in self.animation_ids:
            self.root.after_cancel(anim_id)
        self.animation_ids = []

    def update_solution_board(self):
        """Dynamically create/update a seamless tile board after Start."""
        # Cancel any ongoing animations
        self.cancel_animations()

        # Clear existing grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        if not self.solution_path:
            self.current_state_cells = [[None]*3 for _ in range(3)]
            return

        # Create grid dynamically
        board_size = 240
        cell_size = board_size // 3
        board = self.solution_path[self.step_index][0]
        moved_tile = find_moved_tile(
            self.solution_path[self.step_index - 1][0], board) if self.step_index > 0 else None

        for i in range(3):
            for j in range(3):
                value = board[i][j]
                text = "" if value == 0 else str(value)
                default_bg = "#e0e6e8" if value == 0 else "#1abc9c"
                fg = "#ffffff" if value != 0 else "#2c3e50"
                label = tk.Label(self.grid_frame, text=text, font=("Helvetica", 22, "bold"),
                                 bg=default_bg, fg=fg, width=3, height=1,
                                 borderwidth=0, relief="flat")
                label.place(x=j*cell_size, y=i*cell_size,
                            width=cell_size, height=cell_size)
                self.current_state_cells[i][j] = label

                if moved_tile and board[i][j] == moved_tile:
                    def color_transition(count=6, colors=["#ff6b81", "#ff9f43", "#00cec9", "#3498db", "#1abc9c", "#1abc9c"], label=label):
                        if count > 0 and label.winfo_exists():
                            color_idx = min(5, 6 - count)
                            font_size = 28 if count > 3 else 24
                            label.config(bg=colors[color_idx], fg="#ffffff",
                                         font=("Helvetica", font_size, "bold"),
                                         highlightbackground="#f1c40f", highlightthickness=2)
                            anim_id = self.root.after(
                                100, lambda: color_transition(count - 1, colors, label))
                            self.animation_ids.append(anim_id)
                        else:
                            if label.winfo_exists():
                                label.config(bg=default_bg, fg=fg, font=("Helvetica", 22, "bold"),
                                             highlightbackground="#ecf0f1", highlightthickness=0)
                    color_transition()

    def start_solving(self):
        try:
            start_board = self.get_board_from_entries(self.start_entries)
            goal_board = self.get_board_from_entries(self.goal_entries)
            if not start_board or not goal_board or not is_valid_input(start_board) or not is_valid_input(goal_board):
                messagebox.showerror(
                    "Lỗi", "Trạng thái ban đầu hoặc mục tiêu không hợp lệ.\nNhập số từ 0-8 (0 là ô trống).")
                return
            if not is_solvable(start_board, goal_board):
                messagebox.showerror(
                    "Lỗi", "Trạng thái ban đầu không thể đạt được trạng thái mục tiêu.\nKiểm tra tính khả thi.")
                return
            solution, self.solving_time = solve_puzzle(
                start_board, goal_board, self.algorithm_var.get())
            self.solution_path = get_solution_path(
                solution) if solution else []
            self.step_index = 0
            self.update_board_display()
            self.update_solution_board()
            self.update_time_display()
            self.draw_solution_info()
            self.draw_solution_summary()
            self.solve_button.config(
                state=tk.NORMAL if self.solution_path else tk.DISABLED)
            self.stop_button.config(
                state=tk.NORMAL if self.solution_path else tk.DISABLED)
            self.reset_button.config(
                state=tk.NORMAL if self.solution_path else tk.DISABLED)
            self.forward_button.config(
                state=tk.NORMAL if self.solution_path else tk.DISABLED)
            self.back_button.config(state=tk.DISABLED)
            if not solution:
                messagebox.showinfo(
                    "Kết Quả", f"Không tìm thấy lời giải trong {self.solving_time:.3f} giây.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi giải: {str(e)}")

    def get_board_from_entries(self, entries):
        board = []
        for i in range(3):
            row = []
            for j in range(3):
                value = entries[i][j].get().strip()
                num = 0 if value == "" else int(value)
                if not 0 <= num <= 8:
                    raise ValueError("Giá trị phải từ 0 đến 8")
                row.append(num)
            board.append(row)
        return board

    def update_board_display(self):
        if not self.solution_path:
            return
        board = self.solution_path[self.step_index][0]
        moved_tile = find_moved_tile(
            self.solution_path[self.step_index - 1][0], board) if self.step_index > 0 else None

        for i in range(3):
            for j in range(3):
                default_bg = "#f0f0f0" if board[i][j] == 0 else "#ffffff"
                fg_color = "#000000" if board[i][j] == 0 else "#2c3e50"
                self.start_cells[i][j].config(
                    text="" if board[i][j] == 0 else str(board[i][j]),
                    bg=default_bg, fg=fg_color)

                if moved_tile and board[i][j] == moved_tile:
                    def color_transition(count=6, colors=["#ff6b81", "#ff9f43", "#00cec9", "#3498db", "#1abc9c", "#ffffff"], label=self.start_cells[i][j]):
                        if count > 0 and label.winfo_exists():
                            color_idx = min(5, 6 - count)
                            label.config(bg=colors[color_idx], fg="#ffffff" if color_idx < 5 else fg_color,
                                         font=("Helvetica", 22, "bold"))
                            anim_id = self.root.after(
                                100, lambda: color_transition(count - 1, colors, label))
                            self.animation_ids.append(anim_id)
                        else:
                            if label.winfo_exists():
                                label.config(
                                    bg=default_bg, fg=fg_color, font=("Helvetica", 18))
                    color_transition()

        if self.step_index == len(self.solution_path) - 1 and not self.is_solving:
            messagebox.showinfo(
                "Hoàn Thành", "Đã giải xong! Trạng thái ban đầu khớp với mục tiêu.")

    def step_forward(self):
        if self.step_index < len(self.solution_path) - 1:
            self.step_index += 1
            self.update_board_display()
            self.update_solution_board()
            self.back_button.config(state=tk.NORMAL)
            self.forward_button.config(state=tk.NORMAL if self.step_index < len(
                self.solution_path) - 1 else tk.DISABLED)

    def step_back(self):
        if self.step_index > 0:
            self.step_index -= 1
            self.update_board_display()
            self.update_solution_board()
            self.forward_button.config(state=tk.NORMAL)
            self.back_button.config(
                state=tk.NORMAL if self.step_index > 0 else tk.DISABLED)

    def auto_solve(self):
        if not self.solution_path:
            return
        self.is_solving = True
        self.solve_button.config(state=tk.DISABLED)
        self.back_button.config(state=tk.DISABLED)
        self.forward_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.auto_solve_step()

    def auto_solve_step(self):
        if not self.is_solving or self.step_index >= len(self.solution_path) - 1:
            self.stop_solving()
            return
        self.step_index += 1
        self.update_board_display()
        self.update_solution_board()
        self.root.after(self.speed_delay, self.auto_solve_step)

    def stop_solving(self):
        self.is_solving = False
        self.cancel_animations()  # Ensure animations stop
        self.solve_button.config(
            state=tk.NORMAL if self.solution_path else tk.DISABLED)
        self.reset_button.config(
            state=tk.NORMAL if self.solution_path else tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.back_button.config(
            state=tk.NORMAL if self.step_index > 0 else tk.DISABLED)
        self.forward_button.config(state=tk.NORMAL if self.step_index < len(
            self.solution_path) - 1 else tk.DISABLED)

    def reset_puzzle(self):
        self.stop_solving()
        self.step_index = 0
        self.solution_path = []
        self.solving_time = 0.0
        for i in range(3):
            for j in range(3):
                self.start_entries[i][j].delete(0, tk.END)
        self.update_board_display()
        self.update_solution_board()
        self.update_time_display()
        self.draw_solution_summary()
        self.start_button.config(state=tk.NORMAL)
        self.solve_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.back_button.config(state=tk.DISABLED)
        self.forward_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
