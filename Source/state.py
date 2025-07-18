import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import copy
import heapq
import time
import random
import math
import uuid
import logging
import numpy as np


class PuzzleState:
    def __init__(self, board, moves=0, previous=None, strategy=None):
        self.board = copy.deepcopy(board)
        self.moves = moves
        self.previous = previous
        self.strategy = strategy
        self._hash = None
        self.h = self._calculate_heuristic() if strategy in [
            "Greedy", "A*", "IDA*", "Simple Hill Climbing", "Steepest-Hill Climbing",
            "Stochastic Hill Climbing", "Simulated Annealing", "DFS",
            "Search with Partial Observations", "Backtracking", "AC3",
            "Generate and Test", "Q-Learning", "Genetic Algorithm", "Local Beam Search",
            "AND-OR Graph Search"] else 0

    def __eq__(self, other):
        if not isinstance(other, PuzzleState):
            return False
        return self.board == other.board

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(tuple(row) for row in self.board))
        return self._hash

    def __lt__(self, other):
        h1, h2 = self._calculate_heuristic(), other._calculate_heuristic()
        if self.strategy == "UCS":
            return self.moves < other.moves
        elif self.strategy == "Greedy":
            return h1 < h2
        elif self.strategy in ["Genetic Algorithm", "Local Beam Search"]:
            return h1 < h2
        return (self.moves + h1) < (other.moves + h2)

    def _calculate_heuristic(self):
        goal_pos = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1),
                    6: (1, 2), 7: (2, 0), 8: (2, 1), 0: (2, 2)}
        return sum(abs(i - goal_pos[val][0]) + abs(j - goal_pos[val][1])
                   for i in range(3) for j in range(3) if (val := self.board[i][j]) != 0)


def find_blank(board):
    try:
        if isinstance(board, PuzzleState):
            board = board.board
        elif isinstance(board, tuple):
            logging.warning("find_blank received tuple; converting to list")
            board = [list(row) for row in board]
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return i, j
        logging.error(f"No blank tile in board: {board}")
        raise ValueError("Invalid board: No blank tile (0) found")
    except (TypeError, IndexError) as e:
        logging.error(
            f"Invalid board format in find_blank: {board}, error={str(e)}")
        raise ValueError("Invalid board format")


def is_valid(x, y):
    return 0 <= x < 3 and 0 <= y < 3


def get_new_state(board, old_x, old_y, new_x, new_y):
    new_board = copy.deepcopy(board)
    new_board[old_x][old_y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[old_x][old_y]
    return new_board


def get_possible_moves(board):
    try:
        if isinstance(board, PuzzleState):
            board = board.board
        elif isinstance(board, tuple):
            logging.warning(
                "get_possible_moves received tuple; converting to list")
            board = [list(row) for row in board]
        i, j = find_blank(board)
    except ValueError:
        logging.debug("get_possible_moves: No blank tile found")
        return []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    possible_states = []
    for dx, dy in directions:
        new_i, new_j = i + dx, j + dy
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_board = [[board[i][j] for j in range(3)] for i in range(3)]
            new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
            possible_states.append(new_board)
    return possible_states


def get_hash(board):
    try:
        if isinstance(board, PuzzleState):
            logging.debug("get_hash received PuzzleState; extracting board")
            board = board.board
        elif isinstance(board, tuple):
            logging.warning("get_hash received tuple; converting to list")
            board = [list(row) for row in board]
        board_tuple = tuple(tuple(row) for row in board)
        return hash(board_tuple)
    except (TypeError, IndexError) as e:
        logging.error(f"Invalid board for get_hash: {board}, error={str(e)}")
        raise ValueError("Board must be a 3x3 list or tuple of lists/tuples")


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


def is_valid_move(prev_board, next_board):
    if not prev_board or not next_board:
        logging.debug("is_valid_move received empty board")
        return False
    try:
        if isinstance(prev_board, PuzzleState):
            prev_board = prev_board.board
        elif isinstance(prev_board, tuple):
            logging.warning(
                "is_valid_move received tuple as prev_board; converting")
            prev_board = [list(row) for row in prev_board]
        if isinstance(next_board, PuzzleState):
            next_board = next_board.board
        elif isinstance(next_board, tuple):
            logging.warning(
                "is_valid_move received tuple as next_board; converting")
            next_board = [list(row) for row in next_board]
        prev_board = [[prev_board[i][j] for j in range(3)] for i in range(3)]
        next_board = [[next_board[i][j] for j in range(3)] for i in range(3)]
        blank_i, blank_j = find_blank(prev_board)
        next_blank_i, next_blank_j = find_blank(next_board)
        return (abs(blank_i - next_blank_i) + abs(blank_j - next_blank_j) == 1 and
                all(prev_board[i][j] == next_board[i][j] for i in range(3) for j in range(3)
                    if (i, j) not in [(blank_i, blank_j), (next_blank_i, next_blank_j)]))
    except (TypeError, IndexError, ValueError) as e:
        logging.error(
            f"Invalid boards in is_valid_move: prev={prev_board}, next={next_board}, error={str(e)}")
        return False


def ac3_solve(start_state, goal_state, max_depth=100):
    start_time = time.time()
    if isinstance(start_state, tuple):
        start_state = [list(row) for row in start_state]
    if isinstance(goal_state, tuple):
        goal_state = [list(row) for row in goal_state]
    if not is_valid_input(start_state) or not is_valid_input(goal_state):
        logging.error(
            "Invalid board: Must contain exactly one of each number 0-8")
        return None, time.time() - start_time, []
    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None, time.time() - start_time, []

    def setup_csp(start_board, max_depth):
        csp = {
            'variables': list(range(max_depth)),
            'domains': {i: [] for i in range(max_depth)},
            'neighbors': {i: [i+1] for i in range(max_depth-1)},
            'constraints': lambda arc, values: is_valid_move(values[0], values[1])
        }
        csp['neighbors'][max_depth-1] = []
        csp['domains'][0] = [copy.deepcopy(start_board)]
        queue = deque([(PuzzleState(start_board, strategy="AC3"), 0)])
        visited = set()
        max_states_per_depth = 10000
        logging.debug(f"Starting CSP setup with start_board: {start_board}")

        while queue:
            state, depth = queue.popleft()
            board_hash = get_hash(state.board)
            if board_hash in visited:
                continue
            visited.add(board_hash)
            if depth + 1 >= max_depth:
                continue
            possible_moves = get_possible_moves(state.board)
            move_states = [
                PuzzleState(new_board, state.moves + 1, state, "AC3")
                for new_board in possible_moves
                if is_valid_move(state.board, new_board)
            ]
            move_states.sort(key=lambda s: s.h)
            for new_state in move_states[:max_states_per_depth]:
                new_hash = get_hash(new_state.board)
                if new_hash not in visited:
                    csp['domains'][depth + 1].append(new_state.board)
                    queue.append((new_state, depth + 1))
            if any(new_state.board == goal_state for new_state in move_states):
                csp['domains'][min(depth + 1, max_depth - 1)
                               ].append(goal_state)
                logging.debug(f"Goal state found at depth {depth + 1}")
        for i in range(max_depth // 2, max_depth):
            if goal_state not in csp['domains'][i]:
                csp['domains'][i].append(copy.deepcopy(goal_state))
                logging.debug(f"Added goal state to domain[{i}]")
            for _ in range(5):
                intermediate_board = generate_random_solvable_state(
                    goal_state, max_moves=5)
                if intermediate_board not in csp['domains'][i]:
                    csp['domains'][i].append(intermediate_board)
                    logging.debug(
                        f"Added intermediate state to domain[{i}]: {intermediate_board}")
        domain_sizes = [len(csp['domains'][i]) for i in range(max_depth)]
        logging.debug(f"CSP domain sizes: {domain_sizes}")
        return csp

    def ac3(csp):
        queue = deque([(i, j) for i in csp['variables']
                      for j in csp['neighbors'][i]])
        initial_sizes = [len(csp['domains'][i]) for i in csp['variables']]
        logging.debug(f"Initial domain sizes before AC3: {initial_sizes}")
        while queue:
            xi, xj = queue.popleft()
            revised = False
            domain_xi = csp['domains'][xi].copy()
            if xi == 8:
                logging.debug(
                    f"Processing Domain[{xi}], initial boards: {len(domain_xi)}")
            for x in domain_xi:
                if xi < 10:
                    continue
                constraint_satisfied = False
                for y in csp['domains'][xj]:
                    if is_valid_move(x, y):
                        constraint_satisfied = True
                        break
                    elif xi == 8:
                        logging.debug(
                            f"No valid move from {x} to {y} in Domain[{xj}]")
                if not constraint_satisfied:
                    if len(csp['domains'][xi]) > 1:
                        csp['domains'][xi].remove(x)
                        revised = True
                        logging.debug(f"Removed board from domain[{xi}]: {x}")
            if revised:
                if not csp['domains'][xi]:
                    logging.warning(f"AC3: Domain[{xi}] became empty")
                    return False
                if xi == 8:
                    logging.debug(
                        f"Domain[{xi}] after revision: {len(csp['domains'][xi])} boards")
                for xk in [k for k in csp['variables'] if xi in csp['neighbors'][k] and k != xj]:
                    queue.append((xk, xi))
        final_sizes = [len(csp['domains'][i]) for i in csp['variables']]
        logging.debug(f"Final domain sizes after AC3: {final_sizes}")
        return True

    def backtrack(csp, assignment, depth):
        if depth > 0 and assignment[-1].board == goal_state:
            logging.debug(
                f"Backtrack: Goal reached at depth {depth}, moves {assignment[-1].moves}")
            return assignment[-1]
        if depth >= max_depth or not csp['domains'][depth]:
            logging.debug(
                f"Backtrack: Failed at depth {depth}, domain empty={not csp['domains'][depth]}")
            return None
        sorted_boards = sorted(
            csp['domains'][depth], key=lambda b: manhattan_distance(b, goal_state))
        for board in sorted_boards:
            if depth == 0 or is_valid_move(assignment[-1].board, board):
                new_state = PuzzleState(
                    board,
                    assignment[-1].moves + 1 if assignment else 0,
                    assignment[-1] if assignment else None,
                    strategy="AC3"
                )
                logging.debug(
                    f"Backtrack: Depth {depth}, trying board: {board}")
                result = backtrack(csp, assignment + [new_state], depth + 1)
                if result:
                    return result
        logging.debug(f"Backtrack: No solution at depth {depth}")
        return None

    try:
        csp = setup_csp(start_state, max_depth)
        goal_present = any(goal_state in csp['domains'][i]
                           for i in range(max_depth))
        if not goal_present:
            logging.error(
                "AC3: Goal state not present in any domain after setup")
            raise ValueError("Goal state not reachable")
        if not ac3(csp):
            logging.error(
                "AC3: Constraint propagation failed, falling back to BFS")
            raise ValueError("Constraint propagation failed")
        if not all(csp['domains'][i] for i in range(max_depth)):
            logging.error("AC3: One or more domains empty after propagation")
            raise ValueError("Empty domains after AC3")
        start_state_obj = PuzzleState(start_state, strategy="AC3")
        solution = backtrack(csp, [start_state_obj], 0)
        total_time = time.time() - start_time
        if solution:
            logging.info(
                f"AC3: Solution found in {solution.moves} moves, {total_time:.3f} seconds")
            return solution, total_time, []
        else:
            logging.error(f"AC3: Backtracking failed, falling back to BFS")
            raise ValueError("Backtracking failed")
    except Exception as e:
        logging.error(f"Error in ac3_solve: {str(e)}, falling back to BFS")
        solution, bfs_time, history = solve_puzzle(
            start_state, goal_state, "BFS")
        total_time = time.time() - start_time + bfs_time
        return solution, total_time, history


def backtracking_solve(start_state, goal_state, max_depth=50):
    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None

    def backtrack(state, depth, visited):
        if state.board == goal_state:
            return state
        if depth >= max_depth:
            return None

        state_hash = get_hash(state.board)
        if state_hash in visited:
            return None
        visited.add(state_hash)

        for new_board in get_possible_moves(state.board):
            if is_valid_move(state.board, new_board):
                new_state = PuzzleState(
                    new_board, state.moves + 1, state, "Backtracking")
                result = backtrack(new_state, depth + 1, visited.copy())
                if result:
                    return result
        return None

    start = PuzzleState(start_state, strategy="Backtracking")
    result = backtrack(start, 0, set())
    if result:
        logging.debug(f"Backtracking found solution in {result.moves} moves")
    else:
        logging.debug("Backtracking failed to find solution")
    return result


def generate_and_test_solve(start_state, goal_state, max_steps=1000, max_restarts=10):
    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None, time.time(), []

    # Convert start_state to list if it's a tuple
    if isinstance(start_state, tuple):
        logging.warning(
            "generate_and_test_solve received tuple; converting to list")
        start_state = [list(row) for row in start_state]

    start_time = time.time()
    best_state = None
    best_heuristic = float('inf')
    solution_path = []

    for restart in range(max_restarts):
        # Use the original start_state for the first restart
        if restart == 0:
            current_state = PuzzleState(
                start_state, strategy="Generate and Test")
            logging.debug(
                f"Generate and Test: Using original start_state: {start_state}")
        else:
            # Generate a new solvable state for subsequent restarts
            current_state = PuzzleState(
                generate_random_solvable_state(goal_state), strategy="Generate and Test")
            logging.debug(
                f"Generate and Test: Restart {restart + 1}, new start_state: {current_state.board}")

        visited = set([get_hash(current_state.board)])
        steps = 0
        local_path = [(current_state.board, current_state.moves)]

        while steps < max_steps:
            if current_state.board == goal_state:
                # Validate path for the first restart
                if restart == 0 and local_path[0][0] == start_state:
                    logging.debug(
                        f"Generate and Test: Solution found in {current_state.moves} moves, path length: {len(local_path)}")
                    return current_state, time.time() - start_time, []
                else:
                    logging.debug(
                        f"Generate and Test: Solution found but path does not start with original start_state")
                    break

            # Update best state
            current_heuristic = manhattan_distance(
                current_state.board, goal_state)
            if current_heuristic < best_heuristic:
                best_state = current_state
                best_heuristic = current_heuristic
                solution_path = local_path[:]
                logging.debug(
                    f"Generate and Test: New best heuristic {best_heuristic} at state {current_state.board}")

            # Generate a new state
            possible_moves = get_possible_moves(current_state.board)
            if not possible_moves:
                logging.debug(
                    f"Generate and Test: No valid moves at state {current_state.board}")
                break

            heuristics = [manhattan_distance(
                new_board, goal_state) for new_board in possible_moves]
            weights = [1.0 / (h + 1) for h in heuristics]
            total = sum(weights)
            probabilities = [w / total for w in weights]
            new_board = random.choices(possible_moves, probabilities, k=1)[0]

            new_state_hash = get_hash(new_board)
            if new_state_hash not in visited:
                visited.add(new_state_hash)
                current_state = PuzzleState(
                    new_board, current_state.moves + 1, current_state, strategy="Generate and Test")
                local_path.append((current_state.board, current_state.moves))
                steps += 1
            else:
                logging.debug(
                    f"Generate and Test: State {new_board} already visited")
                break

            logging.debug(
                f"Generate and Test: Restart {restart + 1}, Step {steps}, Heuristic {current_heuristic}")

        # Try BFS from the best state, ensuring input is a list
        if best_state and best_heuristic < float('inf') and restart == 0:
            best_board = best_state.board
            if isinstance(best_board, tuple):
                logging.warning(
                    "Best state board is tuple; converting to list")
                best_board = [list(row) for row in best_board]
            logging.debug(
                f"Generate and Test: Attempting BFS from best state: {best_board}")
            valid_solution, bfs_time, history = solve_puzzle(
                best_board, goal_state, "BFS")
            if valid_solution and isinstance(valid_solution, PuzzleState):
                reconstructed_path = get_solution_path(valid_solution)
                if reconstructed_path and reconstructed_path[0][0] == start_state:
                    logging.debug(
                        f"Generate and Test: Reconstructed solution with {valid_solution.moves} moves")
                    return valid_solution, time.time() - start_time + bfs_time, []
                else:
                    logging.debug(
                        f"Generate and Test: BFS solution does not start with original start_state")
            else:
                logging.warning(
                    f"Generate and Test: BFS returned invalid solution type: {type(valid_solution)}")

    # Final BFS attempt from original start_state
    logging.debug(
        f"Generate and Test: Final BFS attempt from original start_state: {start_state}")
    valid_solution, bfs_time, history = solve_puzzle(
        start_state, goal_state, "BFS")
    total_time = time.time() - start_time + bfs_time
    if valid_solution and isinstance(valid_solution, PuzzleState):
        logging.debug(
            f"Generate and Test: BFS reconstructed solution with {valid_solution.moves} moves")
        return valid_solution, total_time, []
    else:
        logging.warning(
            f"Generate and Test: Final BFS returned invalid solution type: {type(valid_solution)}")

    logging.debug("Generate and Test: Failed to find any solution")
    return best_state if best_state and isinstance(best_state, PuzzleState) else None, total_time, []


def genetic_algorithm_solve(start, goal, population_size=100, max_generations=1000, mutation_rate=0.1):
    def fitness_fn(individual):
        return manhattan_distance(individual.board, goal)

    def random_selection(population, fitness_fn):
        total_fitness = sum(1.0 / (fitness_fn(ind) + 1) for ind in population)
        if total_fitness == 0:
            return random.choice(population)
        weights = [(1.0 / (fitness_fn(ind) + 1)) /
                   total_fitness for ind in population]
        return random.choices(population, weights=weights, k=1)[0]

    def reproduce(x, y):
        n = 9  # Length of flattened board
        c = random.randint(0, n - 1)
        flat_x = [num for row in x.board for num in row]
        flat_y = [num for row in y.board for num in row]
        child_flat = flat_x[:c] + flat_y[c:]
        # Ensure child has all numbers 0-8 exactly once
        used = set(child_flat[:c])
        remaining = [num for num in flat_y if num not in used]
        child_flat[c:] = remaining[:n - c]
        # Convert back to 3x3 board
        child_board = [[child_flat[i * 3 + j]
                        for j in range(3)] for i in range(3)]
        return PuzzleState(child_board, x.moves + 1, x, strategy="Genetic Algorithm")

    def mutate(individual, mutation_rate):
        if random.random() < mutation_rate:
            moves = get_possible_moves(individual.board)
            if moves:
                new_board = random.choice(moves)
                return PuzzleState(new_board, individual.moves + 1, individual, strategy="Genetic Algorithm")
        return individual

    if not is_solvable(start, goal):
        logging.info("Puzzle is not solvable")
        return None, time.time(), []

    start_time = time.time()
    # Initialize population, ensuring start_state is included
    # Include original start state
    population = [PuzzleState(start, strategy="Genetic Algorithm")]
    for _ in range(population_size - 1):  # Generate remaining population
        current_board = copy.deepcopy(start)
        current_state = PuzzleState(
            current_board, strategy="Genetic Algorithm")
        for _ in range(random.randint(5, 15)):  # Random moves to diversify
            moves = get_possible_moves(current_state.board)
            if moves:
                new_board = random.choice(moves)
                current_state = PuzzleState(
                    new_board, current_state.moves + 1, current_state, strategy="Genetic Algorithm")
        population.append(current_state)

    for generation in range(max_generations):
        new_population = []
        for _ in range(population_size):
            x = random_selection(population, fitness_fn)
            y = random_selection(population, fitness_fn)
            child = reproduce(x, y)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        # Find the best individual
        best_individual = min(population, key=fitness_fn)
        if best_individual.board == goal:
            logging.debug(
                f"Genetic Algorithm found solution in generation {generation}, moves {best_individual.moves}")
            return best_individual, time.time() - start_time, []

        logging.debug(
            f"Genetic Algorithm: Generation {generation}, best fitness {fitness_fn(best_individual)}")

    # Return the best individual if no solution found
    best_individual = min(population, key=fitness_fn)
    logging.debug("Genetic Algorithm did not find exact solution")
    # Try to reconstruct a valid path to the goal using BFS
    valid_solution, solve_time, history = solve_puzzle(
        best_individual.board, goal, "BFS")
    total_time = time.time() - start_time
    if valid_solution:
        logging.debug(
            f"Genetic Algorithm: Reconstructed valid solution with {valid_solution.moves} moves")
        return valid_solution, total_time, []
    return best_individual, total_time, []


def and_or_graph_search(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None, time.time(), []

    def or_search(state, g, f_limit, visited, start_time, timeout=20):
        try:
            if time.time() - start_time > timeout:
                logging.debug("AND-OR Graph Search timed out")
                return None, float('inf')
            if not isinstance(state, PuzzleState):
                logging.warning(
                    f"Invalid state type in or_search: {type(state)} - {state}")
                if isinstance(state, (list, tuple)):
                    state = PuzzleState(state, strategy="AND-OR Graph Search")
                    logging.debug(f"Converted to PuzzleState: {state.board}")
                else:
                    logging.error(
                        f"Cannot convert state to PuzzleState: {state}")
                    return None, float('inf')
            logging.debug(
                f"or_search: g={g}, f_limit={f_limit}, state={state.board}")
            if state.board == goal_state:
                return state, state.h
            state_hash = get_hash(state.board)
            logging.debug(f"State hash: {state_hash}")
            if state_hash in visited and visited[state_hash] <= g:
                logging.debug(
                    f"State already visited with lower g: {visited[state_hash]}")
                return None, float('inf')
            visited[state_hash] = g
            f = g + state.h
            if f > f_limit:
                return None, f
            min_f = float('inf')
            possible_moves = get_possible_moves(state.board)
            for new_board in possible_moves:
                if is_valid_move(state.board, new_board):
                    new_state = PuzzleState(
                        new_board, state.moves + 1, state, "AND-OR Graph Search")
                    if not isinstance(new_state, PuzzleState):
                        logging.error(
                            f"Failed to create valid PuzzleState: {type(new_state)} - {new_state}")
                        continue
                    result, new_f = or_search(
                        new_state, g + 1, f_limit, visited.copy(), start_time, timeout)
                    if result:
                        return result, f_limit
                    min_f = min(min_f, new_f)
            return None, min_f
        except Exception as e:
            logging.error(f"Error in or_search: {str(e)}")
            return None, float('inf')

    start_time = time.time()
    logging.debug(
        f"start_state type: {type(start_state)}, value: {start_state}")
    logging.debug(f"goal_state type: {type(goal_state)}, value: {goal_state}")
    try:
        start = PuzzleState(start_state, strategy="AND-OR Graph Search")
        f_limit = start.h
        visited = {}
        max_iterations = 50

        for iteration in range(max_iterations):
            logging.debug(
                f"Starting iteration {iteration + 1}, f_limit={f_limit}")
            result, new_f = or_search(start, 0, f_limit, visited, start_time)
            if result:
                total_time = time.time() - start_time
                logging.debug(
                    f"AND-OR Graph Search found solution in {result.moves} moves, {total_time:.3f} seconds")
                return result, total_time, []
            if new_f == float('inf'):
                logging.debug(
                    "AND-OR Graph Search: No solution within f_limit")
                break
            f_limit = new_f + 1
            logging.debug(
                f"AND-OR Graph Search: Iteration {iteration + 1}, new f_limit={f_limit}")

        total_time = time.time() - start_time
        logging.debug(
            f"AND-OR Graph Search failed to find solution in {total_time:.3f} seconds")
        return None, total_time, []
    except Exception as e:
        logging.error(f"Error in and_or_graph_search: {str(e)}")
        return None, time.time() - start_time, []


def search_with_partial_observations(start_state, goal_state):
    def get_observable_state(board):
        try:
            i, j = find_blank(board)
        except ValueError:
            return frozenset()
        observable = {}
        observable[(i, j)] = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if is_valid(ni, nj):
                observable[(ni, nj)] = board[ni][nj]
        return frozenset(observable.items())

    def belief_state_heuristic(belief):
        return min(manhattan_distance(state.board, goal_state) for state in belief)

    def apply_action(belief, action):
        new_belief = set()
        for state in belief:
            i, j = find_blank(state.board)
            if (i, j) == (action[0], action[1]):
                new_board = get_new_state(
                    state.board, action[0], action[1], action[2], action[3])
                new_state = PuzzleState(
                    new_board, state.moves + 1, state, "Search with Partial Observations")
                new_belief.add(new_state)
        return new_belief

    def get_possible_actions(board):
        try:
            i, j = find_blank(board)
        except ValueError:
            return []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        actions = []
        for dx, dy in directions:
            new_i, new_j = i + dx, j + dy
            if is_valid(new_i, new_j):
                actions.append((i, j, new_i, new_j))
        return actions

    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None, 0, []

    start_time = time.time()
    initial_state = PuzzleState(
        start_state, strategy="Search with Partial Observations")
    initial_belief = {initial_state}
    queue = []
    heapq.heappush(queue, (0, str(uuid.uuid4()),
                   initial_belief, initial_state))
    visited = set()
    g_scores = {frozenset([initial_state]): 0}
    observable_history = []
    state_history = [initial_state]

    while queue:
        _, _, current_belief, current_state = heapq.heappop(queue)
        belief_key = frozenset(current_belief)
        if belief_key in visited:
            continue
        visited.add(belief_key)

        observable_states = {get_observable_state(
            state.board) for state in current_belief}
        observable_history.append(observable_states)
        state_history.append(current_state)
        logging.debug(
            f"Step {len(observable_history)}: Observable states {observable_states}, Current board {current_state.board}")

        for state in current_belief:
            if state.board == goal_state:
                solution_path = get_solution_path(state)
                while len(observable_history) < len(solution_path):
                    observable_history.append(observable_states)
                    logging.debug(
                        f"Padding observable_history with {observable_states}")
                logging.debug(
                    f"Search with Partial Observations found solution in {state.moves} moves")
                return state, time.time() - start_time, observable_history

        actions = set()
        for state in current_belief:
            actions.update(get_possible_actions(state.board))

        for action in actions:
            new_belief = apply_action(current_belief, action)
            if not new_belief:
                continue
            new_belief_key = frozenset(new_belief)
            if new_belief_key in visited:
                continue
            new_g = min(state.moves for state in new_belief)
            if new_belief_key not in g_scores or new_g < g_scores[new_belief_key]:
                g_scores[new_belief_key] = new_g
                h = belief_state_heuristic(new_belief)
                f = new_g + h
                new_state = next(iter(new_belief))
                heapq.heappush(
                    queue, (f, str(uuid.uuid4()), new_belief, new_state))

    logging.debug("Search with Partial Observations failed to find solution")
    return None, time.time() - start_time, observable_history


def search_with_no_observation(start_state, goal_state):
    def board_to_tuple(board):
        return tuple(num for row in board for num in row)

    def tuple_to_board(board_tuple):
        return [[board_tuple[i * 3 + j] for j in range(3)] for i in range(3)]

    def find_blank_in_tuple(board_tuple):
        try:
            idx = board_tuple.index(0)
            return idx // 3, idx % 3
        except ValueError:
            logging.error("No blank tile found in board tuple")
            raise ValueError("Invalid board tuple: No blank tile (0) found")

    def apply_action_to_board(board_tuple, action):
        i, j = find_blank_in_tuple(board_tuple)
        di, dj = action
        new_i, new_j = i + di, j + dj
        if not (0 <= new_i < 3 and 0 <= new_j < 3):
            return None
        idx = i * 3 + j
        new_idx = new_i * 3 + new_j
        board_list = list(board_tuple)
        board_list[idx], board_list[new_idx] = board_list[new_idx], board_list[idx]
        return tuple(board_list)

    def apply_action(belief, action):
        new_belief = set()
        for board_tuple in belief:
            new_board_tuple = apply_action_to_board(board_tuple, action)
            if new_board_tuple:
                new_belief.add(new_board_tuple)
        return frozenset(new_belief)

    def is_goal(belief):
        return len(belief) == 1 and goal_tuple in belief

    def heuristic(belief):
        return min(manhattan_distance(tuple_to_board(b), goal_state) for b in belief)

    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None, 0, []

    start_time = time.time()
    goal_tuple = board_to_tuple(goal_state)
    initial_belief = {board_to_tuple(start_state)}
    initial_heuristic = manhattan_distance(start_state, goal_state)
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = []
    heapq.heappush(queue, (0, str(uuid.uuid4()),
                   frozenset(initial_belief), []))
    visited = set()
    max_steps = 2000
    max_belief_size = 50
    heuristic_threshold = initial_heuristic * 2

    while queue and max_steps > 0:
        _, _, belief, path = heapq.heappop(queue)
        if belief in visited:
            continue
        visited.add(belief)
        belief_size = len(belief)
        h = heuristic(belief)
        logging.debug(
            f"Search with No Observation: Belief size {belief_size}, heuristic {h}, path length {len(path)}")
        if belief_size > max_belief_size or h > heuristic_threshold:
            logging.debug(
                f"Search with No Observation: Pruning belief, size {belief_size}, heuristic {h}")
            continue
        if is_goal(belief):
            current_board = start_state
            current_state = PuzzleState(
                current_board, 0, None, "Search with No Observation")
            for action in path:
                i, j = find_blank(current_board)
                di, dj = action
                new_i, new_j = i + di, j + dj
                if is_valid(new_i, new_j):
                    new_board = get_new_state(
                        current_board, i, j, new_i, new_j)
                    current_state = PuzzleState(
                        new_board, current_state.moves + 1, current_state, "Search with No Observation")
                    current_board = new_board
            if current_board == goal_state:
                logging.debug(
                    f"Search with No Observation: Solution found in {current_state.moves} moves")
                return current_state, time.time() - start_time, []
            logging.debug(
                "Search with No Observation: Goal belief reached but path invalid")
            return None, time.time() - start_time, []
        for action in ACTIONS:
            new_belief = apply_action(belief, action)
            if new_belief and new_belief not in visited:
                h = heuristic(new_belief)
                f = len(path) + h
                heapq.heappush(queue, (f, str(uuid.uuid4()),
                               new_belief, path + [action]))
        max_steps -= 1
    logging.debug(
        f"Search with No Observation: Failed to find solution in {time.time() - start_time:.2f} seconds")
    return None, time.time() - start_time, []


def q_learning_solve(start_state, goal_state, episodes=5000, max_steps=100):
    start_time = time.time()
    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None, time.time() - start_time, []

    def board_to_tuple(board):
        return tuple(tuple(row) for row in board)

    def get_action_from_move(current_board, next_board):
        try:
            blank_i, blank_j = find_blank(current_board)
            next_blank_i, next_blank_j = find_blank(next_board)
        except ValueError:
            return None
        if next_blank_i == blank_i - 1:
            return "up"
        elif next_blank_i == blank_i + 1:
            return "down"
        elif next_blank_j == blank_j - 1:
            return "left"
        elif next_blank_j == blank_j + 1:
            return "right"
        return None

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    q_table = {}
    actions = ["up", "down", "left", "right"]

    def get_q_value(state, action):
        state_tuple = board_to_tuple(state)
        key = (state_tuple, action)
        if key not in q_table:
            q_table[key] = 0.0
        return q_table[key]

    def choose_action(state, possible_moves, explore=True):
        state_tuple = board_to_tuple(state)
        valid_actions = [get_action_from_move(
            state, move) for move in possible_moves]
        valid_actions = [a for a in valid_actions if a]
        if not valid_actions:
            return None
        if explore and random.random() < epsilon:
            return random.choice(valid_actions)
        q_values = [(get_q_value(state, action), action)
                    for action in valid_actions]
        max_q = max(q_values, key=lambda x: x[0])[0]
        best_actions = [a for q, a in q_values if q == max_q]
        return random.choice(best_actions)

    # Training
    for episode in range(episodes):
        current_board = copy.deepcopy(start_state)
        current_state = PuzzleState(current_board, strategy="Q-Learning")
        prev_h = manhattan_distance(current_board, goal_state)

        for _ in range(max_steps):
            if current_board == goal_state:
                break
            possible_moves = get_possible_moves(current_board)
            action = choose_action(current_board, possible_moves, explore=True)
            if not action:
                break
            blank_i, blank_j = find_blank(current_board)
            new_i, new_j = blank_i, blank_j
            if action == "up":
                new_i -= 1
            elif action == "down":
                new_i += 1
            elif action == "left":
                new_j -= 1
            elif action == "right":
                new_j += 1
            if not is_valid(new_i, new_j):
                break
            new_board = get_new_state(
                current_board, blank_i, blank_j, new_i, new_j)
            new_h = manhattan_distance(new_board, goal_state)
            # Reward: +100 for goal, -1 for increasing heuristic, +1 for decreasing
            reward = 100 if new_board == goal_state else (
                1 if new_h < prev_h else -1)
            next_q_values = [get_q_value(new_board, get_action_from_move(new_board, move))
                             for move in get_possible_moves(new_board) if get_action_from_move(new_board, move)]
            max_next_q = max(next_q_values) if next_q_values else 0
            state_tuple = board_to_tuple(current_board)
            q_table[(state_tuple, action)] = (1 - alpha) * get_q_value(current_board, action) + \
                alpha * (reward + gamma * max_next_q)
            current_state = PuzzleState(
                new_board, current_state.moves + 1, current_state, strategy="Q-Learning")
            current_board = new_board
            prev_h = new_h
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 1000 == 0:
            logging.debug(
                f"Q-Learning: Episode {episode}, epsilon {epsilon:.3f}")

    # Testing
    current_board = copy.deepcopy(start_state)
    current_state = PuzzleState(current_board, strategy="Q-Learning")
    visited = set([get_hash(current_board)])
    steps = 0
    max_steps = 200

    while current_board != goal_state and steps < max_steps:
        possible_moves = get_possible_moves(current_board)
        action = choose_action(current_board, possible_moves, explore=False)
        if not action:
            break
        blank_i, blank_j = find_blank(current_board)
        new_i, new_j = blank_i, blank_j
        if action == "up":
            new_i -= 1
        elif action == "down":
            new_i += 1
        elif action == "left":
            new_j -= 1
        elif action == "right":
            new_j += 1
        if not is_valid(new_i, new_j):
            break
        new_board = get_new_state(
            current_board, blank_i, blank_j, new_i, new_j)
        new_hash = get_hash(new_board)
        if new_hash in visited:
            logging.debug("Q-Learning: Cycle detected during testing")
            break
        visited.add(new_hash)
        current_state = PuzzleState(
            new_board, current_state.moves + 1, current_state, strategy="Q-Learning")
        current_board = new_board
        steps += 1

    if current_board == goal_state:
        logging.debug(
            f"Q-Learning found solution in {current_state.moves} moves")
        return current_state, time.time() - start_time, []
    else:
        logging.debug("Q-Learning failed to find solution")
        return None, time.time() - start_time, []


def local_beam_search_solve(start_state, goal_state, beam_width=4, max_iterations=1000):
    def performance(state, goal):
        return manhattan_distance(state.board, goal)

    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None, time.time(), []

    start_time = time.time()
    best_hypothesis = PuzzleState(start_state, strategy="Local Beam Search")
    candidate_hypotheses = [best_hypothesis]
    best_performance = performance(best_hypothesis, goal_state)
    visited = set([get_hash(start_state)])
    iteration = 0

    while candidate_hypotheses and iteration < max_iterations:
        # Generate all possible successors
        new_candidate_hypotheses = []
        for hypothesis in candidate_hypotheses:
            possible_moves = get_possible_moves(hypothesis.board)
            for new_board in possible_moves:
                new_state_hash = get_hash(new_board)
                if new_state_hash not in visited:
                    new_state = PuzzleState(
                        new_board,
                        hypothesis.moves + 1,
                        hypothesis,
                        strategy="Local Beam Search"
                    )
                    if is_valid_move(hypothesis.board, new_board):
                        new_candidate_hypotheses.append(new_state)
                        visited.add(new_state_hash)

        # Update best hypothesis
        for new_hyp in new_candidate_hypotheses:
            current_perf = performance(new_hyp, goal_state)
            if current_perf < best_performance:
                best_hypothesis = new_hyp
                best_performance = current_perf
                logging.debug(
                    f"Local Beam Search: New best heuristic {best_performance} at iteration {iteration}")

            # Check if goal is reached
            if new_hyp.board == goal_state:
                logging.debug(
                    f"Local Beam Search: Solution found in {new_hyp.moves} moves at iteration {iteration}")
                return new_hyp, time.time() - start_time, []

        # Select top k hypotheses based on performance
        new_candidate_hypotheses.sort(key=lambda x: performance(x, goal_state))
        candidate_hypotheses = new_candidate_hypotheses[:beam_width]

        iteration += 1
        logging.debug(
            f"Local Beam Search: Iteration {iteration}, candidates {len(candidate_hypotheses)}")

    # If no solution found, return the best hypothesis
    logging.debug(
        "Local Beam Search: Did not find solution, attempting BFS reconstruction")
    total_time = time.time() - start_time
    if best_hypothesis.board == goal_state:
        return best_hypothesis, total_time, []

    # Reconstruct a valid path to the goal using BFS
    valid_solution, bfs_time, history = solve_puzzle(
        best_hypothesis.board, goal_state, "BFS")
    total_time += bfs_time
    if valid_solution:
        logging.debug(
            f"Local Beam Search: Reconstructed valid solution with {valid_solution.moves} moves")
        return valid_solution, total_time, []

    logging.debug("Local Beam Search: No valid solution found even with BFS")
    return best_hypothesis, total_time, []


def generate_random_solvable_state(goal_state):
    max_attempts = 100
    for _ in range(max_attempts):
        current_board = copy.deepcopy(goal_state)
        moves = 20
        for _ in range(moves):
            possible_moves = get_possible_moves(current_board)
            if possible_moves:
                current_board = random.choice(possible_moves)
        if is_solvable(current_board, goal_state) and current_board != goal_state:
            return current_board
    logging.error(
        f"Failed to generate solvable state after {max_attempts} attempts")
    raise ValueError(
        f"Không thể tạo trạng thái ngẫu nhiên khả thi sau {max_attempts} lần thử.")


def solve_puzzle(start_state, goal_state, strategy="BFS"):
    start_time = time.time()
    if not is_solvable(start_state, goal_state):
        logging.info("Puzzle is not solvable")
        return None, time.time() - start_time, []

    logging.debug(f"Input start_state for {strategy}: {start_state}")

    if strategy == "DFS":
        stack = [PuzzleState(start_state, strategy="DFS")]
        visited = {}
        states_explored = [0]

        while stack:
            current = stack.pop()
            states_explored[0] += 1

            state_hash = hash(tuple(tuple(row) for row in current.board))
            if state_hash in visited and visited[state_hash] <= current.moves:
                continue

            visited[state_hash] = current.moves

            if current.board == goal_state:
                logging.debug(
                    f"DFS found solution in {current.moves} moves, explored {states_explored[0]} states")
                return current, time.time() - start_time, []

            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy="DFS")
                new_state_hash = hash(tuple(tuple(row) for row in new_board))
                if new_state_hash not in visited or visited[new_state_hash] > new_state.moves:
                    stack.append(new_state)

        logging.debug(
            f"DFS failed to find solution, explored {states_explored[0]} states")
        return None, time.time() - start_time, []

    elif strategy == "BFS":
        queue = deque([PuzzleState(start_state, strategy="BFS")])
        visited = set()
        while queue:
            current = queue.popleft()
            if current.board == goal_state:
                logging.debug(f"BFS found solution in {current.moves} moves")
                return current, time.time() - start_time, []
            if current in visited:
                continue
            visited.add(current)
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy="BFS")
                if new_state not in visited:
                    queue.append(new_state)
        logging.debug("BFS failed to find solution")
        return None, time.time() - start_time, []

    elif strategy == "UCS":
        queue = []
        heapq.heappush(queue, PuzzleState(start_state, strategy="UCS"))
        visited = set()
        while queue:
            current = heapq.heappop(queue)
            if current.board == goal_state:
                logging.debug(f"UCS found solution in {current.moves} moves")
                return current, time.time() - start_time, []
            if current in visited:
                continue
            visited.add(current)
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy="UCS")
                if new_state not in visited:
                    heapq.heappush(queue, new_state)
        logging.debug("UCS failed to find solution")
        return None, time.time() - start_time, []

    elif strategy in ["Greedy", "A*"]:
        queue = []
        heapq.heappush(queue, PuzzleState(start_state, strategy=strategy))
        visited = set()
        while queue:
            current = heapq.heappop(queue)
            if current.board == goal_state:
                logging.debug(
                    f"{strategy} found solution in {current.moves} moves")
                return current, time.time() - start_time, []
            if current in visited:
                continue
            visited.add(current)
            for new_board in get_possible_moves(current.board):
                new_state = PuzzleState(
                    new_board, current.moves + 1, current, strategy)
                if new_state not in visited:
                    heapq.heappush(queue, new_state)
        logging.debug(f"{strategy} failed to find solution")
        return None, time.time() - start_time, []

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

        start = PuzzleState(start_state, strategy="IDA*")
        cost_limit = start.h
        while True:
            visited = set()
            result, new_limit = search(start, cost_limit, visited)
            if result:
                logging.debug(f"IDA* found solution in {result.moves} moves")
                return result, time.time() - start_time, []
            if new_limit == float('inf'):
                logging.debug("IDA* failed to find solution")
                return None, time.time() - start_time, []
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

        start = PuzzleState(start_state, strategy="IDS")
        depth = 0
        while depth < 100:
            visited = set()
            result = dfs_limited(start, depth, visited)
            if result:
                logging.debug(
                    f"IDS found solution in {result.moves} moves at depth {depth}")
                return result, time.time() - start_time, []
            depth += 1
        logging.debug("IDS failed to find solution")
        return None, time.time() - start_time, []

    elif strategy == "Simple Hill Climbing":
        original_start_state = copy.deepcopy(start_state)
        max_restarts = 50
        max_steps = 1000
        global_visited = set()
        best_state = None
        best_heuristic = float('inf')
        initial_state = PuzzleState(original_start_state, strategy=strategy)
        # Start with initial state
        solution_path = [(initial_state.board, initial_state.moves)]

        def perturb_state(state, max_moves=10):
            current_board = copy.deepcopy(state)
            current_state = PuzzleState(current_board, strategy=strategy)
            for _ in range(random.randint(1, max_moves)):
                moves = get_possible_moves(current_state.board)
                if moves:
                    new_board = random.choice(moves)
                    current_state = PuzzleState(
                        new_board, current_state.moves + 1, current_state, strategy=strategy)
            return current_state.board

        for restart in range(max_restarts):
            working_state = copy.deepcopy(
                original_start_state) if restart == 0 else perturb_state(original_start_state)
            logging.debug(
                f"Simple Hill Climbing: Restart {restart + 1}, working_state: {working_state}")
            current = PuzzleState(working_state, strategy=strategy)
            visited = set()
            steps = 0

            while current.board != goal_state and steps < max_steps:
                state_hash = get_hash(current.board)
                if state_hash in visited:
                    logging.debug(
                        f"Simple Hill Climbing: Cycle detected at state {current.board}")
                    break
                visited.add(state_hash)
                global_visited.add(state_hash)
                best_neighbor = None
                best_heuristic_neighbor = current.h
                neighbors = get_possible_moves(current.board)
                if not neighbors:
                    logging.debug(
                        f"Simple Hill Climbing: No valid neighbors for state {current.board}")
                    break
                random.shuffle(neighbors)
                for new_board in neighbors:
                    new_state = PuzzleState(
                        new_board, current.moves + 1, current, strategy)
                    if new_state.h <= best_heuristic_neighbor and get_hash(new_board) not in global_visited:
                        best_heuristic_neighbor = new_state.h
                        best_neighbor = new_state
                        break  # Take first equal or better neighbor
                if best_neighbor:
                    current = best_neighbor
                    solution_path.append((current.board, current.moves))
                    if current.h < best_heuristic:
                        best_state = current
                        best_heuristic = current.h
                else:
                    logging.debug(
                        f"Simple Hill Climbing: No better or equal neighbor for state {current.board}")
                    break
                steps += 1

            if current.board == goal_state:
                logging.debug(
                    f"Simple Hill Climbing: Solution found in {current.moves} moves")
                logging.debug(
                    f"Solution path initial state: {solution_path[0][0]}")
                return current, time.time() - start_time, []

        logging.debug(
            f"Simple Hill Climbing: Best state: {best_state.board if best_state else None}")
        logging.debug(f"Solution path initial state: {solution_path[0][0]}")
        # Reconstruct path from start_state to goal_state using BFS
        valid_solution, bfs_time, history = solve_puzzle(
            original_start_state, goal_state, "BFS")
        total_time = time.time() - start_time + bfs_time
        if valid_solution:
            logging.debug(
                f"Simple Hill Climbing: Reconstructed valid solution with {valid_solution.moves} moves")
            return valid_solution, total_time, []
        logging.debug("Simple Hill Climbing: BFS reconstruction failed")
        return best_state, time.time() - start_time, []

    elif strategy == "Steepest-Hill Climbing":
        original_start_state = copy.deepcopy(start_state)
        max_restarts = 50
        max_steps = 1000
        global_visited = set()
        best_state = None
        best_heuristic = float('inf')
        initial_state = PuzzleState(original_start_state, strategy=strategy)
        # Start with initial state
        solution_path = [(initial_state.board, initial_state.moves)]

        def perturb_state(state, max_moves=10):
            current_board = copy.deepcopy(state)
            current_state = PuzzleState(current_board, strategy=strategy)
            for _ in range(random.randint(1, max_moves)):
                moves = get_possible_moves(current_state.board)
                if moves:
                    new_board = random.choice(moves)
                    current_state = PuzzleState(
                        new_board, current_state.moves + 1, current_state, strategy=strategy)
            return current_state.board

        for restart in range(max_restarts):
            working_state = copy.deepcopy(
                original_start_state) if restart == 0 else perturb_state(original_start_state)
            logging.debug(
                f"Steepest-Hill Climbing: Restart {restart + 1}, working_state: {working_state}")
            current = PuzzleState(working_state, strategy=strategy)
            visited = set()
            steps = 0

            while current.board != goal_state and steps < max_steps:
                state_hash = get_hash(current.board)
                if state_hash in visited:
                    logging.debug(
                        f"Steepest-Hill Climbing: Cycle detected at state {current.board}")
                    break
                visited.add(state_hash)
                global_visited.add(state_hash)
                best_neighbor = None
                best_heuristic_neighbor = float('inf')
                neighbors = get_possible_moves(current.board)
                if not neighbors:
                    logging.debug(
                        f"Steepest-Hill Climbing: No valid neighbors for state {current.board}")
                    break
                random.shuffle(neighbors)
                for new_board in neighbors:
                    new_state = PuzzleState(
                        new_board, current.moves + 1, current, strategy)
                    if new_state.h < best_heuristic_neighbor and get_hash(new_board) not in global_visited:
                        best_heuristic_neighbor = new_state.h
                        best_neighbor = new_state
                if best_neighbor and best_heuristic_neighbor < current.h:
                    current = best_neighbor
                    solution_path.append((current.board, current.moves))
                    if current.h < best_heuristic:
                        best_state = current
                        best_heuristic = current.h
                else:
                    logging.debug(
                        f"Steepest-Hill Climbing: No strictly better neighbor for state {current.board}")
                    break
                steps += 1

            if current.board == goal_state:
                logging.debug(
                    f"Steepest-Hill Climbing: Solution found in {current.moves} moves")
                logging.debug(
                    f"Solution path initial state: {solution_path[0][0]}")
                return current, time.time() - start_time, []

        logging.debug(
            f"Steepest-Hill Climbing: Best state: {best_state.board if best_state else None}")
        logging.debug(f"Solution path initial state: {solution_path[0][0]}")
        # Reconstruct path from start_state to goal_state using BFS
        valid_solution, bfs_time, history = solve_puzzle(
            original_start_state, goal_state, "BFS")
        total_time = time.time() - start_time + bfs_time
        if valid_solution:
            logging.debug(
                f"Steepest-Hill Climbing: Reconstructed valid solution with {valid_solution.moves} moves")
            return valid_solution, total_time, []
        logging.debug("Steepest-Hill Climbing: BFS reconstruction failed")
        return best_state, time.time() - start_time, []

    elif strategy == "Stochastic Hill Climbing":
        original_start_state = copy.deepcopy(start_state)
        max_restarts = 200
        max_steps = 1000

        def perturb_state(state, max_moves=5):
            current_board = copy.deepcopy(state)
            current_state = PuzzleState(current_board, strategy=strategy)
            for _ in range(random.randint(1, max_moves)):
                moves = get_possible_moves(current_state.board)
                if moves:
                    new_board = random.choice(moves)
                    current_state = PuzzleState(
                        new_board, current_state.moves + 1, current_state, strategy=strategy)
            return current_state.board

        for restart in range(max_restarts):
            # Use original start_state for first restart, perturb it for subsequent restarts
            working_state = copy.deepcopy(
                original_start_state) if restart == 0 else perturb_state(original_start_state)
            current = PuzzleState(working_state, strategy=strategy)
            visited = set()

            for step in range(max_steps):
                if current.board == goal_state:
                    logging.debug(
                        f"Stochastic Hill Climbing: Solution found in {current.moves} moves")
                    return current, time.time() - start_time, []
                state_hash = get_hash(current.board)
                visited.add(state_hash)
                next_states = [PuzzleState(new_board, current.moves + 1, current, strategy)
                               for new_board in get_possible_moves(current.board)]
                if not next_states:
                    logging.debug(
                        f"Stochastic Hill Climbing: No moves available at state {current.board}")
                    break
                heuristics = [state.h for state in next_states]
                max_h = max(heuristics) if heuristics else 0
                weights = [max_h - h + 1 for h in heuristics]
                total = sum(weights)
                if total > 0:
                    probabilities = [w / total for w in weights]
                    current = random.choices(
                        next_states, probabilities, k=1)[0]
                else:
                    current = random.choice(next_states)
            logging.debug(f"Stochastic Hill Climbing: Restart {restart + 1}")

        logging.debug("Stochastic Hill Climbing: Failed to find solution")
        return None, time.time() - start_time, []

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
                logging.debug(
                    f"Simulated Annealing found solution in {current.moves} moves")
                return current, time.time() - start_time, []

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
            logging.debug(
                f"Simulated Annealing found solution in {best_state.moves} moves")
            return best_state, time.time() - start_time, []

        if best_state.h < current.h:
            logging.debug("Simulated Annealing returning best state")
            return best_state, time.time() - start_time, []

        logging.debug("Simulated Annealing failed to find solution")
        return current if current.h < float('inf') else None, time.time() - start_time, []

    elif strategy == "Local Beam Search":
        solution, solving_time, observable_history = local_beam_search_solve(
            start_state, goal_state)
        if solution and solution.board == goal_state:
            logging.debug(
                f"Local Beam Search found solution in {solution.moves} moves")
        else:
            logging.debug("Local Beam Search did not find exact solution")
        return solution, solving_time, observable_history

    elif strategy == "Genetic Algorithm":
        solution, solving_time, observable_history = genetic_algorithm_solve(
            start_state, goal_state)
        if solution and solution.board == goal_state:
            logging.debug(
                f"Genetic Algorithm found solution in {solution.moves} moves")
        else:
            logging.debug("Genetic Algorithm did not find exact solution")
        return solution, solving_time, observable_history

    elif strategy == "AND-OR Graph Search":
        solution, solving_time, observable_history = and_or_graph_search(
            start_state, goal_state)
        return solution, solving_time, observable_history

    elif strategy == "Search with No Observation":
        return search_with_no_observation(start_state, goal_state)

    elif strategy == "Search with Partial Observations":
        return search_with_partial_observations(start_state, goal_state)

    elif strategy == "Backtracking":
        solution = backtracking_solve(start_state, goal_state)
        return solution, time.time() - start_time, []

    elif strategy == "AC3":
        solution, solving_time, observable_history = ac3_solve(
            start_state, goal_state)
        return solution, solving_time, observable_history

    elif strategy == "Generate and Test":
        solution, solving_time, observable_history = generate_and_test_solve(
            start_state, goal_state)
        return solution, solving_time, observable_history

    elif strategy == "Q-Learning":
        return q_learning_solve(start_state, goal_state)

    logging.debug(f"Unknown strategy {strategy}")
    return None, time.time() - start_time, []


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
    try:
        return next((i, j) for i in range(3) for j in range(3) if board[i][j] == 0)
    except StopIteration:
        logging.error("No blank tile found in board")
        raise ValueError("Invalid board: No blank tile (0) found")


def is_valid_input(board):
    return sorted(num for row in board for num in row) == list(range(9))


def board_to_string(board):
    return '\n'.join([' '.join([str(num) for num in row]) for row in board])


class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trò Chơi 8-Puzzle")
        self.root.state('zoomed')
        self.root.bind("<Escape>", lambda event: self.root.state('normal'))
        self.solving_time = 0.0
        self.speed_delay = 50
        self.observable_history = []
        self.cumulative_observable = {}
        self.current_state_cells = [[None]*3 for _ in range(3)]
        self.animation_ids = []
        self.current_board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.solution_path = []
        self.step_index = 0
        self.is_solving = False
        self.goal_board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.setup_ui()

    def setup_ui(self):
        self.root.configure(bg="#f0f2f5")
        main_frame = tk.Frame(self.root, bg="#ffffff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        left_frame = tk.Frame(main_frame, bg="#2c3e50")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_frame = tk.Frame(main_frame, bg="#ffffff")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.setup_left(left_frame)
        self.setup_right(right_frame)

    def setup_left(self, parent):
        tk.Label(parent, text="Trò Chơi 8-Puzzle", font=("Helvetica", 20, "bold"),
                 fg="#ffffff", bg="#2c3e50", pady=10).pack(fill=tk.X)
        boards_frame = tk.Frame(parent, bg="#2c3e50")
        boards_frame.pack(pady=10, padx=10)
        start_frame = tk.Frame(boards_frame, bg="#3498db")
        start_frame.pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(start_frame, text="Trạng Thái Ban Đầu", font=("Helvetica", 12, "bold"),
                 bg="#3498db", fg="#ffffff").pack(pady=4)
        self.start_cells = [[None]*3 for _ in range(3)]
        self.start_entries = [[None]*3 for _ in range(3)]
        self.setup_grid(start_frame, self.start_entries,
                        self.start_cells, "#3498db")
        goal_frame = tk.Frame(boards_frame, bg="#2ecc71")
        goal_frame.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(goal_frame, text="Trạng Thái Mục Tiêu", font=("Helvetica", 12, "bold"),
                 bg="#2ecc71", fg="#ffffff").pack(pady=4)
        self.goal_cells = [[None]*3 for _ in range(3)]
        self.goal_entries = [[None]*3 for _ in range(3)]
        self.setup_grid(goal_frame, self.goal_entries,
                        self.goal_cells, "#2ecc71")
        for i in range(3):
            for j in range(3):
                value = self.goal_board[i][j]
                self.goal_entries[i][j].insert(
                    0, str(value) if value != 0 else "")
                if self.goal_cells[i][j]:
                    self.goal_cells[i][j].config(
                        text="" if value == 0 else str(value),
                        bg="#f0f0f0" if value == 0 else "#ffffff",
                        fg="#000000" if value == 0 else "#2c3e50"
                    )
        algorithm_groups = {
            "Tìm kiếm có thông tin": ["Greedy", "A*", "IDA*"],
            "Tìm kiếm không có thông tin": ["BFS", "DFS", "UCS", "IDS"],
            "Tìm kiếm cục bộ": ["Simple Hill Climbing", "Steepest-Hill Climbing",
                                "Stochastic Hill Climbing", "Simulated Annealing",
                                "Local Beam Search", "Genetic Algorithm"],
            "Tìm kiếm trong môi trường phức tạp": ["AND-OR Graph Search", "Search with No Observation",
                                                   "Search with Partial Observations"],
            "Tìm kiếm ràng buộc": ["Backtracking", "AC3", "Generate and Test"],
            "Tìm kiếm học tăng cường": ["Q-Learning"]
        }
        control_frame = tk.Frame(parent, bg="#2c3e50")
        control_frame.pack(fill=tk.BOTH, pady=10, padx=10)
        tk.Label(control_frame, text="Nhóm Thuật Toán:", font=("Helvetica", 12, "bold"),
                 bg="#2c3e50", fg="#ecf0f1").pack(anchor="w", pady=(5, 3))
        algo_frame = tk.Frame(control_frame, bg="#2c3e50")
        algo_frame.pack(fill=tk.X, pady=3)
        algo_frame.grid_columnconfigure(0, weight=1)
        algo_frame.grid_columnconfigure(1, weight=1)
        for idx, (group_name, algorithms) in enumerate(algorithm_groups.items()):
            row = idx // 2
            col = idx % 2
            group_button = tk.Button(algo_frame, text=group_name,
                                     bg="#1abc9c", fg="#ffffff",
                                     font=("Segoe UI", 10, "bold"),
                                     width=16, height=1, bd=1, relief="raised",
                                     command=lambda gn=group_name: self.show_algorithm_menu(gn))
            group_button.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Modern.TCombobox",
                        fieldbackground="#1abc9c",
                        background="#1abc9c",
                        foreground="#ffffff",
                        arrowcolor="#ffffff",
                        font=("Helvetica", 10, "bold"),
                        padding=5,
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
                        font=("Helvetica", 10),
                        borderwidth=0)
        style.configure("Modern.TCombobox*Listbox*Frame",
                        background="#2c3e50")
        self.algorithm_var = tk.StringVar(value="BFS")
        self.algo_submenu = ttk.Combobox(control_frame, textvariable=self.algorithm_var,
                                         values=algorithm_groups["Tìm kiếm không có thông tin"],
                                         style="Modern.TCombobox", state="readonly", width=18)
        self.algo_submenu.pack(fill=tk.X, pady=5)
        self.algo_submenu.bind("<<ComboboxSelected>>",
                               self.on_algorithm_change)
        button_container = tk.Frame(control_frame, bg="#2c3e50")
        button_container.pack(fill=tk.X, pady=6)
        button_container.grid_columnconfigure(0, weight=1)
        button_container.grid_columnconfigure(1, weight=1)
        for i in range(4):
            button_container.grid_rowconfigure(i, weight=1)
        button_configs = [
            ("Ngẫu Nhiên", self.random_initial_state,
             "#ff69b4", 0, 0, tk.NORMAL, "random_button"),
            ("Bắt Đầu", self.start_solving, "#4682b4",
             1, 0, tk.NORMAL, "start_button"),
            ("Giải", self.auto_solve, "#66cdaa",
             2, 0, tk.DISABLED, "solve_button"),
            ("Dừng", self.stop_solving, "#ff6347",
             3, 0, tk.DISABLED, "stop_button"),
            ("Quay Lại", self.step_back, "#ffa07a",
             0, 1, tk.DISABLED, "back_button"),
            ("Tiến Tới", self.step_forward, "#9370db",
             1, 1, tk.DISABLED, "forward_button"),
            ("Đặt Lại", self.reset_puzzle, "#a9a9a9",
             2, 1, tk.DISABLED, "reset_button"),
            ("Tốc Độ", self.select_speed, "#ffd700",
             3, 1, tk.DISABLED, "speed_button")
        ]
        for text, command, bg, row, col, state, attr_name in button_configs:
            button = tk.Button(button_container, text=text, command=command,
                               bg=bg, fg="#ffffff", font=("Segoe UI", 10, "bold"),
                               width=8, height=1, bd=1, relief="raised", state=state)
            button.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            setattr(self, attr_name, button)
        self.speed_var = tk.StringVar(value="Bình Thường (Normal)")
        self.speed_menu = ttk.OptionMenu(button_container, self.speed_var, "Bình Thường (Normal)",
                                         "Nhanh (Fast)", "Bình Thường (Normal)", "Chậm (Slow)", "Bỏ Qua (Skip)",
                                         command=self.set_speed, style="Modern.TCombobox")
        self.speed_menu.grid(row=3, column=1, padx=3, pady=3, sticky="ew")
        self.speed_menu.grid_remove()
        self.bind_input_navigation()

    def setup_right(self, parent):
        tk.Label(parent, text="Trạng Thái Hiện Tại", font=("Helvetica", 20, "bold"),
                 bg="#ffffff", fg="#2c3e50").pack(pady=10)
        board_frame = tk.Frame(parent, bg="#ffffff")
        board_frame.pack(pady=10)
        current_frame = tk.Frame(board_frame, bg="#1abc9c")
        current_frame.pack()
        tk.Label(current_frame, text="Bảng Hiện Tại", font=("Helvetica", 12, "bold"),
                 bg="#1abc9c", fg="#ffffff").pack(pady=4)
        self.setup_grid(current_frame, None,
                        self.current_state_cells, "#1abc9c")
        self.update_board_display()
        info_frame = tk.Frame(parent, bg="#ffffff")
        info_frame.pack(fill=tk.X, pady=10)
        self.time_label = tk.Label(info_frame, text="Thời Gian: 0.000 giây",
                                   font=("Helvetica", 12), bg="#ffffff", fg="#2c3e50")
        self.time_label.pack(anchor="w")
        self.steps_label = tk.Label(info_frame, text="Số Bước: 0",
                                    font=("Helvetica", 12), bg="#ffffff", fg="#2c3e50")
        self.steps_label.pack(anchor="w")
        summary_frame = tk.Frame(parent, bg="#ffffff")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(summary_frame, text="Tóm Tắt Lời Giải", font=("Helvetica", 14, "bold"),
                 bg="#ffffff", fg="#2c3e50").pack(anchor="w")
        self.summary_text = tk.Text(summary_frame, height=15, font=("Helvetica", 10),
                                    bg="#f0f0f0", fg="#2c3e50", bd=1, relief="sunken")
        self.summary_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.summary_text.config(state=tk.DISABLED)

    def on_algorithm_change(self, event=None):
        special_algorithms = {
            "Backtracking", "AC3", "Generate and Test", "AND-OR Graph Search",
            "Search with No Observation", "Search with Partial Observations", "Q-Learning"
        }
        selected_algorithm = self.algorithm_var.get()
        state = tk.DISABLED if selected_algorithm in special_algorithms else tk.NORMAL
        for i in range(3):
            for j in range(3):
                self.start_entries[i][j].config(state=state)
                self.goal_entries[i][j].config(state=state)
        if selected_algorithm in special_algorithms:
            self.random_initial_state()
            for i in range(3):
                for j in range(3):
                    self.goal_entries[i][j].delete(0, tk.END)
                    value = self.goal_board[i][j]
                    self.goal_entries[i][j].insert(
                        0, str(value) if value != 0 else "")
                    if self.goal_cells[i][j]:
                        self.goal_cells[i][j].config(
                            text="" if value == 0 else str(value),
                            bg="#f0f0f0" if value == 0 else "#ffffff",
                            fg="#000000" if value == 0 else "#2c3e50"
                        )

    def generate_belief_states(self):
        belief_states = []
        used_hashes = set()
        max_attempts = 100
        goal_board = self.goal_board
        for _ in range(4):
            for attempt in range(max_attempts):
                state = generate_random_solvable_state(goal_board)
                state_hash = tuple(tuple(row)
                                   for row in state)
                if state_hash not in used_hashes and state != goal_board:
                    belief_states.append(state)
                    used_hashes.add(state_hash)
                    break
            else:
                logging.warning("Could not generate unique solvable state")
                return []
        return belief_states

    def start_solving(self):
        special_algorithms = {
            "Backtracking", "AC3", "Generate and Test", "AND-OR Graph Search",
            "Search with No Observation", "Search with Partial Observations", "Q-Learning"
        }
        algorithm = self.algorithm_var.get()
        if algorithm in special_algorithms:
            belief_states = self.generate_belief_states()
            if not belief_states:
                messagebox.showerror(
                    "Lỗi", "Không thể tạo các trạng thái niềm tin.")
                return
            solutions = []
            for state in belief_states:
                result = solve_puzzle(state, self.goal_board, algorithm)
                if result is None:
                    messagebox.showerror(
                        "Lỗi", f"Không tìm thấy lời giải cho trạng thái niềm tin với {algorithm}.")
                    return
                solution, solving_time, observable_history = result
                solution_path = get_solution_path(solution) if solution else []
                result_text = tk.Text(height=8, font=(
                    "Helvetica", 10), bg="#ffffff", fg="#2c3e50", bd=0)
                solutions.append({
                    "initial_state": state,
                    "solution_path": solution_path,
                    "solving_time": solving_time,
                    "observable_history": observable_history,
                    "result_text": result_text
                })
            BeliefStatesWindow(self.root, solutions,
                               algorithm, self.goal_board)
            self.solution_path = solutions[0]["solution_path"]
            self.solving_time = solutions[0]["solving_time"]
            self.observable_history = solutions[0]["observable_history"]
            self.step_index = 0
            for i in range(3):
                for j in range(3):
                    value = belief_states[0][i][j]
                    self.start_entries[i][j].delete(0, tk.END)
                    self.start_entries[i][j].insert(
                        0, str(value) if value != 0 else "")
                    if self.start_cells[i][j]:
                        self.start_cells[i][j].config(
                            text="" if value == 0 else str(value),
                            bg="#f0f0f0" if value == 0 else "#ffffff",
                            fg="#000000" if value == 0 else "#2c3e50"
                        )
        else:
            try:
                start_board = self.get_board_from_entries(self.start_entries)
                goal_board = self.get_board_from_entries(self.goal_entries)
                if not start_board or not goal_board:
                    messagebox.showerror(
                        "Lỗi", "Trạng thái ban đầu hoặc mục tiêu không hợp lệ.")
                    return
                if not is_solvable(start_board, goal_board):
                    messagebox.showerror(
                        "Lỗi", "Trạng thái ban đầu không khả thi.")
                    return
                result = solve_puzzle(start_board, goal_board, algorithm)
                if result is None:
                    messagebox.showerror(
                        "Lỗi", f"Không tìm thấy lời giải với {algorithm}.")
                    return
                solution, self.solving_time, self.observable_history = result
                self.solution_path = get_solution_path(
                    solution) if solution else []
                self.step_index = 0
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi giải: {str(e)}")
                return
        self.update_board_display()
        self.update_solution_board()
        self.update_time_display()
        self.draw_solution_info()
        self.draw_solution_summary()
        self.update_button_states()
        if not self.solution_path:
            messagebox.showinfo(
                "Kết Quả", f"Không tìm thấy lời giải trong {self.solving_time:.3f} giây.")

    def auto_solve(self):
        if not self.solution_path:
            messagebox.showinfo("Thông Báo", "Không có lời giải để chạy.")
            return
        self.is_solving = True
        self.solve_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.back_button.config(state=tk.DISABLED)
        self.forward_button.config(state=tk.DISABLED)
        self.auto_solve_step()

    def auto_solve_step(self):
        if not self.is_solving or self.step_index >= len(self.solution_path) - 1:
            self.stop_solving()
            return
        self.step_index += 1
        self.cancel_animations()
        self.update_board_display()
        self.draw_solution_summary()
        self.root.update()
        self.root.after(self.speed_delay, self.auto_solve_step)

    def stop_solving(self):
        self.is_solving = False
        self.cancel_animations()
        self.update_button_states()

    def step_forward(self):
        if self.step_index < len(self.solution_path) - 1:
            self.step_index += 1
            self.cancel_animations()
            self.update_board_display()
            self.draw_solution_summary()
            self.update_button_states()

    def step_back(self):
        if self.step_index > 0:
            self.step_index -= 1
            self.cancel_animations()
            self.update_board_display()
            self.draw_solution_summary()
            self.update_button_states()

    def cancel_animations(self):
        for anim_id in self.animation_ids:
            self.root.after_cancel(anim_id)
        self.animation_ids = []

    def update_button_states(self):
        self.solve_button.config(
            state=tk.NORMAL if self.solution_path and not self.is_solving else tk.DISABLED)
        self.stop_button.config(
            state=tk.NORMAL if self.is_solving else tk.DISABLED)
        self.back_button.config(
            state=tk.NORMAL if self.step_index > 0 else tk.DISABLED)
        self.forward_button.config(state=tk.NORMAL if self.step_index < len(
            self.solution_path) - 1 else tk.DISABLED)
        self.reset_button.config(
            state=tk.NORMAL if self.solution_path else tk.DISABLED)
        self.speed_button.config(
            state=tk.NORMAL if self.solution_path else tk.DISABLED)

    def setup_grid(self, parent, entries, cells=None, bg_color="#ffffff"):
        grid = tk.Frame(parent, bg=bg_color)
        grid.pack(pady=5, padx=5)
        for i in range(3):
            for j in range(3):
                frame = tk.Frame(grid, width=60, height=60, bg="#ffffff",
                                 highlightbackground="#ffffff", highlightthickness=2, bd=3, relief="ridge")
                frame.grid(row=i, column=j, padx=2, pady=2)
                frame.grid_propagate(False)
                if entries is not None:
                    entry = tk.Entry(frame, width=2, font=("Helvetica", 15),
                                     justify="center", bg="#ffffff", fg="#2c3e50", borderwidth=0)
                    entry.place(relx=0.5, rely=0.5, anchor="center")
                    entries[i][j] = entry
                if cells is not None:
                    label = tk.Label(frame, text="", font=("Helvetica", 15),
                                     bg="#ffffff", fg="#2c3e50")
                    label.place(relx=0.5, rely=0.5, anchor="center")
                    cells[i][j] = label

    def random_initial_state(self):
        try:
            random_board = generate_random_solvable_state(self.goal_board)
            for i in range(3):
                for j in range(3):
                    self.start_entries[i][j].delete(0, tk.END)
                    self.start_entries[i][j].insert(
                        0, str(random_board[i][j]) if random_board[i][j] != 0 else "")
                    if self.start_cells[i][j]:
                        self.start_cells[i][j].config(
                            text="" if random_board[i][j] == 0 else str(
                                random_board[i][j]),
                            bg="#f0f0f0" if random_board[i][j] == 0 else "#ffffff",
                            fg="#000000" if random_board[i][j] == 0 else "#2c3e50"
                        )
        except Exception as e:
            messagebox.showerror(
                "Lỗi", f"Lỗi khi tạo trạng thái ngẫu nhiên: {str(e)}")

    def update_board_display(self):
        if not self.solution_path:
            board = self.current_board
            for i in range(3):
                for j in range(3):
                    value = board[i][j]
                    text = "" if value == 0 else str(value)
                    default_bg = "#f0f0f0" if value == 0 else "#ffffff"
                    fg = "#000000" if value == 0 else "#2c3e50"
                    if self.current_state_cells[i][j]:
                        self.current_state_cells[i][j].config(
                            text=text, bg=default_bg, fg=fg)
            return
        board = self.solution_path[self.step_index][0]
        moved_tile = find_moved_tile(
            self.solution_path[self.step_index -
                               1][0] if self.step_index > 0 else board,
            board
        )
        for i in range(3):
            for j in range(3):
                value = board[i][j]
                text = "" if value == 0 else str(value)
                default_bg = "#f0f0f0" if value == 0 else "#ffffff"
                fg = "#000000" if value == 0 else "#2c3e50"
                if self.current_state_cells[i][j]:
                    self.current_state_cells[i][j].config(
                        text=text, bg=default_bg, fg=fg)
                    if moved_tile and value == moved_tile and self.speed_delay > 0:
                        self.current_state_cells[i][j].config(
                            bg="#ff6b81", fg="#ffffff")
                        anim_id = self.root.after(
                            300,
                            lambda l=self.current_state_cells[i][j], bg=default_bg, fg=fg:
                            l.config(
                                bg=bg, fg=fg) if l.winfo_exists() else None
                        )
                        self.animation_ids.append(anim_id)

    def get_board_from_entries(self, entries):
        try:
            board = [[0]*3 for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    val = entries[i][j].get().strip()
                    if val == "":
                        val = "0"
                    board[i][j] = int(val)
            return board
        except ValueError:
            return None

    def draw_solution_summary(self):
        if not hasattr(self, 'summary_text'):
            return
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        if not self.solution_path:
            self.summary_text.insert(tk.END, "Không tìm thấy lời giải.")
        else:
            for i in range(len(self.solution_path)):
                board, moves = self.solution_path[i]
                board_str = '\n'.join(
                    [' '.join([str(num) for num in row]) for row in board])
                if i == 0:
                    step_text = f"Bước 0:\n{board_str}\n"
                else:
                    prev_board = self.solution_path[i-1][0]
                    moved_tile = find_moved_tile(prev_board, board)
                    step_text = f"Bước {i}: Di chuyển ô {moved_tile}\n{board_str}\n"
                tag_name = f"step_{i}"
                self.summary_text.insert(tk.END, step_text, tag_name)
                self.summary_text.tag_configure(
                    tag_name, font=("Helvetica", 10))
                if i == self.step_index:
                    self.summary_text.tag_configure(
                        tag_name, background="#1abc9c", foreground="#ffffff")
        self.summary_text.config(state=tk.DISABLED)

    def update_time_display(self):
        if not hasattr(self, 'time_label'):
            return
        self.time_label.config(text=f"Thời Gian: {self.solving_time:.3f} giây")

    def draw_solution_info(self):
        if not hasattr(self, 'steps_label'):
            return
        steps = len(self.solution_path) - 1 if self.solution_path else 0
        self.steps_label.config(text=f"Số Bước: {steps}")

    def update_solution_board(self):
        self.update_board_display()

    def show_algorithm_menu(self, group_name):
        algorithm_groups = {
            "Tìm kiếm có thông tin": ["Greedy", "A*", "IDA*"],
            "Tìm kiếm không có thông tin": ["BFS", "DFS", "UCS", "IDS"],
            "Tìm kiếm cục bộ": ["Simple Hill Climbing", "Steepest-Hill Climbing",
                                "Stochastic Hill Climbing", "Simulated Annealing",
                                "Local Beam Search", "Genetic Algorithm"],
            "Tìm kiếm trong môi trường phức tạp": ["AND-OR Graph Search", "Search with No Observation",
                                                   "Search with Partial Observations"],
            "Tìm kiếm ràng buộc": ["Backtracking", "AC3", "Generate and Test"],
            "Tìm kiếm học tăng cường": ["Q-Learning"]
        }
        self.algo_submenu['values'] = algorithm_groups.get(group_name, [])
        if algorithm_groups.get(group_name):
            self.algorithm_var.set(algorithm_groups[group_name][0])

    def set_speed(self, value):
        speed_map = {
            "Nhanh (Fast)": 25,
            "Bình Thường (Normal)": 50,
            "Chậm (Slow)": 100,
            "Bỏ Qua (Skip)": 0
        }
        self.speed_delay = speed_map.get(value, 50)

    def bind_input_navigation(self):
        pass

    def select_speed(self):
        self.speed_menu.grid()
        self.speed_button.config(state=tk.DISABLED)

    def reset_puzzle(self):
        self.solution_path = []
        self.step_index = 0
        self.is_solving = False
        self.cancel_animations()
        self.update_board_display()
        self.draw_solution_summary()
        self.update_button_states()


class BeliefStatesWindow:
    def __init__(self, parent, solutions, algorithm, goal_board):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Giải 8-Puzzle với {algorithm} - 4 Niềm Tin")
        self.window.geometry("1200x800")
        self.window.configure(bg="#f0f2f5")
        self.solutions = solutions
        self.algorithm = algorithm
        self.goal_board = goal_board
        self.step_indices = [0] * 4
        self.is_solving = [False] * 4
        self.speed_delay = 50
        self.current_state_cells = [
            [[None]*3 for _ in range(3)] for _ in range(4)]
        self.animation_ids = [[] for _ in range(4)]
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.window, bg="#ffffff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        tk.Label(main_frame, text=f"Giải 8-Puzzle với {self.algorithm}",
                 font=("Helvetica", 20, "bold"), bg="#ffffff", fg="#2c3e50").pack(pady=10)
        boards_frame = tk.Frame(main_frame, bg="#ffffff")
        boards_frame.pack(fill=tk.BOTH, expand=True)
        for idx in range(4):
            belief_frame = tk.Frame(
                boards_frame, bg="#ecf0f1", bd=2, relief=tk.GROOVE)
            belief_frame.grid(row=idx//2, column=idx %
                              2, padx=5, pady=5, sticky="nsew")
            boards_frame.grid_columnconfigure(idx % 2, weight=1)
            boards_frame.grid_rowconfigure(idx//2, weight=1)
            self.setup_belief_frame(belief_frame, idx)

    def setup_belief_frame(self, parent, idx):
        tk.Label(parent, text=f"Niềm Tin {idx+1}", font=("Helvetica", 14, "bold"),
                 bg="#ecf0f1", fg="#2c3e50").pack(pady=5)
        grid_frame = tk.Frame(parent, bg="#3498db")
        grid_frame.pack(pady=5)
        self.current_state_cells[idx] = [[None]*3 for _ in range(3)]
        self.setup_grid(grid_frame, idx)
        info_frame = tk.Frame(parent, bg="#ecf0f1")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        solution = self.solutions[idx]
        tk.Label(info_frame, text=f"Thời Gian: {solution['solving_time']:.3f} giây",
                 font=("Helvetica", 12), bg="#ecf0f1", fg="#2c3e50").pack(anchor="w")
        tk.Label(info_frame, text=f"Số Bước: {len(solution['solution_path'])-1 if solution['solution_path'] else 0}",
                 font=("Helvetica", 12), bg="#ecf0f1", fg="#2c3e50").pack(anchor="w")
        summary_frame = tk.Frame(parent, bg="#ecf0f1")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.solutions[idx]["result_text"] = tk.Text(summary_frame, height=8, font=("Helvetica", 10),
                                                     bg="#ffffff", fg="#2c3e50", bd=0)
        self.solutions[idx]["result_text"].pack(fill=tk.BOTH, expand=True)
        self.update_summary(idx)
        self.solutions[idx]["result_text"].config(state=tk.DISABLED)
        button_frame = tk.Frame(parent, bg="#ecf0f1")
        button_frame.pack(fill=tk.X, pady=5)
        tk.Button(button_frame, text="Giải", command=lambda: self.auto_solve(idx),
                  bg="#66cdaa", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Dừng", command=lambda: self.stop_solving(idx),
                  bg="#ff6347", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Tiến", command=lambda: self.step_forward(idx),
                  bg="#9370db", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Lùi", command=lambda: self.step_back(idx),
                  bg="#ffa07a", fg="#ffffff", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=2)

    def setup_grid(self, parent, belief_idx):
        grid = tk.Frame(parent, bg="#3498db")
        grid.pack(pady=5, padx=5)
        for i in range(3):
            for j in range(3):
                frame = tk.Frame(grid, width=60, height=60, bg="#ffffff",
                                 highlightbackground="#ffffff", highlightthickness=2, bd=3, relief="ridge")
                frame.grid(row=i, column=j, padx=2, pady=2)
                frame.grid_propagate(False)
                label = tk.Label(frame, text="", font=("Helvetica", 15),
                                 bg="#ffffff", fg="#2c3e50")
                label.place(relx=0.5, rely=0.5, anchor="center")
                self.current_state_cells[belief_idx][i][j] = label
        self.update_board_display(belief_idx)

    def update_board_display(self, belief_idx):
        solution = self.solutions[belief_idx]
        if not solution["solution_path"]:
            return
        board = solution["solution_path"][self.step_indices[belief_idx]][0]
        moved_tile = find_moved_tile(
            solution["solution_path"][self.step_indices[belief_idx] -
                                      1][0] if self.step_indices[belief_idx] > 0 else board,
            board
        )
        for i in range(3):
            for j in range(3):
                value = board[i][j]
                text = "" if value == 0 else str(value)
                default_bg = "#f0f0f0" if value == 0 else "#ffffff"
                fg = "#000000" if value == 0 else "#2c3e50"
                self.current_state_cells[belief_idx][i][j].config(
                    text=text, bg=default_bg, fg=fg
                )
                if moved_tile and value == moved_tile and self.speed_delay > 0:
                    self.current_state_cells[belief_idx][i][j].config(
                        bg="#ff6b81", fg="#ffffff")
                    anim_id = self.window.after(
                        300,
                        lambda l=self.current_state_cells[belief_idx][i][j], bg=default_bg, fg=fg:
                        l.config(bg=bg, fg=fg) if l.winfo_exists() else None
                    )
                    self.animation_ids[belief_idx].append(anim_id)

    def update_summary(self, belief_idx):
        result_text = self.solutions[belief_idx]["result_text"]
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        solution = self.solutions[belief_idx]
        if not solution["solution_path"]:
            result_text.insert(tk.END, "Không tìm thấy lời giải.")
        else:
            for i in range(len(solution["solution_path"])):
                board, moves = solution["solution_path"][i]
                board_str = '\n'.join(
                    [' '.join([str(num) for num in row]) for row in board])
                if i == 0:
                    step_text = f"Bước 0:\n{board_str}\n"
                else:
                    prev_board = solution["solution_path"][i-1][0]
                    moved_tile = find_moved_tile(prev_board, board)
                    step_text = f"Bước {i}: Di chuyển ô {moved_tile}\n{board_str}\n"
                tag_name = f"step_{i}"
                result_text.insert(tk.END, step_text, tag_name)
                result_text.tag_configure(tag_name, font=("Helvetica", 10))
                if i == self.step_indices[belief_idx]:
                    result_text.tag_configure(
                        tag_name, background="#1abc9c", foreground="#ffffff")
        result_text.config(state=tk.DISABLED)

    def cancel_animations(self, belief_idx):
        for anim_id in self.animation_ids[belief_idx]:
            self.window.after_cancel(anim_id)
        self.animation_ids[belief_idx] = []

    def step_forward(self, belief_idx):
        if self.step_indices[belief_idx] < len(self.solutions[belief_idx]["solution_path"]) - 1:
            self.step_indices[belief_idx] += 1
            self.cancel_animations(belief_idx)
            self.update_board_display(belief_idx)
            self.update_summary(belief_idx)

    def step_back(self, belief_idx):
        if self.step_indices[belief_idx] > 0:
            self.step_indices[belief_idx] -= 1
            self.cancel_animations(belief_idx)
            self.update_board_display(belief_idx)
            self.update_summary(belief_idx)

    def auto_solve(self, belief_idx):
        if not self.solutions[belief_idx]["solution_path"]:
            messagebox.showinfo(
                "Thông Báo", f"Không có lời giải cho Niềm Tin {belief_idx+1}.")
            return
        self.is_solving[belief_idx] = True
        self.auto_solve_step(belief_idx)

    def auto_solve_step(self, belief_idx):
        if not self.is_solving[belief_idx] or self.step_indices[belief_idx] >= len(self.solutions[belief_idx]["solution_path"]) - 1:
            self.stop_solving(belief_idx)
            return
        self.step_indices[belief_idx] += 1
        self.cancel_animations(belief_idx)
        self.update_board_display(belief_idx)
        self.update_summary(belief_idx)
        self.window.update()
        self.window.after(self.speed_delay,
                          lambda: self.auto_solve_step(belief_idx))

    def stop_solving(self, belief_idx):
        self.is_solving[belief_idx] = False
        self.cancel_animations(belief_idx)


def main():
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
