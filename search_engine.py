import numpy as np
from PIL import Image, ImageDraw
from abc import ABC, abstractmethod
import heapq
import neural_network

# --- TASK 1.1: DISCRETIZACIÓN ---
def discretize_image(image_path, tile_size=10):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    cols, rows = width // tile_size, height // tile_size
    
    grid = np.zeros((rows, cols), dtype=object)
    start_pos = None
    goals = []

    for r in range(rows):
        for c in range(cols):
            box = (c * tile_size, r * tile_size, (c + 1) * tile_size, (r + 1) * tile_size)
            tile = img.crop(box)
            avg_color = np.array(tile).mean(axis=(0, 1))
            
            # Identificación basada en las instrucciones [cite: 23, 24, 25, 26, 41]
            if avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:
                grid[r, c] = 'WALL'
            elif avg_color[0] > 200 and avg_color[1] < 100 and avg_color[2] < 100:
                grid[r, c] = 'START'; start_pos = (r, c)
            elif avg_color[1] > 200 and avg_color[0] < 100 and avg_color[2] < 100:
                grid[r, c] = 'GOAL'; goals.append((r, c))
            else:
                grid[r, c] = 'PATH'
    return grid, start_pos, goals, img

# --- TASK 1.2: FRAMEWORK DE BÚSQUEDA (POO) [cite: 48, 49, 50] ---
class SearchProblem(ABC):
    @abstractmethod
    def get_start_state(self): pass
    @abstractmethod
    def is_goal_state(self, state): pass
    @abstractmethod
    def get_successors(self, state): pass

class MazeProblem(SearchProblem):
    def __init__(self, grid, start, goals, img=None, tile_size=10):
        self.grid = grid
        self.start = start
        self.goals = goals
        self.img = img
        self.tile_size = tile_size

    def get_start_state(self): return self.start
    def is_goal_state(self, state): return state in self.goals

    def get_successors(self, state):
        successors = []
        r, c = state
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]:
                if self.grid[nr, nc] != 'WALL':
                    successors.append((nr, nc))
        return successors

# --- ALGORITMOS DE BÚSQUEDA [cite: 51, 53, 54, 57] ---
def generic_search(problem, frontier):
    start_node = (problem.get_start_state(), [])
    frontier.append(start_node)
    explored = set()

    while frontier:
        state, path = (frontier.pop() if isinstance(frontier, list) else frontier.popleft())
        if problem.is_goal_state(state): return path + [state]
        
        if state not in explored:
            explored.add(state)
            for next_state in problem.get_successors(state):
                frontier.append((next_state, path + [state]))
    return None

def bfs(problem): # BFS usa Queue (FIFO) [cite: 53]
    from collections import deque
    return generic_search(problem, deque())

def dfs(problem): # DFS usa Stack (LIFO) [cite: 54]
    return generic_search(problem, [])

def a_star(problem, heuristic, mlp=None, idx_to_label=None): # A* usa Priority Queue [cite: 56, 57]
    start = problem.get_start_state()
    frontier = [(0 + heuristic(start, problem.goals[0]), 0, start, [])]
    explored = {} # state: cost

    while frontier:
        f, g, state, path = heapq.heappop(frontier)
        if problem.is_goal_state(state): return path + [state]
        
        if state not in explored or g < explored[state]:
            explored[state] = g
            for next_state in problem.get_successors(state):
                step_cost = 1
                if mlp is not None and idx_to_label is not None and problem.img is not None:
                    # Inferencia en vivo
                    ts = problem.tile_size
                    r, c = next_state
                    box = (c * ts, r * ts, (c + 1) * ts, (r + 1) * ts)
                    tile = problem.img.crop(box)
                    avg_color = np.array(tile).mean(axis=(0, 1))
                    
                    color_name = neural_network.predict_color(avg_color, mlp, idx_to_label)
                    step_cost = neural_network.get_cost_for_color(color_name)

                new_g = g + step_cost
                new_f = new_g + heuristic(next_state, problem.goals[0])
                heapq.heappush(frontier, (new_f, new_g, next_state, path + [state]))
    return None

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# --- VISUALIZACIÓN [cite: 55, 59] ---
def draw_solution(grid, path, title):
    print(f"\n--- {title} ---")
    rows, cols = grid.shape
    for r in range(rows):
        line = ""
        for c in range(cols):
            if (r, c) in path: line += "· "
            elif grid[r, c] == 'WALL': line += "■ "
            elif grid[r, c] == 'START': line += "S "
            elif grid[r, c] == 'GOAL': line += "G "
            else: line += "  "
        print(line)

# --- EJECUCIÓN PRINCIPAL ---
filename = 'Prueba Lab1.bmp'
tile_size = 15
grid, start, goals, original_img = discretize_image(filename, tile_size=tile_size)
problem = MazeProblem(grid, start, goals, img=original_img, tile_size=tile_size)

# Load neural network model if available
try:
    mlp, idx_to_label = neural_network.load_inference_model()
    print("Modelo de Red Neuronal cargado exitosamente.")
except Exception as e:
    print("No se pudo cargar la red neuronal (¿ya la entrenaste?). Usando costo 1 por defecto.")
    mlp, idx_to_label = None, None

# Ejecutar y mostrar resultados
path_bfs = bfs(problem)
draw_solution(grid, path_bfs, "RESULTADO BFS")

path_astar = a_star(problem, manhattan_dist, mlp, idx_to_label)
draw_solution(grid, path_astar, "RESULTADO A* (Con Red Neuronal)")

# --- EJECUCIÓN DEMO MAZE ---
import os
if os.path.exists("demo_maze.bmp"):
    print("\n--- Ejecutando Laberinto Demo (Azul vs Gris) ---")
    d_filename = 'demo_maze.bmp'
    d_tile_size = 10
    d_grid, d_start, d_goals, d_original_img = discretize_image(d_filename, tile_size=d_tile_size)
    d_problem = MazeProblem(d_grid, d_start, d_goals, img=d_original_img, tile_size=d_tile_size)
    d_path_astar = a_star(d_problem, manhattan_dist, mlp, idx_to_label)
    draw_solution(d_grid, d_path_astar, "RESULTADO A* (DEMO MAZE)")