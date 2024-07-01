import random
import lectorVRPTW as lector
import time
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


class VRPTW_ACO:
    def __init__(self, filename: str, seed, n_ants: int = 10, alpha: float = 1, beta: float = 5, rho: float = 0.8):
        self.datos = lector.lectorVRPTW(filename)
        self.customer_data = None
        random.seed(seed)

        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.pheromones = None  # Feromonas
        self.archive = []  # Soluciones del pareto, fitness

        self.pheromone_history = []
        self.trails_history = []
        self.archive_history = []

        self.v_value = []

        # Pesos para las funciones de fitness
        self.vehiculos_weight = 1
        self.time_weight = 1
        self.time_penalty_weight = 1
        self.space_penalty_weight = 1

    def initialize(self):
        self.datos.leer_fichero()
        self.customer_data = np.column_stack(
            (np.array(self.datos.coord), np.array(self.datos.demand), np.array(self.datos.ready_time),
             np.array(self.datos.due_date))
        )
        self.pheromones = np.ones(self.customer_data.shape)

    def optimize(self, max_evaluations: int = 1000):
        t_ini = time.time()
        self.initialize()
        n_evaluations = 0
        while n_evaluations < max_evaluations:
            print(f'Evaluacion {n_evaluations}')
            trails = []
            for _ in range(self.n_ants):
                n_evaluations += 1
                solution = self._construct_solution()
                fitness_vehiculos, fitness_totaltime = self._evaluate(solution)
                self._update_archive(solution, fitness_vehiculos, fitness_totaltime)
                print(fitness_vehiculos, fitness_totaltime)
                trails.append((solution, fitness_vehiculos, fitness_totaltime))
            self._update_pheromones(trails)
        t_final = time.time()
        print(f'Tiempo final: {t_final - t_ini}')

        self.routes_graph()  # Llama a la función para graficar las rutas
        self.pareto_graph()  # Llama a la función para graficar la frontera de Pareto

    def _construct_solution(self) -> list[list[int]]:
        solution = [[0]]  # Empieza en el depósito
        partial_solution = [0]
        current_capacity = 0

        while True:
            if partial_solution[-1] == 0 and len(partial_solution) > 1:
                solution.append([0])
                partial_solution = [0]
                current_capacity = 0
            candidates = self._get_candidates(solution)
            if len(candidates) == 0:
                break
            probabilities = self._calculate_probabilities(candidates, partial_solution)
            try:
                chosen = int(np.random.choice(candidates, p=probabilities))
                demand = self.datos.demand[chosen]
                if current_capacity + demand <= self.datos.vehicle_capacity:
                    solution[-1].append(chosen)
                    partial_solution.append(chosen)
                    current_capacity += demand
                else:
                    solution[-1].append(0)  # Regresa al depósito si se excede la capacidad
                    solution.append([0])
                    partial_solution = [0]
                    current_capacity = 0
            except ValueError:
                print(f"Error en probabilidades: {probabilities}")
                exit(12)

        for route in solution:
            if route[-1] != 0:
                route.append(0)  # Asegura que cada ruta regresa al depósito

        restructured_sol = [route for route in solution if len(route) > 2]  # Elimina rutas vacías o redundantes
        print(f"Solución construida: {restructured_sol}")
        return restructured_sol

    def _get_candidates(self, solution: list[list[int]]) -> np.ndarray:
        visited = set([cust for subroute in solution for cust in subroute if cust != 0])
        all_customers = set(range(1, self.customer_data.shape[0]))
        candidates = list(all_customers - visited)
        return np.array(candidates)

    def _calculate_probabilities(self, candidates: np.array, solution: list[int]) -> np.ndarray:
        weights = np.array(np.random.dirichlet(np.ones(self.customer_data.shape[1]), size=1))
        pheromones = np.sum(weights[:] * self.pheromones[candidates, :], axis=1) ** self.alpha
        heuristics = self._heuristic(candidates, solution) ** self.beta
        total = np.sum(pheromones * heuristics)

        if total == 0:
            probabilities = np.ones(len(candidates)) / len(candidates)
        else:
            probabilities = (pheromones * heuristics) / total

        return np.array(probabilities)

    def _heuristic(self, candidates: np.array, path: list[int]) -> np.ndarray:
        heuristics = []
        t, _ = self.datos.calculate_path_time(path, False)
        for candidate in candidates:
            h = d = self.datos.calculate_distance(path[-1], candidate)
            ts = t + d
            if ts > self.datos.due_date[candidate]:
                h += ((ts - self.datos.due_date[candidate]) + 2) ** 10
            elif ts < self.datos.ready_time[candidate]:
                h += self.datos.ready_time[candidate] - ts
            c = self.datos.calculate_occupied_space(path + [candidate])
            if c > self.datos.vehicle_capacity:
                h += ((c - self.datos.vehicle_capacity) * 10) ** 10
            h = 1 / h
            heuristics.append(h * 1024)
        return np.array(heuristics)

    def _update_pheromones(self, trails: list[tuple[list[list[int]], float, float]]):
        evaporation = 1 - self.rho
        self.pheromones *= evaporation
        for k in range(2):  # Solo tenemos dos valores de fitness
            if k == 0:
                trails.sort(key=lambda x: x[1])
            else:
                trails.sort(key=lambda x: x[2])
            bs, _, _ = trails[0]
            sbs, _, _ = trails[1]
            b_sol = self._flatten_solution(bs)
            sb_sol = self._flatten_solution(sbs)
            for i, cust in enumerate(b_sol):
                self.pheromones[cust, k] += 10 - np.clip((10 / len(b_sol) * i), None, 10)
            for i, cust in enumerate(sb_sol):
                self.pheromones[cust, k] += 5 - np.clip((5 / len(sb_sol) * i), None, 5)
            self.pheromones[0, :] = 1 / self.datos.n_customers

    def _flatten_solution(self, solution: list[list[int]]) -> np.ndarray:
        flat_solution = [0]
        for elem in solution:
            flat_solution.extend(elem)
        return np.array(flat_solution)

    def _evaluate(self, solution: list[list[int]]) -> tuple:
        vehiculos_usados = len(solution)
        total_time = 0
        time_penalty = 0
        space_penalty = 0

        vehiculos_usados = 0
        total_time, time_penalty = 0, 0
        space_penalty = 0

        vehicle_capacity = self.datos.vehicle_capacity

        for vehicle in solution:
            if not vehicle:
                continue
            vehiculos_usados += 1
            a, b = self.datos.calculate_path_time(vehicle)
            total_time += a
            time_penalty += b
            volumen_rest = vehicle_capacity - self.datos.calculate_occupied_space(vehicle)
            if volumen_rest < 0:
                space_penalty += abs(volumen_rest)

        fitness_vehiculos = (vehiculos_usados * self.vehiculos_weight * 2 +
                             self.time_weight / 2 * total_time +
                             self.time_penalty_weight * time_penalty +
                             self.space_penalty_weight * space_penalty)

        fitness_totaltime = (vehiculos_usados * self.vehiculos_weight / 2 +
                             self.time_weight * 2 * total_time +
                             self.time_penalty_weight * time_penalty +
                             self.space_penalty_weight * space_penalty)

        return fitness_vehiculos, fitness_totaltime

    def _update_archive(self, solution: list[list[int]], fitness_vehiculos: float, fitness_totaltime: float):
        was_added = False
        if not self.archive:
            self.archive.append((solution, fitness_vehiculos, fitness_totaltime))
            print("Añadida primera solución al archivo.")
            return

        # Filtrar soluciones dominadas y mantener las no dominadas
        new_archive = []
        for existing_solution, existing_fit_veh, existing_fit_time in self.archive:
            # Si la nueva solución no es dominada por la existente
            if not (existing_fit_veh <= fitness_vehiculos and existing_fit_time <= fitness_totaltime):
                new_archive.append((existing_solution, existing_fit_veh, existing_fit_time))
            # Si la nueva solución domina la existente
            if fitness_vehiculos <= existing_fit_veh and fitness_totaltime <= existing_fit_time:
                was_added = True

        # Añadir la nueva solución al archivo si no fue dominada por ninguna otra
        if not was_added:
            new_archive.append((solution, fitness_vehiculos, fitness_totaltime))
            print("Añadida nueva solución al archivo.")

        self.archive = new_archive
        print(f"Total soluciones en el archivo: {len(self.archive)}")

    def calculate_v_value(self):
        if not self.archive:
            print("No hay soluciones en el archivo para calcular el valor V.")
            return
        # Asegurarse de que hay suficientes valores para formar un polígono
        self.v_value = [(fitness_vehiculos, fitness_totaltime) for _, fitness_vehiculos, fitness_totaltime in
                        self.archive if fitness_vehiculos > 0 and fitness_totaltime > 0]
        if len(self.v_value) < 4:
            print("No hay suficientes valores V para formar un polígono.")
        else:
            print(f"Valor V calculado: {self.v_value}")

    def v_value_graph(self):
        if not self.v_value:
            print("No hay valores V calculados para graficar.")
            return

        if len(self.v_value) < 4:
            print("No hay suficientes valores V para formar un polígono.")
            return

        x_coords = [coord[0] for coord in self.v_value]
        y_coords = [coord[1] for coord in self.v_value]

        v_final_value = Polygon(self.v_value)

        plt.figure(figsize=(8, 6))
        color_random = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', color=color_random, label='Área')
        plt.fill(x_coords, y_coords, color=color_random, alpha=0.5)
        plt.xlabel('Fitness de vehículos')
        plt.ylabel('Fitness de tiempo')
        plt.title(f'Área solución: {v_final_value.area}')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    def pareto_graph(self):
        self.archive_history.append(self.archive)
        for i, arch in enumerate(self.archive_history):
            if arch:
                plt.figure(figsize=(8, 6))
                plt.scatter(*zip(*[(fit_veh, fit_time) for _, fit_veh, fit_time in arch]), color='b')
                plt.xlabel('Fitness de vehículos')
                plt.ylabel('Fitness de tiempo')
                plt.title(f'Evolución de la frontera de Pareto - Iteración {i}')
                plt.grid(True)
                plt.show()

    def routes_graph(self):
        for i, (solution, fitness_vehiculos, fitness_totaltime) in enumerate(self.archive):
            plt.figure(figsize=(10, 8))
            for route in solution:
                x = [self.datos.coord[customer][0] for customer in route]
                y = [self.datos.coord[customer][1] for customer in route]
                plt.plot(x, y, marker='o', linestyle='-', label=f'Vehículo {i + 1}')
            plt.xlabel('Coordenada X')
            plt.ylabel('Coordenada Y')
            plt.title(f'Ruta de los vehículos - Fitness total time {fitness_totaltime}')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()

    def best_solution_graph(self):
        if not self.archive:
            print("No hay soluciones en el archivo para graficar.")
            return

        # Graficar mejor solución en términos de fitness de vehículos
        best_veh_solution = min(self.archive, key=lambda x: x[1])
        solution_veh, fitness_vehiculos, _ = best_veh_solution
        plt.figure(figsize=(10, 8))
        for vehicle, route in enumerate(solution_veh):
            x = [self.datos.coord[customer][0] for customer in route]
            y = [self.datos.coord[customer][1] for customer in route]
            plt.plot(x, y, marker='o', linestyle='-', label=f'Vehículo {vehicle + 1}')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.title(f'Mejor solución por fitness de vehículos - Fitness vehículos {fitness_vehiculos}')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # Graficar mejor solución en términos de fitness de tiempo total
        best_time_solution = min(self.archive, key=lambda x: x[2])
        solution_time, _, fitness_totaltime = best_time_solution
        plt.figure(figsize=(10, 8))
        for vehicle, route in enumerate(solution_time):
            x = [self.datos.coord[customer][0] for customer in route]
            y = [self.datos.coord[customer][1] for customer in route]
            plt.plot(x, y, marker='o', linestyle='-', label=f'Vehículo {vehicle + 1}')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.title(f'Mejor solución por fitness de tiempo total - Fitness tiempo total {fitness_totaltime}')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
