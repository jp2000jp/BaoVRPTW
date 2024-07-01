import random
import time

import lectorVRPTW as lector
from copy import deepcopy

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np


class VRPTWevo:
    individuos = []
    long_cromosoma = 0

    p_mut: float
    p_submut = [0.0] * 5
    total_submut: int

    calidades_medias: list[float]
    calidades_minimas: list[float]

    def __init__(self, filename: str, seed: int, n_individuos=250, stop_generations=200, tournament_size=5,
                 p_cruce=0.7, mul_p_mut=1, auto_orden=False,
                 p_mut_shuf=0.1, p_mut_chg=0.4, p_mut_xchg=0.4, p_mut_rm_vhc=0.1, p_mut_x_vhc=0.1,
                 vehicle_w=3, time_w=3, total_time_w=3, time_penalty_w=3, space_penalty_w=10):
        self.filename: str = filename
        self.datos = lector.lectorVRPTW(filename)

        #Parametros de evolucion
        random.seed(seed)
        self.n_individuos: int = n_individuos
        self.stop_generations: int = stop_generations

        self.tournament_size: int = tournament_size
        self.p_cruce = p_cruce
        self.mul_p_mut = mul_p_mut
        self.auto_orden: bool = auto_orden

        #Probabilidades de submutaciones
        self.p_submut[0] = p_mut_shuf
        self.p_submut[1] = p_mut_chg
        self.p_submut[2] = p_mut_xchg
        self.p_submut[3] = p_mut_rm_vhc
        self.p_submut[4] = p_mut_x_vhc

        #Fitness de todos
        self.fitness = []
        #V_value del pareto
        self.v_value = []

        #Solucion con la mmejor fitness
        self.best_fitness_solution = None

        #Soluciones validas
        self.solutions = []
        self.solutions_scores = []

        #Mejores soluciones validas
        self.bests_solutions = []
        self.bests_solutions_fitness = []
        self.bests_solutions_scores = []

        # Pesos fitness
        self.vehicle_w = vehicle_w
        self.time_w = time_w
        self.total_time_w = total_time_w
        self.time_penalty_w = time_penalty_w
        self.space_penalty_w = space_penalty_w

        #Inicializar
        self.initialize()

    def initialize(self):
        self.datos.leer_fichero()
        self.long_cromosoma = self.datos.n_vehicles
        self.individuos = [[] for _ in range(self.n_individuos)]

        self.p_mut = (self.mul_p_mut / self.long_cromosoma)
        self.total_submut = sum(self.p_submut)

        ant_p = 0
        for i, p in enumerate(self.p_submut):
            ant_p = self.p_submut[i] = p / self.total_submut + ant_p

        self.calidades_minimas = []
        self.calidades_medias = []

        #Poblacion inicial
        up_limit = self.long_cromosoma - 1
        for individuo in self.individuos:
            for i in range(self.long_cromosoma):
                individuo.append([])
            for i in range(1, self.datos.n_customers + 1):
                rnd = random.randint(0, up_limit)
                individuo[rnd].append(i)

            for i in range(len(individuo)):
                self._order_customers(individuo, i)

        self._full_evaluation()

    def evolve(self):
        inicio = time.time()
        generacion = 0
        generation_count = 0
        past_fit = 0
        while generation_count < self.stop_generations:
            generacion += 1
            print(f'Generacion {generacion}')
            self._next_generation()

            #Cruce
            for i in range(0, self.n_individuos, 2):
                if random.random() < self.p_cruce:
                    x_mut1, x_mut2 = self._cruce_PMX(self.individuos[i], self.individuos[i + 1])
                    self.individuos[i], self.individuos[i + 1] = x_mut1.copy(), x_mut2.copy()

            #Mutaciones
            for i in range(self.n_individuos):
                for j in range(len(self.individuos[i])):
                    if random.random() < self.p_mut:
                        prob = random.random()
                        if prob < self.p_submut[0]:
                            mut = self._mutation_shuffle(self.individuos[i], j)
                        elif prob < self.p_submut[1]:
                            mut = self._mutation_cust_change(self.individuos[i], j)
                        elif prob < self.p_submut[2]:
                            mut = self._mutation_cust_exchange(self.individuos[i], j)
                        elif prob < self.p_submut[3]:
                            mut = self._mutation_remove_vehicle(self.individuos[i], j)
                        else:
                            mut = self._mutation_cross_vhcls(self.individuos[i], j)
                        self.individuos[i] = mut.copy()

            self.individuos.append(self.best_fitness_solution)
            #for solution in self.bests_solutions:
                #self.individuos.append(solution)
            self._full_evaluation()

            best_fit = min(self.fitness)
            if best_fit == past_fit:
                generation_count += 1
            else:
                generation_count = 0
                past_fit = best_fit

        self._pareto_front()
        fin = time.time()

        self.print_bests_solutions()
        self.fitness_graph()
        self.solutions_graph()
        self.bests_solutions_graphs()

        print(f'Tiempo de ejecucion: {fin - inicio}')

    def _next_generation(self):
        next_generation = []
        for i in range(self.n_individuos):
            winner = -1
            winner_fit = float('inf')
            for j in range(self.tournament_size):
                candidate = random.randint(0, len(self.individuos) - 1)
                candidate_fit = self.fitness[candidate]
                if candidate_fit < winner_fit:
                    winner = candidate
                    winner_fit = candidate_fit
            next_generation.append(self.individuos[winner])
        self.individuos = deepcopy(next_generation)

    def _cruce_PMX(self, individuo1: list[list[int]], individuo2: list[list[int]]) -> tuple[list, list]:
        p1 = random.randint(0, len(individuo1) - 1)
        p2 = random.randint(0, len(individuo2) - 1)

        if p1 > p2:
            p1, p2 = p2, p1

        x_individuo1 = deepcopy(individuo1)
        x_individuo2 = deepcopy(individuo2)

        x_num1 = []
        x_num2 = []

        for i in range(p1, p2 + 1):
            m = min(len(individuo1[i]), len(individuo2[i]))
            for j in range(m):
                x_individuo1[i][j], x_individuo2[i][j] = x_individuo2[i][j], x_individuo1[i][j]
                x_num1.append(x_individuo1[i][j])
                x_num2.append(x_individuo2[i][j])

        set1 = set(x_num1)
        set2 = set(x_num2)

        interseccion = set1.intersection(set2)

        for elemento in interseccion:
            x_num1.remove(elemento)
            x_num2.remove(elemento)

        rp_num1 = x_num2.copy()
        rp_num2 = x_num1.copy()

        for i in range(self.long_cromosoma):
            if p1 <= i <= p2:
                m = min(len(x_individuo1[i]), len(x_individuo2[i]))
            else:
                m = 0
            for j in range(m, len(x_individuo1[i])):
                if x_individuo1[i][j] in x_num1:
                    x_num1.remove(x_individuo1[i][j])
                    x_individuo1[i][j] = rp_num1.pop(0)
            for j in range(m, len(x_individuo2[i])):
                if x_individuo2[i][j] in x_num2:
                    x_num2.remove(x_individuo2[i][j])
                    x_individuo2[i][j] = rp_num2.pop(0)

        if self.auto_orden:
            for i in range(len(x_individuo1)):
                self._order_customers(x_individuo1, i)
                self._order_customers(x_individuo2, i)

        return x_individuo1, x_individuo2

    def _mutation_shuffle(self, individuo: list[list[int]], vh: int) -> list:
        m_individuo = deepcopy(individuo)
        random.shuffle(m_individuo[vh])
        return m_individuo

    def _mutation_cust_change(self, individuo: list[list[int]], vh: int) -> list:
        m_individuo = deepcopy(individuo)
        if not m_individuo[vh]:
            return m_individuo
        custpos = random.randint(0, len(m_individuo[vh]) - 1)
        cust = m_individuo[vh].pop(custpos)
        vh2 = random.randint(0, len(m_individuo) - 1)
        m_individuo[vh2].append(cust)
        if self.auto_orden:
            self._order_customers(m_individuo, vh2)
        return m_individuo

    def _mutation_cust_exchange(self, individuo: list[list[int]], vh: int) -> list:
        m_individuo = deepcopy(individuo)
        if not m_individuo[vh]:
            return m_individuo
        cust1 = random.randint(0, len(m_individuo[vh]) - 1)
        vh2, cust2 = self._find_cust_pos(individuo, random.randint(1, self.datos.n_customers))
        m_individuo[vh][cust1], m_individuo[vh2][cust2] = m_individuo[vh2][cust2], m_individuo[vh][cust1]
        if self.auto_orden:
            self._order_customers(m_individuo, vh)
            if vh != vh2:
                self._order_customers(m_individuo, vh2)
        return m_individuo

    def _mutation_remove_vehicle(self, individuo: list[list[int]], vh: int) -> list:
        m_individuo = deepcopy(individuo)
        if not m_individuo[vh] or sum(1 for vhcl in individuo if vhcl) <= 1:
            return m_individuo
        rm_vh = m_individuo[vh].copy()
        m_individuo[vh].clear()

        occupied_vhs = []
        for i, vhcl in enumerate(m_individuo):
            if vhcl:
                occupied_vhs.append(i)

        for i, cust in enumerate(rm_vh):
            m_individuo[random.choice(occupied_vhs)].insert(i, cust)

        if self.auto_orden:
            for i in range(len(m_individuo)):
                self._order_customers(m_individuo, i)

        return m_individuo

    def _mutation_cross_vhcls(self, individuo: list[list[int]], vh: int) -> list:
        m_individuo = deepcopy(individuo)
        vh2 = random.randint(0, len(m_individuo) - 1)
        if vh == vh2:
            return m_individuo
        vh_aux = []
        vh_aux.extend(m_individuo[vh])
        vh_aux.extend(m_individuo[vh2])

        m_individuo[vh].clear()
        m_individuo[vh2].clear()

        for cust in vh_aux:
            r = random.randint(0, 1)
            if r == 0:
                m_individuo[vh].append(cust)
            else:
                m_individuo[vh2].append(cust)

        if self.auto_orden:
            self._order_customers(m_individuo, vh)
            self._order_customers(m_individuo, vh2)
        return m_individuo

    def _order_customers(self, individuo: list[list[int]], vh: int):
        ready_times = []
        # Construye una lista de tuplas (cliente, ready_time)
        for i in individuo[vh]:
            ready_times.append((i, self.datos.due_date[i]))

        # Ordena la lista de tuplas según el segundo elemento (ready_time)
        ready_times.sort(key=lambda x: x[1])

        # Actualiza la ruta del vehículo con la lista ordenada de clientes
        individuo[vh] = [customer for customer, _ in ready_times]

    def _find_cust_pos(self, individuo: list[list[int]], cust: int) -> tuple[int, int] | None:
        for i, vh in enumerate(individuo):
            for j, cl in enumerate(vh):
                if cl == cust:
                    return i, j  # Devolver las coordenadas (fila, columna)
        print("No se ha encontrado el cliente: ", cust)
        return None

    def _full_evaluation(self):
        self.fitness.clear()
        solutions = []
        solutions_scores = []
        scores = []
        for individuo in self.individuos:
            fit, score = self._evaluate(individuo)
            self.fitness.append(fit)
            scores.append(score)
            if score[3] == 0 and score[4] == 0:
                self.solutions.append(deepcopy(individuo))
                self.solutions_scores.append(score)

        #Se queda con el individuo de mejor fitness
        fit, i = min((fitness, idx) for idx, fitness in enumerate(self.fitness))
        self.best_fitness_solution = self.individuos[i]

        self.calidades_minimas.append(fit)
        self.calidades_medias.append(sum(self.fitness) / len(self.fitness))

    def _evaluate(self, individuo: list) -> tuple[float, list]:
        heuristic = 0
        datos = []

        #Variables fitness
        vehiculos_usados = 0
        time, total_time, time_penalty = 0, 0, 0
        empty_space, space_penalty = 0, 0

        vehicle_capacity = self.datos.vehicle_capacity

        for gen in individuo:
            if not gen:
                continue
            vehiculos_usados += 1
            a, b = self.datos.calculate_path_time(gen)
            total_time += a
            time_penalty += b
            if a > time:
                time = a

            volumen_rest = vehicle_capacity - self.datos.calculate_occupied_space(gen)
            if volumen_rest < 0:
                space_penalty += abs(volumen_rest)

        datos = [vehiculos_usados, time, total_time, time_penalty, space_penalty]

        fitness = (vehiculos_usados * self.vehicle_w + time * self.time_w + total_time * self.total_time_w
                   + time_penalty * self.time_penalty_w + space_penalty * self.space_penalty_w)

        return fitness, datos

    def _pareto_front(self):
        self.bests_solutions_fitness.clear()
        self.bests_solutions.clear()
        self.bests_solutions_scores.clear()

        coords = [(solution[0], solution[2]) for solution in self.solutions_scores]
        is_front = [True] * len(coords)
        for i, coord in enumerate(coords):
            if is_front[i]:
                for j, aux_coord in enumerate(coords):
                    if i == j:
                        continue
                    if coord[0] <= aux_coord[0] and coord[1] <= aux_coord[1]:
                        is_front[j] = False
                    elif coord[0] >= aux_coord[0] and coord[1] >= aux_coord[1]:
                        is_front[i] = False

        for i, front in enumerate(is_front):
            if front:
                self.bests_solutions.append(self.solutions[i])
                self.bests_solutions_scores.append(self.solutions_scores[i])
                self.bests_solutions_fitness.append(self._evaluate(self.solutions[i])[0])

    def v_value_graph(self):

        x_coords = [coord[0] for coord in self.v_value]
        y_coords = [coord[1] for coord in self.v_value]

        v_final_value = Polygon(self.v_value)

        plt.figure(figsize=(8, 6))
        color_random = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', color=color_random, label='Área')
        plt.fill(x_coords, y_coords, color=color_random, alpha=0.5)
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')

        plt.title(f'Área del polígono: {v_final_value.area}')

        plt.legend()
        plt.grid(True)

        plt.show()
    def calculate_v_value(self):
        total_time: float = 0
        num_vehicles: int = 0
        time_mean: float = 0
        aux_coordinate_x: int = 0
        v_value_intial_lenght: int = 0
        v_value_full_lenght: int = 0
        v_value = []

        for solution_score in self.bests_solutions_scores:
            total_time = solution_score[2]
            num_vehicles = solution_score[0]
            time_mean = total_time/ self.datos.n_vehicles
            v_value.append((num_vehicles, time_mean))
        v_value.sort(reverse=True)

        if v_value:
            aux_coordinate_x = v_value[-1][0]
        else:
            v_value.insert(0, (self.datos.n_vehicles + 1, self.datos.due_date[0] + 0.1))
            aux_coordinate_x = self.datos.n_vehicles
        v_value_intial_lenght = len(v_value)

        for i in range(0, v_value_intial_lenght * 2):
            if i % 2 == 0:
                v_value.insert(i, (0, 0))

        v_value.insert(0, (self.datos.n_vehicles + 1, self.datos.due_date[0] + 0.1))
        v_value.append((aux_coordinate_x, self.datos.due_date[0] + 0.1))

        v_value_full_lenght = len(v_value)
        for i in range(0, v_value_full_lenght - 2):
            if i % 2 != 0:
                v_value[i] = (v_value[i - 1][0], v_value[i + 1][1])
        print("Array de v_valores finales:", v_value)
        coordinates = [(x, y) for x, y in v_value]
        v_final_value = Polygon(coordinates)
        print("Área del polígono:", v_final_value.area)
        self.v_value = v_value

    def print_population(self):
        print("POBLACION")
        print(f"Numero de individuos: {len(self.individuos)}")
        for i, individuo in enumerate(self.individuos):
            print(f"{i + 1}: {individuo}")
        print()



    def print_bests_solutions(self):
        print("Soluciones encontradas: ", len(self.solutions))
        print("---Mejores soluciones---")
        for i, solution in enumerate(self.bests_solutions):
            print(solution)
            print("Vehiculos \tTiempo \t Tiempo total \tPenalizacion de tiempo \tPenalizacion de espacio")
            print(
                f"{self.bests_solutions_scores[i][0]:>9} {self.bests_solutions_scores[i][1]:>8.2f} {self.bests_solutions_scores[i][2]:>14.2f} "
                f"{self.bests_solutions_scores[i][3]:>24.2f} {self.bests_solutions_scores[i][4]:>24}")
            print(f"Fitness: {self.bests_solutions_fitness[i]}")
            print("--- --- ---")


    def fitness_graph(self):
        plt.plot(self.calidades_minimas, label='Mínimas')
        plt.plot(self.calidades_medias, label='Medias')
        plt.xlabel('Generacion')
        plt.ylabel('Calidad')
        plt.title('Fitness')
        plt.grid(True)
        plt.legend()
        plt.show()

    def bests_solutions_graphs(self):
        for solution in self.bests_solutions:
            for i, route in enumerate(solution):
                if not route:
                    continue
                x = []
                y = []
                route_with_origin = route.copy()
                route_with_origin.insert(0, 0)
                route_with_origin.append(0)
                for cust in route_with_origin:
                    a, b = self.datos.coord[cust]
                    x.append(a)
                    y.append(b)
                dx = [x[i + 1] - x[i] for i in range(len(x) - 1)] + [0]
                dy = [y[i + 1] - y[i] for i in range(len(y) - 1)] + [0]
                color = np.random.rand(3, )
                plt.plot(x, y, marker='o', color=color, linestyle='', markersize=3)
                plt.quiver(x, y, dx, dy, color=color, scale_units='xy', angles='xy', scale=1, width=0.003)

            plt.xlabel('Coordenada X')
            plt.ylabel('Coordenada Y')
            plt.title('Recorrido de los vehículos')
            plt.grid(True, alpha=0.7)
            plt.show()

    def solutions_graph(self):
        if not self.solutions_scores:
            return
        x, y = zip(*[(sc[0], sc[2]) for sc in self.solutions_scores])
        bx, by = zip(*[(sc[0], sc[2]) for sc in self.bests_solutions_scores])
        plt.plot(x, y,  marker='o', linestyle='', markersize=3)
        plt.plot(bx, by, marker='o', linestyle='', markersize=6)
        plt.xlabel('Vehiculos')
        plt.ylabel('Tiempo')
        plt.title('Relacion V/T')
        plt.grid(True, alpha=0.7)
        plt.show()


