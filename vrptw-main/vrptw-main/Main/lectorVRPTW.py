class lectorVRPTW:
    def __init__(self, file_path: str):
        self.file_path = file_path

        self.n_vehicles = 0
        self.vehicle_capacity = 0
        self.n_customers = 0

        self.coord = []
        self.demand = []
        self.ready_time = []
        self.due_date = []
        self.service_time = []

        # (cust_no, (x_coord, y_coord), demand, ready_time, due_date, service_time)
        self.customer_data = []

    def leer_fichero(self):
        vehicle_start_reading = False
        customer_start_reading = False

        with open(self.file_path, 'r') as file:
            for line in file:
                if line == '\n':
                    continue
                if line.startswith("NUMBER"):
                    vehicle_start_reading = True
                    continue
                if line.startswith("CUST"):
                    customer_start_reading = True
                    continue
                if vehicle_start_reading:
                    data = line.split()
                    self.n_vehicles = int(data[0])
                    self.vehicle_capacity = int(data[1])
                    vehicle_start_reading = False
                    continue
                if customer_start_reading:
                    data = line.split()
                    cust_no = int(data[0])
                    x_coord = int(data[1])
                    y_coord = int(data[2])
                    self.coord.append((x_coord, y_coord))
                    self.demand.append(int(data[3]))
                    self.ready_time.append(int(data[4]))
                    self.due_date.append(int(data[5]))
                    self.service_time.append(int(data[6]))
                    self.customer_data.append((cust_no, self.coord[cust_no], self.demand[cust_no],
                                               self.ready_time[cust_no], self.due_date[cust_no],
                                               self.service_time[cust_no]))

        self.n_customers = len(self.customer_data) - 1

    def calculate_distance(self, i: int, j: int) -> float:
        return ((self.coord[i][0] - self.coord[j][0]) ** 2 + (self.coord[i][1] - self.coord[j][1]) ** 2) ** 0.5

    # Devuleve: Tiempo en hacer el recorrido y la penalizacion de tiempo por incumplimiento de las ventanas
    def calculate_path_time(self, customers: list[int], regresar=True) -> tuple[float, float]:
        if not customers:
            return 0.0, 0.0
        last_customer = 0
        time = 0
        time_penalty = 0
        aux_customers = customers.copy()
        if aux_customers[len(aux_customers) - 1] != 0 and regresar:
            aux_customers.append(0)
        for customer in aux_customers:
            distance = self.calculate_distance(customer, last_customer)
            if (distance + time) < self.ready_time[customer]:
                time = self.ready_time[customer]
            else:
                time += distance
                if time > (self.due_date[customer]):
                    time_penalty += time - self.due_date[customer]
            time += self.service_time[customer]
            last_customer = customer
        return time, time_penalty

    #Devuelve lo que ocupan los paquetes en el camion
    def calculate_occupied_space(self, customers: list[int]) -> int:
        return sum(self.demand[customer] for customer in customers)

    def cust_cercanos(self, cust_visited: list[int], customers: list[int]) -> list[int]:
        valid_cust = []
        t, _ = self.calculate_path_time(customers, False)
        last_cust = 0
        if customers:
            last_cust = customers[len(customers) - 1]
        for customer in range(self.n_customers):
            if customer != 0 and customer in cust_visited:
                continue
            extend_t = t + self.calculate_distance(last_cust, customer)
            if self.ready_time[customer] <= extend_t < self.due_date[customer]:
                valid_cust.append(customer)

        return valid_cust

    def print_cliente(self, cust_no: int):
        cliente = self.customer_data[cust_no]
        print("CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME")
        print(
            f"{cliente[0]:>7}{cliente[1][0]:>9}{cliente[1][1]:>10}{cliente[2]:>11}{cliente[3]:>13}{cliente[4]:>10}{cliente[5]:>14}\n")

    def print_list_clientes(self, nums: list[int]):
        print("CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME")
        for num in nums:
            cliente = self.customer_data[num]
            print(
                f"{cliente[0]:>7}{cliente[1][0]:>9}{cliente[1][1]:>10}{cliente[2]:>11}{cliente[3]:>13}{cliente[4]:>10}{cliente[5]:>14}")
        print()

    def print_all_clientes(self):
        print("Numero de vehículos:", self.n_vehicles)
        print("Capacidad de vehículos:", self.vehicle_capacity)
        print("Datos de los clientes:")
        print("CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME")
        for data in self.customer_data:
            print(f"{data[0]:>7}{data[1][0]:>9}{data[1][1]:>10}{data[2]:>11}{data[3]:>13}{data[4]:>10}{data[5]:>14}")
        print()

