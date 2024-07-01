import random

import VRPTWACO as VRPTWACO


def main():
    files = []
    seed = random.randint(-20000000, 20000000)

    #seed = -2003468

    files.append("../Datos/C001.txt") #0

    files.append("../Datos/C101.txt") #1
    files.append("../Datos/C102.txt")
    files.append("../Datos/C103.txt")

    files.append("../Datos/R101.txt") #4
    files.append("../Datos/R102.txt")
    files.append("../Datos/R103.txt")

    files.append("../Datos/C1_2_1.txt") #7
    files.append("../Datos/C1_2_2.txt")
    files.append("../Datos/C1_2_3.txt")

    files.append("../Datos/R1_2_1.txt") #10
    files.append("../Datos/R1_2_2.txt")
    files.append("../Datos/R1_2_3.txt")

    files.append("../Datos/R1_2_4.txt") #13
    files.append("../Datos/R1_4_2.txt") #14
    files.append("../Datos/R2_10_10.txt") #15

    aco = VRPTWACO.VRPTW_ACO(filename=files[12], seed=seed, n_ants=25, alpha=1, beta=10, rho=0.9)

    aco.optimize(200)

    aco.best_solution_graph()


    print("Semilla: ", seed)
    aco.calculate_v_value()
    aco.v_value_graph()


if __name__ == '__main__':
    main()
