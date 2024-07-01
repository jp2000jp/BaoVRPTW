import random

import VRPTWevo as VRPTWevo

def main():
    files = []
    seed = random.randint(-20000000, 20000000)

    #seed = 8397117


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

    evo = VRPTWevo.VRPTWevo(filename=files[2], seed=seed, n_individuos=250, stop_generations=200, tournament_size=5,
                            p_cruce=0.2, mul_p_mut=1, auto_orden=True,
                            p_mut_shuf=0, p_mut_chg=2, p_mut_xchg=3, p_mut_rm_vhc=0.5, p_mut_x_vhc=3,
                            vehicle_w=10000, time_w=0, total_time_w=10, time_penalty_w=2000, space_penalty_w=100)

    evo.evolve()
    print("Semilla: ", seed)
    evo.calculate_v_value()
    evo.v_value_graph()


if __name__ == '__main__':
        main()
