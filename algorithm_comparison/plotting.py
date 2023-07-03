import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import trimesh
import math
import os


def norm(v: np.ndarray) -> float:
    assert v.ndim == 1
    # assert v.dtype != np.complex128
    return ((v * v).sum())**(1/2)

def DL_simple():
    figures_path = "C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/scenario_uturn/DL_experiments"
    scenario = "simple"

    for lr in [1e-2,1e-3,1e-4]:

        label = f"learning rate = {lr}"
        iterations = pickle.load(open(
            f"{figures_path}/{scenario}_iterations_lr{lr}.pickle",
            "rb"))


        returns = pickle.load(open(
            f"{figures_path}/{scenario}_returns_lr{lr}.pickle",
            "rb"))


        ALL_REWARDS = pickle.load(open(
            f"{figures_path}/{scenario}_ALL_REWARDS_lr{lr}.pickle",
            "rb"))


        ALL_REWARDS_FOR_ONE_GAME = pickle.load(open(
            f"{figures_path}/{scenario}_ALL_REWARDS_FOR_ONE_GAME_lr{lr}.pickle",
            "rb"))




        num_iterations = 20000
        eval_interval = 1000
        num_eval_episodes = 10



        iterations = range(0, num_iterations + 1, eval_interval)
        plt.figure("eval")
        plt.plot(returns, label=label)
        plt.title(f"Evaluation, number of episodes = {num_eval_episodes}")
        plt.ylabel('Average reward', fontname="Times New Roman", fontsize="14")
        plt.xlabel('Iterations', fontname="Times New Roman", fontsize="14")
        plt.xticks(fontname="Times New Roman", fontsize="14")
        plt.yticks(fontname="Times New Roman", fontsize="14")
        # plt.ylim(top=250)
        plt.legend()
        plt.grid()

        plt.savefig(
            f"{figures_path}/{scenario}_eval.pdf",
            dpi=700, bbox_inches='tight')

        plt.grid()

        plt.figure("training")
        cumulative_average_reward = np.cumsum(ALL_REWARDS) / (np.arange(len(ALL_REWARDS)) + 1)
        plt.plot(cumulative_average_reward, label=label)
        plt.title("Training", fontname="Times New Roman", fontsize="14")
        plt.ylabel('Average reward', fontname="Times New Roman", fontsize="14")
        plt.xlabel('Iterations')
        plt.xticks(fontname="Times New Roman", fontsize="14")
        plt.yticks(fontname="Times New Roman", fontsize="14")
        # plt.ylim(top=250)
        plt.legend()
        plt.grid()
        plt.savefig(
            f"{figures_path}/{scenario}_training.pdf",
            dpi=700, bbox_inches='tight')
        plt.grid()



        plt.figure("training_for_one_game")
        plt.plot(ALL_REWARDS_FOR_ONE_GAME, label=label)
        plt.title("Training")
        plt.ylabel('Average reward', fontname="Times New Roman", fontsize="14")
        plt.xlabel('Iterations', fontname="Times New Roman", fontsize="14")
        plt.xticks(fontname="Times New Roman", fontsize="14")
        plt.yticks(fontname="Times New Roman", fontsize="14")
        # plt.ylim(top=250)
        plt.legend()
        plt.grid()

        plt.savefig(
            f"{figures_path}/{scenario}_training_for_one_game.pdf",
            dpi=700, bbox_inches='tight')

        plt.grid()

def DL_channel():
    SUBDIVISION = 2
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])


    ARMS_NUMBER_CIR = len(beam_directions)
    context_subdevisions = [3]
    number_of_outputs = ARMS_NUMBER_CIR

    context_sets = []
    figures_path = "C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/scenario_uturn/DL"


    rewards_predicted2 = pickle.load(open(
        f"{figures_path}/rewards_predicted_con_num42_arms2.pickle",
        "rb"))
    features2 = pickle.load(open(
        f"{figures_path}/features_con_num42.pickle",
        "rb"))

    ALL_REWARD2 = pickle.load(open(
        f"{figures_path}/rewards_con_num42_arms2.pickle",
        "rb"))


    features_for_prediction = np.unique(features2)  # np.unique(features)
    rewards_for_plotting = rewards_predicted2[:, 0]
    fig_name = f"prediction"
    plt.figure(fig_name)
    plt.plot(features2, ALL_REWARD2[0], ".", label="Reference")
    plt.plot(features_for_prediction, rewards_for_plotting, "*", label="Predicted")
    # plt.title("")
    plt.ylabel('Reward')
    plt.xlabel('State')
    # plt.ylim(top=250)
    plt.grid()
    plt.legend()

    plt.savefig(
        f"{figures_path}/{fig_name}.jpg",
        dpi=700, bbox_inches='tight')

    # exit()

    oracle = pickle.load(open(
        f"{figures_path}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))

    sequential_search_reward = pickle.load(open(
        f"{figures_path}/sequential_search_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))


    for SUBDIVISION_2 in context_subdevisions:
        icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)
        context_sets.append(np.array(icosphere_context.vertices))

    context_type = ["DOA_plus_previous_beam", "DOA", "previous_beam"] #
    lable_name = ["DOA + previous beam", "DOA", "previous beam"]
    seed = 34
    for cont_set in context_sets:
        test_name = f"many_beams_other_network"
        ALL_REWARD = pickle.load(open(
            f"{figures_path}/{test_name}_rewards_con_num{len(cont_set)}_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))

        features = pickle.load(open(
            f"{figures_path}/{test_name}_features_con_num{len(cont_set)}.pickle",
            "rb"))
        cumulative_oracle = np.cumsum(oracle) / (np.arange(len(oracle)) + 1)
        cumulative_sequential_search_reward = np.cumsum(sequential_search_reward) / (np.arange(len(sequential_search_reward)) + 1)

        fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}_cum_cont{len(cont_set)}_algorithm_comparison"
        plt.figure(fig_name)
        plt.plot(cumulative_oracle, label="Oracle")
        plt.plot(cumulative_sequential_search_reward, label="Sequential search")
        for con_type, l in zip(context_type, lable_name):



            rewards_predicted = pickle.load(open(
                f"{figures_path}/{test_name}_{con_type}_rewards_predicted_con_num{len(cont_set)}_arms{int(ARMS_NUMBER_CIR)}.pickle",
                "rb"))



            def choose_beam(rewards_predicted,ALL_REWARD):
                reward = np.zeros(int(np.shape(rewards_predicted)[1]))
                for it_num in range(int(np.shape(rewards_predicted)[1])):
                    index_max = np.argmax(rewards_predicted[:,it_num])
                    reward[it_num] = ALL_REWARD[index_max, it_num]
                return reward

            rewards = choose_beam(rewards_predicted, ALL_REWARD)
            cumulative_reward = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)


            plt.plot(cumulative_reward, label=f"SL, {l}")

        # plt.title("")
        plt.ylabel('Average cumulative reward')
        plt.xlabel('Sample')
        # plt.ylim(top=250)
        plt.grid()
        plt.legend()

        plt.savefig(
            f"{figures_path}/{fig_name}.pdf",
            dpi=700, bbox_inches='tight')

    plt.show()

def plot_seed_test():
    SUBDIVISION = 1
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])


    ARMS_NUMBER_CIR = len(beam_directions)
    context_subdevisions = [3]
    number_of_outputs = ARMS_NUMBER_CIR

    context_sets = []
    figures_path = "C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/scenario_uturn/DL/seed_testing"


    # rewards_predicted2 = pickle.load(open(
    #     f"{figures_path}/rewards_predicted_con_num42_arms2.pickle",
    #     "rb"))
    # features2 = pickle.load(open(
    #     f"{figures_path}/features_con_num42.pickle",
    #     "rb"))
    #
    # ALL_REWARD2 = pickle.load(open(
    #     f"{figures_path}/rewards_con_num42_arms2.pickle",
    #     "rb"))
    #
    #
    # features_for_prediction = np.unique(features2)  # np.unique(features)
    # rewards_for_plotting = rewards_predicted2[:, 0]
    # fig_name = f"prediction"
    # plt.figure(fig_name)
    # plt.plot(features2, ALL_REWARD2[0], ".", label="Reference")
    # plt.plot(features_for_prediction, rewards_for_plotting, "*", label="Predicted")
    # # plt.title("")
    # plt.ylabel('Reward')
    # plt.xlabel('State')
    # # plt.ylim(top=250)
    # plt.grid()
    # plt.legend()
    #
    # plt.savefig(
    #     f"{figures_path}/{fig_name}.jpg",
    #     dpi=700, bbox_inches='tight')
    #
    # exit()

    oracle = pickle.load(open(
        f"{figures_path}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))

    sequential_search_reward = pickle.load(open(
        f"{figures_path}/sequential_search_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))


    for SUBDIVISION_2 in context_subdevisions:
        icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)
        context_sets.append(np.array(icosphere_context.vertices))

    context_type = ["DOA_plus_previous_beam"] #"DOA", "previous_beam"
    lable_name = ["DOA + previous beam", "DOA", "previous beam"]
    cont_set = context_sets[0]


    cumulative_oracle = np.cumsum(oracle) / (np.arange(len(oracle)) + 1)
    cumulative_sequential_search_reward = np.cumsum(sequential_search_reward) / (
                np.arange(len(sequential_search_reward)) + 1)

    test_name = f"many_beams_different_seeds"

    fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}_cum_cont{len(cont_set)}_algorithm_comparison"
    plt.figure(fig_name)
    plt.plot(cumulative_oracle, label="Oracle")
    plt.plot(cumulative_sequential_search_reward, label="Sequential search")


    for s in range(1,18):

        ALL_REWARD = pickle.load(open(
            f"{figures_path}/{test_name}_{s}_rewards_con_num{len(cont_set)}_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))

        features = pickle.load(open(
            f"{figures_path}/{test_name}_{s}_features_con_num{len(cont_set)}.pickle",
            "rb"))
        for con_type, l in zip(context_type, lable_name):



            rewards_predicted = pickle.load(open(
                f"{figures_path}/{test_name}_{s}_{con_type}_rewards_predicted_con_num{len(cont_set)}_arms{int(ARMS_NUMBER_CIR)}.pickle",
                "rb"))



            def choose_beam(rewards_predicted,ALL_REWARD):
                reward = np.zeros(int(np.shape(rewards_predicted)[1]))
                for it_num in range(int(np.shape(rewards_predicted)[1])):
                    index_max = np.argmax(rewards_predicted[:,it_num])
                    reward[it_num] = ALL_REWARD[index_max, it_num]
                return reward

            rewards = choose_beam(rewards_predicted, ALL_REWARD)
            cumulative_reward = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)


            plt.plot(cumulative_reward, label=f"SL, {l}, seed = {s}")

        # plt.title("")
        plt.ylabel('Average cumulative reward')
        plt.xlabel('Sample')
        # plt.ylim(top=250)
        plt.grid()
        plt.legend(prop={'size': 6})

        plt.savefig(
            f"{figures_path}/{fig_name}.pdf",
            dpi=700, bbox_inches='tight')

    plt.show()

def plot_time():
    SUBDIVISION = 3
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])


    ARMS_NUMBER_CIR = len(beam_directions)
    context_subdevisions = [3]
    number_of_outputs = ARMS_NUMBER_CIR

    context_sets = []
    figures_path = "C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/scenario_uturn"



    sequential_search_time = pickle.load(open(
        f"{figures_path}/exp_expl_time_sequential_search_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))
    test_name = f"exp_vs_expl"
    fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}"
    plt.figure(fig_name)
    plt.plot(sequential_search_time, label="Sequential search")
    for eps in [0.05,0.1,0.15]:
        eps_greedy_time = pickle.load(open(
            f"{figures_path}/exp_expl_time_eps_greedy_arms{int(ARMS_NUMBER_CIR)}_eps{eps}.pickle",
            "rb"))
        plt.plot(eps_greedy_time, label=f"Epsilon greedy strategy, $\epsilon$ = {eps}")




    # plt.title("")
    plt.ylabel('Exploration to exploitation time ratio')
    plt.xlabel('Sample')
    # plt.yscale("log")
    plt.xlim(left=ARMS_NUMBER_CIR*2)
    plt.ylim(top=1, bottom = 0)
    plt.grid()
    plt.legend(prop={'size': 9})

    plt.savefig(
        f"{figures_path}/{fig_name}.pdf",
        dpi=700, bbox_inches='tight')

    plt.show()


def plot_location():

    frames_per_data_frame = 1000 #10000
    FRAME_NUMBER = 38
    ITER_NUMBER_CIR = frames_per_data_frame * FRAME_NUMBER
    ITER_NUMBER_RANDOM = ITER_NUMBER_CIR

    SUBDIVISION = 2
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])

    ARMS_NUMBER_CIR = len(beam_directions)
    SUBDIVISION_2 = 1
    icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)



    NUMBER_OF_ITERATIONS_TRAINING = ITER_NUMBER_CIR #250000
    # scenarios = ["uturn", "LOS_moving", "blockage"]
    scenarios = ["uturn"]
    #context_sets = [np.array(icosphere_context.vertices),np.array([[1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0]]), np.array([[1, 1, 0]])]
    #context_sets = [np.array(icosphere_context.vertices)]
    location_grid = []
    context_sets = [location_grid]
    context_types = ["location","DOA","DOA","DOA","DOA"]
    # algorithm_names = ["EPS_greedy",
    #                    "UCB",
    #                    "THS"]

    cont_params = [15, 12, 42, 162, 642]
    cont_param_signs = ["Grid step", "Number of contexts","Number of contexts","Number of contexts","Number of contexts"]

    algorithm_names = ["EPS_greedy"] #"DQL","EPS_greedy"
    algorithm_legend_names = ["$\epsilon$-Greedy"]
    param_signs = ["$\epsilon$"]
    # parameters = [[0.05, 0.1, 0.15],
    #               [10 ** (-7), 10 ** (-7) * 2, 10 ** (-7) / 2],
    #               [0.2, 0.5]]
    parameters = [[0.15]]





    for sc in scenarios:
        number_of_cycles = 1
        folder_name_figures = f"scenario_{sc}"
        figures_path = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/{folder_name_figures}/context_location"

        oracle = pickle.load(open(
            f"{figures_path}/cumulative_avarage_oracle_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))

        sequential_search_reward = pickle.load(open(
            f"{figures_path}/cumulative_avarage_sequential_search_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))

        test_name = f"Context_location_DOA_comp"
        fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}"
        plt.figure(fig_name)
        plt.plot(sequential_search_reward, label="Sequential search")
        plt.plot(oracle, label="Oracle")
        for con_type, cont_param, cont_param_sigh in zip(context_types, cont_params, cont_param_signs):
            try:
                num_ex_conts = pickle.load(open(f"{figures_path}/number_of_contexts_cont_par{cont_param}.pickle", "rb"))
                print(f"Number of existing contexts for cont param {cont_param}: {num_ex_conts}")
            except:
                print(f"Number of existing contexts for cont param {cont_param} is unknown!")
            for alg_name, pars, algorithm_legend_name, param_sign in zip(algorithm_names, parameters, algorithm_legend_names, param_signs):

                for p in pars:
                    average_reward = pickle.load(open(
                        f"{figures_path}/cumulative_average_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}.pickle",
                        "rb"))
                    #plt.plot(average_reward, label=f"{algorithm_legend_name}, {param_sign} = {p}")
                    plt.plot(average_reward, label=f"{con_type}, {cont_param_sigh} = {cont_param}")

        #plt.title(f"Grid step = {cont_params[0]}, Number of contexts = {num_ex_conts}")
        plt.title(f"{param_sign} = {p}")

        plt.ylabel('Cumulative average reward')
        plt.xlabel('Sample')
        # plt.yscale("log")
        # plt.ylim(top=250)
        plt.grid()
        plt.legend(prop={'size': 9})

        plt.savefig(
            f"{figures_path}/{fig_name}.pdf",
            dpi=700, bbox_inches='tight')

        plt.show()

def plot_exploitation_test():
    frames_per_data_frame = 10000 #10000
    FRAME_NUMBER = 38
    ITER_NUMBER_CIR = frames_per_data_frame * FRAME_NUMBER
    ITER_NUMBER_RANDOM = ITER_NUMBER_CIR

    SUBDIVISION = 2
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])

    ARMS_NUMBER_CIR = len(beam_directions)
    SUBDIVISION_2 = 2
    icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)



    NUMBER_OF_ITERATIONS_TRAINING = ITER_NUMBER_CIR #250000
    # scenarios = ["uturn", "LOS_moving", "blockage"]
    scenarios = ["uturn"]
    #context_sets = [np.array(icosphere_context.vertices),np.array([[1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0]]), np.array([[1, 1, 0]])]
    #context_sets = [np.array(icosphere_context.vertices)]
    location_grid = []
    context_sets = [location_grid]
    context_types = ["DOA", "location"]
    # algorithm_names = ["EPS_greedy",
    #                    "UCB",
    #                    "THS"]

    cont_params = [162,  15]
    cont_param_signs = ["Number of contexts","Grid step"]
    folder_test = "exploitation_test"
    algorithm_names = ["EPS_greedy"] #"DQL","EPS_greedy"
    algorithm_legend_names = ["$\epsilon$-Greedy"]
    param_signs = ["$\epsilon$"]
    # parameters = [[0.05, 0.1, 0.15],
    #               [10 ** (-7), 10 ** (-7) * 2, 10 ** (-7) / 2],
    #               [0.2, 0.5]]
    parameters = [[0.15]]

    figures_path_DL = "C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/scenario_uturn/DL"
    test_name_DL = f"many_beams_other_network"
    DL_state = 642
    ALL_REWARD = pickle.load(open(
        f"{figures_path_DL}/{test_name_DL}_rewards_con_num{DL_state}_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))

    for sc in scenarios:

        rewards_predicted = pickle.load(open(
            f"{figures_path_DL}/{test_name_DL}_DOA_rewards_predicted_con_num{DL_state}_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))

        def choose_beam(rewards_predicted, ALL_REWARD):
            reward = np.zeros(int(np.shape(rewards_predicted)[1]))
            for it_num in range(int(np.shape(rewards_predicted)[1])):
                index_max = np.argmax(rewards_predicted[:, it_num])
                reward[it_num] = ALL_REWARD[index_max, it_num]
            return reward

        rewards_DL = choose_beam(rewards_predicted, ALL_REWARD)
        cumulative_reward_DL = np.cumsum(rewards_DL) / (np.arange(len(rewards_DL)) + 1)







        number_of_cycles = 1
        folder_name_figures = f"scenario_{sc}"
        figures_path = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/{folder_name_figures}/{folder_test}"

        oracle = pickle.load(open(
            f"{figures_path}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))

        avarage_oracle = np.cumsum(oracle) / (np.arange(ITER_NUMBER_CIR) + 1)

        sequential_search_reward = pickle.load(open(
            f"{figures_path}/cumulative_avarage_sequential_search_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))




        test_name = f"exploitation_explorion"
        fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}"
        plt.figure(fig_name)
        plt.plot(sequential_search_reward, label="Sequential search")
        plt.plot(avarage_oracle, label="Oracle")

        for con_type, cont_param, cont_param_sigh in zip(context_types, cont_params, cont_param_signs):
            try:
                num_ex_conts = pickle.load(open(f"{figures_path}/number_of_contexts_cont_par{cont_param}.pickle", "rb"))
                print(f"Number of existing contexts for cont param {cont_param}: {num_ex_conts}")
            except:
                print(f"Number of existing contexts for cont param {cont_param} is unknown!")
            for alg_name, pars, algorithm_legend_name, param_sign in zip(algorithm_names, parameters, algorithm_legend_names, param_signs):

                for p in pars:
                    #plt.plot(average_reward, label=f"{algorithm_legend_name}, {param_sign} = {p}")
                    average_reward = pickle.load(open(
                        f"{figures_path}/cumulative_average_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}.pickle",
                        "rb"))

                    plt.plot(np.array( average_reward), label=f"{algorithm_legend_name}, {param_sign} = {p}, {cont_param_sigh} = {cont_param}")

        #plt.title(f"Grid step = {cont_params[0]}, Number of contexts = {num_ex_conts}")
        # plt.title(f"{param_sign} = {p}")

        plt.ylabel('Cumulative average reward')
        plt.xlabel('Sample')
        # plt.yscale("log")
        # plt.ylim(top=250)
        plt.grid()
        plt.legend(prop={'size': 9})

        plt.savefig(
            f"{figures_path}/{fig_name}.pdf",
            dpi=700, bbox_inches='tight')


#==========================================================

        seq_search_exploitation_reward = pickle.load(open(
            f"{figures_path}/seq_search_exploitation_reward_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))

        seq_search_exploitation_it_num = pickle.load(open(
            f"{figures_path}/seq_search_exploitation_it_num_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))

        test_name = f"exploitation"
        fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}"
        plt.figure(fig_name,figsize=(9, 4))
        oracle = np.array(oracle)
        oracle_average = np.cumsum(np.array(oracle)) / (np.arange(len(oracle)) + 1)
        oracle_for_seq = oracle[np.array(seq_search_exploitation_it_num)]
        oracle_for_seq_average = np.cumsum(np.array(oracle_for_seq)) / (np.arange(len(oracle_for_seq)) + 1)
        seq_search_exploitation_reward_average = np.cumsum(seq_search_exploitation_reward) / (np.arange(len(seq_search_exploitation_reward)) + 1)

        plt.plot(np.array(seq_search_exploitation_it_num), seq_search_exploitation_reward_average, label="Sequential search")
        plt.plot(np.linspace(0,len(avarage_oracle)-1,len(avarage_oracle)), oracle_average, label=f"Oracle")

        plt.plot(cumulative_reward_DL, label=f"SL, DOA, Number of states = {DL_state}")


        for con_type, cont_param, cont_param_sigh in zip(context_types, cont_params, cont_param_signs):
            try:
                num_ex_conts = pickle.load(open(f"{figures_path}/number_of_contexts_cont_par{cont_param}.pickle", "rb"))
                print(f"Number of existing contexts for cont param {cont_param}: {num_ex_conts}")
            except:
                print(f"Number of existing contexts for cont param {cont_param} is unknown!")
            for alg_name, pars, algorithm_legend_name, param_sign in zip(algorithm_names, parameters,
                                                                         algorithm_legend_names, param_signs):

                for p in pars:
                    # test_name = f"exploitation_test_bandit_{p}"
                    # fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}"
                    # plt.figure(fig_name)

                    exloitation_iterations = pickle.load(open(
                        f"{figures_path}/exloitation_iterations_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}.pickle",
                        "rb"))

                    reward_exploitation = pickle.load(open(
                        f"{figures_path}/reward_exploitation_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}.pickle",
                        "rb"))


                    reward_exploitation_average = np.cumsum(reward_exploitation) / (np.arange(len(reward_exploitation)) + 1)
                    plt.plot(np.array(exloitation_iterations), reward_exploitation_average, label=f"{algorithm_legend_name}, {con_type}, {cont_param_sigh} = {cont_param}")
                    # oracle_for_bandit = oracle[np.array(exloitation_iterations)]
                    # oracle_for_bandit_average = np.cumsum(oracle_for_bandit) / (np.arange(len(oracle_for_bandit)) + 1)
                    #
                    # plt.plot(oracle_for_bandit_average, label=f"Oracle, {con_type}, {cont_param_sigh} = {cont_param}")


        # plt.title(f"Grid step = {cont_params[0]}, Number of contexts = {num_ex_conts}")
        # plt.title(f"MAB, {param_sign} = {p}")

        plt.ylabel('Cumulative average reward', fontsize = 14)
        plt.xlabel("Sample", fontsize = 14)
        # plt.yscale("log")
        # plt.ylim(top=250)
        plt.grid()
        plt.legend(prop={'size': 12})
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.savefig(
            f"{figures_path}/{fig_name}.pdf",
            dpi=700, bbox_inches='tight')





    plt.show()

def calculate_act_throughput(reward, exp_iterations, BAND_COEF,BANDWIDTH, noize_dB):

    inst_throughput = BANDWIDTH*np.log2(1 + np.array(reward)/(10**(noize_dB/10)))
    inst_throughput_exploration = inst_throughput[exp_iterations]
    actual_throughput = sum(inst_throughput_exploration) *BAND_COEF
    inst_throughput[exp_iterations] = 0
    actual_throughput += sum(inst_throughput)
    return actual_throughput/len(inst_throughput)

def plot_real_protocol():

    def cumulative_window(arr, window_size):
        new_arr = np.zeros(len(arr)-window_size+1)
        n = 0
        for i in range(window_size,len(arr)+1):
            new_arr[n] = sum(arr[i-window_size:i])/window_size
            n += 1
        return new_arr




    PLOT_WINDOW = False
    # a = [1,2,3,4,5,6,7,8,9]
    # aa = cumulative_window(a,3)
    # exit()
    frames_per_data_frame = 10000 #10000
    FRAME_NUMBER = 38
    NUM_CYCLE = 30
    ITER_NUMBER_CIR = frames_per_data_frame * FRAME_NUMBER * NUM_CYCLE
    ITER_NUMBER_RANDOM = ITER_NUMBER_CIR
    SCENARIO_DURATION = 8
    SCENARIO_DURATION = SCENARIO_DURATION * NUM_CYCLE
    duration_of_one_sample = SCENARIO_DURATION / ITER_NUMBER_RANDOM # 20 mcs 2e-5

    # BS_power_dBi = 25
    UE_power_dBi = 5
    BAND_COEF = 0.036
    BANDWIDTH = 100e6
    noize_dB = -120
    SUBDIVISION = 4

    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])

    ARMS_NUMBER_CIR = len(beam_directions)
    # SUBDIVISION_2 = 2
    # icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)


    NUMBER_OF_ITERATIONS_TRAINING = ITER_NUMBER_CIR #250000
    # scenarios = ["uturn", "LOS_moving", "blockage"]
    scenarios = ["LOS"]
    #context_sets = [np.array(icosphere_context.vertices),np.array([[1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0]]), np.array([[1, 1, 0]])]
    #context_sets = [np.array(icosphere_context.vertices)]
    location_grid = []
    context_sets = [location_grid]
    context_types = ["DOA"]
    # algorithm_names = ["EPS_greedy",
    #                    "UCB",
    #                    "THS"]

    cont_params = [162] #[15, 162]
    con_pars = [162] # [0,  162]
    cont_param_signs = ["Number of contexts"] #["Grid step", "Number of contexts"]

    #algorithm_names = ["EPS_greedy", "UCB"] #"DQL","EPS_greedy"
    algorithm_names = ["EPS_greedy"]  # "DQL","EPS_greedy"

    #algorithm_legend_names = ["$\epsilon$-Greedy", "UCB"]
    algorithm_legend_names = ["$\epsilon$-Greedy"]


    #param_signs = ["$\epsilon$","c"]
    param_signs = ["$\epsilon$"]
    # parameters = [[0.05, 0.1, 0.15],
    #               [10 ** (-7), 10 ** (-7) * 2, 10 ** (-7) / 2],
    #               [0.2, 0.5]]
    #parameters = [[0.15, 0.05, 0.4, 0.8, 0.95], [0.01,0.02, 0.2, 0.5]]
    # parameters = [[ 0.8], [0.01]]
    parameters = [[0.2, 0.4, 0.6, 0.7, 0.8, 0.9]]
    NUMBERs_OF_CONS_SSB = np.array([64]) # 4,8,64    4,8,16,32,64
    Numbers_of_frames_between_SSB = np.array([1,2,4,8,16]) #1,2,4,8,16

    window_size = 5000


    ANTENNA_TYPE = 3
    number_of_cycles = 1
    folder_name_figures = "scenario_LOS_28_calib2"
    # figures_path = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/{folder_name_figures}"

    PATH = f"/home/hciutr/project_voxel_engine/voxel_engine/draft_engine/narvi/{folder_name_figures}/output_type{ANTENNA_TYPE}"
    figures_path = f"{PATH}/figures_zoom"
    try:
        os.makedirs(figures_path)
    except:
        print(f"Folder {figures_path} exists!")

    max_reward = pickle.load(open(
        f"/home/hciutr/project_voxel_engine/voxel_engine/draft_engine/narvi/{folder_name_figures}/max_reward_type{ANTENNA_TYPE}_arms{int(ARMS_NUMBER_CIR)}_it{int(ITER_NUMBER_CIR/NUM_CYCLE)}.pickle",
        "rb"))


    test_name = f"real_protocol_SSB_period"

    oracle = pickle.load(open(
        f"{PATH}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))

    oracle = oracle * max_reward * 10**(UE_power_dBi/10)


    r_a_oracle = calculate_act_throughput(oracle, np.array([0]), BAND_COEF, BANDWIDTH,
                                           noize_dB)
    oracle_dB = 10 * np.log10(oracle)
    # avarage_oracle = cumulative_window(oracle, window_size)
    # avarage_oracle_dBm = 10 * np.log10(avarage_oracle / (10 ** (-3)))


    # r_a_or = calculate_act_throughput(oracle, np.array([]), BAND_COEF,BANDWIDTH, noize_dB)




    TX_locations = pickle.load(open(
        f"{PATH}/TX_locations.pickle",
        "rb"))

    RX_locations = pickle.load(open(
        f"{PATH}/RX_locations.pickle",
        "rb"))
    Dist = np.zeros(len(RX_locations))
    n = 0
    for c_RX, c_TX in zip (RX_locations, TX_locations):
        Dist[n] = norm(c_RX-c_TX)
        n +=1
    fig_name4 = f"Distance"
    plt.figure(fig_name4)
    frames = frames_per_data_frame* np.linspace(0,FRAME_NUMBER-1,FRAME_NUMBER)
    plt.plot(frames*duration_of_one_sample, Dist[0:FRAME_NUMBER], "*")
    plt.ylabel('Distance, m',fontsize=14)
    plt.xlabel("Time, sec",fontsize=14)
    # plt.yscale("log")
    #plt.ylim(0,10)
    plt.grid()
    plt.legend(prop={'size': 12})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(
        f"{figures_path}/{fig_name4}.pdf",
        dpi=700, bbox_inches='tight')

    carrier_frequency = 28e9
    file_name = f"/home/hciutr/project_voxel_engine/voxel_engine/draft_engine/narvi/{folder_name_figures}/CIRS/CIR_scene_frame{1}_grid_step{0.1}_voxel_size{0.5}_freq{carrier_frequency}"
    data = pickle.load(open(f"{file_name}.pickle", "rb"))
    directions_of_arrival_RX = data[0]
    directions_of_arrival_RX = np.array(directions_of_arrival_RX)
    directions_of_arrival_RX_for_antenna = - directions_of_arrival_RX
    E = data[2]  # v/m
    E = np.array(E)
    time_array = data[1]
    Power = np.abs(E)**2

    fig_name5 = f"CIR"
    plt.figure(fig_name5)
    its = np.linspace(0,ITER_NUMBER_CIR-1,ITER_NUMBER_CIR)
    plt.plot(Power, "*")
    plt.ylabel('Power, W',fontsize=14)
    plt.xlabel("Time, sec",fontsize=14)
    # plt.yscale("log")
    #plt.ylim(0,10)
    plt.grid()
    plt.legend(prop={'size': 12})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(
        f"{figures_path}/{fig_name5}.pdf",
        dpi=700, bbox_inches='tight')

    print("plotted")
    try:
        os.makedirs(f"{figures_path}/window")
    except:
        print(f"Folder {figures_path}/window exists!")

    BURST_calib = 4
    PERIOD_calib = 16


    start_it = 0 #170000#251750

    leng = ITER_NUMBER_CIR #370000 - start_it#299250 - start_it
    # chosen_beam_number_seq_search = pickle.load(open(
    #     f"{PATH}/chosen_beam_number_seq_search_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{PERIOD_calib}_consSSB{BURST_calib}.pickle",
    #     "rb"))

    search_true = pickle.load(open(
        f"{PATH}/search_true_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{PERIOD_calib}_consSSB{BURST_calib}.pickle",
        "rb"))

    search_false = pickle.load(open(
        f"{PATH}/search_false_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{PERIOD_calib}_consSSB{BURST_calib}.pickle",
        "rb"))


    threshold_all = pickle.load(open(
        f"{PATH}/threshold_all_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{PERIOD_calib}_consSSB{BURST_calib}.pickle",
        "rb"))
    iter_threshold = pickle.load(open(
        f"{PATH}/iter_threshold_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{PERIOD_calib}_consSSB{BURST_calib}.pickle",
        "rb"))



    # fig_name3 = f"chosen_beam_number_seq_search_arms{ARMS_NUMBER_CIR}_dBm_SSBperiod{PERIOD_calib}_consSSB{BURST_calib}"
    # plt.figure(fig_name3)
    # its = np.linspace(0,ITER_NUMBER_CIR-1,ITER_NUMBER_CIR)
    # plt.plot(its[start_it:start_it+ leng] * duration_of_one_sample, chosen_beam_number_seq_search[start_it:start_it+ leng], ".")
    # plt.plot( its[start_it:start_it +leng] * duration_of_one_sample, np.array(search_false[start_it: start_it+ leng])-2, "o", color = "r", label = "Search disactivated")
    # plt.plot(  its[start_it:start_it + leng]  * duration_of_one_sample, np.array(search_true[start_it:start_it+ leng])-2, ".", color = "g", label = "Search activated")
    # plt.ylabel('Beam number',fontsize=14)
    # plt.xlabel("Time, sec",fontsize=14)
    # # plt.yscale("log")
    # plt.ylim(0,ARMS_NUMBER_CIR + 3)
    # plt.grid()
    # plt.legend(prop={'size': 12})
    # plt.yticks(fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.savefig(
    #     f"{figures_path}/{fig_name3}.png",
    #     dpi=700, bbox_inches='tight')

    fig_name3 = f"threshold_seq_search_arms{ARMS_NUMBER_CIR}_dBm"
    plt.figure(fig_name3)
    print(iter_threshold[start_it:start_it+leng])
    plt.plot(np.array(iter_threshold[start_it:start_it+leng]) * duration_of_one_sample, 10* np.log10(np.array(threshold_all[start_it:start_it+ leng]) * max_reward*10**(UE_power_dBi/10)), ".")


    plt.ylabel('Threshold, dB', fontsize=14)
    plt.xlabel("Time, sec", fontsize=14)
    # plt.yscale("log")
    # plt.ylim(0, ARMS_NUMBER_CIR + 3)
    plt.grid()
    plt.legend(prop={'size': 12})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(
        f"{figures_path}/{fig_name3}.png",
        dpi=700, bbox_inches='tight')


    fig_name3 = f"oracle_{test_name}_arms{ARMS_NUMBER_CIR}_dB"
    plt.figure(fig_name3)

    its = np.linspace(0,ITER_NUMBER_CIR-1,ITER_NUMBER_CIR)
    print(f"length, {ITER_NUMBER_CIR}")
    plt.plot(its[start_it:start_it + leng] * duration_of_one_sample, oracle_dB[start_it:start_it + leng])


    plt.ylabel('Power, dB',fontsize=14)
    plt.xlabel("Time, sec",fontsize=14)
    # plt.yscale("log")
    #plt.ylim(0,10)
    plt.grid()
    plt.legend(prop={'size': 12})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(
        f"{figures_path}/{fig_name3}.pdf",
        dpi=700, bbox_inches='tight')

    best_beam = pickle.load(open(
        f"{PATH}/best_beam_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))

    fig_name3 = f"best_beam_{test_name}_arms{ARMS_NUMBER_CIR}_dBm"
    plt.figure(fig_name3)
    its = np.linspace(0, ITER_NUMBER_CIR - 1, ITER_NUMBER_CIR)
    plt.plot(its[start_it:start_it + leng] * duration_of_one_sample, best_beam[start_it:start_it + leng], ".")
    plt.ylabel('Beam number', fontsize=14)
    plt.xlabel("Time, sec", fontsize=14)
    # plt.yscale("log")
    plt.ylim(550,700)
    plt.grid()
    plt.legend(prop={'size': 12})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(
        f"{figures_path}/{fig_name3}.png",
        dpi=700, bbox_inches='tight')

    try:
        os.makedirs(f"{figures_path}/sel_beams")
    except:
        print(f"Folder {figures_path} exists!")

    # for con_type, cont_param in zip(context_types, con_pars):
    #     for a, pars in zip(algorithm_names, parameters):
    #         for p in pars:
    #             for N_f in Numbers_of_frames_between_SSB:
    #                 for n_b in NUMBERs_OF_CONS_SSB:
    #
    #                     chosen_beam_number_bandit = pickle.load(open(
    #                         f"{PATH}/chosen_arm_type{con_type}_context{cont_param}_{a}_{p}_{ARMS_NUMBER_CIR}_SSBperiod{N_f}_consSSB{n_b}.pickle",
    #                         "rb"))
    #
    #                     fig_name3 = f"chosen_arm_type{con_type}_context{cont_param}_{a}_{p}_{ARMS_NUMBER_CIR}_SSBperiod{N_f}_consSSB{n_b}.pickle"
    #                     plt.figure(fig_name3)
    #                     its = np.linspace(0, ITER_NUMBER_CIR - 1, ITER_NUMBER_CIR)
    #                     plt.plot(its[start_it:start_it + leng] * duration_of_one_sample,
    #                              chosen_beam_number_bandit[start_it:start_it + leng], ".")
    #                     plt.ylabel('Beam number', fontsize=14)
    #                     plt.xlabel("Time, sec", fontsize=14)
    #                     # plt.yscale("log")
    #                     # plt.ylim(0,10)
    #                     plt.grid()
    #                     plt.legend(prop={'size': 12})
    #                     plt.yticks(fontsize=12)
    #                     plt.xticks(fontsize=12)
    #                     plt.savefig(
    #                         f"{figures_path}/sel_beams/{fig_name3}.png",
    #                         dpi=700, bbox_inches='tight')


    r_act_seq = np.zeros(
        (len(NUMBERs_OF_CONS_SSB), len(Numbers_of_frames_between_SSB)))
    for n_b_i, n_b in enumerate(NUMBERs_OF_CONS_SSB):
        print(f"n_b {n_b}")
        fig_name = f"sequential_seqrch_{test_name}_arms{ARMS_NUMBER_CIR}_numCons{n_b}"
        plt.figure(fig_name)
        its = np.linspace(0, ITER_NUMBER_CIR - 1, ITER_NUMBER_CIR)
        oracle = np.array(oracle)
        # plt.plot(avarage_oracle, label="Oracle")
        len_or = np.linspace(0, len(oracle) - 1, len(oracle))

        for nfi, N_f in enumerate(Numbers_of_frames_between_SSB):
            print(f"N_f {N_f}")
            # sequential_search_reward = pickle.load(open(
            #     f"{figures_path}/cumulative_avarage_sequential_search_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{SSB_p}_consSSB{NUMBER_OF_CONS_SSB}.pickle",
            #     "rb"))




            #
            # seq_search_exploitation_it_num = pickle.load(open(
            #     f"{PATH}/seq_search_exploitation_it_num_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{N_f}_consSSB{n_b}.pickle",
            #     "rb"))
            sequential_search_reward = pickle.load(open(
                f"{PATH}/seq_search_reward_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{N_f}_consSSB{n_b}.pickle",
                "rb"))

            # oracle_for_seq = oracle[np.array(seq_search_exploitation_it_num)]
            #oracle_for_seq_average = np.cumsum(np.array(oracle_for_seq)) / (np.arange(len(oracle_for_seq)) + 1)
            # if PLOT_WINDOW:
            #     oracle_for_seq_average_cum_win = cumulative_window(oracle_for_seq, window_size)

            if PLOT_WINDOW:
                oracle_for_seq = cumulative_window(oracle, window_size)
            oracle_for_seq_av = np.cumsum(np.array(oracle)) / (np.arange(len(oracle)) + 1)

            sequential_search_reward = sequential_search_reward * max_reward *10** (UE_power_dBi/10)
            #seq_search_exploitation_reward = cumulative_window(seq_search_exploitation_reward, window_size)
            seq_search_exploitation_reward_av = np.cumsum(np.array(sequential_search_reward)) / (np.arange(len(sequential_search_reward)) + 1)

            # plt.plot(np.array(seq_search_exploitation_it_num[
            #                   window_size - 1:len(seq_search_exploitation_it_num)]) * duration_of_one_sample,
            #          seq_search_exploitation_reward, label=f"SSB period = {SSB_p}")
            #
            # # for el_num,el in enumerate(seq_search_exploitation_reward):
            # #     if el==0:
            # #         print(f"it_num, {el_num* duration_of_one_sample}")
            # plt.plot(np.array(seq_search_exploitation_it_num[
            #                   window_size - 1:len(seq_search_exploitation_it_num)]) * duration_of_one_sample,
            #          oracle_for_seq, label=f"oracle, SSB period = {SSB_p}")
            oracle_for_seq_dBm = 10 * np.log10(oracle_for_seq_av / (10 ** (-3)))
            seq_search_exploitation_reward_dBm = 10 * np.log10(seq_search_exploitation_reward_av / (10 ** (-3)))

            exloitation_iterations_seq =  pickle.load(open(
                f"{PATH}/sequential_search_exploitation_itarations_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{N_f}_consSSB{n_b}.pickle",
                "rb"))

            diff_seq_search = oracle_for_seq_dBm - seq_search_exploitation_reward_dBm
            print(f"exloitation_iterations_seq {len(exloitation_iterations_seq)}")
            r_a_seq_ser = calculate_act_throughput(sequential_search_reward, exloitation_iterations_seq, BAND_COEF, BANDWIDTH, noize_dB)
            r_act_seq[n_b_i, nfi] = r_a_seq_ser

            # seq_search_exploitation_reward_average = np.cumsum(seq_search_exploitation_reward) / (
            #             np.arange(len(seq_search_exploitation_reward)) + 1)
            if PLOT_WINDOW:
                seq_search_exploitation_reward_cum_wind = cumulative_window(sequential_search_reward, window_size)
            if PLOT_WINDOW:
                diff_seq_search_window = 10 * np.log10(oracle_for_seq / (10 ** (-3))) - 10 * np.log10(seq_search_exploitation_reward_cum_wind/ (10 ** (-3)))



            #!!diff_seq_search_new = cumulative_window(diff_seq_search, window_size)



            #plt.plot(np.array(seq_search_exploitation_it_num[window_size-1:len(seq_search_exploitation_it_num)])*duration_of_one_sample,diff_seq_search, label=f"SSB period = {SSB_p}")
            plt.plot(len_or * duration_of_one_sample, diff_seq_search, label=f"$N_f$ = {N_f}")

        #plt.plot(oracle_for_seq_average, label=f"Oracle, SSB period = {SSB_p}")


        line_3dB = np.full(len(len_or), 3)
        plt.plot(len_or * duration_of_one_sample, line_3dB,
                 label=f"loss of 3dB", color="r")
        plt.title("Sequential search, $I_{SB}$ = " + f"{n_b}",fontsize=14)
        plt.ylabel('Power loss, dB',fontsize=14)
        plt.xlabel("Time, sec",fontsize=14)
        # plt.yscale("log")
        plt.ylim(0,10)
        plt.grid()
        plt.legend(prop={'size': 12})
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.savefig(
            f"{figures_path}/{fig_name}.pdf",
            dpi=700, bbox_inches='tight')
        if PLOT_WINDOW:
            fig_name2 = f"sequential_search_windows_arms{ARMS_NUMBER_CIR}_numCons{n_b}"
            plt.figure(fig_name2)
            plt.plot(np.array(len_or[
                              window_size - 1:len(len_or)]) * duration_of_one_sample,
                     diff_seq_search_window, label=f"$N_f$ = {N_f}")
            line_3dB = np.full(len(len_or), 3)
            plt.plot(len_or * duration_of_one_sample, line_3dB,
                     label=f"loss of 3dB", color="r")
            plt.title("Sequential search, $I_{SB}$ = " + f"{n_b}", fontsize=14)
            plt.ylabel('Power loss, dB', fontsize=14)
            plt.xlabel("Time, sec", fontsize=14)
            # plt.yscale("log")
            plt.ylim(0, 10)
            plt.grid()
            plt.legend(prop={'size': 12})
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.savefig(
                f"{figures_path}/window/{fig_name2}.pdf",
                dpi=700, bbox_inches='tight')

        alg_name_i = 0
        for alg_name, pars, algorithm_legend_name, param_sign in zip(algorithm_names, parameters,
                                                                     algorithm_legend_names, param_signs):


            con_type_i = 0
            for con_type, cont_param, cont_param_sigh in zip(context_types, cont_params, cont_param_signs):
                try:
                    num_ex_conts = pickle.load(open(
                        f"{figures_path}/number_of_contexts_cont_par{cont_param}_SSBperiod{N_f}_consSSB{n_b}.pickle",
                        "rb"))
                    print(f"Number of existing contexts for cont param {cont_param}: {num_ex_conts}")
                except:
                    print(f"Number of existing contexts for cont param {cont_param} is unknown!")
                test_name = f"real_protocol_SSB_period"
                r_act_bandit = np.zeros((len(pars),len(NUMBERs_OF_CONS_SSB), len(context_types), len(Numbers_of_frames_between_SSB)))
                fig_name = f"{con_type}_{alg_name}_{cont_param}_{test_name}_arms{ARMS_NUMBER_CIR}_numCons{n_b}"

                plt.figure(fig_name)
                for p_i, p in enumerate(pars):
                    print(p)

                    # plt.plot(avarage_oracle, label="Oracle")
                    for N_f_i, N_f in enumerate(Numbers_of_frames_between_SSB):
                        print(f"bandit N_f {N_f}")
                        # average_reward = pickle.load(open(
                        #     f"{figures_path}/cumulative_average_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{SSB_p}_consSSB{NUMBER_OF_CONS_SSB}.pickle",
                        #     "rb"))
                        # #plt.plot(average_reward, label=f"{algorithm_legend_name}, {param_sign} = {p}")
                        # plt.plot(average_reward, label=f"SSB period = {SSB_p}")
                        # exloitation_iterations = pickle.load(open(
                        #     f"{PATH}/exloitation_iterations_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{N_f}_consSSB{n_b}_seed{1}.pickle",
                        #     "rb"))

                        #diff = np.zeros(len(exloitation_iterations)-window_size+1)
                        diff = np.zeros(len(oracle))
                        number_of_seeds = 10

                        for seed_num in range(1,number_of_seeds+1):
                            # exloitation_iterations = pickle.load(open(
                            #     f"{PATH}/exloitation_iterations_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{N_f}_consSSB{n_b}_seed{seed_num}.pickle",
                            #     "rb"))
                            exloitation_iterations = pickle.load(open(
                                f"{PATH}/exloitation_iterations_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{N_f}_consSSB{n_b}_seed{seed_num}.pickle",
                                "rb"))


                            reward_band = pickle.load(open(
                                f"{PATH}/reward_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{N_f}_consSSB{n_b}_seed{seed_num}.pickle",
                                "rb"))


                            reward_band = reward_band * max_reward * 10**(UE_power_dBi/10)
                            # reward_exploitation_average = np.cumsum(reward_exploitation) / (np.arange(len(reward_exploitation)) + 1)
                            #reward_exploitation_average = cumulative_window(reward_exploitation,window_size)


                            # reward_exploitation_average = 10 * np.log10(
                            #     reward_exploitation_average / (10 ** (-3)))


                            # oracle_for_bandit = oracle[np.array(exloitation_iterations)]
                            # oracle_for_bandit_average = np.cumsum(oracle_for_bandit) / (np.arange(len(oracle_for_bandit)) + 1)

                            #oracle_for_bandit_average = cumulative_window(oracle_for_bandit,window_size)
                            #oracle_for_bandit = cumulative_window(oracle_for_bandit, window_size)
                            # oracle_for_bandit_av = np.cumsum(oracle_for_bandit) / (
                            #             np.arange(len(oracle_for_bandit)) + 1)
                            #reward_exploitation = cumulative_window(reward_exploitation, window_size)
                            reward_av = np.cumsum(reward_band) / (
                                    np.arange(len(reward_band)) + 1)


                            # oracle_for_bandit_dBm = 10 * np.log10(oracle_for_bandit_av / (10 ** (-3)))
                            reward_dBm = 10 * np.log10(reward_av / (10 ** (-3)))
                            diff += oracle_for_seq_dBm - reward_dBm

                        diff = diff/number_of_seeds

                        print(f"exloitation_iterations bandit {len(exloitation_iterations)}")
                        r_a = calculate_act_throughput(reward_band, exloitation_iterations, BAND_COEF,BANDWIDTH, noize_dB)

                        #plt.plot(np.array(exloitation_iterations[window_size-1:len(exloitation_iterations)])*duration_of_one_sample, diff, label=f"SSB period = {SSB_p}")
                        r_act_bandit[p_i, n_b_i, con_type_i, N_f_i] = r_a

                        plt.plot(len_or * duration_of_one_sample, diff,
                                 label=f"$N_f$ = {N_f}, {param_sign} = {p}")
                    #plt.plot(oracle_for_bandit_average, label=f"Oracle, SSB period = {SSB_p}")

                    #plt.title(f"Grid step = {cont_params[0]}, Number of contexts = {num_ex_conts}")
                    line_3dB = np.full(len(len_or), 3)
                    plt.plot(len_or * duration_of_one_sample, line_3dB,
                             label=f"loss of 3dB", color="r")
                    # plt.plot(len_or * duration_of_one_sample, oracle_for_seq_dBm,
                    #          label=f"oracle", color="r")
                    # plt.title(f"{algorithm_legend_name}, {param_sign} = {p}, {cont_param_sigh} = {cont_param}, Number of SSB = {n_b}",fontsize=14)
                    plt.title(f"{algorithm_legend_name}, {cont_param_sigh} = {cont_param}, Number of SB = {n_b}",fontsize=14)
                    plt.ylabel('Power loss, dB',fontsize=14)
                    plt.xlabel("Time, sec",fontsize=14)
                    # plt.yscale("log")
                    plt.ylim(0,10)
                    plt.grid()
                    plt.legend(prop={'size': 12})
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)
                    plt.savefig(
                        f"{figures_path}/{fig_name}.pdf",
                        dpi=700, bbox_inches='tight')
                con_type_i += 1
            alg_name_i += 1
        pickle.dump(r_act_bandit, open(
            f"{figures_path}/r_act_bandit_type{ANTENNA_TYPE}_arms{int(ARMS_NUMBER_CIR)}_it{ITER_NUMBER_CIR}.pickle",
            'wb'))

        for con_type_i in range(0, len(context_types)):

            fig_name2 = f"throughput_arms{ARMS_NUMBER_CIR}_context_type_{context_types[con_type_i]}_algorithm_name{alg_name}_cycle3"
            plt.figure(fig_name2)

            for p_i, p in enumerate(pars):
                for n_b_i in range(0, len(NUMBERs_OF_CONS_SSB)):
                    plt.plot(Numbers_of_frames_between_SSB,
                             r_act_bandit[p_i, n_b_i, con_type_i, :]/1e9,
                             label=f"{param_sign} = {p}") #label=f"Number of SSB= {NUMBERs_OF_CONS_SSB[n_b_i]}

            plt.plot(Numbers_of_frames_between_SSB,
                     np.full((len(Numbers_of_frames_between_SSB)), r_a_oracle/1e9), label=f"Oracle")
            plt.ylabel('Average throughput, Gbps', fontsize=14)
            plt.xlabel("$N_f$, frames", fontsize=14)
            # plt.yscale("log")
            # plt.ylim(0, 10)
            # plt.ylim(0.1e9, 1.6e9)
            plt.grid()
            plt.legend(prop={'size': 12})
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.savefig(
                f"{figures_path}/window/{fig_name2}.pdf",
                dpi=700, bbox_inches='tight')




    fig_name28= f"throughput_seq_search_arms{ARMS_NUMBER_CIR}_cycle3"
    plt.figure(fig_name28)
    for n_b_i in range(0, len(NUMBERs_OF_CONS_SSB)):
        plt.plot(Numbers_of_frames_between_SSB,
                 r_act_seq[n_b_i, :]/1e9, label=f"Number of SB = {NUMBERs_OF_CONS_SSB[n_b_i]}")
    plt.plot(Numbers_of_frames_between_SSB,
             np.full((len(Numbers_of_frames_between_SSB)), r_a_oracle/1e9), label=f"Oracle")
    # plt.plot(Numbers_of_frames_between_SSB,
    #          np.full((len(Numbers_of_frames_between_SSB)), r_a_or), label=f"Burst len = {NUMBERs_OF_CONS_SSB[n_b_i]}")
    plt.ylabel('Average throughput, Gbps', fontsize=14)
    plt.xlabel("$N_f$, frames", fontsize=14)
    # plt.yscale("log")
    # plt.ylim(0.1e9, 1.6e9)
    plt.grid()
    plt.legend(prop={'size': 12})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(
        f"{figures_path}/window/{fig_name28}.pdf",
        dpi=700, bbox_inches='tight')

    #
    # test_name = f"real_protocol_SSB_period"
    # fig_name2 = f"oracle_{test_name}_arms{ARMS_NUMBER_CIR}"
    # plt.figure(fig_name2)
    # plt.plot(len_or * duration_of_one_sample, oracle_for_bandit)
    #
    #
    # plt.ylabel('Power, dB',fontsize=14)
    # plt.xlabel("Time, sec",fontsize=14)
    # # plt.yscale("log")
    # plt.ylim(0,10)
    # plt.grid()
    # plt.legend(prop={'size': 12})
    # plt.yticks(fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.savefig(
    #     f"{figures_path}/{fig_name2}.pdf",
    #     dpi=700, bbox_inches='tight')


        # plt.show()

def calc_beam_wide(ico_subdivision):
    icosphere = trimesh.creation.icosphere(subdivisions=ico_subdivision, radius=1.0, color=None)
    triangle_size = norm(icosphere.vertices[icosphere.edges[0][0]] - icosphere.vertices[icosphere.edges[0][1]])
    cone_angle = math.acos((2 - (triangle_size) ** 2) / 2)
    print(cone_angle*180/math.pi, "o")
    print(len(np.array(icosphere.vertices)))



#DL_channel()
#plot_time()
#plot_exploitation_test()
#plot_exploitation_test()


# calc_beam_wide(4)
# exit()
plot_real_protocol()