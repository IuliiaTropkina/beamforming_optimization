import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import trimesh

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

    frames_per_data_frame = 1000 #10000
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





    for sc in scenarios:
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

                    plt.plot(average_reward, label=f"{con_type}, {cont_param_sigh} = {cont_param}")

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

        test_name = f"exploitation_test_seq"
        fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}"
        plt.figure(fig_name)
        oracle = np.array(oracle)
        oracle_for_seq = oracle[np.array(seq_search_exploitation_it_num)]
        oracle_for_seq_average = np.cumsum(np.array(oracle_for_seq)) / (np.arange(len(oracle_for_seq)) + 1)
        seq_search_exploitation_reward_average = np.cumsum(seq_search_exploitation_reward) / (np.arange(len(seq_search_exploitation_reward)) + 1)

        plt.plot(seq_search_exploitation_reward_average, label="Sequential search")
        plt.plot(oracle_for_seq_average, label=f"Oracle")

        # plt.title(f"Grid step = {cont_params[0]}, Number of contexts = {num_ex_conts}")
        plt.title(f"Sequential search")

        plt.ylabel('Cumulative average reward')
        plt.xlabel('Sample')
        # plt.yscale("log")
        # plt.ylim(top=250)
        plt.grid()
        plt.legend(prop={'size': 9})

        plt.savefig(
            f"{figures_path}/{fig_name}.pdf",
            dpi=700, bbox_inches='tight')


        for con_type, cont_param, cont_param_sigh in zip(context_types, cont_params, cont_param_signs):
            try:
                num_ex_conts = pickle.load(open(f"{figures_path}/number_of_contexts_cont_par{cont_param}.pickle", "rb"))
                print(f"Number of existing contexts for cont param {cont_param}: {num_ex_conts}")
            except:
                print(f"Number of existing contexts for cont param {cont_param} is unknown!")
            for alg_name, pars, algorithm_legend_name, param_sign in zip(algorithm_names, parameters,
                                                                         algorithm_legend_names, param_signs):

                for p in pars:
                    test_name = f"exploitation_test_bandit_{p}"
                    fig_name = f"{test_name}_arms{ARMS_NUMBER_CIR}"
                    plt.figure(fig_name)

                    exloitation_iterations = pickle.load(open(
                        f"{figures_path}/exloitation_iterations_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}.pickle",
                        "rb"))

                    reward_exploitation = pickle.load(open(
                        f"{figures_path}/reward_exploitation_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}.pickle",
                        "rb"))


                    reward_exploitation_average = np.cumsum(reward_exploitation) / (np.arange(len(reward_exploitation)) + 1)
                    plt.plot(reward_exploitation_average, label=f"{con_type}, {cont_param_sigh} = {cont_param}")
                    oracle_for_bandit = oracle[np.array(exloitation_iterations)]
                    oracle_for_bandit_average = np.cumsum(oracle_for_bandit) / (np.arange(len(oracle_for_bandit)) + 1)

                    plt.plot(oracle_for_bandit_average, label=f"Oracle, {con_type}, {cont_param_sigh} = {cont_param}")


                    # plt.title(f"Grid step = {cont_params[0]}, Number of contexts = {num_ex_conts}")
                    plt.title(f"MAB, {param_sign} = {p}")

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



def plot_real_protocol():

    frames_per_data_frame = 1000 #10000
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
    context_types = ["location","DOA"]
    # algorithm_names = ["EPS_greedy",
    #                    "UCB",
    #                    "THS"]

    cont_params = [15, 162]
    cont_param_signs = ["Grid step", "Number of contexts"]

    algorithm_names = ["EPS_greedy"] #"DQL","EPS_greedy"
    algorithm_legend_names = ["$\epsilon$-Greedy"]
    param_signs = ["$\epsilon$"]
    # parameters = [[0.05, 0.1, 0.15],
    #               [10 ** (-7), 10 ** (-7) * 2, 10 ** (-7) / 2],
    #               [0.2, 0.5]]
    parameters = [[0.15]]

    SCENARIO_DURATION = 3
    NUMBER_OF_CONS_SSB = 4
    SSB_period = np.array([5,10,20,40,80,160])
    SSB_period = SSB_period*10**(-3)



    for sc in scenarios:
        number_of_cycles = 1
        folder_test = "real_protocol"
        folder_name_figures = f"scenario_{sc}"
        figures_path = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/{folder_name_figures}/{folder_test}"




        test_name = f"real_protocol_SSB_period"
        fig_name = f"sequential_seqrch{test_name}_arms{ARMS_NUMBER_CIR}_numCons{NUMBER_OF_CONS_SSB}"
        plt.figure(fig_name)
        oracle = pickle.load(open(
            f"{figures_path}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle",
            "rb"))
        avarage_oracle = np.cumsum(oracle) / (np.arange(ITER_NUMBER_CIR) + 1)
        plt.plot(avarage_oracle, label="Oracle")
        for SSB_p in SSB_period:
            sequential_search_reward = pickle.load(open(
                f"{figures_path}/cumulative_avarage_sequential_search_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{SSB_p}_consSSB{NUMBER_OF_CONS_SSB}.pickle",
                "rb"))
            plt.plot(sequential_search_reward, label=f"SSB period = {SSB_p}")
        plt.title(f"Sequential search, Number of SSB = {NUMBER_OF_CONS_SSB}")
        plt.ylabel('Cumulative average reward')
        plt.xlabel('Sample')
        # plt.yscale("log")
        # plt.ylim(top=250)
        plt.grid()
        plt.legend(prop={'size': 9})

        plt.savefig(
            f"{figures_path}/{fig_name}.pdf",
            dpi=700, bbox_inches='tight')






        for con_type, cont_param, cont_param_sigh in zip(context_types, cont_params, cont_param_signs):



            try:
                num_ex_conts = pickle.load(open(f"{figures_path}/number_of_contexts_cont_par{cont_param}.pickle", "rb"))
                print(f"Number of existing contexts for cont param {cont_param}: {num_ex_conts}")
            except:
                print(f"Number of existing contexts for cont param {cont_param} is unknown!")
            for alg_name, pars, algorithm_legend_name, param_sign in zip(algorithm_names, parameters, algorithm_legend_names, param_signs):

                for p in pars:
                    test_name = f"real_protocol_SSB_period"
                    fig_name = f"{con_type}_{alg_name}_{p}_{cont_param}_{test_name}_arms{ARMS_NUMBER_CIR}_numCons{NUMBER_OF_CONS_SSB}"

                    plt.figure(fig_name)
                    plt.plot(avarage_oracle, label="Oracle")
                    for SSB_p in SSB_period:
                        average_reward = pickle.load(open(
                            f"{figures_path}/cumulative_average_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{SSB_p}_consSSB{NUMBER_OF_CONS_SSB}.pickle",
                            "rb"))
                        #plt.plot(average_reward, label=f"{algorithm_legend_name}, {param_sign} = {p}")
                        plt.plot(average_reward, label=f"SSB period = {SSB_p}")




                    #plt.title(f"Grid step = {cont_params[0]}, Number of contexts = {num_ex_conts}")
                    plt.title(f"{algorithm_legend_name}, {param_sign} = {p}, {cont_param_sigh} = {cont_param}, Number of SSB = {NUMBER_OF_CONS_SSB}")

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


#DL_channel()
#plot_time()
#plot_exploitation_test()
plot_real_protocol()