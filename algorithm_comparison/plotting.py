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



    # features_for_prediction = features[0]  # np.unique(features)
    # rewards_for_plotting = rewards_predicted[i, :]
    # fig_name = f"_arm_reward_cont{len(cont_set)}"
    # plt.figure(fig_name)
    # plt.plot(features_for_prediction, rewards_for_plotting, "*", label="predicted")
    # plt.plot(features[0], ALL_REWARD[i], ".", label="reference")
    # # plt.title("")
    # plt.ylabel('Reward')
    # plt.xlabel('Context')
    # # plt.ylim(top=250)
    # plt.grid()
    # plt.legend()
    #
    # plt.savefig(
    #     f"{figures_path}/{fig_name}.pdf",
    #     dpi=700, bbox_inches='tight')



    oracle = pickle.load(open(
        f"{figures_path}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))

    sequential_search_reward = pickle.load(open(
        f"{figures_path}/sequential_search_arms{int(ARMS_NUMBER_CIR)}.pickle",
        "rb"))


    for SUBDIVISION_2 in context_subdevisions:
        icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)
        context_sets.append(np.array(icosphere_context.vertices))

    context_type = ["DOA_plus_previous_beam", "DOA", "previous_beam"] #"DOA", "previous_beam"
    lable_name = ["DOA + previous beam", "DOA", "previous beam"]
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


            plt.plot(cumulative_reward, label=f"DL, {l}")

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


DL_channel()
