from deprecated.a3c_agent import A3Cagent


def main():
    max_episode_num = 100
    env_name = "Pendulum-v0"
    agent = A3Cagent(env_name)
    # start training
    agent.train(max_episode_num)
    agent.plot_result()


if __name__=="__main__":
    main()