from deep_rl import *

def ppo_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('skip', False)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=2.5e-4)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 5
    config.optimization_epochs = 4
    config.mini_batch_size = config.rollout_length * config.num_workers // 4
    config.ppo_ratio_clip = 0.1
    config.log_interval = config.rollout_length * config.num_workers
    config.shared_repr = True
    config.max_steps = int(2e7)
    run_steps(PPOAgent(config))

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(0)
    # select_device(0)

    game = 'BreakoutNoFrameskip-v4'
    # dqn_pixel(game=game)
    # quantile_regression_dqn_pixel(game=game)
    # categorical_dqn_pixel(game=game)
    # rainbow_pixel(game=game)
    ppo_pixel(game=game)
    # n_step_dqn_pixel(game=game)
    # option_critic_pixel(game=game)
    # ppo_pixel(game=game)