from easydict import EasyDict

config = EasyDict()

config.training = True
config.gpu = False
config.soft_max = False
config.memory_size = 1
config.strategy = "m1"
config.player_name = "player2_" + config.strategy + str(config.memory_size)
config.learning_rate = 1e-4
config.num_rounds = 10000
config.save_iteration = config.num_rounds / 10
config.print_iteration = 100
config.update_q_target_frequency = 100
config.batch_size = 1
config.max_batch = 5000
config.switch_iters = config.num_rounds / 5
config.epsilon_decay = config.switch_iters / 3
config.epsilon_start = 0.99
config.epsilon_min = 0.01

config.num_deck = 4
config.num_obv = 23
config.num_memory = 10
config.num_action = 2
config.step_reward = 0.1
config.fail_reward = -1
config.success_reward = 2

config.in_x_dim = config.num_obv + config.num_action
config.out_x_dim = config.in_x_dim
config.hidden_dim = config.out_x_dim
config.batch_size = 10
config.gamma = 0.999

config.num_layer = 5
config.scale_factor = 7
config.q_input_dim = config.num_obv + config.memory_size + config.hidden_dim

