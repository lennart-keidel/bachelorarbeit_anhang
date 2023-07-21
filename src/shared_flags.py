from absl import flags

VALID_GAMES = [
    "leduc_poker",
]


VALID_PLAYERS = [
    "ISMCTS",
    "random",
    "CFRSolver",
    "CFRPlusSolver",
    "CFRBRSolver",
    "DCFRSolver",
    "DeepCFRSolver",
    "XFPSolver",
    "NFSP",
    "PolicyGradient",
]


# MODEL SAVE
flags.DEFINE_string("save_checkpoint_dir", "trained_models", "Directory to save/load the agent models or solvers from")
flags.DEFINE_bool("save_enable", True, "Enable saving the model or solver")
flags.DEFINE_bool("load_enable", True, "Enable loading the model or solver")
flags.DEFINE_integer("save_frequency", 100, "Episode frequency at which the models or solvers are saved")

# TRAINING
flags.DEFINE_integer("training_number_episodes", 17000, "Number of training episodes")
flags.DEFINE_integer("training_evaluation_frequency", 100, "Episode frequency at which models or solvers evaluated")
flags.DEFINE_integer("training_log_frequency", 100, "Episode frequency at which to print/store log")

# GAME
flags.DEFINE_integer("game_number_players", 2, "Number of players")
flags.DEFINE_enum("game", "leduc_poker", VALID_GAMES, "Name of the game")
flags.DEFINE_enum("game_agent_player1", "CFRSolver", VALID_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("game_agent_player2", "CFRPlusSolver", VALID_PLAYERS, "Who controls player 2.")

# WANDB FLAGS
flags.DEFINE_string("wandb_project_name", "OpenSpiel", "Weights & Biases project name")
flags.DEFINE_boolean("wandb_enable", False, "disables Weights & Biases")