
# train CFR, Counterfactual Regret Minimization
python3 src/training/cfr.py >> "logs/CFR_run_$RANDOM.log" 2>&1;

# train CFR-plus, Counterfactual Regret Minimization Plus
python3 src/training/cfr_plus.py >> "logs/CFR_Plus_run_$RANDOM.log" 2>&1;

# train CFR-BR, Counterfactual Regret Minimization against Best Response
python3 src/training/cfr_br.py >> "logs/CFR_BR_run_$RANDOM.log" 2>&1;

# train DCFR, Discounted Counterfactual Regret Minimization
python3 src/training/dcfr.py >> "logs/DCFR_run_$RANDOM.log" 2>&1;

# train XFP, Extensive Form Fictitous Play
python3 src/training/xfp.py >> "logs/XFP_run_$RANDOM.log" 2>&1;

# train NFSP, Neural Fictitous Self Play
python3 src/training/nfsp.py --training_number_episodes="200000000000" --training_evaluation_frequency="10000" --save_frequency="100000" >> "logs/NFSP_run_$RANDOM.log" 2>&1;

# train Deep CFR, Deep Counterfactual Regret Minimization
python3 src/training/deep_cfr.py --training_number_episodes="100000" >> "logs/DeepCFRSolver_run_$RANDOM.log" 2>&1;

# train Policy Gradient with Advantage Actor Critic algorithm
# see loss_algorithm flag for algorithm options
# a2c = Advantage Actor Critic
# rpg = Regret Policy Gradient
# qpg = Q-based Policy Gradient
# rm  = Regret Matching Policy Gradient
loss_algorithm="a2c"
python3 src/training/policy_gradient.py --loss_algorithm="$loss_algorithm" --training_evaluation_frequency="10000" --save_frequency="10000" --training_number_episodes="100000000" >> "logs/PolicyGradient_${loss_algorithm}_run_$RANDOM.log" 2>&1;

# train information set monte-carlo-tree-search
python3 src/training/is_mcts.py >> "logs/ISMCTS_run_$RANDOM.log" 2>&1;