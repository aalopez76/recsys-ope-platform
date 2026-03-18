.PHONY: help setup data train_recbole reward_model bandits train_rl load_data ope ope_full app test lint clean clean-dry

# Default target
help:
	@echo "RecSys OPE Platform v2.0 - Makefile Targets"
	@echo "============================================"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  make setup                    - Install all dependencies (production + dev)"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make data                     - Build OBD artifacts"
	@echo "  make load_data dataset=<name> - Load dataset (synthetic|obd|movielens)"
	@echo ""
	@echo "Training:"
	@echo "  make train_recbole            - Train RecBole baselines"
	@echo "  make reward_model             - Train click reward model"
	@echo "  make bandits                  - Train bandit agents (LinUCB+EpsGreedy)"
	@echo "  make train_rl agent_type=dqn  - Train RL agent (linucb|epsgreedy|dqn|ppo)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make ope                      - Off-Policy Evaluation (all estimators)"
	@echo ""
	@echo "Quality & Testing:"
	@echo "  make test                     - Run all pytest tests with coverage"
	@echo "  make lint                     - Run ruff + black + mypy"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean                    - Remove generated files and cache"

# Install dependencies
setup:
	@echo "Installing dependencies from pyproject.toml..."
	pip install -e ".[dev]"
	@echo "Setup complete!"

# Download and prepare OBD dataset
data:
	@echo "Preparing OBD (Open Bandit Dataset) - Certification Ready..."
	python -m src.data.build_obd_datasets
	@echo "Data preparation complete!"

# Train RecBole models
train_recbole:
	@echo "Training RecBole recommendation models (Pop, BPR, NeuMF, LightGCN)..."
	python -m src.recsys.train_recbole_baselines --models Pop,BPR,NeuMF,LightGCN
	@echo "RecBole training complete! Results in reports/tables/"

# Train Bandit reward model
reward_model:
	@echo "Training reward model (logistic regression on train_idx)..."
	python -m src.bandits.reward_model --seed 42
	@echo "Reward model saved to saved/reward_model/"

# Train bandit agents + export policy
bandits: reward_model
	@echo "Training bandit agents (LinUCB + EpsGreedy)..."
	python -m src.bandits.train_tf_agents --steps 5000 --seed 42 --epsilon 0.1
	@echo "Exporting policy for OPE..."
	python -m src.bandits.export_policy_for_ope --epsilon 0.1
	@echo "Bandits pipeline complete!"

# Run OPE (baselines only)
ope:
	@echo "Running OPE v2 (baselines)..."
	python -m src.ope.run_ope_suite --n-bootstrap 200 --seed 42
	@echo "OPE results: reports/tables/ope_results_sample.csv"

# Run OPE with TFAgent external policy
ope_full: ope
	@echo "Running OPE with TFAgent policy..."
	python -m src.ope.run_ope_suite \
		--out reports/tables/ope_results_with_tf_agents.csv \
		--report reports/tables/ope_report_with_tf_agents.md \
		--external-policy-csv reports/tables/tf_agents_policy_test_action_prob.csv \
		--external-policy-name TFAgent_eps0.1 \
		--n-bootstrap 200 --seed 42
	@echo "OPE with TFAgent complete!"

# Launch Streamlit dashboard
app:
	@echo "Starting Streamlit dashboard..."
	streamlit run src/app/app.py

# Generate comparison reports (alias)
report: ope_full
	@echo "All reports in reports/tables/ and reports/plots/"

# Train RL agent (v2.0) — replaces/extends bandits target
# Usage: make train_rl agent_type=dqn steps=5000
agent_type ?= dqn
steps ?= 5000
train_rl: reward_model
	@echo "Training RL agent ($(agent_type)) for $(steps) steps..."
	python -m src.rl_agents.train_rl \
		--agent-type $(agent_type) \
		--steps $(steps) \
		--seed 42
	@echo "Training complete! Reports in reports/tables/ and reports/plots/"

# Load / inspect a dataset
# Usage: make load_data dataset=obd
dataset ?= synthetic
load_data:
	@echo "Loading dataset: $(dataset)..."
	python -m src.data.data_loader --dataset $(dataset)
	@echo "Done."

# Run tests (all suites)
test:
	@echo "Running all pytest tests with coverage..."
	pytest tests/ -q --cov=src --cov-report=term-missing
	@echo "Tests complete!"

# Lint and format code
lint:
	@echo "Running code quality checks..."
	@echo "1. Running ruff..."
	ruff check src/ tests/ --fix
	@echo "2. Running black..."
	black src/ tests/
	@echo "3. Running mypy..."
	mypy src/
	@echo "Linting complete!"

# Clean generated artifacts (cross-platform)
clean:
	@echo "Cleaning regenerable artifacts..."
	python scripts/clean_artifacts.py --execute
	@echo "Clean complete!"

# Preview clean (dry-run)
clean-dry:
	@echo "Preview of artifacts to clean..."
	python scripts/clean_artifacts.py --dry-run
