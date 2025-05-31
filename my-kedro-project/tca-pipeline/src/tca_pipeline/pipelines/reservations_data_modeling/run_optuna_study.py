# src/tca_pipeline/pipelines/reservations_data_modeling/run_optuna_study.py

import optuna
import yaml
import joblib
import sys
import os

from kedro.config import OmegaConfigLoader
from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project

# So we can import the pipeline module even when running this as a script
sys.path.append("src")

# 🚀 Correct imports!
from tca_pipeline.pipelines.reservations_data_modeling.nodes import (
    create_train_test,
    impute_and_scale,
    prepare_features
)

from tca_pipeline.pipelines.reservations_data_modeling.optuna_objectives import (
    objective_transformer,
    objective_gru
)

# 1️⃣ LOAD PARAMETERS
config_loader = OmegaConfigLoader(conf_source="conf")
params = config_loader["parameters"]

optuna_params = params["optuna_search_space"]
optuna_config = params["optuna_config"]

# 2️⃣ SELECT MODEL TO TUNE
MODEL_TO_TUNE = "gru"  # <<< or "gru"

space = optuna_params[MODEL_TO_TUNE]
n_trials = optuna_config["n_trials"]
seed = optuna_config["seed"]

# 3️⃣ LOAD DATA (BRO THIS WAS THE MISSING PART 🚀)
configure_project("tca_pipeline")

with KedroSession.create() as session:
    context = session.load_context()
    catalog = context.catalog

# 🚀 Load rooms_by_date
df = catalog.load("rooms_by_date")

# 4️⃣ PREPARE FEATURES
df_features = prepare_features(df)

# 5️⃣ IMPUTE & SCALE
features = params["modeling"]["features"]
arr, features_list, scaler = impute_and_scale(df_features, features)

# 6️⃣ DEFINE OBJECTIVE FUNCTION
def objective(trial):
    if MODEL_TO_TUNE == "transformer":
        return objective_transformer(trial, arr, features_list, scaler, space, seed)
    elif MODEL_TO_TUNE == "gru":
        return objective_gru(trial, arr, features_list, scaler, space, seed)
    else:
        raise ValueError(f"Unknown MODEL_TO_TUNE: {MODEL_TO_TUNE}")

# 7️⃣ RUN OPTUNA STUDY
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=n_trials)

# 8️⃣ SAVE STUDY
study_path = f"data/07_optuna/{MODEL_TO_TUNE}_study.pkl"
os.makedirs("data/07_optuna", exist_ok=True)
joblib.dump(study, study_path)
print(f"✅ Saved study to {study_path}")

# 9️⃣ SAVE BEST PARAMS
best_params = study.best_trial.params
best_params_path = f"data/07_optuna/{MODEL_TO_TUNE}_best_params.yml"
with open(best_params_path, "w") as f:
    yaml.dump(best_params, f)
print(f"✅ Saved best params to {best_params_path}")

print(f"BEST MSE LOSS = {study.best_value:.6f}")
print(f"BEST PARAMS = {best_params}")
