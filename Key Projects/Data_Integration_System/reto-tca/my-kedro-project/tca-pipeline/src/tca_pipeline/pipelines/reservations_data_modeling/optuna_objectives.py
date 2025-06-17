# src/tca_pipeline/pipelines/data_modeling/optuna_objectives.py

import os
import yaml
from tca_pipeline.pipelines.reservations_data_modeling.nodes import (
    train_transformer,
    train_gru,
    train_lstm,
    create_train_test
)



def objective_lstm(trial, arr, features_list, scaler, space, seed):
    cfg = {
        "input_dim": len(features_list),
        "hidden_dim": trial.suggest_categorical("hidden_dim", space["hidden_dim"]),
        "num_layers": trial.suggest_int("num_layers", space["num_layers"][0], space["num_layers"][1]),
        "lr": trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True),
        "epochs": 10, 
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "window_size": trial.suggest_int("window_size", space["window_size"][0], space["window_size"][1]),
        "horizon": trial.suggest_int("horizon", space["horizon"][0], space["horizon"][1]),
        "n_splits": 6
    }

    X_train, y_train, X_test, y_test = create_train_test(arr, cfg)

    save_model_path = f"data/06_models/lstm_trial_{trial.number}.pt"

    mae, rmse, wmape = train_lstm(
    X_train, y_train, X_test, y_test, cfg, scaler, arr, seed=seed, save_model_path=save_model_path
)


    mse = rmse ** 2

    print(f"✅ LSTM Trial {trial.number} MSE = {mse:.4f}, WMAPE = {wmape:.4f}")

    return mse

def objective_transformer(trial, arr, features_list, scaler, space, seed):
    # BUILD CFG
    cfg = {
        "input_dim": len(features_list),
        "d_model": trial.suggest_categorical("d_model", space["d_model"]),
        "nhead": trial.suggest_categorical("nhead", space["nhead"]),
        "num_layers": trial.suggest_int("num_layers", space["num_layers"][0], space["num_layers"][1]),
        "lr": trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True),
        "epochs": 10,  # preview only!
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "window_size": trial.suggest_int("window_size", space["window_size"][0], space["window_size"][1]),
        "horizon": trial.suggest_int("horizon", space["horizon"][0], space["horizon"][1]),
        "n_splits": 6
    }

    # CREATE TRAIN/TEST
    X_train, y_train, X_test, y_test = create_train_test(arr, cfg)

    # SAVE MODEL PATH PER TRIAL
    save_model_path = f"data/06_models/transformer_trial_{trial.number}.pt"

    # RUN TRAINING
    mae, rmse, wmape = train_transformer(
        X_train, y_train, X_test, y_test, cfg, scaler, arr, seed=seed,
        save_model_path=save_model_path
    )

    # COMPUTE MSE → this is the objective!
    mse = rmse ** 2

    print(f"✅ Transformer Trial {trial.number} MSE = {mse:.4f}, WMAPE = {wmape:.4f}")

    return mse  # Optuna will optimize MSE

def objective_gru(trial, arr, features_list, scaler, space, seed):
    # BUILD CFG
    cfg = {
        "input_dim": len(features_list),
        "hidden_dim": trial.suggest_categorical("hidden_dim", space["hidden_dim"]),
        "num_layers": trial.suggest_int("num_layers", space["num_layers"][0], space["num_layers"][1]),
        "lr": trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True),
        "epochs": 10,  # preview only!
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "window_size": trial.suggest_int("window_size", space["window_size"][0], space["window_size"][1]),
        "horizon": trial.suggest_int("horizon", space["horizon"][0], space["horizon"][1]),
        "n_splits": 6
    }

    # CREATE TRAIN/TEST
    X_train, y_train, X_test, y_test = create_train_test(arr, cfg)

    # SAVE MODEL PATH PER TRIAL
    save_model_path = f"data/06_models/gru_trial_{trial.number}.pt"

    # RUN TRAINING
    mae, rmse, wmape = train_gru(
    X_train, y_train, X_test, y_test, cfg, scaler, arr, seed=seed, save_model_path=save_model_path
)


    # COMPUTE MSE → this is the objective!
    mse = rmse ** 2

    print(f"✅ GRU Trial {trial.number} MSE = {mse:.4f}, WMAPE = {wmape:.4f}")

    return mse  # Optuna will optimize MSE
