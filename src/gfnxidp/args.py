"""
Configuration file for IDP project using argparse.
Contains settings for dataset, generator, oracle, and proxy.
"""

import argparse
from pathlib import Path


# Path helpers ---------------------------------------------------------------

# args.py is in: src/gfnxidp/args.py
# project root = GFLOWNETXIDPDESIGN/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"
LOGS_DIR = PROJECT_ROOT / "logs"
TOOLS_DIR = PROJECT_ROOT / "src" / "gfnxidp" / "tools"


def get_default_args():
    """
    Returns default Args object without parsing command line.
    Useful for testing and notebooks.
    """
    parser = argparse.ArgumentParser()

    # ---------------- Dataset ----------------
    parser.add_argument(
        "--train_path",
        type=str,
        default=str(DATASETS_DIR / "de_dataset.csv"),
    )
    parser.add_argument("--test_size", type=int, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # ---------------- Generator ----------------
    parser.add_argument("--reward_exp_min", default=1e-6, type=float)
    parser.add_argument("--gen_Z_learning_rate", default=5e-3, type=float)
    parser.add_argument("--generator_model", type=str, default=None)
    parser.add_argument("--gen_leaf_coef", type=float, default=25)
    parser.add_argument("--gen_output_coef", type=float, default=1)
    parser.add_argument("--gen_loss_eps", type=float, default=1e-5)
    parser.add_argument("--gen_max_len", type=float, default=256)
    parser.add_argument("--gen_balanced_loss", type=float, default=1)
    parser.add_argument("--gen_num_hidden", type=float, default=128)
    parser.add_argument("--gen_partition_init", type=float, default=10)
    parser.add_argument("--gen_learning_rate", type=float, default=1e-4)
    parser.add_argument("--gen_L2", type=float, default=1e-4)
    parser.add_argument("--gen_clip", type=float, default=1)
    parser.add_argument("--gen_episodes_per_step", type=int, default=16)
    parser.add_argument("--gen_random_action_prob", type=float, default=0.001)
    parser.add_argument("--gen_sampling_temperature", type=float, default=3.0)
    parser.add_argument("--gen_data_sample_per_step", type=float, default=16)
    parser.add_argument("--gen_num_iterations", type=float, default=500)
    parser.add_argument("--num_sampled_per_round", type=float, default=128)
    parser.add_argument("--num_rounds", type=float, default=1)

    # ---------------- Oracle ----------------
    parser.add_argument(
        "--oracle_mode",
        type=str,
        default="csat",
        choices=["csat", "dg", "deescore"],
    )
    parser.add_argument(
        "--phys_model",
        type=str,
        default=str(TOOLS_DIR / "Models" / "phys_multi.sav"),
    )
    parser.add_argument(
        "--w2v_model",
        type=str,
        default=str(TOOLS_DIR / "Models" / "w2v_multi.sav"),
    )
    parser.add_argument(
        "--swissprot_model",
        type=str,
        default=str(TOOLS_DIR / "Embeddings" / "swissprot_size200_window25.model"),
    )
    parser.add_argument(
        "--csat_model",
        type=str,
        default=str(TOOLS_DIR / "Models" / "model_logcdil_mgml.joblib"),
    )
    parser.add_argument(
        "--dg_model",
        type=str,
        default=str(TOOLS_DIR / "Models" / "model_dG.joblib"),
    )
    parser.add_argument(
        "--nu_model",
        type=str,
        default=str(TOOLS_DIR / "Models" / "svr_model_nu.joblib"),
    )
    parser.add_argument(
        "--iupred2a",
        type=str,
        default=str(TOOLS_DIR / "iupred2a.py"),
    )
    parser.add_argument("--charge_termini", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=293)
    parser.add_argument("--ionic_strength", type=float, default=0.15)
    parser.add_argument(
        "--residues_file",
        type=str,
        default=str(DATASETS_DIR / "residues.csv"),
    )

    # ---------------- Proxy ----------------
    parser.add_argument("--proxy_model", type=str, default=None)
    parser.add_argument("--proxy_weights", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--proxy_num_iterations", type=int, default=10000)
    parser.add_argument("--proxy_num_hid", type=int, default=1024)
    parser.add_argument("--proxy_num_layers", type=int, default=4)
    parser.add_argument("--proxy_dropout", type=float, default=0.1)
    parser.add_argument("--proxy_learning_rate", type=float, default=1e-5)
    parser.add_argument("--proxy_L2", type=float, default=1e-6)
    parser.add_argument("--proxy_early_stop_tol", type=int, default=5)
    parser.add_argument("--proxy_num_per_minibatch", type=float, default=256)
    parser.add_argument("--proxy_early_stop_to_best_params", type=int, default=0)
    parser.add_argument("--proxy_num_dropout_samples", type=int, default=25)
    parser.add_argument(
        "--proxy_mode",
        type=str,
        default="range",
        choices=["gaussian", "range", "direct"],
    )
    parser.add_argument(
        "--target_dg_low",
        type=float,
        default=-3.0,
        help="Lower bound of target ΔG range",
    )
    parser.add_argument(
        "--target_dg_high",
        type=float,
        default=2.5,
        help="Upper bound of target ΔG range",
    )
    parser.add_argument(
        "--preference_direction",
        type=int,
        default=0,
        choices=[-1, 0, 1],
        help="Preference within range: -1=lower, 0=no preference, 1=higher",
    )
    parser.add_argument(
        "--reward_lambda",
        type=float,
        default=1.0,
        help="Decay rate for out-of-range penalties (like λ in A-GFN)",
    )
    parser.add_argument("--reward_min_clip", type=float, default=1e-6)
    parser.add_argument("--reward_max_clip", type=float, default=1.0)


    parser.add_argument('--cys_max_fraction', type=float, default=0.03)
    parser.add_argument('--hydrophobic_max_fraction', type=float, default=0.35)
    parser.add_argument('--aromatic_min_fraction', type=float, default=0.08)
    parser.add_argument('--aromatic_max_fraction', type=float, default=0.12)
    parser.add_argument('--cys_penalty_strength', type=float, default=10.0)
    parser.add_argument('--hydrophobic_penalty_strength', type=float, default=10.0)
    parser.add_argument('--aromatic_penalty_strength', type=float, default=10.0)
    parser.add_argument('--use_disorder_filter', type=int, default=1)
    parser.add_argument('--min_disorder_score', type=float, default=0.7)
    parser.add_argument('--max_ordered_stretch', type=int, default=15)
    parser.add_argument('--disorder_penalty_strength', type=float, default=2.0)


    # ---------------- Logging ----------------
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(LOGS_DIR / "training_log.json"),
    )

    # Parse empty list for defaults only
    args = parser.parse_args([])
    return args
