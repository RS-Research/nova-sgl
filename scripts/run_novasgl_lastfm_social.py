from recbole.quick_start import run_recbole


def main():
    config_dict = {
        "data_path": r"data",

        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "artist_id",

        "load_col": {
            "inter": ["user_id", "artist_id"],
        },

        "filter_inter_by_user_or_item": True,
        "user_inter_num_interval": "[5,inf)",
        "item_inter_num_interval": "[5,inf)",

        "embedding_size": 64,
        "n_layers": 2,
        "social_layers": 1,

        "novelty_weight": 0.10,
        "pop_penalty_weight": 0.05,
        "eta_global": 0.5,
        "eta_social": 0.5,
        "gate_hidden_size": 32,
        "long_tail_quantile": 0.80,

        "cl_weight": 0.05,
        "cl_temp": 0.2,
        "disable_cl_when_no_social": True,

        "social_file": r"data\lastfm\lastfm.net",
        "social_undirected": True,
        "add_social_self_loop": True,
        "debug_social": True,

        "epochs": 100,
        "stopping_step": 20,
        "train_batch_size": 2048,
        "eval_batch_size": 1024,
        "learning_rate": 0.001,
        "reg_weight": 1e-5,

        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        },

        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },

        "metrics": ["Recall", "NDCG", "Hit", "MRR"],
        "topk": [10, 20],
        "valid_metric": "NDCG@20",
        "metric_decimal_place": 4,

        "seed": 2026,
        "reproducibility": True,
        "use_gpu": True,
        "gpu_id": 0,
    }

    result = run_recbole(
        model="NOVASGL",
        dataset="lastfm",
        config_dict=config_dict,
    )

    print("\nFinal result for full social NOVA-SGL on LastFM:")
    print(result)


if __name__ == "__main__":
    main()