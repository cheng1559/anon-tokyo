"""Closed-loop simulation — standalone PPO training script."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop PPO training")
    parser.add_argument("--model", type=str, required=True, choices=["anon_tokyo", "agent_centric", "query_centric"])
    parser.add_argument("--data_root", type=str, default="data/shards")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--num_updates", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="tb_logs/simulation")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: build model, env, PPO loop
    raise NotImplementedError("Simulation training not yet implemented")


if __name__ == "__main__":
    main()
