"""
Thompson Sampling bandit over reranker strategies.
Learns which reranking approach works best per query type.
"""
import numpy as np
from collections import defaultdict


class ThompsonBandit:
    def __init__(self, strategies: list[str]):
        self.strategies = strategies
        # Beta distribution parameters per strategy
        self.alpha = defaultdict(lambda: {s: 1.0 for s in strategies})
        self.beta_ = defaultdict(lambda: {s: 1.0 for s in strategies})
        self.history = []

    def select(self, context: str = "default") -> str:
        """Sample from each arm's Beta distribution, pick highest."""
        samples = {
            s: np.random.beta(self.alpha[context][s], self.beta_[context][s])
            for s in self.strategies
        }
        chosen = max(samples, key=samples.get)
        return chosen

    def update(self, context: str, strategy: str, reward: float):
        """Update Beta parameters. reward in [0, 1]."""
        self.alpha[context][strategy] += reward
        self.beta_[context][strategy] += (1 - reward)
        self.history.append({
            "context": context, "strategy": strategy, "reward": reward,
            "alpha": self.alpha[context][strategy],
            "beta": self.beta_[context][strategy],
        })

    def get_stats(self, context: str = "default") -> dict:
        """Return current estimates per strategy."""
        return {
            s: {
                "mean": self.alpha[context][s] / (self.alpha[context][s] + self.beta_[context][s]),
                "pulls": self.alpha[context][s] + self.beta_[context][s] - 2,
            }
            for s in self.strategies
        }


if __name__ == "__main__":
    print("=== Thompson Sampling Bandit Demo ===\n")

    bandit = ThompsonBandit(["semantic", "attribute", "category"])

    # Simulate: semantic works best for short queries, attribute for long
    np.random.seed(42)
    for i in range(100):
        query_type = "short" if i % 2 == 0 else "long"
        chosen = bandit.select(query_type)

        # Simulated rewards
        if query_type == "short":
            rewards = {"semantic": 0.8, "attribute": 0.5, "category": 0.3}
        else:
            rewards = {"semantic": 0.4, "attribute": 0.9, "category": 0.5}

        reward = rewards[chosen] + np.random.normal(0, 0.1)
        reward = np.clip(reward, 0, 1)
        bandit.update(query_type, chosen, reward)

    print("After 100 rounds:")
    for ctx in ["short", "long"]:
        print(f"\n  Context: {ctx}")
        for s, stats in bandit.get_stats(ctx).items():
            print(f"    {s:12s}  mean={stats['mean']:.3f}  pulls={stats['pulls']:.0f}")

    print("\n  Expected: semantic wins for short, attribute wins for long")
