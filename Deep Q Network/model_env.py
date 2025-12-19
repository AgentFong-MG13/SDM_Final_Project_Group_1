import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random

# ============================
# 1. 全域參數與設定
# ============================

HORIZON = 14
DEMAND_LEVELS = [5000.0, 10000.0, 15000.0, 20000.0, 25000.0]
DEMAND_PROBS  = [ 0.339, 0.244, 0.188, 0.132, 0.097]
UNIT = 1000
@dataclass
class Airline:
    name: str
    unit_cost: float
    daily_capacity: float #單日運能上限，需求按比例分配不能超過此運能上限，目前情境是超出容量的話會直接棄單
    discount_thresholds: List[Tuple[float, float]]

    def get_discount_factor(self, total_weight: float) -> float:
        factor = 1.0
        for thr, disc in sorted(self.discount_thresholds, key=lambda x: x[0]):
            if total_weight >= thr:
                factor = disc
            else:
                break
        return factor

AIRLINES = {
    "A": Airline("A", 45, 15400.0, [(50000.0, 0.97), (80000.0, 0.96), (100000.0, 0.95), (150000.0, 0.92)]),
    "B": Airline("B", 50, 9900.0, [(50000.0, 0.97), (100000.0, 0.95)]),
    "C": Airline("C", 45, 11000.0,  [(50000.0, 0.98),(100000.0, 0.95), (150000.0, 0.93)]),
}
NAMES = list(AIRLINES.keys())

# ============================
# 2. DRL 環境類別定義
# ============================

class AirCargoDQNEnv(gym.Env):
    """
    DQN 版環境：動作空間為 Discrete
    """
    def __init__(self):
        super(AirCargoDQNEnv, self).__init__()
        self.airlines = AIRLINES
        self.max_acc_limits = {
            name: HORIZON * airline.daily_capacity 
            for name, airline in self.airlines.items()
        }
        
        self.MAX_TOTAL_ACCUMULATION = max(self.max_acc_limits.values()) 
        self.MAX_DEMAND = max(DEMAND_LEVELS)

        # 觀察空間
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # 動作映射表
        # self.action_mapping = [
        #     np.array([1.0, 0.0, 0.0]), # Action 0: 優先塞滿 A
        #     np.array([0.0, 1.0, 0.0]), # Action 1: 優先塞滿 B
        #     np.array([0.0, 0.0, 1.0]), # Action 2: 優先塞滿 C
        #     np.array([0.2, 0.4, 0.4]),
        #     np.array([0.4, 0.4, 0.2]),
        #     np.array([0.4, 0.2, 0.4]),
        #     np.array([0.2, 0.8, 0.0]),
        #     np.array([0.8, 0.2, 0.0]),
        #     np.array([0.0, 0.2, 0.8]),
        #     np.array([0.0, 0.8, 0.2]),
        #     np.array([0.8, 0.0, 0.2]),
        #     np.array([0.2, 0.0, 0.8]), # 0-2-8
        #     np.array([0.6, 0.2, 0.2]),
        #     np.array([0.2, 0.6, 0.2]),
        #     np.array([0.2, 0.2, 0.6]), # 0-2-6
        #     np.array([0.4, 0.6, 0.0]),
        #     np.array([0.6, 0.4, 0.0]),
        #     np.array([0.0, 0.4, 0.6]),
        #     np.array([0.0, 0.6, 0.4]),
        #     np.array([0.6, 0.0, 0.4]),
        #     np.array([0.4, 0.0, 0.6])           
        # ]
        self.action_mapping = [
            np.array([1.0, 0.0, 0.0]), # Action 0: 優先塞滿 A
            np.array([0.0, 1.0, 0.0]), # Action 1: 優先塞滿 B
            np.array([0.0, 0.0, 1.0]), # Action 2: 優先塞滿 C
            np.array([0.5, 0.5, 0.0]), # Action 3: A, B 平分
            np.array([0.5, 0.0, 0.5]), # Action 4: A, C 平分
            np.array([0.0, 0.5, 0.5]), # Action 5: B, C 平分           
            np.array([0.34, 0.33, 0.33]),# Action 6: 三家平分
            np.array([0.2, 0.4, 0.4]),
            np.array([0.4, 0.4, 0.2]),
            np.array([0.4, 0.2, 0.4]),
            np.array([0.2, 0.8, 0.0]),
            np.array([0.8, 0.2, 0.0]),
            np.array([0.0, 0.2, 0.8]),
            np.array([0.0, 0.8, 0.2]),
            np.array([0.8, 0.0, 0.2]),
            np.array([0.2, 0.0, 0.8]), # 0-2-8
            np.array([0.6, 0.2, 0.2]),
            np.array([0.2, 0.6, 0.2]),
            np.array([0.2, 0.2, 0.6]), # 0-2-6
            np.array([0.4, 0.6, 0.0]),
            np.array([0.6, 0.4, 0.0]),
            np.array([0.0, 0.4, 0.6]),
            np.array([0.0, 0.6, 0.4]),
            np.array([0.6, 0.0, 0.4]),
            np.array([0.4, 0.0, 0.6]), # 0-4-6
            np.array([0.1, 0.1, 0.8]),
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.8, 0.1]), # 0-1-8
            np.array([0.2, 0.3, 0.5]),
            np.array([0.2, 0.5, 0.3]),
            np.array([0.3, 0.2, 0.5]),
            np.array([0.3, 0.5, 0.2]),
            np.array([0.5, 0.2, 0.3]),
            np.array([0.5, 0.3, 0.2]), # 2-3-5
            np.array([0.2, 0.167, 0.133]),
            np.array([0.2, 0.133, 0.167]),
            np.array([0.133, 0.167, 0.2]),
            np.array([0.133, 0.2, 0.167]),
            np.array([0.167, 0.2, 0.133]),
            np.array([0.167, 0.133, 0.2]), # 2-3-10
            np.array([0.3, 0.3, 0.4]),
            np.array([0.4, 0.3, 0.3]),
            np.array([0.3, 0.4, 0.3]), # 3-3-4
            np.array([0.1, 0.4, 0.5]),
            np.array([0.1, 0.5, 0.4]),
            np.array([0.4, 0.1, 0.5]),
            np.array([0.4, 0.5, 0.1]),
            np.array([0.5, 0.4, 0.1]),
            np.array([0.5, 0.1, 0.4]),# 1-4-5
        ]
        # ---------------------------
        self.action_space = spaces.Discrete(len(self.action_mapping))

        self.current_demand = 0.0
        self.current_day = 0
        self.current_acc = {"A": 0.0, "B": 0.0, "C": 0.0}

    def _get_obs(self) -> np.ndarray:
            day_norm = self.current_day / HORIZON
            acc_norm = np.array([
                self.current_acc[name] / self.max_acc_limits[name]
                for name in NAMES
            ])
            demand_norm = self.current_demand / self.MAX_DEMAND
            return np.concatenate([[day_norm], acc_norm, [demand_norm]], dtype=np.float32)

    def _sample_demand(self) -> float:
        return random.choices(DEMAND_LEVELS, weights=DEMAND_PROBS, k=1)[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 1
        self.current_acc = {"A": 0.0, "B": 0.0, "C": 0.0}
        self.current_demand = self._sample_demand()
        return self._get_obs(), {}

    def terminal_cost(self) -> float:
        total_cost = 0.0
        for name in NAMES:
            airline = self.airlines[name]
            w_kg = self.current_acc[name]
            disc = airline.get_discount_factor(w_kg)
            total_cost += airline.unit_cost * w_kg * disc
        return total_cost

    def step(self, action: int):
        current_demand = self.current_demand
        
        # 取得分配比例
        proportions = self.action_mapping[action]
        
        caps = np.array([self.airlines[name].daily_capacity for name in NAMES])
        target_allocs = proportions * current_demand
        clipped_allocs = np.minimum(target_allocs, caps)
        
        total_allocated = np.sum(clipped_allocs)
        remaining = current_demand - total_allocated
        
        final_allocs_array = clipped_allocs.copy()
        
        if remaining > 1e-5:
            has_space = (caps - clipped_allocs) > 1e-5
            space_proportions = proportions * has_space # 剩餘公司的容量分配權重總和
            sum_space_prop = np.sum(space_proportions)
            
            if sum_space_prop > 0:
                re_alloc = remaining * (space_proportions / sum_space_prop) #依照比例分配，比如0:0:1，如果C分配完，則剩餘會依照1:1分配給A和B
                final_allocs_array += re_alloc
            else:
                valid_indices = np.where(has_space)[0]
                if len(valid_indices) > 0:
                    dist = remaining / len(valid_indices)
                    final_allocs_array[valid_indices] += dist
        
        final_allocs_array = np.minimum(final_allocs_array, caps)
        final_allocs = {name: final_allocs_array[i] for i, name in enumerate(NAMES)}
        
        for name in NAMES:
            self.current_acc[name] += final_allocs[name]

        self.current_day += 1
        terminated = self.current_day > HORIZON
        
        reward = 0.0
        if terminated:
            reward = -self.terminal_cost()

        if not terminated:
            self.current_demand = self._sample_demand()
        else:
            self.current_demand = 0.0

        info = {
            "allocations": final_allocs,
            "step_demand": current_demand
        }
        
        return self._get_obs(), reward, terminated, False, info