import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random
import time
import torch
from stable_baselines3 import DQN  # 改用 DQN
from stable_baselines3.common.env_util import make_vec_env
from model_env import AirCargoDQNEnv, NAMES

# 模擬與驗證 (DQN 版)
def run_simulation(model, env, num_episodes):
    all_costs = []
    daily_allocations_df = None

    for i in range(num_episodes):
        # --- Reset 處理 ---
        reset_out = env.reset()
        
        # 判斷是 Gym 還是 VecEnv
        if isinstance(reset_out, tuple): 
            # 標準 Gym API 回傳 (obs, info)
            obs = reset_out[0]
        else:
            # VecEnv API 通常只回傳 obs
            obs = reset_out

        # 如果是 VecEnv (n_envs > 1)，obs 會是多維陣列，我們只取第一個環境
        # 修正邏輯：只要它是陣列且維度符合 batch 特徵，就取 [0]
        if isinstance(obs, np.ndarray) and obs.ndim > 1: 
             obs = obs[0]

        done = False
        total_reward = 0
        episode_allocs = []

        while not done:
            # Predict
            action, _ = model.predict(obs, deterministic=True)

            # Step
            step_out = env.step(action)
            
            # 統一處理回傳值格式 (Gym 0.26+ vs Old Gym vs VecEnv)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
            else:
                next_obs, reward, done_flag, info = step_out
                terminated = done_flag
                truncated = False

            # === 關鍵修正開始：處理向量化環境的回傳值 ===
            # 檢查 reward 是否為陣列/列表 (代表這是 VecEnv)
            if isinstance(reward, (np.ndarray, list)):
                # 強制只取第 0 個環境的數據
                reward = reward[0]
                terminated = terminated[0]
                truncated = truncated[0]
                info = info[0]
                
                # 處理 Observation
                if isinstance(next_obs, np.ndarray):
                    next_obs = next_obs[0]
            # === 關鍵修正結束 ===

            # 更新 obs
            obs = next_obs

            # 現在 terminated 已經是純布林值 (Scalar)，可以安全轉換
            done = bool(terminated) or bool(truncated)
            total_reward += reward

            # 只記錄第一次模擬 (i==0) 的詳細數據
            if i == 0:
                alloc = info.get('allocations', {}).copy()
                alloc['Day'] = env.current_day - 1
                alloc['Demand'] = info.get('step_demand', 0.0)
                
                # action 可能是陣列，轉回 int
                if isinstance(action, (np.ndarray, list)):
                    real_action = int(action) if np.ndim(action) == 0 else int(action[0])
                else:
                    real_action = int(action)
                    
                alloc['Strategy_Idx'] = real_action
                episode_allocs.append(alloc)

        final_cost = -total_reward
        all_costs.append(final_cost)

        if i == 0:
            daily_allocations_df = pd.DataFrame(episode_allocs)
            daily_allocations_df.set_index('Day', inplace=True)
            daily_allocations_df['Acc_A'] = daily_allocations_df['A'].cumsum()
            daily_allocations_df['Acc_B'] = daily_allocations_df['B'].cumsum()
            daily_allocations_df['Acc_C'] = daily_allocations_df['C'].cumsum()
            
            # 建立 Total Row
            daily_allocations_df.loc['Total'] = np.nan
            cols_sum = ['A', 'B', 'C', 'Demand']
            daily_allocations_df.loc['Total', cols_sum] = daily_allocations_df[cols_sum].sum()
            
            # 填補累積量與總成本
            daily_allocations_df.loc['Total', 'Acc_A'] = daily_allocations_df['Acc_A'].iloc[-2]
            daily_allocations_df.loc['Total', 'Acc_B'] = daily_allocations_df['Acc_B'].iloc[-2]
            daily_allocations_df.loc['Total', 'Acc_C'] = daily_allocations_df['Acc_C'].iloc[-2]
            
            daily_allocations_df['Total_Cost'] = np.nan
            daily_allocations_df.loc['Total', 'Total_Cost'] = final_cost

    avg_cost = np.mean(all_costs)

    print("\n==============================================")
    print(f"DQN 執行 {num_episodes} 次模擬結果：")
    print(f"平均成本：NTD {avg_cost:,.2f} 元")
    print("==============================================")
    print("\n一次模擬的每日最佳分配策略 (與策略代碼)：")

    try:
        # 如果有安裝 tabulate 套件，顯示會更漂亮
        print(daily_allocations_df.round(0).to_markdown(numalign="left", stralign="left"))
    except (ImportError, AttributeError):
        print(daily_allocations_df.round(0))

    daily_allocations_df.to_csv('dqn_daily_allocation_results.csv')

# 執行測試
if __name__ == "__main__":
    # 訓練設定
    TOTAL_TIMESTEPS = 6000000 # 原本300000，一天就是一步，最佳6000000
    train_env = make_vec_env(AirCargoDQNEnv, n_envs=16) # 設成8的推論效果好像比較好
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"開始訓練 DQN 模型，總步數: {TOTAL_TIMESTEPS}...")
    print(f"使用設備: {DEVICE}")
    start_time = time.time()

    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=1e-4,
        buffer_size=1200000,#原本50000
        learning_starts=100000,#原本5000，最佳100000
        batch_size=512, #原本128
        tau=1.0,
        gamma=1.0,
        exploration_fraction=0.3,#原本0.5
        exploration_final_eps=0.05,
        target_update_interval=10000, #target network 更新頻率
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=0,
        device=DEVICE
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("dqn_policy.zip")
    end_time = time.time()
    print(f"訓練完成！耗時: {end_time - start_time:.2f} 秒")
    test_env = AirCargoDQNEnv()
    run_simulation(model, test_env, 1000)
