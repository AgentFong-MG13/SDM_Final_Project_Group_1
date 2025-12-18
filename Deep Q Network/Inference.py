import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random
from stable_baselines3 import DQN
from model_env import AirCargoDQNEnv, HORIZON, NAMES
import time

# 推論與分析程式碼

def run_inference_analysis(model_path, num_simulations):
    # 載入環境與模型
    env = AirCargoDQNEnv()
    try:
        model = DQN.load(model_path)
        print(f"成功載入模型: {model_path}")
    except FileNotFoundError:
        print(f"找不到模型檔案: {model_path}，請確認檔名是否正確。")
        return

    all_episodes_data = []

    print(f"正在執行 {num_simulations} 次模擬...")
    
    for ep in range(num_simulations):
        obs, _ = env.reset()
        done = False
        episode_log = []
        total_reward = 0
        
        while not done:
            # deterministic=True 代表使用訓練好的最佳策略，不進行隨機探索
            action, _ = model.predict(obs, deterministic=True)
            
            # 執行一步
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            
            # 記錄這一步的資訊
            alloc = info['allocations']
            episode_log.append({
                'Day': env.current_day - 1,
                'Demand': info['step_demand'],
                'Strategy': action.item(), # 策略 ID
                'A_kg': alloc['A'],
                'B_kg': alloc['B'],
                'C_kg': alloc['C']
            })
        
        # 成本是獎勵的負值
        total_cost = -total_reward
        all_episodes_data.append({
            'episode_id': ep,
            'total_cost': total_cost,
            'daily_data': episode_log
        })

    # 排序找出最佳與最差
    sorted_episodes = sorted(all_episodes_data, key=lambda x: x['total_cost'])
    
    best_episode = sorted_episodes[0]
    worst_episode = sorted_episodes[-1]
    avg_cost = np.mean([e['total_cost'] for e in all_episodes_data])

    print("\n" + "="*50)
    print(f"模擬統計 ({num_simulations} 次)")
    print("="*50)
    print(f"平均成本: NTD {avg_cost:,.2f}")
    print(f"最低成本: NTD {best_episode['total_cost']:,.2f}")
    print(f"最高成本: NTD {worst_episode['total_cost']:,.2f}")

    # 輔助函式：將每日數據轉為 DataFrame 並計算累計值
    def process_episode_df(episode_data):
        df = pd.DataFrame(episode_data['daily_data'])
        df.set_index('Day', inplace=True)
        # 計算累積量
        df['Acc_A'] = df['A_kg'].cumsum()
        df['Acc_B'] = df['B_kg'].cumsum()
        df['Acc_C'] = df['C_kg'].cumsum()
        return df

    # 顯示最佳結果
    print("\n" + "="*50)
    print("【最佳分配策略 (Best Case)】")
    print("="*50)
    print(f"總成本: NTD {best_episode['total_cost']:,.2f}")
    df_best = process_episode_df(best_episode)
    # 增加一個總計行方便觀看
    df_best.loc['TOTAL'] = df_best.sum(numeric_only=True)
    # 修正總計行的 Strategy 和累積值顯示
    df_best.loc['TOTAL', 'Strategy'] = np.nan
    df_best.loc['TOTAL', 'Acc_A'] = df_best.loc[14, 'Acc_A']
    df_best.loc['TOTAL', 'Acc_B'] = df_best.loc[14, 'Acc_B']
    df_best.loc['TOTAL', 'Acc_C'] = df_best.loc[14, 'Acc_C']
    
    # 簡單的策略說明對照
    # strategy_desc = {
    #     0: "全A", 1: "全B", 2: "全C", 3: "A,B半", 4: "A,C半", 5: "B,C半", 
    #     6: "均分", 7: "0.2/0.4/0.4", 8: "0.4/0.4/0.2", 9: "0.4/0.2/0.4"
    # }
    
    # 格式化輸出
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 0)
    
    print(df_best[['Demand', 'Strategy', 'A_kg', 'B_kg', 'C_kg', 'Acc_A', 'Acc_B', 'Acc_C']])
    print("-" * 30)
    # print("策略代碼對照:", strategy_desc)

    # 顯示最差結果
    print("\n" + "="*50)
    print("【最差分配策略 (Worst Case - 通常是隨機需求極高導致)】")
    print("="*50)
    print(f"總成本: NTD {worst_episode['total_cost']:,.2f}")
    df_worst = process_episode_df(worst_episode)
    df_worst.loc['TOTAL'] = df_worst.sum(numeric_only=True)
    df_worst.loc['TOTAL', 'Strategy'] = np.nan
    df_worst.loc['TOTAL', 'Acc_A'] = df_worst.loc[14, 'Acc_A']
    df_worst.loc['TOTAL', 'Acc_B'] = df_worst.loc[14, 'Acc_B']
    df_worst.loc['TOTAL', 'Acc_C'] = df_worst.loc[14, 'Acc_C']
    
    print(df_worst[['Demand', 'Strategy', 'A_kg', 'B_kg', 'C_kg', 'Acc_A', 'Acc_B', 'Acc_C']])

    # 儲存最佳解到 CSV
    df_best.to_csv("best_allocation_strategy.csv")
    print("\n[Info] 最佳分配策略已儲存至 best_allocation_strategy.csv")

if __name__ == "__main__":
    start_time = time.time()
    num_simulations = 1000
    run_inference_analysis("dqn_policy.zip", num_simulations)
    end_time = time.time()
    print(f"模擬{num_simulations}次共耗時: {end_time - start_time:.2f} 秒")
