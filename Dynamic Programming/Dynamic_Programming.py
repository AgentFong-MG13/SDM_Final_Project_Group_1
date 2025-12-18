'''
- 需求為 n 個離散點 (需為 a 的倍數，發生率加總為 1)
- 決策以 a 為單位

- initial_expected_cost 最佳決策下的期望成本
- demo_demands 用一條具體路徑測試 policy 的行為
'''

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Dict
import time
import random

# ============================
# 基本設定：時間、單位、需求分佈
# ============================

UNIT = 1000         # 決策 / 狀態 使用的重量「一單位」(kg)
HORIZON = 14
DEMAND_LEVELS = [5000.0, 10000.0, 15000.0, 20000.0, 25000.0]
DEMAND_PROBS  = [ 0.339, 0.244, 0.188, 0.132, 0.097]


@dataclass
class Airline:
    """
    表示一家航空公司：
    - name: 名稱 (A/B/C)
    - unit_cost: 未折扣前的每公斤運費
    - daily_capacity: 每天最多可運的重量 (kg / day)
    - discount_thresholds: 折扣門檻列表，格式為 (threshold_kg, discount_factor)
      例如 [(10000, 0.95), (20000, 0.9)] 表示：
        若累積量 >  0: 暫時 factor=1.0
        若累積量 > 10000: factor=0.95
        若累積量 > 20000: factor=0.90
    """
    name: str
    unit_cost: float
    daily_capacity: float          # kg / day
    discount_thresholds: List[Tuple[float, float]]  # (threshold_kg, discount_factor)

    def get_discount_factor(self, total_weight: float) -> float:
        """
        給定期末累積運量 total_weight (kg)，回傳對應折扣因子。
        規則：按 threshold 由小到大排序，當 total_weight > thr 時更新 factor。
             最後留下的是滿足最高門檻的折扣。
        """
        factor = 1.0  # 預設不打折

        # 先依門檻重量排序，避免打亂順序
        for thr, disc in sorted(self.discount_thresholds, key=lambda x: x[0]):
            if total_weight > thr:
                factor = disc
            else:
                # 一旦 total_weight 沒有超過當前門檻，就可以結束
                break
        return factor


def build_airlines() -> Dict[str, Airline]:
    """
    建立三家航空公司的參數
    回傳 dict: key 是 'A'/'B'/'C', value 是 Airline 物件
    """
    return {
        "A": Airline(
            name="A",
            unit_cost=45,
            daily_capacity=15400,
            discount_thresholds=[
                (50000, 0.97),
                (80000, 0.96),
                (100000, 0.95),
                (150000, 0.92),
            ],
        ),
        "B": Airline(
            name="B",
            unit_cost=50,
            daily_capacity=9900,
            discount_thresholds=[
                (50000, 0.97),
                (100000, 0.95),
            ],
        ),
        "C": Airline(
            name="C",
            unit_cost=45,
            daily_capacity=11000,
            discount_thresholds=[
                (50000, 0.98),
                (100000, 0.95),
                (150000, 0.93),
            ],
        ),
    }

# 建立全域的航空公司字典，和其名稱列表
AIRLINES = build_airlines()
NAMES = list(AIRLINES.keys())


# ============================
# 工具：終端成本 (terminal cost)
# ============================

def terminal_cost_from_units(
    w_units: Dict[str, int],
    unit: float = UNIT,
) -> float:
    """
    在期末 (t > HORIZON) 時根據累積重量（以 UNIT 為單位的整數），
    計算折扣後的「總成本」。

    參數：
    - w_units: 各家航空公司累積的重量(單位數)，例如 {"A": 10, "B": 5, "C": 3}
               代表 A=10*UNIT kg, B=5*UNIT kg...
    - unit   : 每一單位的公斤數（預設=1000)

    回傳：
    - total_cost: 加總三家折扣後的運費
    """
    total_cost = 0.0
    for name in NAMES:
        airline = AIRLINES[name]
        w_kg = w_units[name] * unit                     # 單位數 * UNIT → 換成 kg
        disc = airline.get_discount_factor(w_kg)        # 依累積量取得折扣係數
        total_cost += airline.unit_cost * w_kg * disc   # 總成本 = 單價 * 重量 * 折扣
    return total_cost


# ============================
# 行動集合：以 UNIT 為步長，枚舉所有 (A,B,C)
# ============================

def generate_actions_by_unit(
    demand_kg: float,
    airlines: Dict[str, Airline],
    unit: float = UNIT,
) -> List[Dict[str, float]]:
    """
    用 UNIT 當一格，枚舉所有對三家航空公司 A/B/C 的離散分配。

    例如：
    - demand=8000, unit=1000 → d_units=8
    - 枚舉所有 (kA, kB, kC) 是非負整數，且 kA+kB+kC=8
      並且  kA≤A 當日容量(單位數), kB≤B 當日容量, kC≤C 當日容量

    回傳：
    - actions: List[Dict[str,float]]，每個元素是一個 dict
               如: {"A": 2000.0, "B": 3000.0, "C": 3000.0}
    """
    # 將 demand 換成「單位數」整數
    d_units = int(round(demand_kg / unit))

    # 每家今天最多可拿幾格 unit (由 daily_capacity 限制）
    cap_units = {
        name: int(airline.daily_capacity // unit)
        for name, airline in airlines.items()
    }

    actions: List[Dict[str, float]] = []

    # 這裡假設只有 A, B, C 三家（若未來擴充公司數，需改成遞迴枚舉）
    maxA = min(cap_units["A"], d_units)
    for kA in range(maxA + 1):
        remaining_after_A = d_units - kA
        maxB = min(cap_units["B"], remaining_after_A)
        for kB in range(maxB + 1):
            # 由總量限制，kC = 剩下的單位
            kC = remaining_after_A - kB
            # 需確認 C 的容量也足夠
            if 0 <= kC <= cap_units["C"]:
                # 轉回 kg
                alloc = {
                    "A": kA * unit,
                    "B": kB * unit,
                    "C": kC * unit,
                }
                actions.append(alloc)

    return actions


# ============================
# Value function: V(t, wA_units, wB_units, wC_units)
# ============================

@lru_cache(maxsize=None)
def optimal_expected_cost(
    t: int,
    wA_units: int,
    wB_units: int,
    wC_units: int,
) -> float:
    """
    V(t, wA_units, wB_units, wC_units)
    = 第 t 天開始（尚未處理 day t) 時，
      在目前累積重量（以 UNIT 為單位）為 wA_units, wB_units, wC_units 的情況下，
      從第 t 天一路做最優決策到期末的「最小期望成本」。

    狀態說明：
    - t: 目前要決策的是第 t 天 (1..HORIZON)。若 t > HORIZON, 表示已過完所有天。
    - wA_units: A 的累積運量 (unit 數)，實際重量 = wA_units * UNIT
    - wB_units、wC_units 同理。
    """

    # 印出 debug 方便檢查遞迴深度（可視情況關掉）
    #print("[Debug] t = " + str(t))

    # ====== 終端條件：超過最後一天，直接算期末折扣成本 ======
    if t > HORIZON:
        w_units = {"A": wA_units, "B": wB_units, "C": wC_units}
        return terminal_cost_from_units(w_units, unit=UNIT)

    # ====== 一般情況：還有第 t 天要決策 ======
    expected_value = 0.0

    # 對需求的期望（對每一種可能 demand_kg，乘上對應的機率 prob）
    for demand_kg, prob in zip(DEMAND_LEVELS, DEMAND_PROBS):

        #print("[Debug] demand_kg, prob = " + str(demand_kg) + " " + str(prob))

        # 為這個 demand 枚舉所有可行 action
        actions = generate_actions_by_unit(demand_kg, AIRLINES, unit=UNIT)
        
        # best_for_this_d：在這個特定需求量下，最好的（最小）未來成本
        best_for_this_d = float("inf")

        for alloc in actions:
            xA_kg = alloc["A"]
            xB_kg = alloc["B"]
            xC_kg = alloc["C"]

            # 更新累積（以 UNIT 為單位）
            new_wA_units = wA_units + int(round(xA_kg / UNIT))
            new_wB_units = wB_units + int(round(xB_kg / UNIT))
            new_wC_units = wC_units + int(round(xC_kg / UNIT))

            # 所以這裡只有「未來的期望成本(future value)」。
            future_cost = optimal_expected_cost(t + 1, new_wA_units, new_wB_units, new_wC_units)

            # 在這個 demand_kg 下，只取最小的 future_cost
            if future_cost < best_for_this_d:
                best_for_this_d = future_cost

        # 對 demand 做期望：sum_d [ prob(d) * bestCost(d) ]
        expected_value += prob * best_for_this_d

    return expected_value


# ============================
# 給定實際 demand 時的「最佳行動」
# ============================

def best_action_given_demand(
    t: int,
    wA_units: int,
    wB_units: int,
    wC_units: int,
    demand_kg: float,
) -> Dict[str, float]:
    """
    真實運行時，你會知道今天的 demand_kg。
    這個函式在 state=(t, wA_units, wB_units, wC_units) 下，
    針對「已知的 demand_kg」找出最好的分配 (A,B,C)。

    傳回值：
    - best_alloc: dict, 格式 {"A": xA_kg, "B": xB_kg, "C": xC_kg}
    """
    # 先列出所有對於這個 demand_kg 的可行分配
    actions = generate_actions_by_unit(demand_kg, AIRLINES, unit=UNIT)

    best_cost = float("inf")
    best_alloc = None

    for alloc in actions:
        xA_kg = alloc["A"]
        xB_kg = alloc["B"]
        xC_kg = alloc["C"]

        # 更新累積重量（以 unit 表示）
        new_wA_units = wA_units + int(round(xA_kg / UNIT))
        new_wB_units = wB_units + int(round(xB_kg / UNIT))
        new_wC_units = wC_units + int(round(xC_kg / UNIT))

        # 未來的期望成本：從 t+1 開始到期末
        future_cost = optimal_expected_cost(t + 1, new_wA_units, new_wB_units, new_wC_units)

        # 找出使 future_cost 最小的那個行動
        if future_cost < best_cost:
            best_cost = future_cost
            best_alloc = alloc

    return best_alloc


# ============================
# 觀察 MDP 決策
# ============================

def mdp_policy(
    state: Dict[str, float],     # 累積重量 (kg)
    demand: float,               # 今天的需求 (kg)
    day: int,
) -> Dict[str, float]:
    """
    給 Baseline/Rollout 同樣格式的介面：
      state: {"A":累積kg, "B":..., "C":...}
      demand: 今天需求 (kg)
      day: 今天是第幾天 (1~HORIZON)

    回傳: {"A":今天分配kg, "B":..., "C":...}
    """
    # 1) 把累積重量換成以 UNIT 為單位的整數
    wA_units = int(round(state.get("A", 0.0) / UNIT))
    wB_units = int(round(state.get("B", 0.0) / UNIT))
    wC_units = int(round(state.get("C", 0.0) / UNIT))

    # 2) 為了跟 DP 對齊，把 demand 也 round 到 UNIT
    demand_rounded = round(demand / UNIT) * UNIT

    # 3) 用 DP 算出這個 state + demand 下的最佳分配
    alloc = best_action_given_demand(
        t=day,
        wA_units=wA_units,
        wB_units=wB_units,
        wC_units=wC_units,
        demand_kg=demand_rounded,
    )

    return alloc  # {"A":kg, "B":kg, "C":kg}


# ============================
# Demo：計算 V(1,0,0,0)，並用一條 demand 路徑模擬決策
# ============================

def main():

    # 1. 算初始 state 的最小期望成本
    print("[Debug] 算初始 state 的最小期望成本")

    # 時間紀錄
    start_time = time.perf_counter()

    # V(1,0,0,0)：從第 1 天開始，三家累積貨量都是 0 時，
    # 若每天的 demand 依 (DEMAND_LEVELS, DEMAND_PROBS) 分佈，
    # 並且每天都做最優決策，最終的「期望運費成本」。
    initial_expected_cost  = optimal_expected_cost(1, 0, 0, 0) # 最佳決策下的期望成本

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Initial Expected Cost(1, wA=0, wB=0, wC=0) = {initial_expected_cost :.2f}")
    print(f"Time to solve Initial Expected Cost = {elapsed:.4f} seconds")

    # 2. 示範一條具體路徑：
    #    這裡純粹用「每天 demand 都是 8000 kg」來測試 policy 的行為。
    #    (你可以改成真的 sample）
    demo_demands = [
        random.choices(DEMAND_LEVELS, weights=DEMAND_PROBS)[0]
        for _ in range(HORIZON)
]

    # 初始的累積重量（以 UNIT 為單位）都為 0
    wA_units = wB_units = wC_units = 0

    print("\n=== One greedy path under DP policy (demand = 8000kg each day) ===")
    for t in range(1, HORIZON + 1):
        #print("[Debug] t = " + str(t))
        d = demo_demands[t - 1]

        # 在當前 state 下，針對「已知今天 demand=d」找最佳分配
        alloc = best_action_given_demand(t, wA_units, wB_units, wC_units, d)

        xA = alloc["A"]
        xB = alloc["B"]
        xC = alloc["C"]

        print(
            f"Day {t:2d} | demand={d:5.0f}kg | "
            f"A={xA:5.0f}kg, B={xB:5.0f}kg, C={xC:5.0f}kg"
        )

        # 更新 state（累積重量以 UNIT 為單位）
        wA_units += int(round(xA / UNIT))
        wB_units += int(round(xB / UNIT))
        wC_units += int(round(xC / UNIT))

    # 根據這條實際路徑的累積量，計算期末總成本（折扣後）
    final_cost = terminal_cost_from_units(
        {"A": wA_units, "B": wB_units, "C": wC_units},
        unit=UNIT,
    )
    print(f"\nTerminal cost for this path = {final_cost:.2f}")


if __name__ == "__main__":
    main()
