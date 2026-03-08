from functools import lru_cache
import random
from dataclasses import dataclass
import os
import statistics


def _linspace(start: float, stop: float, num: int):
    if num <= 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _analytic_expected_time_seconds(
    *,
    n_: int,
    t_s_: float,
    t_q_: float,
    t_m_: float,
    p_pre_: float,
    p_a1_: float,
    gamma_: float,
    p_q_true_: float,
    p_q_false_: float,
    eps: float = 1e-12,
    max_scans: int = 500,
):
    """解析计算 E[T] 与 E[#scans]（与 expected_time() 的逻辑一致，但参数化且不依赖全局变量）。"""

    def p_a_i(i: int) -> float:
        return p_a1_ * (gamma_ ** (i - 1))

    def per_scan_stats(*, order_present: bool):
        if not order_present:
            e_scan = n_ * (t_s_ + p_q_false_ * t_q_)
            return e_scan, 0.0

        s_scan = p_q_true_
        e_success = ((n_ + 1) / 2) * t_s_ + ((n_ - 1) / 2) * (p_q_false_ * t_q_) + t_q_
        e_fail = n_ * t_s_ + (n_ - 1) * (p_q_false_ * t_q_)
        e_scan = s_scan * e_success + (1 - s_scan) * e_fail
        return e_scan, s_scan

    def expected_given_presence(order_present: bool):
        reach_prob = 1.0
        total = 0.0
        expected_scans = 0.0
        phone_prob = 0.0
        success_in_scan_prob = 0.0

        for i in range(1, max_scans + 1):
            if reach_prob < eps:
                break

            expected_scans += reach_prob

            scan_e, scan_s = per_scan_stats(order_present=order_present)
            total += reach_prob * scan_e
            success_in_scan_prob += reach_prob * scan_s

            fail_prob = reach_prob * (1 - scan_s)
            take_phone_prob = fail_prob * (1 - p_a_i(i))
            if take_phone_prob > 0:
                total += take_phone_prob * (t_m_ + t_q_)
                phone_prob += take_phone_prob

            reach_prob = fail_prob * p_a_i(i)

        return {
            "expected_seconds": total,
            "expected_scans": expected_scans,
            "prob_phone": phone_prob,
            "prob_success_in_scan": success_in_scan_prob,
            "prob_reach_trunc_tail": reach_prob,
        }

    present = expected_given_presence(True)
    stolen = expected_given_presence(False)

    mixed_expected = p_pre_ * present["expected_seconds"] + (1 - p_pre_) * stolen["expected_seconds"]
    mixed_expected_scans = p_pre_ * present["expected_scans"] + (1 - p_pre_) * stolen["expected_scans"]

    return {
        "expected_seconds": mixed_expected,
        "expected_scans": mixed_expected_scans,
        "present": present,
        "stolen": stolen,
    }

n = 125  # 假设外卖架上有100到150份外卖

t_s = 0.1  # 假设扫描时间在0.1s到0.3s之间，取中间值
t_q = 6  # 假设查询时间在6s到8s之间，取中间值
t_m = 25  # 假设拿出手机获取外卖位置的时间为25s

p_pre = 0.95  # 假设外卖没被偷的先验概率为0.95
p_a1 = 0.5  # 假设第一次扫描后再次扫描的概率为0.5

# 扫描模型：
# - 扫到“正确外卖”时触发查询的概率
# - 扫到“错误外卖”时误触发查询的概率
p_q_true = 0.40
p_q_false = 0.01

gamma = 0.9  # 每次扫描后再次扫描的概率下降的程度

# 概率函数递归公式
@lru_cache(maxsize=None)
def p_a(i):
    if i == 1:
        return p_a1
    else:
        return p_a(i-1) * gamma
    
def expected_time():
    """估算总期望耗时（单位：秒）。
    """

    def per_scan_stats(i: int, *, order_present: bool):
        """返回(本次扫描的期望耗时, 本次扫描内成功概率)。

        模型：
        - 若外卖存在：扫到正确外卖时，以 p_q_true 查询并立刻成功；否则继续。
          在错误外卖处，以 p_q_false 概率误查询（耗时 t_q）但不会结束。
        - 若外卖不存在：所有外卖都是“错误外卖”，只能浪费时间，成功概率为 0。

        
        """
        if not order_present:
            # 外卖被偷：所有 n 个位置都是错误外卖
            e = n * (t_s + p_q_false * t_q)
            return e, 0.0

        # 外卖存在：正确外卖在本次扫描序列中的位置视为均匀随机 1..n
        s = p_q_true

        # 成功时：终止在正确外卖处（期望位置 (n+1)/2）
        e_success = ((n + 1) / 2) * t_s + ((n - 1) / 2) * (p_q_false * t_q) + t_q

        # 失败时：完整扫完 n 个；只有 n-1 个错误外卖会产生误查询
        e_fail = n * t_s + (n - 1) * (p_q_false * t_q)

        e = s * e_success + (1 - s) * e_fail
        return e, s

    def expected_given_presence(order_present: bool):
        # reach_prob: 到达“第 i 次扫描开始”的概率
        reach_prob = 1.0
        total = 0.0
        phone_prob = 0.0
        success_in_scan_prob = 0.0
        expected_scans = 0.0

        # 用一个截断迭代做无限和（尾项足够小就停）
        eps = 1e-12
        max_scans = 500

        for i in range(1, max_scans + 1):
            if reach_prob < eps:
                break

            # 只要到达了第 i 次扫描开始，就算进行了一次扫描
            expected_scans += reach_prob

            scan_e, scan_s = per_scan_stats(i, order_present=order_present)
            total += reach_prob * scan_e
            success_in_scan_prob += reach_prob * scan_s

            # 扫描失败后（未结束）才会做“再扫/掏手机”决策
            fail_prob = reach_prob * (1 - scan_s)
            take_phone_prob = fail_prob * (1 - p_a(i))
            if take_phone_prob > 0:
                total += take_phone_prob * (t_m + t_q)
                phone_prob += take_phone_prob

            # 进入下一次扫描
            reach_prob = fail_prob * p_a(i)

        return {
            "expected_seconds": total,
            "prob_phone": phone_prob,
            "prob_success_in_scan": success_in_scan_prob,
            "expected_scans": expected_scans,
            "prob_reach_trunc_tail": reach_prob,
            "scans_iterated": i,
        }

    present = expected_given_presence(True)
    stolen = expected_given_presence(False)
    mixed_expected = p_pre * present["expected_seconds"] + (1 - p_pre) * stolen["expected_seconds"]
    mixed_expected_scans = p_pre * present["expected_scans"] + (1 - p_pre) * stolen["expected_scans"]

    return {
        "expected_seconds": mixed_expected,
        "expected_scans": mixed_expected_scans,
        "present": present,
        "stolen": stolen,
    }


@dataclass(frozen=True)
class SimulationResult:
    total_seconds: float
    scans: int
    order_present: bool
    used_phone: bool
    found_during_scan: bool


def simulate_once(rng: random.Random) -> SimulationResult:
    """模拟一次拿外卖流程，返回总耗时与扫描轮数等信息。"""
    order_present = rng.random() < p_pre

    total = 0.0
    scans = 0
    used_phone = False
    found_during_scan = False

    i = 1
    while True:
        scans += 1

        if order_present:
            # 正确外卖在 1..n 均匀随机
            correct_pos = rng.randint(1, n)
        else:
            correct_pos = None

        success = False
        for pos in range(1, n + 1):
            total += t_s

            if order_present and pos == correct_pos:
                # 扫到正确外卖：以 p_q_true 触发查询，查询则成功
                if rng.random() < p_q_true:
                    total += t_q
                    success = True
                    break
            else:
                # 扫到错误外卖：以 p_q_false 误触发查询（必失败）
                if rng.random() < p_q_false:
                    total += t_q

        if success:
            found_during_scan = True
            break

        # 本轮扫描没找到：决定再扫或掏手机
        if rng.random() < p_a(i):
            i += 1
            continue

        used_phone = True
        total += t_m + t_q
        break

    return SimulationResult(
        total_seconds=total,
        scans=scans,
        order_present=order_present,
        used_phone=used_phone,
        found_during_scan=found_during_scan,
    )


def monte_carlo(trials: int = 1000, seed: int = 0, *, threshold_seconds: float | None = None):
    rng = random.Random(seed)
    results = [simulate_once(rng) for _ in range(trials)]
    times = [r.total_seconds for r in results]
    scan_counts = [r.scans for r in results]
    phone_rate = sum(1 for r in results if r.used_phone) / trials
    present_rate = sum(1 for r in results if r.order_present) / trials
    found_rate = sum(1 for r in results if r.found_during_scan) / trials

    below_threshold_count = None
    below_threshold_rate = None
    if threshold_seconds is not None:
        below_threshold_count = sum(1 for t in times if t < threshold_seconds)
        below_threshold_rate = below_threshold_count / trials

    return {
        "trials": trials,
        "seed": seed,
        "threshold_seconds": threshold_seconds,
        "below_threshold_count": below_threshold_count,
        "below_threshold_rate": below_threshold_rate,
        "results": results,
        "times": times,
        "scan_counts": scan_counts,
        "mean_time": sum(times) / trials,
        "median_time": statistics.median(times),
        "mean_scans": sum(scan_counts) / trials,
        "phone_rate": phone_rate,
        "present_rate": present_rate,
        "found_rate": found_rate,
    }


def plot_monte_carlo(
    mc,
    *,
    bins: int = 30,
    out_path: str | None = None,
    show: bool = True,
    threshold_seconds: float | None = None,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "缺少 matplotlib，无法画图。请先安装：pip install matplotlib"
        ) from e

    times = mc["times"]
    scan_counts = mc["scan_counts"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(times, bins=bins)
    axes[0].set_title(f"Total time histogram (N={mc['trials']})")
    axes[0].set_xlabel("seconds")
    axes[0].set_ylabel("count")

    if threshold_seconds is not None:
        axes[0].axvline(threshold_seconds, linestyle="--", linewidth=2)
        axes[0].text(
            threshold_seconds,
            axes[0].get_ylim()[1] * 0.95,
            f"x={threshold_seconds}",
            rotation=90,
            va="top",
            ha="right",
        )

    axes[1].hist(scan_counts, bins=range(1, max(scan_counts) + 2), align="left", rwidth=0.9)
    axes[1].set_title("Scan rounds histogram")
    axes[1].set_xlabel("#scans")
    axes[1].set_ylabel("count")

    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), f"monte_carlo_{mc['trials']}.png")
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_expected_time_sweeps(*, threshold_seconds: float = 32):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "缺少 matplotlib，无法画图。请先安装：pip install matplotlib"
        ) from e

    base = {
        "n_": n,
        "t_s_": t_s,
        "t_q_": t_q,
        "t_m_": t_m,
        "p_pre_": p_pre,
        "p_a1_": p_a1,
        "gamma_": gamma,
        "p_q_true_": p_q_true,
        "p_q_false_": p_q_false,
    }

    def eval_expected(**overrides):
        params = dict(base)
        params.update(overrides)
        return _analytic_expected_time_seconds(**params)["expected_seconds"]

    sweeps = [
        ("n", list(range(50, 150, 5)), lambda v: {"n_": int(v)}),
        ("t_s", _linspace(0.01, 0.30, 40), lambda v: {"t_s_": float(v)}),
        ("t_q", _linspace(2.0, 8.0, 40), lambda v: {"t_q_": float(v)}),
        ("p_q_true", _linspace(0.05, 0.95, 46), lambda v: {"p_q_true_": float(v)}),
        ("p_q_false", _linspace(0.0, 0.05, 51), lambda v: {"p_q_false_": float(v)}),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes_flat = [ax for row in axes for ax in row]

    for idx, (name, values, override_fn) in enumerate(sweeps):
        ax = axes_flat[idx]
        ys = [eval_expected(**override_fn(v)) for v in values]

        ax.plot(values, ys)
        ax.axhline(threshold_seconds, linestyle="--", linewidth=2)
        ax.set_title(f"E[T] vs {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("seconds")

        # 标出当前基准点
        base_x = {
            "n": base["n_"],
            "t_s": base["t_s_"],
            "t_q": base["t_q_"],
            "p_q_true": base["p_q_true_"],
            "p_q_false": base["p_q_false_"],
        }[name]
        base_y = eval_expected()
        ax.scatter([base_x], [base_y])

    # 最后一个子图留空
    axes_flat[-1].axis("off")

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "expected_time_sweeps.png")
    fig.savefig(out_path, dpi=180)
    print(f"Saved sweep plot to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    result = expected_time()
    e = result["expected_seconds"]
    print(f"E[T] ≈ {e:.2f} s  ({e/60:.2f} min)")
    print(f"E[#scans] ≈ {result['expected_scans']:.4f}")

    print(f"params: n={n}, t_s={t_s}, t_q={t_q}, t_m={t_m}, p_pre={p_pre}")
    print(f"scan model: p_q_true={p_q_true}, p_q_false={p_q_false}; rescan: p_a1={p_a1}, gamma={gamma}")

    p = result["present"]
    s = result["stolen"]
    print("-- condition on present (not stolen) --")
    print(f"E[T|present] ≈ {p['expected_seconds']:.2f} s  ({p['expected_seconds']/60:.2f} min)")
    print(f"E[#scans|present] ≈ {p['expected_scans']:.4f}")
    print(f"P(phone|present) ≈ {p['prob_phone']:.4f}; P(success during scans|present) ≈ {p['prob_success_in_scan']:.4f}")
    print("-- condition on stolen --")
    print(f"E[T|stolen] ≈ {s['expected_seconds']:.2f} s  ({s['expected_seconds']/60:.2f} min)")
    print(f"E[#scans|stolen] ≈ {s['expected_scans']:.4f}")
    print(f"P(phone|stolen) ≈ {s['prob_phone']:.4f}")

    # 蒙特卡洛仿真 + 可视化
    trials = 10000
    seed = 0
    threshold_seconds = 32
    mc = monte_carlo(trials=trials, seed=seed, threshold_seconds=threshold_seconds)
    print("-- monte carlo --")
    print(f"MC mean time ≈ {mc['mean_time']:.2f} s  ({mc['mean_time']/60:.2f} min)")
    print(f"MC median time ≈ {mc['median_time']:.2f} s  ({mc['median_time']/60:.2f} min)")
    print(f"MC mean scans ≈ {mc['mean_scans']:.4f}")
    print(f"MC phone rate ≈ {mc['phone_rate']:.4f}; present rate ≈ {mc['present_rate']:.4f}; found rate ≈ {mc['found_rate']:.4f}")
    if mc["below_threshold_count"] is not None:
        print(
            f"MC P(T<{threshold_seconds}s) ≈ {mc['below_threshold_rate']:.4f} "
            f"({mc['below_threshold_count']}/{mc['trials']})"
        )

    try:
        # 在部分终端/无 GUI 环境下 show() 可能不可见，因此默认保存图片
        plot_monte_carlo(mc, show=False, threshold_seconds=threshold_seconds)
    except RuntimeError as err:
        print(str(err))

    try:
        plot_expected_time_sweeps(threshold_seconds=threshold_seconds)
    except RuntimeError as err:
        print(str(err))

