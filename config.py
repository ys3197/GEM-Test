"""Shared configuration: domains, paths, and the constants the pipeline agrees on."""

from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"

# 四个 Amazon 品类，当作四个 "surface"。
#
# 选择标准是 `python -m data.probe` 量出来的，不是文件大小。Amazon Reviews 的
# 稀疏程度和体积无关：All_Beauty 有 70 万条评论、63 万个用户，平均每人 1.11 次，
# k-core(5) 之后只剩 357 个用户——一个没有序列的域，序列模型无从学起。
#
#   domain                     rows/user   k5 users
#   All_Beauty                   1.11          357   ✗ 一次性购买为主
#   Handmade_Products            1.13           89   ✗
#   Digital_Music                1.29           20   ✗
#   Software                     1.88      149,625   ✓
#   Video_Games                  1.67       98,906   ✓
#   Musical_Instruments          1.71       59,941   ✓
#   Industrial_and_Scientific    1.51       54,567   ✓
#
# 四个入选域的买家类型和复购节奏各不相同——数字商品 / 数字娱乐 / 实体爱好品 /
# B2B 耗材。若四个域太像，"跨域学习"和"合并成一个域"就没有区别，
# 迁移实验的对照会失去意义。
DOMAINS = [
    "Software",                   # ~1.74 GB  数字商品，无物流，偏 B2B
    "Video_Games",                # ~2.50 GB  数字娱乐，高单价，强季节性
    "Musical_Instruments",        # ~1.45 GB  实体爱好品，长决策周期
    "Industrial_and_Scientific",  # ~2.19 GB  B2B 耗材，补货型复购
]

# 只跑通管道时用（~1.74 GB，四个域里最小的一个）
DOMAINS_SMOKE = ["Software"]

# ── k-core 过滤 ────────────────────────────────────────────────
# 推荐系统的标准做法：反复丢掉交互过少的用户和商品，直到收敛。
# 目的不是"清洗脏数据"，而是保证每个用户都有一条**够长的序列**可学。
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

# ── 序列 ──────────────────────────────────────────────────────
# 单个样本喂给序列模型的最大事件数。32 是 analysis/padding_waste.py 量出来的：
# 四个域的 p99 都在 30 左右，取 32 覆盖约 99% 的用户而不至于补出大量空位。
MAX_SEQ_LEN = 32

RANDOM_SEED = 42

# ── 时间差分桶 ────────────────────────────────────────────────
# 历史事件距离候选事件多久。分桶而不是喂浮点，理由和价格一样：
# 事件模型要把它当成一个可嵌入的属性，和商品属性一起线性压缩。
# 边界取得不均匀是有意的——"1 小时前"和"2 小时前"的差别，
# 远大于"3 年前"和"3 年零 1 小时前"。
TIME_GAP_BUCKETS_DAYS = [
    0.0417,   # 1 小时
    0.25, 1, 3, 7, 14, 30, 60, 90, 180, 365, 730, 1825,
]

# ── 迁移实验的时间窗口（M3 / M4）────────────────────────────────
# 三个不相交的区间：教师训到 T-k，学生训 [T-W, T]，两者都在 [T, T+v] 上评测。
#
# 学生窗口**宽度固定**，不随 k 变。最初的设计是 `[T-k, T]`——这同时犯了两个错：
# k=0 时窗口为空，k=3 时窗口里只有 355 个样本。陈旧性说的是教师的**滞后**，
# 不是学生的数据量，两者在生产环境里本来就是独立的。
#
# k 的取值不能照搬 GEM 的天级尺度。量出来的教师数据损失：
#
#   k=7    -0.14%      k=365   -8.11%
#   k=30   -0.54%      k=730   -17.22%
#   k=90   -1.54%      k=1095  -28.13%
#
# Meta 的一周是海量数据加上快速换代的广告创意；Amazon 评论的一周是 0.14% 的数据，
# 商品目录几乎没动。所以这里的单位不是"天"，而是"教师缺失了多少知识"——
# 沿用 {0,3,7,14,30} 会得到五个几乎相同的数字，那是假阴性，不是结论。
SPLIT_DATE = "2022-01-01"
STUDENT_WINDOW_DAYS = 365     # 每域约 44k-56k 个位置
TEACHER_WINDOW_DAYS = 730     # 滑动窗口：各 k 下 430k-524k，且越旧的窗口略多
VALID_WINDOW_DAYS = 90        # 选 epoch 用，每域约 10k
EVAL_WINDOW_DAYS = 180        # 只报告用，每域约 19k-25k

# valid 和 eval 必须分开。早期版本在 eval 窗口上挑最好的 epoch，又把同一个数字
# 报出来——那是在测试集上做模型选择，而且各 arm 的轨迹噪声不同，偏差大小也不同，
# 恰好会污染 arm 之间的比较，也就是整个实验要测的东西。
STALENESS_DAYS = [0, 90, 365, 730, 1095]

# 种子噪声。同一 k、同一教师，只换学生的初始化种子：
#   seed 42    A 0.2288   C 0.2356   (+2.99%)
#   seed 1337  A 0.2357   C 0.2355   (-0.10%)
# arm 之间的效应量是 1-3%，和噪声同量级，所以单次运行读不出任何结论。
# 只把 CPU fp32 换成 GPU bf16 就足以让 A/C 的胜负翻转。
SEEDS = [42, 1337, 7, 2024, 31337]
