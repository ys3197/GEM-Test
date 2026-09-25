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
