#!/usr/bin/env python
"""语料装配入口。

用法::

    python prepare_dual_data.py                    # 默认 2000/2000/4000
    python prepare_dual_data.py --n-code 500      # 小规模试跑
    python prepare_dual_data.py --val-ratio 0.1   # 切出 10% 验证集

本脚本是 :mod:`dbl.prepare` 的薄封装，实际逻辑在包内以便被测试导入。
"""

from __future__ import annotations

from dbl.prepare import main

if __name__ == "__main__":
    main()
