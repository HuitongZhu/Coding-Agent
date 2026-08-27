"""
Coding Agent 程序入口
用法：python main.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，保证 agent_core 可被正确导入
# （无论从项目根目录还是子目录执行均有效）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_core import run_coding_agent


SEP = "=" * 50


def main() -> None:
    print(SEP)
    print("  极简手写 Coding Agent")
    print("  输入任务后回车开始执行，输入 quit 退出")
    print(SEP)

    while True:
        try:
            user_task = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_task:
            continue
        if user_task.lower() in ("quit", "exit", "q"):
            print("退出。")
            break

        result = run_coding_agent(user_task)

        print(f"\n{SEP}")
        print("  [Agent 最终结果]")
        print(result)
        print(SEP)


if __name__ == "__main__":
    main()
