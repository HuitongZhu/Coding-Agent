# 简单版本 Coding Agent

从零实现基于 OpenAI 原生 Tool Calling 协议的 Coding Agent。

---

## 环境安装

```bash
pip install openai python-dotenv
```

---

## .env 配置

在项目根目录创建 `.env` 文件：


| 变量 | 说明 | 必填 |
|---|---|---|
| `OPENAI_API_KEY` | 模型 API 密钥 | 是 |
| `OPENAI_API_BASE` | OpenAI 兼容网关地址 | 是 |
| `OPENAI_MODEL_NAME` | 模型名称 | 否，默认 gpt-4o-mini |
| `CHAT_MAX_TOKENS` | 单次响应最大 token | 否，默认 4096 |
| `WORKSPACE` | 工作区根目录（沙箱边界） | 否，默认项目根 |

---

## 运行方式

```bash
python main.py
```

交互式输入任务，Agent 自动完成。输入 `quit` 退出。

示例：
```
> 写hello.py，打印"hello coding agent"，运行这个脚本 

--------------------------------------------------
  [Round 1] >> 请求 LLM
--------------------------------------------------

--------------------------------------------------
  [Round 1] >> 工具调用计划
  * write_file({'content': 'print("hello coding agent")\n', 'path': 'hello.py'})  [id=call_47f3310b46e14c30b9768ac5]
  * run_shell({'command': 'python hello.py'})  [id=call_85eec0cc44cc4fe2ac70b862]
--------------------------------------------------

--------------------------------------------------
  [Round 1] [OK] 工具执行：write_file
[write_file 成功] 已写入 E:\nd_Programming_Agent\hello.py（覆盖）
--------------------------------------------------

--------------------------------------------------
  [Round 1] [OK] 工具执行：run_shell
[run_shell 退出码=0]
stdout:
hello coding agent

--------------------------------------------------

--------------------------------------------------
  [Round 2] >> 请求 LLM
--------------------------------------------------

--------------------------------------------------
  [Round 2] >> 任务完成（LLM 直接回答）


已创建 `hello.py` 并运行成功，输出：

hello coding agent

--------------------------------------------------

==================================================
  [Agent 最终结果]


已创建 `hello.py` 并运行成功，输出：

hello coding agent

==================================================
```

---

## 实现功能

### 核心架构

```
main.py                 程序入口（交互循环）
  └── agent_core.py     核心模块
        ├── Message / AgentContext   对话上下文管理
        ├── TOOLS_SCHEMA             手写 JSON Schema（无框架装饰器）
        ├── read_file / write_file   文件读写（含沙箱路径校验）
        ├── run_shell                命令执行（黑名单+白名单双重校验）
        ├── invoke_tool              工具调度分发
        └── run_coding_agent()       ReAct 循环主入口
```

### 关键特性

1. **无框架依赖**：仅 `openai` + `python-dotenv`，零 Agent 框架
2. **原生 Tool Calling**：手写 JSON Schema，直接对接 OpenAI function calling 协议
3. **ReAct 循环**：LLM → 解析 tool_calls → 分发执行 → 结果回灌 → 判断终止
4. **工作区沙箱**：文件路径校验必须在 WORKSPACE 内，`..` 穿越自动拦截
5. **命令安全管控**：`run_shell` 黑名单+白名单双重过滤，危险命令直接拒绝
6. **循环保护**：最多 8 轮，超出强制终止

---

## 文件说明

- `agent_core.py`（624 行）：核心模块，含上下文管理、工具定义与调度、ReAct 循环
- `main.py`（41 行）：程序入口，交互循环
- `.env.example`：环境变量模板
- `.gitignore`：排除 `.env`、日志、Python 编译产物

---

## 安全设计

1. 文件路径通过 `Path.resolve().relative_to(workspace)` 校验，越界拦截
2. Shell 命令禁止 `rm -rf`、`format`、`shutdown` 等危险前缀
3. 仅允许白名单命令：`python` / `git` / `ls` / `grep` / `pytest` 等
4. 命令强制在 WORKSPACE 目录内执行（cwd 锁定）

---

