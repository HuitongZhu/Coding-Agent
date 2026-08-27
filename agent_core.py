"""
极简手写 Coding Agent 核心模块
仅依赖：openai、python-dotenv
无 agent 框架 / SDK
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import openai
import dotenv

# ── 常量 ──────────────────────────────────────────────────────

# Agent 最大循环轮数，防止无限调用模型
MAX_AGENT_ROUNDS = 8


# ── 环境变量加载 ──────────────────────────────────────────────

def load_env() -> None:
    """
    加载项目根目录 .env 文件（若存在）。
    以执行脚本的工作目录为准（os.getcwd），而非本模块所在目录，
    保证无论从哪个入口运行都能找到项目根 .env。
    """
    dotenv.load_dotenv(
        dotenv_path=os.path.join(os.getcwd(), ".env"),
        override=False,
    )


# ── 数据结构 ─────────────────────────────────────────────────

@dataclass
class Message:
    """对话消息，完整保留 OpenAI 工具调用协议字段"""
    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


@dataclass
class AgentContext:
    """Agent 运行上下文，支持多实例并发"""
    system_prompt: str
    history: list[Message] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)

    @property
    def chat_max_tokens(self) -> int:
        """单次响应最大输出 token"""
        _ensure_env()
        return int(os.getenv("CHAT_MAX_TOKENS", "4096"))

    @property
    def model(self) -> str:
        _ensure_env()
        return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        msg = Message(role=role, content=content, **kwargs)
        self.history.append(msg)

    def add_user(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant(self, content: str, tool_calls: Optional[list[dict]] = None) -> None:
        self.add_message("assistant", content, tool_calls=tool_calls)

    def add_tool_result(self, content: str, tool_call_id: str) -> None:
        self.add_message("tool", content, tool_call_id=tool_call_id)

    def to_api_messages(self) -> list[dict]:
        """
        将历史消息序列化为 OpenAI API 要求的 dict 格式。
        仅保留有值的字段，确保 tool_calls / tool_call_id 正确传递。
        """
        result = []
        for m in self.history:
            d: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                d["tool_calls"] = m.tool_calls
            result.append(d)
        return result


# ── 环境变量读取 ──────────────────────────────────────────────

_LOADED = False


def _ensure_env() -> None:
    """延迟加载 .env，确保在任何入口下都能读取到项目本地配置"""
    global _LOADED
    if not _LOADED:
        load_env()
        _LOADED = True


def get_api_key() -> str:
    _ensure_env()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "未设置 OPENAI_API_KEY，请检查项目根目录 .env 文件"
        )
    return key


def get_api_base() -> str:
    _ensure_env()
    return os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


def get_workspace() -> Path:
    """
    获取工作区根目录（沙箱边界）。
    优先读 WORKSPACE 环境变量，默认为项目根目录（脚本 cwd）。
    """
    _ensure_env()
    ws = os.getenv("WORKSPACE", os.getcwd())
    return Path(ws).resolve()


# ── OpenAI 兼容接口封装 ──────────────────────────────────────

def create_client() -> openai.OpenAI:
    """创建 OpenAI 客户端（指向兼容网关）"""
    _ensure_env()
    return openai.OpenAI(
        api_key=get_api_key(),
        base_url=get_api_base(),
    )


def chat_completion(
    client: openai.OpenAI,
    context: AgentContext,
    **kwargs: Any,
) -> openai.types.chat.chat_completion.ChatCompletion:
    """
    发送对话请求并返回完整响应。
    通过 context.to_api_messages() 确保 tool_calls / tool_call_id
    完整传递给模型，支持原生工具调用协议。
    """
    messages = context.to_api_messages()
    return client.chat.completions.create(
        model=context.model,
        messages=messages,
        max_tokens=context.chat_max_tokens,
        **kwargs,
    )


def parse_tool_calls(
    response: openai.types.chat.chat_completion.ChatCompletion,
) -> list[dict]:
    """
    从响应中提取 tool_calls 列表。
    增加空校验，避免 choices 为空时 IndexError。
    """
    choices = response.choices
    if not choices:
        return []
    choice = choices[0]
    if choice.message.tool_calls:
        return [tc.model_dump() for tc in choice.message.tool_calls]
    return []


# ── 工具定义（手写 JSON Schema，无框架装饰器）────────────────

# 工具 JSON Schema 列表，直接传入 OpenAI tool calling 协议。
# 每个工具的 description 均提示模型仅在 WORKSPACE 内操作，
# 便于模型理解沙箱边界。
TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "读取工作区内的本地文件内容并返回文本。"
                "仅允许读取 WORKSPACE 目录及其子目录下的文件，"
                "越界路径将被拒绝。适用于读取代码、配置文件、日志等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径，相对于项目工作区（WORKSPACE）的相对路径，例如 'src/main.py'",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "文件编码，默认 utf-8",
                        "enum": ["utf-8", "latin-1", "gbk"],
                        "default": "utf-8",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "将文本内容写入工作区内的本地文件。"
                "文件不存在则自动创建（含父目录），存在则覆盖。"
                "仅允许在工作区目录内操作，越界路径将被拒绝。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径，相对于项目工作区（WORKSPACE）的相对路径，例如 'output/result.txt'",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文件内容",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "文件编码，默认 utf-8",
                        "enum": ["utf-8", "latin-1", "gbk"],
                        "default": "utf-8",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "是否以追加模式写入，默认 False（覆盖）",
                        "default": False,
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "在工作区内执行 Shell 命令，返回 stdout、stderr 和退出码。"
                "注意：仅允许执行安全的只读或项目构建类命令，"
                "禁止执行 rm -rf、格式盘、涉及系统敏感路径的命令。"
                "实际安全校验由函数内部执行前拦截。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 Shell 命令（单条命令，不使用 & 或 ; 拼接）",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "命令超时秒数，默认 30",
                        "default": 30,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


# ── 沙箱校验工具 ──────────────────────────────────────────────

def _assert_in_workspace(rel_path: str) -> tuple[Optional[str], Optional[Path]]:
    """
    校验相对路径解析后是否仍在 WORKSPACE 内部。
    若越界则返回错误信息字符串（不抛异常，方便 Agent 消费）。
    返回：(error_msg, resolved_path)。error_msg 非空表示校验失败。
    """
    workspace = get_workspace()
    # 将相对路径拼接并 resolve（处理 .. 等穿越尝试）
    resolved = (workspace / rel_path).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return (
            f"[沙箱拦截] 路径 '{rel_path}' 解析后为 {resolved}，"
            f"已超出 WORKSPACE 边界 {workspace}，操作被拒绝。"
        ), None
    return None, resolved


# 危险命令前缀黑名单，防止破坏性操作
_DANGEROUS_PREFIXES: tuple[str, ...] = (
    "rm -rf", "rm -r", "del /f /s", "format",
    "shutdown", "reboot", "poweroff", "halt",
    "dd if=", "mkfs", "wget http", "curl http",
)

# 允许执行的命令前缀白名单
_SAFE_PREFIXES: tuple[str, ...] = (
    "python", "pip", "node", "npm", "git",
    "ls", "dir", "cat", "echo", "mkdir",
    "cp", "mv", "head", "tail", "find",
    "wc", "grep", "sort", "uniq", "diff",
    "pytest", "python -m", "python3", "pip3",
    "tree", "pwd",
)


def _validate_shell_command(command: str) -> Optional[str]:
    """
    校验 shell 命令是否安全。
    返回 None 表示通过，返回错误字符串表示拒绝。
    """
    stripped = command.strip()

    # 1. 黑名单：危险前缀直接拒绝
    for prefix in _DANGEROUS_PREFIXES:
        if stripped.lower().startswith(prefix):
            return f"[run_shell 拦截] 危险命令被拒绝：{stripped}"

    # 2. 白名单：仅允许安全前缀
    matched = any(stripped.lower().startswith(p) for p in _SAFE_PREFIXES)
    if not matched:
        return (
            f"[run_shell 拦截] 命令不在安全白名单内：{stripped}\n"
            f"允许的前缀：{_SAFE_PREFIXES}"
        )

    return None


# ── 工具实现（纯函数，无框架依赖）────────────────────────────

_TOOL_REGISTRY: dict[str, Any] = {}


def _register_tool(func: Any) -> Any:
    """内部注册：将函数添加到工具调度表"""
    _TOOL_REGISTRY[func.__name__] = func
    return func


@_register_tool
def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    读取工作区内的文件内容。
    沙箱校验失败或 IO 异常均返回结构化错误信息。
    """
    err, resolved = _assert_in_workspace(path)
    if err:
        return err

    try:
        content = resolved.read_text(encoding=encoding)
        return content
    except FileNotFoundError:
        return f"[read_file 错误] 文件不存在：{path}"
    except PermissionError:
        return f"[read_file 错误] 权限不足，无法读取：{path}"
    except UnicodeDecodeError as e:
        return f"[read_file 错误] 编码错误（{encoding}）：{e}"
    except OSError as e:
        return f"[read_file 错误] IO 异常：{e}"


@_register_tool
def write_file(
    path: str,
    content: str,
    encoding: str = "utf-8",
    append: bool = False,
) -> str:
    """
    写入文件内容（覆盖或追加）。
    自动创建父目录；沙箱校验失败或 IO 异常均返回结构化错误。
    """
    err, resolved = _assert_in_workspace(path)
    if err:
        return err

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding, newline="")
        return f"[write_file 成功] 已写入 {resolved}（{'追加' if append else '覆盖'}）"
    except PermissionError:
        return f"[write_file 错误] 权限不足，无法写入：{path}"
    except OSError as e:
        return f"[write_file 错误] IO 异常：{e}"


@_register_tool
def run_shell(command: str, timeout: int = 30) -> str:
    """
    执行 Shell 命令，返回合并输出与退出码。
    执行前进行安全校验（黑名单 + 白名单）；超时/异常统一转为结构化信息。
    """
    err = _validate_shell_command(command)
    if err:
        return err

    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=str(get_workspace()),  # 强制在工作区内执行
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"stderr:\n{result.stderr}")
        output = "\n".join(output_parts) if output_parts else "(无输出)"
        return f"[run_shell 退出码={result.returncode}]\n{output}"
    except subprocess.TimeoutExpired:
        return f"[run_shell 错误] 命令超时（>{timeout}s）：{command}"
    except FileNotFoundError:
        return f"[run_shell 错误] Shell 不可用或命令不存在：{command}"
    except OSError as e:
        return f"[run_shell 错误] OS 异常：{e}"


# ── 工具调度 ──────────────────────────────────────────────────

def invoke_tool(tool_name: str, tool_args: dict) -> str:
    """
    根据工具名查找并执行对应函数，返回结果字符串。
    未注册工具或参数解析失败时返回友好错误信息。
    """
    func = _TOOL_REGISTRY.get(tool_name)
    if func is None:
        registered = ", ".join(_TOOL_REGISTRY.keys())
        return f"[invoke_tool 错误] 未注册工具 '{tool_name}'，可用工具：{registered}"
    try:
        return func(**tool_args)
    except TypeError as e:
        return f"[invoke_tool 错误] 参数不匹配：{e}"
    except Exception as e:
        return f"[invoke_tool 错误] {tool_name} 执行异常：{type(e).__name__}: {e}"


# ── Agent 主循环 ──────────────────────────────────────────────

def _print_round_log(round_num: int, heading: str, body: str = "") -> None:
    """统一控制台日志输出格式（兼容 Windows 终端，不使用 emoji）"""
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  [Round {round_num}] {heading}")
    if body:
        print(body)
    print(sep)


def run_coding_agent(user_task: str) -> str:
    """
    Coding Agent 主入口：ReAct 循环。
    - 每轮调用 LLM，解析 tool_calls
    - 有 tool_calls：分发调用本地工具，将结果追加到上下文，继续下一轮
    - 无 tool_calls：LLM 直接给出最终回答，循环结束
    - 最多 MAX_AGENT_ROUNDS 轮，防止无限循环
    - 最终返回 LLM 的最终文本响应
    """
    _ensure_env()

    # 1. 创建 OpenAI 客户端
    client = create_client()

    # 2. 构建 Agent 上下文
    system_text = (
        "你是一个 Coding Agent，可以在工作区内读写文件、执行命令。\n"
        f"工作区根目录：{get_workspace()}\n"
        "你只能在工作区目录内操作。每次拿到工具结果后，根据结果决定下一步行动。"
        "如果任务已完成，直接输出最终答案，不要再次调用工具。"
    )
    context = AgentContext(system_prompt=system_text)
    context.add_user(user_task)

    # 3. 调用 LLM 时附带工具定义
    call_kwargs: dict[str, Any] = {
        "tools": TOOLS_SCHEMA,
        "tool_choice": "auto",  # 允许模型自主决定是否调用工具
    }

    final_answer = ""

    for round_num in range(1, MAX_AGENT_ROUNDS + 1):
        # ── 调用 LLM ──────────────────────────────────────────
        _print_round_log(round_num, ">> 请求 LLM")

        try:
            response = chat_completion(client, context, **call_kwargs)
        except Exception as e:
            err_msg = f"[Agent 错误] LLM 调用失败：{type(e).__name__}: {e}"
            _print_round_log(round_num, "!! LLM 调用异常", err_msg)
            return err_msg

        # ── 解析工具调用 ──────────────────────────────────────
        tool_calls = parse_tool_calls(response)

        # 无工具调用 → LLM 直接回答，提取最终文本并结束
        if not tool_calls:
            final_answer = response.choices[0].message.content or ""
            _print_round_log(
                round_num,
                ">> 任务完成（LLM 直接回答）",
                final_answer,
            )
            break

        # ── 打印本轮 LLM 意图 ─────────────────────────────────
        intents = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            args_str = fn.get("arguments", "")
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {"_parse_error": args_str}
            tc_id = tc.get("id", "")
            intents.append(f"  · {name}({args})  [id={tc_id}]")
        _print_round_log(round_num, ">> 工具调用计划", "\n".join(intents))

        # ── 将 assistant 消息（含 tool_calls）追加到上下文 ─────
        context.add_assistant(
            content=response.choices[0].message.content or "",
            tool_calls=tool_calls,
        )

        # ── 分发调用工具，结果逐条追加 ────────────────────────
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                tool_args = json.loads(fn.get("arguments", "{}") or "{}")
            except json.JSONDecodeError:
                tool_args = {}
            tc_id = tc.get("id", "")

            result = invoke_tool(tool_name, tool_args)

            # 将工具结果追加为 tool 消息
            context.add_tool_result(content=result, tool_call_id=tc_id)

            status = "[OK]" if not result.startswith("[错误") else "[WARN]"
            _print_round_log(
                round_num,
                f"{status} 工具执行：{tool_name}",
                result,
            )

        # 进入下一轮
        continue
    else:
        # 超出最大轮数仍未收敛
        _print_round_log(MAX_AGENT_ROUNDS, ">> 达到最大轮数，强制终止")
        final_answer = f"[Agent 终止] 已达最大轮数（{MAX_AGENT_ROUNDS}），任务未完成。"

    return final_answer
