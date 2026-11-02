<div align="center">

<img src="logo.png" width="132" alt="ActiveReply logo" />

# 主动回复与记忆（Reply & Memory）

**让 Bot 自己决定何时开口、@ 谁，并真正记住该记的事。**

[![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.0.0-4b8bf5)](https://docs.astrbot.app/dev/star/plugin-new.html)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776ab)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-v1.1.0--beta.1-12a37a)](./CHANGELOG.md)
[![GitHub](https://img.shields.io/badge/GitHub-astrbot__plugin__Reply--Memory-181717?logo=github)](https://github.com/AethenNet/astrbot_plugin_Reply-Memory)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](./LICENSE)

**支持平台：OneBot（aiocqhttp）** · **插件市场分类：工具、AI 增强**

</div>

---

## 目录

- [简介](#简介)
- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装](#安装)
- [使用前检查](#使用前检查)
- [指令](#指令)
- [配置](#配置)
- [记忆系统](#记忆系统)
- [群聊上下文与回复](#群聊上下文与回复)
- [控制台](#控制台)
- [数据与备份](#数据与备份)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [开发与构建](#开发与构建)
- [致谢](#致谢)
- [许可证](#许可证)

## 简介

ActiveReply 是面向 AstrBot 的群聊增强插件。它把「主动回复」「群聊上下文」「长期记忆 RAG」「用户屏蔽」和「可视化控制台」组合在一起，让 Bot 能按群聊语境决定是否开口、如何 @ 或引用，并记住值得长期保留的信息。

当前版本：**v1.1.0-beta.1**。

- 代码仓库：[AethenNet/astrbot_plugin_Reply-Memory](https://github.com/AethenNet/astrbot_plugin_Reply-Memory)
- 下载地址：[最新 Release](https://github.com/AethenNet/astrbot_plugin_Reply-Memory/releases/latest)

## 功能特性

| 功能 | 说明 |
|---|---|
| 消息主动回复 | 概率触发或模型判定；累计 N 条消息后兜底回复；支持黑白名单 |
| 群聊上下文增强 | 记录昵称、QQ、时间、消息号；支持图片转述；记忆不依赖此开关 |
| 智能回复 | 解析 `<mention id="..."/>`、`<quote id="..."/>`，转换为真实 @ 与引用 |
| 自动记忆 | 关键词快速通道 + 后台模型抽取；独立记忆时间线，不受对话格式影响 |
| 记忆 RAG | 写入、语义检索、去重、冲突失效、生命周期管理与自检 |
| 引用归属 | 引用/@ 内容不会被误认成当前发言人；证据与角色只从正文/消息头提取 |
| 用户屏蔽 | 全局黑名单 + 群级屏蔽；支持原因、时长、屏蔽提示；管理员豁免 |
| 联网搜索 | `grok_web_search` 工具，按需检索实时信息 |
| 图片理解 | `enhance_use_image` 按需查看历史图片并回填描述 |
| 回复调试 | `/replytest` 开启后，每次回复附带 token、缓存命中率与插件日志 |
| 可视化页面 | 状态、群配置、记忆 RAG、屏蔽设置、全局设置；深浅色自动跟随 |

## 环境要求

- AstrBot **>= 4.0.0**
- Python **>= 3.10**
- OneBot v11 适配器（`aiocqhttp`）
- 使用记忆 RAG 时：需要在 AstrBot 中配置一个可用的 **Embedding 提供商**

## 安装

### 方式一：手动安装

1. 将 `astrbot_plugin_active_reply` 整个文件夹放入 AstrBot 的 `data/plugins/`；
2. 重启 AstrBot，或在插件页重载插件；
3. 在插件配置中按需开启「记忆 RAG」「消息主动回复」「联网搜索」等功能。

### 方式二：Release

从 [GitHub Releases](https://github.com/AethenNet/astrbot_plugin_Reply-Memory/releases/latest) 下载最新 zip，解压后放入 `data/plugins/`。

> 升级时请保留 `data/plugin_data/astrbot_plugin_active_reply/` 下的数据文件；插件启动时会自动执行配置迁移。

## 使用前检查

建议先完成以下设置，避免出现「不回复 / 重复回复 / 记忆不可用」：

1. **关闭 AstrBot 内置群聊上下文感知**（`group_icl_enable`），避免与本插件重复注入历史；
2. **关闭 AstrBot 内置主动回复**（`provider_settings.active_reply.enable`），否则同一条消息可能回复两次；
3. **关闭 AstrBot 内置引用回复**（`reply_with_quote`），让插件按需引用；
4. **开启 AstrBot 用户识别**（`provider_settings.identifier`），历史中的用户 ID 与角色显示依赖它；
5. 想用真实 @ 与引用：在插件配置中开启「群聊回复增强 → React Pattern」与「智能回复」；
6. 想用记忆 RAG：先在 AstrBot 中配置 Embedding 提供商；
7. 想用联网搜索：在插件「联网搜索」中填写专用 `provider_id`（不会回退到会话模型）。

## 指令

| 指令 | 权限 | 说明 |
|---|---|---|
| `/执行回复` | 管理员 | 立即执行一次主动回复或模型判定，并发送执行日志（含 token 用量） |
| `/replytest true\|false` | 管理员 | 开启/关闭本群回复调试；别名 `/回复调试 开\|关` |
| `/ban <时间> <用户ID> <原因>` | 管理员 | 在本群屏蔽用户 |
| `/ban-all <时间> <用户ID> <原因>` | 管理员 | 全局屏蔽用户（含私聊） |
| `/remove <用户ID>`、`/pass <用户ID>` | 管理员 | 解除本群屏蔽 |
| `/remove-all <用户ID>`、`/pass-all <用户ID>` | 管理员 | 解除全部屏蔽（全局 + 群级） |
| `/总结记忆` | 管理员 | 立即总结当前会话；支持 `/总结记忆 quiet` 只回一行结果 |

时间格式：`0` 表示永久；支持 `1d2h30m` 组合；纯数字按分钟计算。

示例：

    /ban 0 123456789 机器人
    /ban-all 7d 987654321 广告
    /remove 123456789
    /pass-all 987654321

## 配置

插件全局配置分为 6 个区块：

| 区块 | 关键项 |
|---|---|
| 消息主动回复 | 启用、回复方式（概率/模型判定）、概率、触发积累、黑白名单 |
| 群聊上下文增强 | 历史开关、消息上限、包含用户信息、包含标签、图片转述 |
| 群聊回复增强 | React Pattern、角色显示、智能回复、用户屏蔽 |
| 联网搜索 | 启用、专用提供商、请求模式、系统提示词、来源输出、最大来源数 |
| 记忆 RAG | 记忆控制、记忆生成、记忆注入、记忆评估 |
| 调用配置 | LRU 上限、图片转述超时、模型判定超时 |

群级配置可以覆盖主动回复、上下文、回复增强和记忆 RAG 的部分字段。**群级只会保存与全局不同的字段**；未修改的字段继续继承全局，避免默认值静默覆盖全局开关。

## 记忆系统

### 独立时间线

记忆使用独立的 `memory_chats` 历史，与对话模型使用的 `session_chats` 分开：

- 记忆行头固定为 `[昵称/QQ号/YYYY-MM-DD HH:MM:SS]`，始终包含发送者 ID 和完整日期；
- 不受「群聊上下文增强」「包含用户信息」「包含标签」等对话开关影响；
- 记忆库的 `memory_time` 取窗口内最后一条有效消息的真实发送时间，而不是摘要任务执行时间；
- 抽取提示词会附带当前时间与时区，并明确禁止照抄示例日期。

> 旧版本已经写入的记忆不会自动重算时间；新写入的记忆使用新的时间逻辑。

### 自动记忆

自动记忆由两条通道组成：

1. **关键词快速通道**：消息包含「记住 / 记一下 / 帮我记 / 记着 / 别忘了」时，后台写入一条记忆；
2. **模型抽取通道**：累计消息达到「累计消息记忆」（默认 50 条），且超过「抽取间隔」（默认 10 分钟）后，后台总结最近片段；对话停顿超过「对话间隔切窗」（默认 60 分钟）也会触发。

自动记忆**不依赖**群聊上下文增强或主动回复开关。只要记忆 RAG 与自动产生记忆开启，即使关闭上下文/主动回复，仍会记录记忆历史并尝试抽取。

### 置信度

置信度表示这段对话作为长期记忆的价值，不使用「摘要正确率」。分档如下：

| 档位 | 分值 | 含义 |
|---|---|---|
| major | 0.90-1.00 | 重要群体决策、群规、全员投票、正式承诺、金钱/安全/重大日程等 |
| discussion | 0.70-0.80 | 多人参与且有明确结论的有价值讨论 |
| daily | 0.50-0.60 | 日常讨论、普通约饭/约玩、临时安排、一般信息 |
| chatter | 0.30-0.40 | 简单闲聊、临时状态、玩笑、推测 |
| spam | 0.00-0.20 | 灌水、寒暄、表情、复读 |

- 默认最低置信度为 **0.50**；日常讨论可以写入，简单闲聊默认过滤；
- 管理员执行 `/总结记忆` 属于显式操作，不受自动抽取门槛限制；
- 模型输出的高低分会被限制在所选档位内；缺少档位时高分按保守策略处理。

### 引用与发言者

- 每条消息的发言人只看行首 `[昵称/QQ/时间]` 或 `[Bot: ...]`；
- 引用内容会改写为 `（引用 X 的旧消息：「…」）` 附在正文末尾，不会被当成当前发言人的话；
- `（@ X）` 只表示提及，不代表 X 当前发言；
- 摘要证据（quotes）与角色（role_ids）只从当前消息正文/消息头提取，不能直接摘录引用旧消息。

### 群级配置与自动修复

旧版本保存群级独立配置时，可能把记忆 RAG 的默认 `False` 一并写入，导致该群记忆被静默关闭。当前版本会：

- 保存群配置时只写入与全局不同的字段；
- 启动时检查旧 `groups.json`，移除「7 个记忆字段全部为旧默认值」的自动填充覆盖；
- 用户真正修改过的群级记忆字段会保留。

## 群聊上下文与回复

- 群聊历史按配置记录为 `[昵称/QQ/时间](角色) #msg消息号: 内容`；
- 普通模式将历史追加到 system prompt；React 模式通过临时内容块注入，不改写 `req.prompt`、不清空 `req.contexts`，保留上下文缓存前缀；
- 主动回复支持概率与模型判定；模型判定使用独立的非持久化请求；
- 插件会关闭本轮流式输出，确保 `<mention/>` / `<quote/>` 在发送前完成解析；
- 发送前通过 `on_decorating_result` 转换为真实 @/引用组件，发送前/后均可做标签清理兜底。

## 控制台

在 AstrBot 插件详情页打开插件视图：

- **状态**：开关状态、记忆统计、Embedding 可用性、最近活跃群；
- **群配置**：按群覆盖主动回复/上下文/回复增强/记忆 RAG，只保存差异字段；
- **记忆 RAG**：统计、筛选、分页、详情、删除、回收站撤销、自检；顶部「自动记忆」面板可查看开关、抽取模型、累计条数、间隔、置信度、作用域；
- **屏蔽设置**：添加/解除屏蔽，支持 ID/原因搜索；
- **全局设置**：按 `_conf_schema.json` 渲染，保存后即时生效；
- **主题**：跟随系统 / 日间 / 夜间；操作按钮点击后 1 秒锁定并显示形变加载反馈。

## 数据与备份

运行数据目录：`AstrBot/data/plugin_data/astrbot_plugin_active_reply/`

| 文件 | 内容 |
|---|---|
| `memory_rag.db` | 长期记忆 SQLite（记忆、角色、回收站） |
| `groups.json` | 群级覆盖配置 |
| `user_blocks.json` | 用户屏蔽记录 |

升级插件时请保留以上文件；插件会在启动时做向后兼容迁移。删除 `memory_rag.db` 会丢失全部长期记忆。

## 常见问题

### 自动记忆没有触发

按顺序检查：

1. 「记忆 RAG → 记忆控制 → 启用」是否开启；
2. 「自动产生记忆」是否开启；
3. 当前群是否在群级配置中单独关闭了记忆；
4. AstrBot 是否配置了可用的 Embedding 提供商（启动日志会给出警告）；
5. 累计消息是否达到「累计消息记忆」且超过「抽取间隔」；
6. 查看插件日志中的「自动记忆触发」「自动记忆配置」「没有可用的 Embedding 提供商」等提示。

### 记忆时间不对

- 新记忆使用 AstrBot 全局时区下的真实发送时间；请确认 AstrBot 的全局时区设置正确；
- 旧版本写入的历史记忆不会被批量改写，可删除后重新总结；
- 摘要中的日期以记忆历史行头时间为准，示例日期不会被当作真实日期。

### `/总结记忆` 提示消息不足

`/总结记忆` 至少需要 3 条记忆历史。重启插件后运行时历史会清空，需要先产生新的聊天消息再总结。

### 引用或 @ 没有正确发出

确认「群聊回复增强 → React Pattern」与「智能回复」开启，并确认本轮没有被强制流式输出；插件会在发送前把标签转换为平台组件。

## 项目结构

| 路径 | 职责 |
|---|---|
| `main.py` | 插件入口、事件流程、提示词注入、工具、页面路由、记忆/屏蔽/调试逻辑 |
| `config_model.py` | 配置默认值、类型转换、全局/群级合并、默认提示词与迁移辅助 |
| `runtime_state.py` | 进程内短期状态：对话历史、记忆历史、计数、LRU |
| `group_store.py` | 群级覆盖配置 `groups.json` |
| `block_store.py` | 用户屏蔽记录 `user_blocks.json` |
| `memory_store.py` | 长期记忆 SQLite：表结构、检索、去重、生命周期 |
| `tags.py` | `<mention/>` / `<quote/>` 标签解析与组件转换 |
| `page_api.py`、`web_compat.py` | 控制台后端接口与 AstrBot 新旧网页 API 兼容 |
| `_conf_schema.json` | 插件配置表单定义 |
| `views/console/` | 控制台前端与图标资源 |

## 开发与构建

    # 静态检查
    ruff check .
    ruff format --check .
    python3 -m compileall -q .
    node --check views/console/app.js

    # 打包（排除缓存）
    zip -r astrbot_plugin_active_reply.zip astrbot_plugin_active_reply \
      -x "*/__pycache__/*" "*.pyc" "*/.ruff_cache/*"

发布包不包含测试目录；本地开发时建议为事件流程、记忆抽取、标签转换、群配置合并与页面接口补充回归测试。

## 相关链接

- [代码仓库](https://github.com/AethenNet/astrbot_plugin_Reply-Memory)
- [Releases](https://github.com/AethenNet/astrbot_plugin_Reply-Memory/releases/latest)
- [Issue 反馈](https://github.com/AethenNet/astrbot_plugin_Reply-Memory/issues)

## 致谢

- 界面图标来自 [Morphicons](https://github.com/guillermolg00/morphicons) 的 Lucide 图标集（MIT / Lucide ISC）；菜单按钮使用 Morphicons 引擎做形变动画；
- 功能设计参考社区插件 `astrbot_plugin_astrbot_enhance_mode`（其仓库未声明开源协议，本项目仅参考交互与功能思路，未复制其代码）。

## 许可证

本项目使用 [GNU Affero General Public License v3.0](./LICENSE)（AGPL-3.0）。

你可以自由使用、修改和分发本项目，但分发修改后的版本时必须继续使用 AGPL-3.0，并向通过网络与本项目交互的用户提供对应源代码。

> 第三方图标资源（`views/console/vendor/morphicons`）仍遵循其原有的 MIT / ISC 许可证；这些第三方许可证不受本项目 AGPL 协议影响。
