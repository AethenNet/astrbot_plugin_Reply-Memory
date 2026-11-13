/**
 * 主动回复（ActiveReply）控制台前端。
 *
 * 页面通过 AstrBot 插件视图 bridge 与后端通信：
 *   window.AstrBotPluginView.apiGet(endpoint, params)
 *   window.AstrBotPluginView.apiPost(endpoint, body)
 * 后端统一返回 { success: true, data } 或 { success: false, error }。
 *
 * 交互采用“请求更新 state、render 函数重绘 DOM”的单向数据流。表单输入
 * 先写入 state 中的编辑副本，只有点击保存后才提交后端；重新打开页面或
 * 重新加载配置时，以后端返回值覆盖本地副本，避免把尚未保存的改动误当成
 * 已持久化数据。各视图函数只负责当前页面区域，避免一次操作重建无关页面。
 */

const UI_BUILD = 36;
console.info(`[ActiveReply] UI build ${UI_BUILD}`);

const bridge = window.AstrBotPluginView || window.AstrBotPluginPage || null;

/**
 * 键盘检测基线（r3.3w）：iframe 被宿主 adjustResize 时 innerHeight 与
 * visualViewport.height 会同步变小，两者差值恒为 0，不能用差值判断键盘。
 * 这里记录「无键盘时」的视口高度基线，当前高度明显小于基线才认为键盘展开。
 */
let keyboardBaselineHeight = 0;
let keyboardBaselineTimer = 0;

function currentViewportHeight() {
  const vv = window.visualViewport;
  const inner = Number(window.innerHeight) || 0;
  const visual = vv ? Number(vv.height) || 0 : inner;
  return Math.min(inner, visual);
}

function isInputFocused() {
  const focused = document.activeElement;
  if (!focused) return false;
  const tag = String(focused.tagName || "").toUpperCase();
  return tag === "INPUT" || tag === "TEXTAREA" || Boolean(focused.isContentEditable);
}

/** 软键盘是否处于展开状态。 */
function isKeyboardViewportOpen() {
  if (keyboardBaselineHeight <= 0) return false;
  return currentViewportHeight() < keyboardBaselineHeight - 60;
}

/** 延后重算键盘基线（键盘收起/旋转后调用；输入框聚焦时不更新）。 */
function scheduleBaselineRefresh() {
  clearTimeout(keyboardBaselineTimer);
  keyboardBaselineTimer = window.setTimeout(() => {
    if (isInputFocused()) return;
    const height = currentViewportHeight();
    if (height > 0) keyboardBaselineHeight = height;
  }, 220);
}

/** 重新计算顶部胶囊占位高度（键盘/旋转改变视口后需要刷新）。 */
function refreshTopbarHeight() {
  const controls = document.querySelector(".floating-controls");
  if (!controls) return;
  // 抽屉/详情打开时 .floating-controls 带 transform，getBoundingClientRect 不可信。
  if (
    document.body.classList.contains("drawer-open") ||
    document.body.classList.contains("detail-open")
  ) {
    return;
  }
  const pane = document.getElementById("main-pane");
  if (pane && pane.classList.contains("is-pushed")) return;
  const rect = controls.getBoundingClientRect();
  if (!rect || rect.height <= 0) return;
  document.documentElement.style.setProperty(
    "--topbar-h",
    `${Math.round(rect.bottom)}px`
  );
}

/**
 * 页面唯一的前端状态源：视图数据、编辑中的配置值与记忆列表筛选条件均保存在此处。
 * 读取接口后先更新 state，再由 render 函数将其映射到 DOM。
 */
const state = {
  view: "overview",
  overview: null,
  groups: [],
  groupsLoaded: false,
  groupSearch: "",
  currentGroupId: "",
  currentGroup: null,
  currentTab: "active_reply",
  globalTab: "active_reply",
  providers: [],
  embeddingProviders: [],
  schema: null,
  globalValues: null,
  autoMemory: null,
  blocks: null,
  groupBlocks: [],
  themeMode: "system",
  systemDark: false,
  blockState: {
    global: { search: "", expanded: false },
    group: { search: "", expanded: false },
  },
  rag: {
    page: 1,
    pageSize: 20,
    total: 0,
    hasMore: false,
    keyword: "",
    scope: "",
    role: "",
    memoryType: "",
    kind: "",
    needsReview: false,
    identity: "",
    // r3.3l：生命周期筛选（即将过期/已过期/低价值将归档/长期未访问）。
    lifecycle: "",
    // r3.3：改为「只看已归档」开关，请求时传 archived_only=1。
    archivedOnly: false,
    // 批量选择：当前页条目与已选 ID。
    selectMode: false,
    selected: new Set(),
    items: [],
    loadedOnce: false,
    scopes: [],
    // r3.3l：服务端返回的当前生效遗忘规则（用于顶部摘要与时间线）。
    rules: null,
  },
  // r3.3：开发者选项鉴权状态；令牌只保存在内存中，进入/离开页面都会清空。
  dev: { token: "", authed: false },
};

const PROVIDER_PATHS = new Set([
  "active_reply.model_choice_provider_id",
  "group_context_enhancement.image_caption_provider_id",
]);

const SECTION_ICONS = {
  active_reply: "message",
  group_context_enhancement: "layers",
  group_reply_enhancement: "sparkles",
  web_search: "search",
  memory_rag: "database",
  runtime: "sliders",
};

/** 按图标名生成引用 SVG sprite 的标记。 */
function icon(name) {
  return `<span class="icon icon-${name}" aria-hidden="true"></span>`;
}

const FIELD_GROUPS = [
  { prefix: "active_reply.", tab: "active_reply", title: "消息主动回复" },
  {
    prefix: "group_context_enhancement.",
    tab: "group_context_enhancement",
    title: "群聊上下文增强",
  },
  {
    prefix: "group_reply_enhancement.",
    tab: "group_reply_enhancement",
    title: "群聊回复增强",
  },
  {
    prefix: "memory_rag.",
    tab: "memory_rag",
    title: "记忆",
  },
];

/** 通过元素 ID 获取控制台节点，减少重复的 DOM 查询写法。 */
const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* 基础工具                                                            */
/* ------------------------------------------------------------------ */
/**
 * 对不可信文本执行 HTML 转义，供模板字符串插入标记前使用。
 * @param {*} value 任意待展示值。
 * @returns {string} 转义后的文本。
 */
const HTML_ESCAPE_RE = /[&<>"']/g;
const HTML_ESCAPE_MAP = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(value) {
  return String(value ?? "").replace(
    HTML_ESCAPE_RE,
    (char) => HTML_ESCAPE_MAP[char]
  );
}

let toastTimer = null;

/**
 * 页面运行在受限 iframe 中，window.confirm 可能被禁用，
 * 因此用页面内的弹窗代替。
 */
function askConfirm(message, options = {}) {
  return new Promise((resolve) => {
    $("modal-text").textContent = message;
    const modal = $("modal");
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
    const okBtn = $("modal-ok");
    const cancelBtn = $("modal-cancel");
    okBtn.classList.toggle("danger", Boolean(options.danger));
    const finish = (value) => {
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
      okBtn.removeEventListener("click", onOk);
      cancelBtn.removeEventListener("click", onCancel);
      // 确认按钮不是提交按钮：点完立即解除加载锁，避免挡住下一次确认弹窗。
      unlockButton(okBtn);
      resolve(value);
    };
    const onOk = () => finish(true);
    const onCancel = () => finish(false);
    okBtn.addEventListener("click", onOk);
    cancelBtn.addEventListener("click", onCancel);
  });
}

/** 显示短暂的成功或错误提示，并重置之前启动的隐藏计时器。 */
function toast(message, isError = false) {
  const node = $("toast");
  node.textContent = message;
  node.classList.toggle("error", Boolean(isError));
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 2400);
}

/** 复制文本；Clipboard API 在 iframe/WebView 中可能被拒绝，回退到 execCommand。 */
async function copyTextToClipboard(text) {
  const value = String(text ?? "");
  let clipboardError = null;
  if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
    try {
      await navigator.clipboard.writeText(value);
      return;
    } catch (error) {
      clipboardError = error;
    }
  }
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "-9999px";
  textarea.style.left = "-9999px";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  try {
    textarea.focus();
    textarea.select();
    textarea.setSelectionRange(0, value.length);
    if (!document.execCommand("copy")) {
      throw clipboardError || new Error("execCommand copy failed");
    }
  } finally {
    textarea.remove();
  }
}

/** 进行中的接口请求数；按钮加载锁据此判断“请求是否真的结束”（F42）。 */
let apiInFlight = 0;
/** 已发起的接口请求总数，用于判断点击后是否真的发出了请求。 */
let apiStartedCount = 0;

/**
 * 经 AstrBot 插件视图 bridge 调用接口，并统一处理 success/error 包装。
 * @throws {Error} 页面未在 AstrBot 内打开，或接口返回失败时抛出。
 */
async function callApi(method, endpoint, payload) {
  // bridge 不存在通常表示用户绕过 AstrBot 单独打开静态页面；此时明确失败，
  // 不伪造空响应，避免渲染成“请求成功但没有数据”的状态。
  if (!bridge) {
    throw new Error("请在 AstrBot 面板中打开该页面");
  }
  apiInFlight += 1;
  apiStartedCount += 1;
  try {
    const response =
      method === "GET"
        ? await bridge.apiGet(endpoint, payload || {})
        : await bridge.apiPost(endpoint, payload || {});
    if (response && response.success === false) {
      throw new Error(response.error || "请求失败");
    }
    if (response && response.data !== undefined) {
      return response.data;
    }
    return response;
  } finally {
    apiInFlight = Math.max(0, apiInFlight - 1);
    // 请求结束（成功或失败）后立即尝试解锁；未满最短展示时长的按钮由
    // 轮询计时器在到点后释放。
    releaseFinishedButtonLocks();
  }
}

/** 将计数格式化为当前语言环境的千位分隔数字。 */
function formatNumber(value) {
  return Number(value || 0).toLocaleString("zh-CN");
}

/** 把秒级时间戳格式化为时刻文本；缺失或非法值返回 "-"（今天只显示 HH:MM）。 */
function formatClockTime(seconds) {
  const value = Number(seconds || 0);
  if (!Number.isFinite(value) || value <= 0) return "-";
  const date = new Date(value * 1000);
  if (Number.isNaN(date.getTime())) return "-";
  const pad = (part) => String(part).padStart(2, "0");
  const clock = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  return sameDay ? clock : `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${clock}`;
}

/** 清除配置提示末尾多余的中文/英文句号，统一表单文案样式。 */
function cleanHint(text) {
  return String(text || "").trim().replace(/[。.]+$/, "");
}

/** 按点分隔路径读取 schema 对应的嵌套配置值。 */
function getPath(obj, path) {
  let current = obj;
  for (const key of String(path).split(".")) {
    if (!current || typeof current !== "object") return undefined;
    current = current[key];
  }
  return current;
}

/** 按点分隔路径创建缺失的对象节点并写入表单字段值。 */
function setPath(obj, path, value) {
  const keys = String(path).split(".");
  let current = obj;
  for (let i = 0; i < keys.length - 1; i += 1) {
    if (!current[keys[i]] || typeof current[keys[i]] !== "object") {
      current[keys[i]] = {};
    }
    current = current[keys[i]];
  }
  current[keys[keys.length - 1]] = value;
}

/** 固定占位图：任何头像处理失败时的兜底，保证页面渲染不中断。 */
const SAFE_AVATAR_FALLBACK = "./assets/avatar-fallback.svg";

/** 生成 QQ 群头像地址；只接受纯数字群号，其它值回退本地占位图。 */
function avatarUrl(groupId) {
  const gid = String(groupId || "").trim();
  return /^\d{5,20}$/.test(gid)
    ? `https://p.qlogo.cn/gh/${gid}/${gid}/100`
    : "";
}

/** 头像加载失败时使用 views/console/assets 下的静态占位图。 */
function fallbackAvatar(_name) {
  return SAFE_AVATAR_FALLBACK;
}

/* ------------------------------------------------------------------ */
/* 主题                                                                */
/* ------------------------------------------------------------------ */
const THEME_KEY = "areply.theme";
const THEME_CYCLE = ["system", "light", "dark"];
const THEME_LABELS = {
  system: "跟随系统",
  light: "日间模式",
  dark: "夜间模式",
};
const THEME_ICON_CLASSES = {
  system: "icon-theme-system",
  light: "icon-theme-light",
  dark: "icon-theme-dark",
};

/** 将 AstrBot 面板提供的主题状态映射到 body 属性与状态徽标。 */
function applyTheme(isDark) {
  document.body.dataset.theme = isDark ? "dark" : "light";
  $("theme-chip").textContent = isDark ? "深色" : "浅色";
}

/** 读取本地保存的主题偏好；默认跟随系统。 */
function loadThemeMode() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (THEME_CYCLE.includes(saved)) return saved;
  } catch (error) {
    console.warn("读取主题偏好失败", error);
  }
  return "system";
}

/** 把主题图标切换到对应形状（带 Morphicons 形变）。 */
function setThemeIcon(mode) {
  const key = THEME_CYCLE.includes(mode) ? mode : "system";
  if (themeMorph && MORPH_PATHS[`theme-${key}`]) {
    try {
      themeMorph.morphTo(MORPH_PATHS[`theme-${key}`]);
      return;
    } catch (error) {
      themeMorph = null;
    }
  }
  const next = THEME_ICON_CLASSES[key];
  const morphPath = document.getElementById("theme-morph-path");
  if (morphPath && MORPH_PATHS[`theme-${key}`]) {
    morphPath.setAttribute("d", MORPH_PATHS[`theme-${key}`]);
    return;
  }
  const node = $("theme-icon");
  if (!node) return;
  for (const cls of Object.values(THEME_ICON_CLASSES)) {
    node.classList.toggle(cls, cls === next);
  }
}

/** 应用主题模式：system 跟随系统，light/dark 为手动覆盖。 */
function applyThemeMode(mode, persist = false) {
  const nextMode = THEME_CYCLE.includes(mode) ? mode : "system";
  state.themeMode = nextMode;
  const isDark =
    nextMode === "dark" || (nextMode === "system" && Boolean(state.systemDark));
  applyTheme(isDark);
  setThemeIcon(nextMode);
  const button = $("theme-btn");
  if (button) {
    button.title = `主题：${THEME_LABELS[nextMode]}（点击切换）`;
    button.setAttribute("aria-label", `当前主题：${THEME_LABELS[nextMode]}，点击切换`);
  }
  if (persist) {
    try {
      localStorage.setItem(THEME_KEY, nextMode);
    } catch (error) {
      console.warn("保存主题偏好失败", error);
    }
  }
}

/** 循环切换：跟随系统 → 日间 → 夜间 → 跟随系统。 */
function cycleTheme() {
  const index = THEME_CYCLE.indexOf(state.themeMode);
  const next = THEME_CYCLE[(index + 1) % THEME_CYCLE.length];
  applyThemeMode(next, true);
  toast(`主题：${THEME_LABELS[next]}`);
}

/* ------------------------------------------------------------------ */
/* 抽屉与视图切换                                                      */
/* ------------------------------------------------------------------ */
/** 切换主视图与导航选中态，并按页面类型触发对应数据加载。 */
let scrolledRaf = 0;

/** 在 Tab 栏前放一个固定锚点；无论 Tab 是否搬走，都用它的位置判定。 */
function ensureTabsAnchor(head) {
  if (head._tabsAnchor && head._tabsAnchor.isConnected) return head._tabsAnchor;
  const anchor = document.createElement("div");
  anchor.className = "tabs-flow-anchor";
  head.insertAdjacentElement("beforebegin", anchor);
  head._tabsAnchor = anchor;
  return anchor;
}

/** 给离开原位置的 Tab 栏加占位，避免正文跳动。 */
function ensureTabsGap(head, height = 0) {
  let gap = document.createElement("div");
  gap.className = "tabs-flow-gap";
  gap.style.height = `${Math.max(0, Math.round(height))}px`;
  head.insertAdjacentElement("afterend", gap);
  return gap;
}

/** 将 Tab 栏搬回内容流原位置。 */
function restoreTabsHead(head) {
  if (!head || !head._tabsHome) return;
  head._tabsHome.insertBefore(head, head._tabsGap);
  if (head._tabsGap) head._tabsGap.remove();
  delete head._tabsHome;
  delete head._tabsGap;
  head.classList.remove("tabs-in-topbar");
}

function restoreAllTabsHeads() {
  document.querySelectorAll(".sticky-head.tabs-in-topbar").forEach(restoreTabsHead);
}

/** FLIP 包裹：位移用 translate，长度用 Tab 表面的 scaleX 视觉收缩，不裁切、不改布局。 */
function animateTabsChange(head, change) {
  head._tabsBusy = true;
  const oldRect = head.getBoundingClientRect();
  change();
  const newRect = head.getBoundingClientRect();
  if (!oldRect.width || !oldRect.height || !newRect.width || !newRect.height) {
    head._tabsBusy = false;
    return;
  }
  const dx = oldRect.left - newRect.left;
  const dy = oldRect.top - newRect.top;
  const startScale = Math.max(0.2, Math.min(1.8, oldRect.width / newRect.width));
  const surface = head.querySelector(".seg") || head;
  surface.getAnimations?.().forEach((animation) => animation.cancel());
  surface.style.transformOrigin = "left center";
  let finished = false;
  let tabAnimations = [];
  const finish = () => {
    if (finished) return;
    finished = true;
    tabAnimations.forEach((animation) => animation.cancel());
    tabAnimations = [];
    head._tabsBusy = false;
    head.style.transform = "";
    surface.style.transform = "";
    surface.style.transformOrigin = "";
    window.setTimeout(() => updateScrolledClass(), 0);
  };
  if (typeof head.animate === "function") {
    tabAnimations = [
      head.animate(
        [
          { transform: `translate(${dx}px, ${dy}px)` },
          { transform: "translate(0, 0)" },
        ],
        {
          duration: 500,
          easing: "cubic-bezier(0.22, 0.7, 0.2, 1)",
          fill: "both",
        }
      ),
      surface.animate(
        [{ transform: `scaleX(${startScale})` }, { transform: "scaleX(1)" }],
        {
          duration: 500,
          easing: "cubic-bezier(0.22, 0.7, 0.2, 1)",
          fill: "both",
        }
      ),
    ];
    window.setTimeout(finish, 620);
    return;
  }
  head.style.transition = "none";
  head.style.transform = `translate(${dx}px, ${dy}px)`;
  surface.style.transform = `scaleX(${startScale})`;
  void head.offsetWidth;
  requestAnimationFrame(() => {
    head.style.transition = "transform 0.5s cubic-bezier(0.22, 0.7, 0.2, 1)";
    surface.style.transition = "transform 0.5s cubic-bezier(0.22, 0.7, 0.2, 1)";
    head.style.transform = "";
    surface.style.transform = "scaleX(1)";
    head.addEventListener("transitionend", finish, { once: true });
    window.setTimeout(finish, 620);
  });
}

/** 把 Tab 栏搬进浮动顶栏，位于菜单和主题按钮之间。 */
function mountTabsHead(head) {
  if (!head || head._tabsHome) return;
  const home = head.parentElement;
  if (!home) return;
  const topbar = document.querySelector(".floating-controls");
  const theme = topbar ? topbar.querySelector(".topbar-right") : null;
  if (!topbar || !theme) return;
  const oldRect = head.getBoundingClientRect();
  const style = getComputedStyle(head);
  const flowHeight =
    head.offsetHeight +
    (parseFloat(style.marginTop) || 0) +
    (parseFloat(style.marginBottom) || 0);
  ensureTabsAnchor(head);
  const gap = ensureTabsGap(head, flowHeight);
  gap.style.overflowAnchor = "none";
  head.style.overflowAnchor = "none";
  head._tabsHome = home;
  head._tabsGap = gap;
  head.classList.add("tabs-in-topbar");
  animateTabsChange(head, () => {
    topbar.insertBefore(head, theme);
  });
}

/** 滚动事件本身已按帧合并，直接响应，不再额外延迟。 */
function updateTabsAtTop() {
  applyTabsAtTop();
}

/** 根据当前位置决定 Tab 是否搬入/搬出顶栏。 */
function applyTabsAtTop() {
  const active = document.querySelector(".view.active");
  const ownHead = active ? active.querySelector(".sticky-head") : null;
  const mounted = document.querySelector(
    ".floating-controls > .sticky-head.tabs-in-topbar"
  );
  const head =
    ownHead ||
    (mounted && active && mounted._tabsHome && active.contains(mounted._tabsHome)
      ? mounted
      : null);
  const tabs = head ? head.querySelector(".tabs.seg") : null;
  const capsule = document.querySelector(".menu-capsule");
  const capBottom = capsule ? capsule.getBoundingClientRect().bottom : 0;

  // 手机端保留原方案：只用原 Tab 位置计算顶部渐隐，不做搬移。
  if (!isDesktopLayout()) {
    restoreAllTabsHeads();
    const mobileTabs = active ? active.querySelector(".tabs.seg") : null;
    let atTop = false;
    let fadeH = 0;
    if (mobileTabs) {
      const rect = mobileTabs.getBoundingClientRect();
      atTop = rect.top <= capBottom + rect.height + 8;
      fadeH = Math.max(capBottom + 10, rect.bottom + 10);
    }
    document.body.classList.toggle("tabs-at-top", atTop);
    if (atTop) {
      document.body.style.setProperty("--tabs-fade-h", `${Math.round(fadeH)}px`);
    } else {
      document.body.style.removeProperty("--tabs-fade-h");
    }
    return;
  }

  if (isKeyboardViewportOpen()) return;
  if (head && head._tabsBusy) return;
  // 切页后旧页的 Tab 可能还挂在顶栏，先搬回旧页。
  document.querySelectorAll(".sticky-head.tabs-in-topbar").forEach((node) => {
    if (node !== head) restoreTabsHead(node);
  });
  if (!tabs || !head) {
    restoreAllTabsHeads();
    document.body.classList.remove("tabs-at-top");
    return;
  }
  // 直接看固定锚点是否进入顶栏；锚点不随 Tab 搬移而变化。
  // 动画期间的 busy 锁避免同一帧反复触发。
  const anchorRect = ensureTabsAnchor(head).getBoundingClientRect();
  const atTop = anchorRect.top <= capBottom + 8;
  if (atTop && !head._tabsHome) {
    mountTabsHead(head);
  } else if (!atTop && head._tabsHome) {
    animateTabsChange(head, () => restoreTabsHead(head));
  }
  document.body.classList.toggle("tabs-at-top", atTop);
  if (atTop) {
    document.body.style.setProperty(
      "--tabs-fade-h",
      `${Math.round(capBottom + 10)}px`
    );
  } else {
    document.body.style.removeProperty("--tabs-fade-h");
  }
}

/** 页面是否已滚离顶部：控制顶部渐隐，避免正文穿到浮层按钮下面。 */
function pageScrollTop() {
  const pane = document.querySelector("#main-pane");
  if (pane) {
    const overflowY = getComputedStyle(pane).overflowY;
    if (
      (overflowY === "auto" || overflowY === "scroll") &&
      pane.scrollHeight > pane.clientHeight
    ) {
      return pane.scrollTop;
    }
  }
  return window.scrollY || 0;
}

/** 群信息条从原群头像位置缩放移动到顶栏，或反向退回。 */
function animateGroupMiniHead(show) {
  const mini = $("group-mini-head");
  const avatar = $("group-avatar");
  if (!mini || !avatar) return;
  const avatarRect = avatar.getBoundingClientRect();
  const topbar = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--topbar-h")
  ) || 56;
  if (!avatarRect.width || !avatarRect.height) {
    mini.classList.toggle("is-visible", show);
    mini.setAttribute("aria-hidden", show ? "false" : "true");
    return;
  }
  // 大卡已滚出顶栏时不再从屏幕外飞入，交给 CSS 在顶栏原位淡入/淡出。
  if (avatarRect.bottom <= topbar + 4) {
    if (!show) {
      mini.classList.remove("is-visible");
      mini.setAttribute("aria-hidden", "true");
    }
    return;
  }
  const targetRect = mini.getBoundingClientRect();
  const fromRect = show ? avatarRect : targetRect;
  const toRect = show ? targetRect : avatarRect;
  if (!fromRect.width || !toRect.width) return;
  const dx = fromRect.left - toRect.left;
  const dy = fromRect.top - toRect.top;
  const sx = fromRect.width / toRect.width;
  const sy = fromRect.height / toRect.height;
  mini.getAnimations?.().forEach((item) => item.cancel());
  mini.style.transition = "none";
  if (typeof mini.animate === "function") {
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      mini.style.transition = "none";
      if (!show) {
        mini.classList.remove("is-visible");
        mini.setAttribute("aria-hidden", "true");
      }
      window.setTimeout(() => {
        mini.style.transition = "";
      }, 20);
    };
    const animation = mini.animate(
      [
        {
          transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})`,
          opacity: show ? 0.35 : 1,
        },
        { transform: "translate(0, 0) scale(1)", opacity: show ? 1 : 0 },
      ],
      {
        duration: 500,
        easing: "cubic-bezier(0.22, 1, 0.36, 1)",
        fill: "both",
      }
    );
    animation.onfinish = () => {
      animation.cancel();
      finish();
    };
    window.setTimeout(finish, 650);
  }
}

/** 手机端群页：群信息滚过顶栏后，把头像/群名/群号放进顶栏按钮中间。 */
function updateGroupHeadStuck() {
  const card = $("group-head-card");
  const mini = $("group-mini-head");
  const panel = $("group-panel");
  const active =
    !isDesktopLayout() &&
    state.view === "groups" &&
    panel &&
    !panel.classList.contains("hidden") &&
    card;
  if (!active) {
    if (mini) {
      mini.classList.remove("is-visible");
      mini.setAttribute("aria-hidden", "true");
    }
    document.body.classList.remove("group-head-stuck");
    return;
  }
  const topbar = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--topbar-h")
  ) || 56;
  const tabs = document.querySelector(".view.active .tabs.seg");
  // 等 Tab 栏接近/贴到顶栏时，再把群信息大卡收成胶囊。
  const stuck = Boolean(
    tabs && tabs.getBoundingClientRect().top <= topbar + 80
  );
  const wasVisible = mini.classList.contains("is-visible");
  document.body.classList.toggle("group-head-stuck", stuck);
  if (stuck !== wasVisible) {
    mini.classList.toggle("is-visible", stuck);
    mini.setAttribute("aria-hidden", stuck ? "false" : "true");
  }
}

function updateScrolledClass() {
  const pane = document.querySelector("#main-pane");
  const paneScrolled = Boolean(
    pane &&
      (getComputedStyle(pane).overflowY === "auto" ||
        getComputedStyle(pane).overflowY === "scroll") &&
      pane.scrollTop > 8
  );
  document.body.classList.toggle("scrolled", paneScrolled || window.scrollY > 8);
  updateGroupHeadStuck();
  updateTabsAtTop();
}

let panelClosingTimer = 0;

/** 面板回位期间隐藏顶部渐隐，等卡片/遮罩动效结束再按滚动状态淡入，避免闪一下。 */
function beginPanelClose() {
  document.body.classList.add("panel-closing");
  clearTimeout(panelClosingTimer);
  panelClosingTimer = setTimeout(() => {
    document.body.classList.remove("panel-closing");
    updateScrolledClass();
  }, 520);
}

function scheduleScrolledClass() {
  if (scrolledRaf) return;
  scrolledRaf = requestAnimationFrame(() => {
    scrolledRaf = 0;
    updateScrolledClass();
  });
}

function switchView(name) {
  state.view = name;
  document.querySelectorAll(".view").forEach((node) => {
    node.classList.toggle("active", node.id === `view-${name}`);
  });
  document.querySelectorAll(".nav-item").forEach((node) => {
    node.classList.toggle("active", node.dataset.view === name);
  });
  updateScrolledClass();
  pushScrollY = 0;
  if (!isDesktopLayout()) closeDrawer();
  // r3.4o：切页不再自动滚动到表单位置，避免设置页 Tab 栏被强制吸顶跳转；
  // 页面保持进入前的滚动位置。
  if (apiInFlight === 0 && lockedButtons.size) {
    Array.from(lockedButtons.keys()).forEach((button) => unlockButton(button));
  }
  recoverUiState("switch-view");
  // r3.3：开发者页每次进入都必须重新输入密钥，离开时立即清掉本地令牌；
  // dev 分支不自动加载任何数据，只把焦点放到密钥输入框。
  resetDevGate();
  // r3.3w：不再自动 focus 密钥输入框——无手势自动 focus 会直接弹系统键盘，
  // 是开发者页触发视口错位/键盘卡死的高频路径。
  if (name === "overview") {
    void loadOverview();
  } else if (name === "groups") {
    void loadGroups();
  } else if (name === "rag") {
    syncRagArchivedButton();
    void loadAutoMemory();
    void loadRagStats(true);
    void loadRagList();
  } else if (name === "blocks") {
    void loadBlocks();
  } else if (name === "settings") {
    void loadGlobal();
  }
}

const MORPH_BASE = "./assets/morph";
const MORPH_PATHS = {};
const BUTTON_MORPH_PATHS = {};
const BUTTON_ICON_NAMES = [
  "activity", "archive", "arrow-right", "check", "close", "database", "eye",
  "funnel", "inbox", "layers", "left", "message", "refresh", "reset", "right",
  "search", "select", "settings", "shield", "sliders", "sparkles", "theme-dark",
  "theme-light", "theme-system", "trash", "users", "wrench",
];
let menuMorph = null;
let themeMorph = null;
let morphicons = null;

/** 从 views/console/assets/morph 的运行时副本读取单路径 SVG，供 Morphicons 形变。 */
async function loadMorphPath(name) {
  const response = await fetch(`${MORPH_BASE}/${name}.svg`, { cache: "force-cache" });
  if (!response.ok) throw new Error(`morph ${name} ${response.status}`);
  const text = await response.text();
  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  const d = doc.querySelector("path")?.getAttribute("d") || "";
  if (!d) throw new Error(`morph ${name} has no path`);
  return d;
}

/** 读取一个图标 SVG 的全部 path 数据（多子路径合并成一个 d）。 */
async function loadIconPath(name) {
  const response = await fetch(`./assets/icons/i-${name}.svg`, {
    cache: "force-cache",
  });
  if (!response.ok) throw new Error(`icon ${name} ${response.status}`);
  const text = await response.text();
  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  const d = [...doc.querySelectorAll("path")]
    .map((node) => node.getAttribute("d") || "")
    .filter(Boolean)
    .join(" ");
  if (!d) throw new Error(`icon ${name} has no path`);
  return d;
}

/** 预加载按钮图标路径，供加载态 Morphicons 使用。 */
async function preloadButtonMorphPaths() {
  if (BUTTON_MORPH_PATHS.refresh) return;
  const values = await Promise.all(
    BUTTON_ICON_NAMES.map(async (name) => {
      try {
        BUTTON_MORPH_PATHS[name] = await loadIconPath(name);
      } catch (error) {
        // 单个图标缺失时只用刷新图标兜底。
      }
    })
  );
  return values;
}

function createMorphIcon(hostId, pathId, d) {
  const host = $(hostId);
  if (!host) return null;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "icon morph-icon");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.id = pathId;
  path.setAttribute("d", d);
  svg.appendChild(path);
  host.replaceWith(svg);
  return path;
}

/** 加载外部 morph 路径并恢复菜单/主题形变动效。 */
async function initMorphIcons() {
  try {
    const names = ["menu", "close", "theme-system", "theme-light", "theme-dark"];
    const values = await Promise.all(names.map((name) => loadMorphPath(name)));
    names.forEach((name, index) => {
      MORPH_PATHS[name] = values[index];
    });
    const { createMorph } = await import("./vendor/morphicons/dom.js");
    morphicons = { createMorph };
    await preloadButtonMorphPaths();
    const menuPath = createMorphIcon("menu-icon", "menu-morph-path", MORPH_PATHS.menu);
    if (menuPath) {
      menuMorph = createMorph(menuPath, MORPH_PATHS.menu);
      menuMorph.set(MORPH_PATHS.menu);
    }
    const themePath = createMorphIcon(
      "theme-icon",
      "theme-morph-path",
      MORPH_PATHS[`theme-${state.themeMode}`] || MORPH_PATHS["theme-system"]
    );
    if (themePath) {
      themeMorph = createMorph(
        themePath,
        MORPH_PATHS[`theme-${state.themeMode}`] || MORPH_PATHS["theme-system"]
      );
      themeMorph.set(
        MORPH_PATHS[`theme-${state.themeMode}`] || MORPH_PATHS["theme-system"]
      );
    }
  } catch (error) {
    // 文件不可达或旧 WebView 不支持动画时，保留 CSS mask 静态图标。
    console.warn("[ActiveReply] Morphicons 外部路径加载失败", error);
  }
}

/** 将汉堡菜单图标在 menu/close 之间切换。 */
function setMenuMorph(open) {
  const next = open ? MORPH_PATHS.close : MORPH_PATHS.menu;
  if (menuMorph && next) {
    try {
      menuMorph.morphTo(next);
      return;
    } catch (error) {
      menuMorph = null;
    }
  }
  const morphPath = document.getElementById("menu-morph-path");
  if (morphPath && next) morphPath.setAttribute("d", next);
  const node = $("menu-icon");
  if (node) {
    node.classList.toggle("icon-menu", !open);
    node.classList.toggle("icon-close", open);
  }
}

let pushScrollY = 0;

/** 卡片锁成 100dvh 时，用内容位移补偿页面滚动，视觉上停在原位。 */
function setContentScrollOffset(scrollY) {
  const content = document.querySelector(
    "#main-pane > .content, #main-pane .content-viewport > .content"
  );
  if (!content) return;
  content.style.transform = scrollY ? `translateY(${-scrollY}px)` : "";
}

/** PC 正文视口宽度用显式 px 动画，避免 100% → calc 不可插值导致瞬移。 */
function animateContentViewport(viewport, fromWidth, toWidth) {
  if (!viewport || !fromWidth || !toWidth || fromWidth === toWidth) return;
  viewport.getAnimations?.().forEach((animation) => animation.cancel());
  viewport.style.width = `${fromWidth}px`;
  void viewport.offsetWidth;
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    viewport.style.width = "";
  };
  if (typeof viewport.animate === "function") {
    const animation = viewport.animate(
      [{ width: `${fromWidth}px` }, { width: `${toWidth}px` }],
      {
        duration: 500,
        easing: "cubic-bezier(0.16, 1, 0.3, 1)",
        fill: "both",
      }
    );
    animation.onfinish = () => {
      animation.cancel();
      finish();
    };
    window.setTimeout(finish, 620);
    return;
  }
  viewport.style.transition = "width 0.5s cubic-bezier(0.16, 1, 0.3, 1)";
  requestAnimationFrame(() => {
    viewport.style.width = `${toWidth}px`;
    window.setTimeout(() => {
      viewport.style.transition = "";
      finish();
    }, 520);
  });
}

/** 展开侧边栏并同步遮罩与无障碍状态。 */
function openDrawer() {
  // 键盘未收起时不做程序化滚动 + transform（WebView 视口错位高发组合）。
  if (document.activeElement && document.activeElement !== document.body) {
    document.activeElement.blur?.();
  }
  // 挤压效果:主内容区加 .is-pushed 向右平移并缩小成卡片;保持原来的卡片缩小动画
  if (!$("drawer").classList.contains("open")) {
    pushScrollY = window.scrollY || 0;
  }
  if (!isDesktopLayout()) window.scrollTo(0, 0);
  updateScrolledClass();
  $("main-pane").classList.add("is-pushed");
  if (isDesktopLayout()) {
    // PC 保持窗口滚动位置，body 锁滚动即可，避免先跳顶再补偿造成闪动。
    setContentScrollOffset(0);
  } else {
    // 手机端沿用内容位移补偿。
    setContentScrollOffset(pushScrollY);
  }
  updateScrolledClass();
  document.body.classList.add("drawer-open");
  $("drawer").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
  $("scrim").classList.add("show");
  $("menu-btn").setAttribute("aria-expanded", "true");
  $("menu-btn").setAttribute("aria-label", "关闭菜单");
  setMenuMorph(true);
}

/** 关闭侧边栏并恢复遮罩与按钮无障碍状态。 */
function closeDrawer() {
  const wasOpen = $("drawer").classList.contains("open");
  beginPanelClose();
  const pane = $("main-pane");
  // 手势中断等异常路径也要清掉内联位移,否则卡片会停在半开位置
  pane.style.transition = "";
  pane.style.transform = "";
  pane.style.width = "";
  pane.style.borderRadius = "";
  pane.style.boxShadow = "";
  const veil = document.querySelector(".pane-veil");
  if (veil) veil.style.opacity = "";
  const drawer = $("drawer");
  if (drawer) {
    drawer.style.transition = "";
    drawer.style.transform = "";
  }
  clearFloatingControlsInline();
  // 关键：关闭时必须移除推挤状态，否则抽屉已收起但主页仍停在右侧。
  pane.classList.remove("is-pushed");
  if (!isDesktopLayout()) pane.scrollTop = 0;
  setContentScrollOffset(0);
  document.body.classList.remove("drawer-open");
  if (wasOpen && !isDesktopLayout()) window.scrollTo(0, pushScrollY);
  updateScrolledClass();
  $("drawer").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  $("scrim").classList.remove("show");
  $("menu-btn").setAttribute("aria-expanded", "false");
  $("menu-btn").setAttribute("aria-label", "打开菜单");
  setMenuMorph(false);
}

/* ------------------------------------------------------------------ */
/* 挤压手势：从屏幕左边缘右滑，把主面板推开露出抽屉                    */
/* ------------------------------------------------------------------ */
const OPEN_RATIO = 0.78; // 抽屉占屏宽比例(与 --drawer-w 保持一致)
const PANE_RADIUS = 22; // 推开后的卡片圆角(与 CSS --pane-radius 一致)
const DRAWER_SCALE = 0.95; // Push Drawer：侧栏 95%→100%，主面板 100%→95%
const OPEN_THRESHOLD = 0.3; // 滑过屏宽 30% 视为完全打开
let dragState = null;

/** 完全打开时主面板应右移的距离。 */
function paneOpenWidth() {
  const drawer = $("drawer");
  return drawer ? Math.round(drawer.offsetWidth) : Math.round(window.innerWidth * OPEN_RATIO);
}

function isDesktopLayout() {
  return window.matchMedia("(min-width: 900px)").matches;
}

/** 抽屉当前是否处于打开状态。 */
function isDrawerOpen() {
  return $("drawer").classList.contains("open");
}

/** 拖拽过程中实时设置主面板位移、圆角与阴影(随距离渐变)。 */
function dragPaneTo(offsetX) {
  const pane = $("main-pane");
  const width = paneOpenWidth() || 1;
  const offset = Math.max(0, Math.min(offsetX, width));
  const progress = offset / width;
  pane.style.transition = "none";
  const scale = 1 - (1 - DRAWER_SCALE) * progress;
  if (isDesktopLayout()) {
    // PC：主面板收窄到剩余宽度，只挤压、不把右侧移出视口。
    pane.style.width = `calc(100vw - ${offset}px)`;
    pane.style.transform = `translateX(${offset}px)`;
  } else {
    pane.style.transform = `translateX(${offset}px) scale(${scale.toFixed(3)})`;
  }
  const drawer = $("drawer");
  if (drawer) {
    drawer.style.transition = "none";
    drawer.style.transform = `translateX(${(-(1 - progress) * 100).toFixed(2)}%) scale(${(DRAWER_SCALE + (1 - DRAWER_SCALE) * progress).toFixed(3)})`;
  }
  pane.style.borderRadius =
    progress > 0 ? `${Math.round(PANE_RADIUS * progress)}px` : "";
  pane.style.boxShadow =
    progress > 0
      ? `-12px 0 44px rgba(0, 0, 0, ${(0.12 * progress).toFixed(3)})`
      : "";
  const veil = document.querySelector(".pane-veil");
  if (veil) {
    veil.style.transition = "none";
    veil.style.opacity = progress.toFixed(3);
  }
  const controls = document.querySelector(".floating-controls");
  if (controls) {
    controls.style.transition = "none";
    if (isDesktopLayout()) {
      controls.style.left = `${offset + 20}px`;
      controls.style.transform = "none";
    } else {
      controls.style.transform = `translateX(${offset}px) scale(${scale.toFixed(3)})`;
    }
  }
}

/** 清理浮层按钮的拖拽内联样式；开关动作结束后交还给 body 类。 */
function clearFloatingControlsInline() {
  const controls = document.querySelector(".floating-controls");
  if (!controls) return;
  controls.style.transition = "";
  controls.style.transform = "";
}

/** 手势结束：清理内联样式,超过阈值则完全打开,否则回弹关闭。 */
function endPaneDrag(commit) {
  const pane = $("main-pane");
  pane.style.transition = "";
  pane.style.transform = "";
  pane.style.width = "";
  pane.style.borderRadius = "";
  pane.style.boxShadow = "";
  const drawer = $("drawer");
  if (drawer) {
    drawer.style.transition = "";
    drawer.style.transform = "";
  }
  const veil = document.querySelector(".pane-veil");
  if (veil) {
    veil.style.transition = "";
    veil.style.opacity = "";
  }
  const controls = document.querySelector(".floating-controls");
  if (controls) controls.style.left = "";
  clearFloatingControlsInline();
  if (commit) openDrawer();
  else closeDrawer();
}

/** 绑定 Pointer Events 手势(同时覆盖鼠标与触摸)。 */
function bindPushGesture() {
  const pane = $("main-pane");
  const edge = $("pane-edge");
  if (!pane || !edge) return;
  edge.addEventListener("pointerdown", (event) => {
    if (isDesktopLayout() || isDrawerOpen()) return;
    if (event.pointerType === "mouse" && event.button !== 0) return;
    dragState = { startX: event.clientX, moved: 0 };
    try {
      edge.setPointerCapture(event.pointerId);
    } catch (error) {
      /* 老浏览器不支持指针捕获时忽略 */
    }
  });
  edge.addEventListener("pointermove", (event) => {
    if (!dragState) return;
    const delta = event.clientX - dragState.startX;
    dragState.moved = Math.max(dragState.moved, delta);
    dragPaneTo(delta);
    if (delta > 8 && event.cancelable) event.preventDefault();
  });
  edge.addEventListener("pointerup", (event) => {
    if (!dragState) return;
    const delta = Math.max(dragState.moved, event.clientX - dragState.startX);
    dragState = null;
    endPaneDrag(delta > window.innerWidth * OPEN_THRESHOLD);
  });
  edge.addEventListener("pointercancel", () => {
    if (!dragState) return;
    dragState = null;
    endPaneDrag(false);
  });
}

/** 从推开状态左滑收回：在露出的卡片区域拖动，松手按 30% 阈值吸附。 */
function bindCloseGesture() {
  const scrim = $("scrim");
  if (!scrim) return;
  let drag = null;
  let suppressClick = false;
  let clickTimer = null;

  scrim.addEventListener("pointerdown", (event) => {
    if (isDesktopLayout() || !isDrawerOpen()) return;
    if ($("detail").classList.contains("open")) return; // 详情打开时只接受点击关闭
    if (event.pointerType === "mouse" && event.button !== 0) return;
    drag = { startX: event.clientX, moved: 0 };
    try {
      scrim.setPointerCapture(event.pointerId);
    } catch (error) {
      /* 老浏览器不支持指针捕获时忽略 */
    }
  });
  scrim.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const delta = drag.startX - event.clientX;
    if (delta <= 0) return;
    drag.moved = Math.max(drag.moved, delta);
    if (drag.moved > 8) {
      suppressClick = true;
      clearTimeout(clickTimer);
    }
    dragPaneTo(paneOpenWidth() - delta);
    if (event.cancelable) event.preventDefault();
  });
  scrim.addEventListener("pointerup", (event) => {
    if (!drag) return;
    const moved = Math.max(drag.moved, drag.startX - event.clientX);
    drag = null;
    if (moved <= 8) return; // 没移动：当作点击，交给 click 关闭
    endPaneDrag(moved <= window.innerWidth * OPEN_THRESHOLD);
    // 拖拽后浏览器不一定会补发 click，超时后自动解除抑制
    clearTimeout(clickTimer);
    clickTimer = setTimeout(() => {
      suppressClick = false;
    }, 400);
  });
  scrim.addEventListener("pointercancel", () => {
    if (!drag) return;
    drag = null;
    endPaneDrag(true);
  });
  scrim.addEventListener("click", () => {
    if (isDesktopLayout()) return;
    if (suppressClick) {
      suppressClick = false;
      return;
    }
    if ($("detail").classList.contains("open")) {
      closeDetail();
      return;
    }
    closeDrawer();
  });
}

/* ------------------------------------------------------------------ */
/* 状态（概览）                                                        */
/* ------------------------------------------------------------------ */
/** 组合 Embedding 模型名和 provider ID，供状态区及提示文字显示。 */
function embeddingLabel(item) {
  const model = String(item.model || "").trim();
  const id = String(item.id || "").trim();
  if (model && id && model !== id) return `${model}（ID: ${id}）`;
  return model || id || "";
}

/** 将 Embedding 可用状态转换为记忆 RAG 状态卡的简短说明。 */
function ragEmbeddingHint(embedding) {
  if (!embedding || !embedding.available) return "未配置 Embedding 提供商";
  const providers = embedding.providers || [];
  if (providers.length > 1) return `Embedding 就绪（${providers.length} 个）`;
  const first = providers[0];
  return `Embedding：${first ? embeddingLabel(first) : "—"}`;
}

/** 在全局配置的 Embedding 控件下列出已发现的 provider 和模型名。 */
function decorateEmbeddingHint() {
  if (!state.embeddingProviders.length) return;
  // 字段实际路径为 memory_rag.memory_control.embedding_provider_id（schema 缩进
  // 过一层），旧选择器里的 memory_rag.embedding_provider_id 永远匹配不到，
  // 提示因此从未显示。这里改用后缀匹配，schema 再次搬迁也不会失效。
  const control = document.querySelector(
    '#global-form [data-global-row$="embedding_provider_id"] .control'
  );
  if (!control || control.querySelector(".embedding-options")) return;
  const text = state.embeddingProviders
    .map((item) => embeddingLabel(item))
    .filter(Boolean)
    .join("、");
  const hint = document.createElement("span");
  hint.className = "field-hint embedding-options";
  hint.textContent = `当前可用：${text}（留空自动选择第一个可用）`;
  control.appendChild(hint);
}

/** 生成一张统一样式的状态卡 HTML，调用方传入已确定的展示内容。 */
function statusCard(label, value, hint, tone = "") {
  // 参考图 B:左指标名 15px + 次级说明,右数值 17px/600 + 可选彩色小徽标(tone: on / off)
  return `
    <div class="row status-row">
      <div class="row-main">
        <span class="row-label">${escapeHtml(label)}</span>
        ${hint ? `<span class="row-sub">${escapeHtml(hint)}</span>` : ""}
      </div>
      <div class="row-value ${tone}">${tone ? `<i class="badge ${tone}"></i>` : ""}${value}</div>
    </div>`;
}

/** 渲染状态页「本轮记忆」读数：召回耗时、注入条数、降级原因与后台抽取。 */
function renderMemoryMetrics(metrics) {
  const container = $("overview-metrics");
  if (!container) return;
  // 数据来自 /overview 的 data.metrics（不是 data.stats）；缺字段一律按 0/空对象降级。
  const data = metrics && typeof metrics === "object" ? metrics : {};
  const autoExtract =
    data.auto_extract && typeof data.auto_extract === "object"
      ? data.auto_extract
      : {};
  const number = (value) => {
    const parsed = Number(value || 0);
    return Number.isFinite(parsed) ? parsed : 0;
  };
  const recallMs = number(data.recall_ms);
  const memoryHits = number(data.memory_hits);
  const deadlineHits = number(data.recall_deadline_hit);
  const embeddingTimeouts = number(data.embedding_timeout);
  const pendingMessages = number(autoExtract.pending_messages);
  // 分阶段耗时与并行丢弃：page_api 白名单后才会出现，缺失时按 0 处理且不占位。
  const embeddingWasted = number(data.embedding_wasted);
  const lexicalMs = number(data.lexical_ms);
  const embeddingMs = number(data.embedding_ms);
  const vectorMs = number(data.vector_ms);
  const lexicalTimeouts = number(data.lexical_timeout);

  const reasons = [];
  if (deadlineHits > 0) reasons.push(`召回超时 ${formatNumber(deadlineHits)} 次`);
  if (embeddingTimeouts > 0) {
    reasons.push(`Embedding 超时 ${formatNumber(embeddingTimeouts)} 次`);
  }
  if (lexicalTimeouts > 0) {
    reasons.push(`词法超时 ${formatNumber(lexicalTimeouts)} 次`);
  }
  const recallHints = [`本轮注入 ${formatNumber(memoryHits)} 条记忆`];
  if (lexicalMs > 0) recallHints.push(`词法 ${formatNumber(lexicalMs)}ms`);
  if (embeddingMs > 0) recallHints.push(`向量化 ${formatNumber(embeddingMs)}ms`);
  if (vectorMs > 0) recallHints.push(`语义 ${formatNumber(vectorMs)}ms`);
  const degraded = Boolean(data.degraded);
  const hasMetrics = Object.keys(data).length > 0 && Boolean(data.origin);
  let statusBadge = '<span class="tag on">正常</span>';
  if (!hasMetrics) statusBadge = '<span class="tag off">—</span>';
  else if (degraded) statusBadge = '<span class="tag warn">降级</span>';
  const statusHints = [];
  if (degraded) statusHints.push(reasons.length ? reasons.join(" · ") : "本轮发生降级");
  if (embeddingWasted > 0) {
    statusHints.push(`并行丢弃 ${formatNumber(embeddingWasted)} 次`);
  }
  let statusHint = statusHints.length ? statusHints.join(" · ") : "本轮未触发降级";
  if (!hasMetrics) statusHint = "暂无本轮召回数据";

  container.innerHTML = [
    statusCard("记忆召回", `${formatNumber(recallMs)} ms`, recallHints.join(" · ")),
    statusCard("召回状态", statusBadge, statusHint),
    statusCard(
      "后台抽取",
      `${formatNumber(pendingMessages)} 条待抽取`,
      `最近抽取：${formatClockTime(autoExtract.last_at)}`
    ),
  ].join("");
}

/** 获取插件概览数据并刷新状态页与侧边栏群列表。 */
async function loadOverview(force = false) {
  const requestId = (state.overviewSeq || 0) + 1;
  state.overviewSeq = requestId;
  try {
    const data = await callApi("GET", "overview", force ? { force: 1 } : {});
    if (requestId !== state.overviewSeq) return;
    state.overview = data;
    state.groups = data.groups || [];
    state.providers = data.providers || [];
    state.embeddingProviders =
      (data.embedding && data.embedding.providers) || [];
    renderOverview(data);
    renderDrawerGroups();
  } catch (error) {
    if (requestId === state.overviewSeq) {
      toast(error.message || "加载状态失败", true);
    }
  }
}

/** 根据概览响应渲染状态卡、全局开关、记忆统计和最近活跃群。 */
function renderOverview(data) {
  const s = data.switches || {};
  const stats = data.stats || {};
  // 时区展示已移除
  $("drawer-version").textContent = `v${data.version || "-"} · UI${UI_BUILD}`;

  const modeLabel = s.mode === "model_choice" ? "模型判定" : "概率";
  const onOff = (flag) => (flag ? "on" : "off");
  $("status-grid").innerHTML = [
    statusCard(
      "主动回复",
      s.active_reply ? "已开启" : "已关闭",
      s.active_reply ? `当前方式：${modeLabel}` : "可在群配置中单独开启",
      onOff(s.active_reply)
    ),
    statusCard(
      "群聊上下文增强",
      s.context ? "已开启" : "已关闭",
      s.react_pattern ? "React Pattern 已开启" : "React Pattern 未开启",
      onOff(s.context)
    ),
    statusCard(
      "智能回复",
      s.smart_reply ? "生效中" : "未生效",
      s.react_pattern ? "@ 与引用解析" : "需要 React Pattern",
      onOff(s.smart_reply)
    ),
    statusCard(
      "记忆条目",
      formatNumber(stats.total_memories),
      `${(data.groups || []).length} 个已知群`
    ),
    statusCard(
      "记忆",
      s.memory_rag ? "已开启" : "已关闭",
      ragEmbeddingHint(data.embedding),
      onOff(s.memory_rag)
    ),
    statusCard(
      "联网搜索",
      s.web_search ? "已开启" : "已关闭",
      s.web_search ? "模型可调用 grok_web_search" : "需要时可在全局设置中启用",
      onOff(s.web_search)
    ),
  ].join("");

  // 本轮记忆读数:数据来源是 /overview 的 data.metrics,不是 data.stats。
  renderMemoryMetrics(data.metrics);

  // 记忆库统计:一张分组卡内的多行;0 条时切换为参考图 A 的空状态(静态节点只切 hidden)
  const statsEmpty = !Number(stats.total_memories);
  $("overview-stats").classList.toggle("hidden", statsEmpty);
  $("overview-stats-empty").classList.toggle("hidden", !statsEmpty);
  $("overview-stats-refresh").onclick = () => void loadOverview(true);
  $("overview-stats").innerHTML = [
    ["记忆总数", stats.total_memories],
    ["群范围", stats.group_scope_count],
    ["群数量", stats.group_id_count],
    ["角色数", stats.role_count],
  ]
    .map(
      ([title, value]) => `
      <div class="row stat-line">
        <span class="row-label">${escapeHtml(title)}</span>
        <span class="row-value">${formatNumber(value)}</span>
      </div>`
    )
    .join("");

  // 全局开关:布尔值渲染成只读(灰化)iOS 开关,不再用彩色徽标
  const switches = [
    ["主动回复", s.active_reply],
    ["群聊上下文增强", s.context],
    ["React Pattern", s.react_pattern],
    ["智能回复", s.smart_reply],
    ["联网搜索", s.web_search],
    ["记忆", s.memory_rag],
  ];
  $("overview-switches").innerHTML = switches
    .map(
      ([label, enabled]) => `
      <li class="row">
        <span class="row-label">${escapeHtml(label)}</span>
        <label class="switch readonly" aria-label="${escapeHtml(label)}：${enabled ? "开启" : "关闭"}">
          <input type="checkbox" disabled ${enabled ? "checked" : ""} />
          <span class="slider"></span>
        </label>
      </li>`
    )
    .join("");

  // 最近活跃群:横向条(头像 40 + 群名 13);无数据时显示空状态并复用「前往群配置」
  const groups = (data.groups || []).filter((item) => item.messaged);
  $("overview-groups").parentElement.classList.toggle("hidden", !groups.length);
  $("overview-groups-empty").classList.toggle("hidden", groups.length > 0);
  $("overview-groups-go").onclick = () => {
    state.currentGroupId = "";
    switchView("groups");
  };
  $("overview-groups").innerHTML = groups
    .map((item) => {
      const label = item.name || item.group_id;
      return `
        <button class="group-pill" data-gid="${escapeHtml(item.group_id)}" title="${escapeHtml(label)}">
          <img class="mini-avatar" data-avatar="${escapeHtml(item.group_id)}"
               data-name="${escapeHtml(label)}" alt="" />
          <span>${escapeHtml(label)}</span>
        </button>`;
    })
    .join("");

  $("overview-groups")
    .querySelectorAll(".group-pill")
    .forEach((btn) => {
      const img = btn.querySelector("img[data-avatar]");
      if (img) bindAvatar(img, img.dataset.avatar, img.dataset.name);
      btn.addEventListener("click", () => {
        switchView("groups");
        void openGroup(btn.dataset.gid);
      });
    });
}

/* ------------------------------------------------------------------ */
/* 群配置                                                              */
/* ------------------------------------------------------------------ */
/** 返回群列表中优先展示的群名，没有可用名称时退回“未命名群”。 */
function groupDisplayName(item) {
  return item && item.name ? item.name : "未命名群";
}

/**
 * 绑定群头像：优先加载 QQ 群头像，失败或群号不合法时用本地 SVG 占位图。
 *
 * 头像加载失败只影响当前图片，不能中断整页渲染（此前 emoji 群名会触发
 * `URI malformed`，导致群列表加载报错）。
 */
function bindAvatar(img, groupId, name) {
  if (!img) return;
  let fallback = SAFE_AVATAR_FALLBACK;
  try {
    fallback = fallbackAvatar(name || groupId);
  } catch (error) {
    fallback = SAFE_AVATAR_FALLBACK;
  }
  const setFallback = () => {
    img.onerror = null;
    img.src = fallback;
  };
  try {
    img.onerror = setFallback;
    // 不把控制台页面地址作为 Referer 发给头像 CDN。
    img.referrerPolicy = "no-referrer";
    img.loading = "lazy";
    const remote = avatarUrl(groupId);
    img.src = remote || fallback;
  } catch (error) {
    setFallback();
  }
}

/** 渲染抽屉内的快捷群列表，并绑定头像回退和打开群配置事件。 */
function renderDrawerGroups() {
  const container = $("drawer-groups");
  if (!container) return;
  const groups = state.groups.slice(0, 40);
  $("drawer-group-count").textContent = String(state.groups.length);
  if (!groups.length) {
    container.innerHTML =
      '<p class="muted small pad">暂无群，收到群消息后会自动出现</p>';
    return;
  }
  container.innerHTML = groups
    .map(
      (item) => `
      <button class="drawer-group ${
        item.group_id === state.currentGroupId ? "active" : ""
      }" data-gid="${escapeHtml(item.group_id)}">
        <img class="mini-avatar" data-avatar="${escapeHtml(item.group_id)}"
             data-name="${escapeHtml(groupDisplayName(item))}" alt="" />
        <span class="drawer-group-name">${escapeHtml(groupDisplayName(item))}</span>
        <span class="dot ${item.active_reply ? "on" : ""}"></span>
      </button>`
    )
    .join("");
  container.querySelectorAll(".drawer-group").forEach((btn) => {
    const img = btn.querySelector("img[data-avatar]");
    if (img) bindAvatar(img, img.dataset.avatar, img.dataset.name);
    btn.addEventListener("click", () => {
      switchView("groups");
      void openGroup(btn.dataset.gid);
    });
  });
}

/** 按搜索词过滤群列表、渲染条目并绑定群配置打开事件。 */
function renderGroupList() {
  const keyword = state.groupSearch.trim().toLowerCase();
  const groups = state.groups.filter((item) => {
    if (!keyword) return true;
    return (
      String(item.group_id).includes(keyword) ||
      String(item.name || "").toLowerCase().includes(keyword)
    );
  });

  const container = $("group-items");
  if (!groups.length) {
    // 空列表:居中空状态(参考图 A),按钮复用现有「清除搜索 / 刷新」逻辑,不新增接口
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">${icon("users")}</div>
        <h3>${keyword ? "没有匹配的群" : "暂无群"}</h3>
        <p class="muted">${
          keyword
            ? "换个关键词试试，或直接在右侧输入群号打开"
            : "收到群消息后会自动出现，也可以在右侧输入群号"
        }</p>
        <button type="button" class="primary pill-btn" id="group-empty-action">${
          keyword ? "清除搜索" : "刷新列表"
        }</button>
      </div>`;
    $("group-empty-action").addEventListener("click", () => {
      if (keyword) {
        state.groupSearch = "";
        $("group-search").value = "";
        renderGroupList();
      } else {
        void loadGroups(true);
      }
    });
    renderDrawerGroups();
    return;
  }

  container.innerHTML = groups
    .map(
      (item) => `
      <button class="group-item ${
        item.group_id === state.currentGroupId ? "active" : ""
      }" data-gid="${escapeHtml(item.group_id)}">
        <img class="mini-avatar" data-avatar="${escapeHtml(item.group_id)}"
             data-name="${escapeHtml(groupDisplayName(item))}" alt="" />
        <span class="group-item-text">
          <span class="gname">${escapeHtml(groupDisplayName(item))}</span>
          <span class="gid">${escapeHtml(item.group_id)}</span>
        </span>
        <span class="dot ${item.active_reply ? "on" : ""}"></span>
        ${icon("right").replace('class="icon"', 'class="icon chevron"')}
      </button>`
    )
    .join("");

  container.querySelectorAll(".mini-avatar").forEach((img) => {
    bindAvatar(img, img.dataset.avatar, img.dataset.name);
  });
  container.querySelectorAll(".group-item").forEach((btn) => {
    btn.addEventListener("click", () => void openGroup(btn.dataset.gid));
  });
  renderDrawerGroups();
}

/** 获取群目录；已有缓存时直接重绘，force 用于请求后端强制刷新。 */
async function loadGroups(force = false) {
  if (state.groups.length && !force) {
    renderGroupList();
    return;
  }
  const requestId = (state.groupsSeq || 0) + 1;
  state.groupsSeq = requestId;
  // 只有首次加载显示骨架；保存后强制刷新沿用现有列表。
  if (!state.groupsLoaded) showGroupsSkeleton();
  try {
    const data = await callApi("GET", "groups", force ? { force: 1 } : {});
    if (requestId !== state.groupsSeq) return;
    state.groups = data.groups || [];
    state.groupsLoaded = true;
    renderGroupList();
    renderDrawerGroups();
  } catch (error) {
    if (requestId === state.groupsSeq) {
      toast(error.message || "加载群列表失败", true);
    }
  }
}

/** 按回复模式决定群级表单的条件字段是否可见。 */
function fieldVisible(field, values) {
  if (field.path === "active_reply.possibility") {
    return values["active_reply.mode"] === "probability";
  }
  if (
    field.path === "active_reply.model_history_messages" ||
    field.path === "active_reply.model_choice_provider_id"
  ) {
    return values["active_reply.mode"] === "model_choice";
  }
  return true;
}

/** 生成 embedding provider 下拉；值为 provider ID，留空表示自动取第一个可用。 */
function embeddingProviderOptions(selected) {
  const providers = state.embeddingProviders || [];
  const options = [`<option value="">自动（第一个可用）</option>`];
  providers.forEach((item) => {
    const id = String(item.id || "").trim();
    if (!id) return;
    const label = embeddingLabel(item) || id;
    options.push(
      `<option value="${escapeHtml(id)}" ${
        id === selected ? "selected" : ""
      }>${escapeHtml(label)}</option>`
    );
  });
  // 配置里可能是当前不可用的旧 ID，保留它以免保存时被悄悄改掉。
  if (selected && !providers.some((item) => String(item.id || "").trim() === selected)) {
    options.push(
      `<option value="${escapeHtml(selected)}" selected>${escapeHtml(selected)}</option>`
    );
  }
  return options.join("");
}

/** 生成 provider 下拉选项，同时保留当前配置中已不可用的旧 ID。 */
function providerOptions(selected) {
  const options = [
    `<option value="">跟随会话默认</option>`,
    ...state.providers.map(
      (item) =>
        `<option value="${escapeHtml(item.id)}" ${
          item.id === selected ? "selected" : ""
        }>${escapeHtml(item.label || item.id)}</option>`
    ),
  ];
  if (selected && !state.providers.some((item) => item.id === selected)) {
    options.push(
      `<option value="${escapeHtml(selected)}" selected>${escapeHtml(selected)}</option>`
    );
  }
  return options.join("");
}

/** 设置行布局:开关行(右对齐)/ 行内行(下拉、滑块、短文本)/ 堆叠行(多行文本)。 */
function rowLayout(type) {
  if (type === "bool") return "row-switch";
  if (type === "text" || type === "list" || type === "dict") return "row-stack";
  if (type === "int" || type === "float") return "row-range";
  return "row-inline";
}

/** 按群覆盖字段元数据生成开关、选项、滑块或文本控件。 */
function buildControl(field, value) {
  const path = field.path;
  if (field.kind === "bool") {
    return `
      <label class="switch">
        <input type="checkbox" data-path="${path}" ${value ? "checked" : ""} />
        <span class="slider"></span>
      </label>`;
  }
  if (field.kind === "enum") {
    const options = field.options
      .map(
        (option, index) =>
          `<option value="${escapeHtml(option)}" ${
            option === value ? "selected" : ""
          }>${escapeHtml(field.labels[index] || option)}</option>`
      )
      .join("");
    return `<select data-path="${path}" data-kind="enum">${options}</select>`;
  }
  if (PROVIDER_PATHS.has(path)) {
    return `<select data-path="${path}" data-kind="str">${providerOptions(
      value || ""
    )}</select>`;
  }
  if (field.kind === "int" || field.kind === "float") {
    const step = field.step || (field.kind === "int" ? 1 : 0.1);
    return `
      <input type="range" data-path="${path}" data-kind="${field.kind}"
        min="${escapeHtml(field.min)}" max="${escapeHtml(field.max)}"
        step="${escapeHtml(step)}" value="${escapeHtml(value)}" />
      <span class="range-value" data-for="${path}">${escapeHtml(value)}</span>`;
  }
  if (field.kind === "text") {
    return `<textarea data-path="${path}" data-kind="text">${escapeHtml(value || "")}</textarea>`;
  }
  return `<input type="text" data-path="${path}" data-kind="str" value="${escapeHtml(value || "")}" />`;
}

/** 同步顶栏的独立配置按钮高亮。 */
function syncGroupTopActions() {
  const button = $("group-mini-independent");
  const toggle = $("independent-toggle");
  if (!button || !toggle) return;
  const active = Boolean(toggle.checked);
  button.classList.toggle("active", active);
  button.setAttribute("aria-pressed", active ? "true" : "false");
  const saveButton = $("group-mini-save");
  if (saveButton) saveButton.disabled = $("save-group")?.disabled || false;
}

/** 组合当前群的三组配置表单，并重新绑定控件与 Tab 事件。 */
function renderGroupPanel() {
  const data = state.currentGroup;
  if (!data) return;
  const values = data.values || {};
  const independent = Boolean(data.independent);

  $("group-empty").classList.add("hidden");
  $("group-panel").classList.remove("hidden");
  $("group-name").textContent = data.name || "未命名群";
  $("group-id").textContent = String(data.group_id);
  bindAvatar($("group-avatar"), data.group_id, data.name || "");
  $("group-mini-name").textContent = data.name || "未命名群";
  $("group-mini-id").textContent = String(data.group_id);
  bindAvatar($("group-mini-avatar"), data.group_id, data.name || "");
  $("independent-toggle").checked = independent;
  syncGroupTopActions();
  $("save-hint").textContent = independent
    ? "当前为本群独立配置，会覆盖全局对应项"
    : "当前跟随全局配置；改完后先打开「独立配置」再保存";
  // 按钮保持可点击：未勾选「独立配置」时点击会触发红闪抖动提示

  updateGroupHeadStuck();
  const panels = FIELD_GROUPS.map((group) => {
    const fields = (data.fields || []).filter((field) =>
      field.path.startsWith(group.prefix)
    );
    const rows = fields
      .map(
        (field) => `
        <div class="row form-row ${rowLayout(field.kind)}" data-row="${field.path}" ${
          fieldVisible(field, values) ? "" : 'style="display:none"'
        }>
          <label>${escapeHtml(field.label)}</label>
          <div class="control">${buildControl(field, values[field.path])}</div>
          ${
            field.hint
              ? `<p class="row-hint">${escapeHtml(cleanHint(field.hint))}</p>`
              : ""
          }
        </div>`
      )
      .join("");
    return `<div class="tab-panel ${
      state.currentTab === group.tab ? "active" : ""
    }" data-panel="${group.tab}"><h4>${escapeHtml(
      group.title
    )}</h4><div class="form-group section-card group-card">${rows}</div></div>`;
  }).join("");

  const blockPanel = `
    <div class="tab-panel ${state.currentTab === "user_block" ? "active" : ""}" data-panel="user_block">
      <p class="muted small">只作用于本群，被屏蔽用户的消息会被直接忽略</p>
      <div class="block-form compact">
        <label class="block-field"><span>用户 ID</span>
          <input id="group-block-user" class="input mono" placeholder="QQ 号或字母数字" /></label>
        <label class="block-field"><span>屏蔽原因</span>
          <input id="group-block-reason" class="input" placeholder="例如：广告、刷屏" /></label>
        <label class="block-field"><span>屏蔽时间</span>
          <input id="group-block-duration" class="input mono" value="0" placeholder="0=永久，或 30m" /></label>
        <button id="group-block-add" class="primary small-btn" type="button">本群屏蔽</button>
      </div>
      <div class="block-toolbar">
        <input id="group-block-search" class="input" placeholder="搜索用户 ID 或原因" />
        <button id="group-block-toggle" class="ghost small-btn" type="button">展开</button>
      </div>
      <div id="group-block-list" class="block-list collapsed">
        <p class="muted small pad">加载中…</p>
      </div>
    </div>`;

  $("group-form").innerHTML = panels + blockPanel;
  bindFormEvents();
  switchTab(state.currentTab);
  ensureTabVisible("group-tabs");
  const groupAdd = $("group-block-add");
  if (groupAdd) groupAdd.addEventListener("click", () => void onBlockAddGroup());
  const groupSearch = $("group-block-search");
  if (groupSearch) {
    groupSearch.value = state.blockState.group.search;
    groupSearch.addEventListener("input", (event) => {
      state.blockState.group.search = event.target.value;
      if (event.target.value.trim()) state.blockState.group.expanded = true;
      renderGroupBlocks();
    });
  }
  const groupToggle = $("group-block-toggle");
  if (groupToggle) {
    groupToggle.addEventListener("click", () => {
      state.blockState.group.expanded = !state.blockState.group.expanded;
      renderGroupBlocks();
    });
  }
  if (state.currentTab === "user_block") void loadGroupBlocks();
}

/** 在下一帧将当前活动 Tab 滚入其横向滚动容器的可视范围。 */
function ensureTabVisible(containerId) {
  requestAnimationFrame(() => {
    const container = $(containerId);
    const el = document.querySelector(`#${containerId} .tab.active`);
    if (!container || !el) return;
    // r3.3t：只做 Tab 条横向滚动，绝不触发页面纵向滚动（scrollIntoView 会连带滚动页面）。
    const containerRect = container.getBoundingClientRect();
    const tabRect = el.getBoundingClientRect();
    if (tabRect.left < containerRect.left) {
      container.scrollLeft -= containerRect.left - tabRect.left;
    } else if (tabRect.right > containerRect.right) {
      container.scrollLeft += tabRect.right - containerRect.right;
    }
  });
}

/** 平滑滚动横向 Tab，使刚切换的分区落在容器中央。 */
function centerTab(tab) {
  if (!tab) return;
  const container = tab.closest(".tabs");
  if (!container) return;
  const containerRect = container.getBoundingClientRect();
  const tabRect = tab.getBoundingClientRect();
  const target =
    container.scrollLeft +
    (tabRect.left - containerRect.left) -
    (containerRect.width - tabRect.width) / 2;
  try {
    container.scrollTo({ left: Math.max(0, target), behavior: "smooth" });
  } catch (error) {
    container.scrollLeft = Math.max(0, target);
  }
}

/** 通过移除并重加 CSS 类重新播放分区切换过渡动画。 */
function playTabSwitch(el) {
  if (!el) return;
  el.classList.remove("tab-switch");
  void el.offsetWidth;
  el.classList.add("tab-switch");
}

/** 更新群级配置 Tab 与对应内容面板的活动状态。 */
function switchTab(name) {
  state.currentTab = name;
  // r3.3t：切换 Tab 不再滚动页面；只切换面板，保持用户当前滚动位置。
  document.querySelectorAll("#group-tabs .tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll("#group-form .tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === name);
  });
  if (name === "user_block") void loadGroupBlocks();
}

/** 将群级表单控件的修改同步回 state.currentGroup.values。 */
function bindFormEvents() {
  const form = $("group-form");
  form.querySelectorAll("[data-path]").forEach((input) => {
    const handler = () => {
      const path = input.dataset.path;
      if (!state.currentGroup) return;
      let value;
      if (input.type === "checkbox") {
        value = input.checked;
      } else if (input.type === "range") {
        value = Number(input.value);
        const label = form.querySelector(`.range-value[data-for="${path}"]`);
        if (label) label.textContent = value;
      } else {
        value = input.value;
      }
      state.currentGroup.values[path] = value;
      if (path === "active_reply.mode") {
        refreshConditionalRows();
      }
    };
    input.addEventListener("input", handler);
    input.addEventListener("change", handler);
  });
}

/** 按当前群级配置值重新显示或隐藏与模式相关的字段行。 */
function refreshConditionalRows() {
  const data = state.currentGroup;
  if (!data) return;
  data.fields.forEach((field) => {
    const row = document.querySelector(`[data-row="${field.path}"]`);
    if (row) {
      row.style.display = fieldVisible(field, data.values) ? "" : "none";
    }
  });
}

/** 提示用户先开启群独立配置，不在跟随全局状态下执行写入操作。 */
function shakeIndependent() {
  const wrap = $("independent-wrap");
  wrap.classList.remove("shake");
  void wrap.offsetWidth;
  wrap.classList.add("shake");
  setTimeout(() => wrap.classList.remove("shake"), 800);
  toast("当前未开启「独立配置」，请先勾选后再保存或恢复", true);
}

/** 加载指定群配置，更新当前群状态并渲染可编辑的群面板。 */
async function openGroup(groupId) {
  // 先记录当前选择并更新目录选中态，再取后端权威配置；编辑值与全局值
  // 都复制成新对象，后续表单输入只改前端编辑副本，不会污染响应缓存。
  state.currentGroupId = String(groupId);
  renderGroupList();
  renderDrawerGroups();
  try {
    const data = await callApi("GET", `groups/${encodeURIComponent(groupId)}`);
    // 快速切换 A→B 时，A 的慢响应不能覆盖 B 的当前状态。
    if (state.currentGroupId !== String(groupId)) return;
    state.currentGroup = {
      ...data,
      values: { ...(data.values || {}) },
      global: { ...(data.global || {}) },
    };
    if (Array.isArray(data.providers) && data.providers.length) {
      state.providers = data.providers;
    }
    renderGroupPanel();
  } catch (error) {
    if (state.currentGroupId === String(groupId)) {
      toast(error.message || "加载群配置失败", true);
    }
  }
}

/** 校验本群处于独立模式后提交覆盖值，并刷新列表与面板状态。 */
async function saveGroup() {
  const data = state.currentGroup;
  if (!data) return;
  const independent = $("independent-toggle").checked;
  if (!independent) {
    // 跟随全局的群不保存覆盖项；通过视觉提示告诉用户先切换到独立模式。
    shakeIndependent();
    return;
  }
  const button = $("save-group");
  button.disabled = true;
  try {
    // 仅提交群级值与名称备注，成功后重新取目录和该群配置，确保页面展示
    // 的状态来自已持久化的数据，而不是仅沿用本地表单快照。
    await callApi("POST", `groups/${encodeURIComponent(data.group_id)}/save`, {
      independent: true,
      note: data.name || "",
      values: data.values,
    });
    toast("本群配置已保存");
    await loadGroups(true);
    if (state.currentGroupId === String(data.group_id)) {
      await openGroup(data.group_id);
    }
  } catch (error) {
    toast(error.message || "保存失败", true);
  } finally {
    // 保存结束必须恢复按钮；之前误用了未定义的 toggle，异常导致按钮永久禁用。
    button.disabled = false;
    syncGroupTopActions();
  }
}

/** 将当前群表单重置为 schema 默认值；更改仅在用户保存后写入。 */
function resetGroup() {
  const data = state.currentGroup;
  if (!data) return;
  if (!$("independent-toggle").checked) {
    shakeIndependent();
    return;
  }
  (data.fields || []).forEach((field) => {
    data.values[field.path] = field.default;
  });
  renderGroupPanel();
  toast("已填入默认值，点击「保存」后生效");
}

/** 二次确认后删除群级覆盖配置，使指定群重新跟随全局值。 */
async function followGlobal(groupId) {
  const confirmed = await askConfirm(
    `取消独立配置会删除该群的自有配置，之后跟随全局配置。继续？`
  );
  if (!confirmed) return;
  try {
    await callApi("POST", `groups/${encodeURIComponent(groupId)}/reset`, {});
    toast("已取消独立配置，改为跟随全局");
    await loadGroups(true);
    await openGroup(groupId);
  } catch (error) {
    toast(error.message || "操作失败", true);
  }
}

/* ------------------------------------------------------------------ */
/* 全局设置（按 _conf_schema.json 渲染）                                */
/* ------------------------------------------------------------------ */
/** 检查 schema 条件对象中的每个键值是否与当前配置区块相符。 */
function conditionOk(meta, scope) {
  if (!meta || !meta.condition) return true;
  return Object.entries(meta.condition).every(
    ([key, expected]) => scope && scope[key] === expected
  );
}

/** 根据全局配置 schema 类型生成对应输入控件及其 data 属性。 */
function buildSchemaControl(meta, value, path) {
  const type = String(meta.type || "string");
  if (type === "bool") {
    return `
      <label class="switch">
        <input type="checkbox" data-global-path="${path}" data-global-type="bool" ${
          value ? "checked" : ""
        } />
        <span class="slider"></span>
      </label>`;
  }
  // _special 有两种取值：select_provider（文本模型）与
  // select_provider:embedding（Embedding 提供商）。schema 里 Embedding 用的是
  // 带后缀的写法，旧代码只认全等 "select_provider"，导致该字段退化成纯文本框。
  const special = String(meta._special || "");
  if (special === "select_provider" || special.startsWith("select_provider:")) {
    const kind = special.split(":")[1] || "";
    const options =
      kind === "embedding"
        ? embeddingProviderOptions(value || "")
        : providerOptions(value || "");
    return `<select data-global-path="${path}" data-global-type="str">${options}</select>`;
  }
  if (Array.isArray(meta.options) && meta.options.length) {
    const labels = Array.isArray(meta.labels) ? meta.labels : [];
    const options = meta.options
      .map(
        (option, index) =>
          `<option value="${escapeHtml(option)}" ${
            option === value ? "selected" : ""
          }>${escapeHtml(labels[index] || option)}</option>`
      )
      .join("");
    return `<select data-global-path="${path}" data-global-type="str">${options}</select>`;
  }
  if (type === "int" || type === "float") {
    const slider = meta.slider || {};
    const min = Number(slider.min ?? 0);
    const max = Number(slider.max ?? 100);
    const step = Number(slider.step ?? (type === "int" ? 1 : 0.1));
    return `
      <input type="range" data-global-path="${path}" data-global-type="${type}"
        min="${escapeHtml(min)}" max="${escapeHtml(max)}"
        step="${escapeHtml(step)}" value="${escapeHtml(value ?? min)}" />
      <span class="range-value" data-for-global="${path}">${escapeHtml(value ?? min)}</span>`;
  }
  if (type === "text") {
    return `<textarea data-global-path="${path}" data-global-type="text">${escapeHtml(
      value || ""
    )}</textarea>`;
  }
  if (type === "list") {
    const items = Array.isArray(value) ? value.join("\n") : "";
    return `<textarea data-global-path="${path}" data-global-type="list" placeholder="每行一个，或用逗号分隔">${escapeHtml(
      items
    )}</textarea>`;
  }
  if (type === "dict") {
    return `<textarea data-global-path="${path}" data-global-type="dict">${escapeHtml(
      JSON.stringify(value || {}, null, 2)
    )}</textarea>`;
  }
  return `<input type="text" data-global-path="${path}" data-global-type="str" value="${escapeHtml(
    value ?? ""
  )}" />`;
}

/** 递归渲染 schema 对象与字段行，并在初始状态隐藏条件字段。 */
/** 与记忆页面板重复的全局配置：只保留面板入口，避免两处改同一个值。 */
const HIDDEN_GLOBAL_PATHS = new Set([
  "memory_rag.memory_generation.auto_extract_enable",
  // 记忆修复设置已移到开发者选项的记忆修复卡片旁。
  "memory_rag.memory_generation.memory_kind_repair_confidence",
  "memory_rag.memory_generation.memory_kind_reverse_repair",
]);

/** 页面内数字键盘状态（r3.3w）：替代系统软键盘，避免输入后 WebView 视口错位。 */
let editingRangeValue = false;

/** 点击滑动控件旁的数值：原位切换成原生数字输入框编辑。 */
function startRangeValueEdit(span) {
  if (!span || editingRangeValue) return;
  const isGlobal = span.hasAttribute("data-for-global");
  const path = isGlobal
    ? span.getAttribute("data-for-global")
    : span.getAttribute("data-for");
  if (!path) return;
  const selector = isGlobal
    ? '#global-form input[type="range"][data-global-path]'
    : '#group-form input[type="range"][data-path]';
  const slider = Array.from(document.querySelectorAll(selector)).find((item) =>
    isGlobal ? item.dataset.globalPath === path : item.dataset.path === path
  );
  if (!slider) return;
  const min = Number(slider.min);
  const max = Number(slider.max);
  const kind = slider.dataset.kind || slider.dataset.globalType || "float";
  const original = span.textContent;
  const input = document.createElement("input");
  input.type = "number";
  input.className = "range-value-input";
  input.min = String(min);
  input.max = String(max);
  // 不按滑杆 step 取整，允许任意数值（最多 3 位小数）。
  input.step = "any";
  input.inputMode = "decimal";
  input.value = String(slider.value);
  editingRangeValue = true;
  span.classList.add("editing");
  span.textContent = "";
  span.appendChild(input);
  // r3.4s：只唤起键盘，不滚动页面。
  const editScrollY = Math.max(0, Number(window.scrollY) || 0);
  try {
    input.focus({ preventScroll: true });
  } catch (error) {
    input.focus();
  }
  input.select();
  requestAnimationFrame(() => {
    if (Math.abs(window.scrollY - editScrollY) > 1) {
      window.scrollTo(0, editScrollY);
    }
  });
  let done = false;
  const finish = (commit) => {
    if (done) return;
    done = true;
    editingRangeValue = false;
    span.classList.remove("editing");
    if (!commit) {
      span.textContent = original;
      return;
    }
    let value = Number(input.value);
    if (!Number.isFinite(value)) {
      span.textContent = original;
      return;
    }
    value = Math.min(max, Math.max(min, value));
    value = Math.round(value * 1000) / 1000;
    const display = String(value);
    const stepAttr = slider.getAttribute("step");
    slider.dataset.exactValue = display;
    slider.step = "any";
    slider.value = display;
    span.textContent = display;
    slider.dataset.programmaticEdit = "1";
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));
    delete slider.dataset.programmaticEdit;
    if (stepAttr !== null) slider.setAttribute("step", stepAttr);
    else slider.removeAttribute("step");
    attachRangeExactListener(slider, path, isGlobal);
    // 显示层按类型规范化：int 显示整数，float 最多 3 位小数。
    span.textContent = kind === "int" ? String(Math.round(value)) : display;
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      finish(false);
    }
  });
  input.addEventListener("blur", () => finish(true));
}

/** 为滑块安装「拖拽时清理手输精确值 + 规范化显示」的监听（只装一次）。 */
function attachRangeExactListener(slider, path, isGlobal) {
  if (slider.dataset.exactListener === "1") return;
  slider.dataset.exactListener = "1";
  slider.addEventListener("input", () => {
    if (slider.dataset.programmaticEdit === "1") return;
    delete slider.dataset.exactValue;
    const numeric = Number(slider.value);
    if (!Number.isFinite(numeric)) return;
    const lo = Number(slider.min);
    const hi = Number(slider.max);
    let normalized = Math.min(hi, Math.max(lo, numeric));
    const kind = slider.dataset.kind || slider.dataset.globalType || "float";
    normalized =
      kind === "int"
        ? Math.round(normalized)
        : Math.round(normalized * 1000) / 1000;
    const label = isGlobal
      ? document.querySelector(`.range-value[data-for-global="${path}"]`)
      : document.querySelector(`.range-value[data-for="${path}"]`);
    if (label) label.textContent = String(normalized);
  });
}

/** 渲染全局表单时的 schema 区块。 */
function renderSchemaBlock(schema, scopeValues, prefix) {
  // 参考图 B:连续的叶子字段合成一张分组卡(.group-card);嵌套 object 的标题放在卡外上方并降一级
  let html = "";
  let rows = "";
  const flush = () => {
    if (!rows) return;
    html += `<div class="form-group section-card group-card">${rows}</div>`;
    rows = "";
  };
  for (const [key, meta] of Object.entries(schema)) {
    if (!meta || typeof meta !== "object") continue;
    const path = prefix ? `${prefix}.${key}` : key;
    if (HIDDEN_GLOBAL_PATHS.has(path)) continue;
    if (meta.type === "object" && meta.items) {
      const child =
        scopeValues[key] && typeof scopeValues[key] === "object"
          ? scopeValues[key]
          : {};
      flush();
      const nestedHint = meta.hint
        ? `<p class="nested-hint">${escapeHtml(cleanHint(meta.hint))}</p>`
        : "";
      html += `<div class="form-group nested"><h4>${escapeHtml(
        meta.description || key
      )}</h4>${nestedHint}${renderSchemaBlock(meta.items, child, path)}</div>`;
      continue;
    }
    // 条件字段也要渲染出来，只是默认隐藏：这样用户切换下拉时能立刻显示/隐藏
    const condition = meta.condition;
    const hasCondition =
      condition && typeof condition === "object" && Object.keys(condition).length > 0;
    const visible = !hasCondition || conditionOk(meta, scopeValues);
    const condAttr = hasCondition
      ? ` data-cond="${escapeHtml(JSON.stringify(condition))}"`
      : "";
    const style = visible ? "" : ' style="display:none"';
    const hint = meta.hint
      ? `<p class="row-hint">${escapeHtml(cleanHint(meta.hint))}</p>`
      : "";
    const layout = rowLayout(String(meta.type || "string"));
    rows += `
      <div class="row form-row ${layout}" data-global-row="${path}"${condAttr}${style}>
        <label>${escapeHtml(meta.description || key)}</label>
        <div class="control">${buildSchemaControl(meta, scopeValues[key], path)}</div>
        ${hint}
      </div>`;
  }
  flush();
  return html;
}

/** 收集当前面板值并重绘活动全局配置区块，随后恢复输入事件绑定。 */
function rerenderActiveGlobalPanel() {
  const sections = globalSections();
  const entry = sections.find(([key]) => key === state.globalTab) || sections[0];
  if (!entry) return;
  const [key, meta] = entry;
  const values = collectGlobalValues();
  const scope =
    values[key] && typeof values[key] === "object" ? values[key] : {};
  const panel = document.querySelector(
    `#global-form .tab-panel[data-gpanel="${key}"]`
  );
  if (!panel) return;
  panel.innerHTML =
    `<h4>${escapeHtml(meta.description || key)}</h4>` +
    renderSchemaBlock(meta.items, scope, key);
  bindGlobalInputs();
  refreshGlobalConditions();
  decorateEmbeddingHint();
}

/** 根据已填写的全局配置值同步 schema 条件字段的显示状态。 */
function refreshGlobalConditions() {
  const values = collectGlobalValues();
  document.querySelectorAll("#global-form [data-cond]").forEach((row) => {
    let condition = {};
    try {
      condition = JSON.parse(row.dataset.cond || "{}");
    } catch (error) {
      condition = {};
    }
    const path = row.dataset.globalRow || "";
    const scopePath = path.split(".").slice(0, -1).join(".");
    const scope = (scopePath ? getPath(values, scopePath) : values) || {};
    const ok = Object.entries(condition).every(
      ([key, expected]) => scope[key] === expected
    );
    row.style.display = ok ? "" : "none";
  });
}

/** 绑定全局表单输入监听，更新滑块标签并及时刷新条件字段。 */
function bindGlobalInputs() {
  document.querySelectorAll("#global-form [data-global-path]").forEach((input) => {
    if (input.type === "range") {
      const label = document.querySelector(
        `.range-value[data-for-global="${input.dataset.globalPath}"]`
      );
      input.addEventListener("input", () => {
        if (label) label.textContent = input.value;
      });
    }
    const markDirty = () => {
      state.globalDirty = true;
      state.globalEditVersion = (state.globalEditVersion || 0) + 1;
      if (!(state.globalDirtyPaths instanceof Set)) {
        state.globalDirtyPaths = new Set();
      }
      if (input.dataset.globalPath) {
        state.globalDirtyPaths.add(input.dataset.globalPath);
      }
    };
    input.addEventListener("input", markDirty);
    input.addEventListener("change", markDirty);
    const rerun = () => refreshGlobalConditions();
    input.addEventListener("change", rerun);
    if (input.tagName === "SELECT" || input.type === "checkbox") {
      input.addEventListener("input", rerun);
    }
  });
  document.querySelectorAll("#global-form [data-global-path]").forEach((input) => {
    if (
      input.dataset.globalPath.endsWith("memory_control.enable") ||
      input.dataset.globalPath === "memory_rag.enable"
    ) {
      input.addEventListener("change", syncMemoryMaster);
    }
  });
  syncMemoryMaster();
}

/** 记忆总开关：关闭时禁用记忆 RAG 下的其他控件。 */
function syncMemoryMaster() {
  const master =
    document.querySelector(
      '#global-form [data-global-path="memory_rag.memory_control.enable"]'
    ) ||
    document.querySelector('#global-form [data-global-path="memory_rag.enable"]');
  if (!master) return;
  const disabled = !master.checked;
  document
    .querySelectorAll('#global-form [data-global-path^="memory_rag."]')
    .forEach((input) => {
      if (input === master) return;
      input.disabled = disabled;
      const row = input.closest(".form-row");
      if (row) row.classList.toggle("disabled-row", disabled);
    });
}

/** 从加载的 schema 中提取可作为一级 Tab 的对象配置区块。 */
function globalSections() {
  return Object.entries(state.schema || {}).filter(
    ([, meta]) => meta && meta.type === "object" && meta.items
  );
}

/** 切换全局设置的活动 Tab 和相应配置面板。 */
function switchGlobalTab(key) {
  state.globalTab = key;
  // r3.3t：切换 Tab 不再滚动页面；只切换面板，保持用户当前滚动位置。
  document.querySelectorAll("#global-tabs .tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.gtab === key);
  });
  document.querySelectorAll("#global-form .tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.gpanel === key);
  });
}

/** 依据 schema 渲染全局配置 Tab 和递归字段表单。 */
function renderGlobalForm() {
  const sections = globalSections();
  if (!sections.length) {
    $("global-tabs").innerHTML = "";
    $("global-form").innerHTML =
      '<p class="muted small pad">没有可用的配置 schema</p>';
    return;
  }

  $("global-tabs").innerHTML = sections
    .map(
      ([key, meta], index) => `
      <button class="tab ${index === 0 ? "active" : ""}" data-gtab="${escapeHtml(
        key
      )}" role="tab">${icon(SECTION_ICONS[key] || "sliders")}${escapeHtml(meta.description || key)}</button>`
    )
    .join("");

  $("global-form").innerHTML = sections
    .map(([key, meta], index) => {
      const scope =
        state.globalValues[key] && typeof state.globalValues[key] === "object"
          ? state.globalValues[key]
          : {};
      const body = renderSchemaBlock(meta.items, scope, key);
      return `<div class="tab-panel ${
        index === 0 ? "active" : ""
      }" data-gpanel="${escapeHtml(key)}"><h4>${escapeHtml(
        meta.description || key
      )}</h4>${body}</div>`;
    })
    .join("");

  bindGlobalInputs();
  const activeKey = sections.some(([key]) => key === state.globalTab)
    ? state.globalTab
    : sections[0][0];
  switchGlobalTab(activeKey);
  ensureTabVisible("global-tabs");
  // 表单重绘会丢掉动态提示，重绘后重新补上 Embedding 可用列表。
  decorateEmbeddingHint();
}

/** 递归提取 schema 默认值并复制数组/对象，避免共享引用被表单修改。 */
function schemaDefaults(schema) {
  const defaults = {};
  for (const [key, meta] of Object.entries(schema || {})) {
    if (!meta || typeof meta !== "object") continue;
    if (meta.type === "object" && meta.items) {
      defaults[key] = schemaDefaults(meta.items);
      continue;
    }
    const value = meta.default;
    defaults[key] =
      value && typeof value === "object" ? JSON.parse(JSON.stringify(value)) : value;
  }
  return defaults;
}

/** 从后端载入 schema、全局配置和 provider 选项后刷新设置表单。 */
async function loadGlobal() {
  const requestId = (state.globalLoadSeq || 0) + 1;
  state.globalLoadSeq = requestId;
  try {
    const data = await callApi("GET", "config");
    if (requestId !== state.globalLoadSeq) return;
    const serverValues = data.values || {};
    if (state.globalDirty) {
      // 保留用户编辑过的字段；其余字段仍接受最新服务端值。
      const dirtyPaths =
        state.globalDirtyPaths instanceof Set ? state.globalDirtyPaths : new Set();
      if (!dirtyPaths.size) return;
      const edited = collectGlobalValues();
      for (const path of dirtyPaths) {
        setPath(serverValues, path, getPath(edited, path));
      }
    }
    state.schema = data.schema || {};
    state.globalValues = serverValues;
    state.providers = data.providers || state.providers;
    state.embeddingProviders =
      data.embedding_providers || state.embeddingProviders;
    if (!state.globalDirty) {
      state.globalEditVersion = state.globalEditVersion || 0;
      state.globalDirtyPaths = new Set();
    }
    renderGlobalForm();
    $("global-hint").textContent = "";
  } catch (error) {
    if (requestId !== state.globalLoadSeq || state.globalDirty) return;
    $("global-tabs").innerHTML = "";
    $("global-form").innerHTML = `<p class="muted small pad">加载失败：${escapeHtml(
      error.message || "未知错误"
    )}</p>`;
  }
}

/** 将编辑中的全局值重置为 schema 默认值，不在此处立即保存。 */
function resetGlobalDefaults() {
  if (!state.schema || !Object.keys(state.schema).length) {
    toast("配置尚未加载完成，请稍候", true);
    return;
  }
  state.globalValues = schemaDefaults(state.schema);
  state.globalDirty = true;
  state.globalDirtyPaths = new Set();
  state.globalEditVersion = (state.globalEditVersion || 0) + 1;
  renderGlobalForm();
  $("global-hint").textContent = "已填入默认值，点击「保存」后生效";
  toast("已填入默认值，点击「保存」后生效");
}

/** 从 DOM 控件重建完整嵌套配置对象，供服务端 schema 校验。 */
function collectGlobalValues() {
  const values = {};
  document.querySelectorAll("#global-form [data-global-path]").forEach((input) => {
    const path = input.dataset.globalPath;
    const type = input.dataset.globalType;
    let value;
    if (type === "bool") {
      value = input.checked;
    } else if (type === "int" || type === "float") {
      // r3.3u：用户手输的精确值优先（避免 range 恢复 step 后被吸附）。
      value =
        input.dataset.exactValue !== undefined
          ? Number(input.dataset.exactValue)
          : Number(input.value);
    } else if (type === "list") {
      value = input.value
        .split(/[\n,，]/)
        .map((item) => item.trim())
        .filter(Boolean);
    } else if (type === "dict") {
      try {
        value = JSON.parse(input.value || "{}");
      } catch (error) {
        value = {};
      }
    } else {
      value = input.value;
    }
    setPath(values, path, value);
  });
  // 把当前 DOM 编辑值同步回状态，避免条件字段重绘时用旧值覆盖未保存文本。
  state.globalValues = values;
  return values;
}

/** 提交收集后的全局设置，展示写盘警告，并刷新配置及概览。 */
async function saveGlobal() {
  const button = $("save-global");
  button.disabled = true;
  $("global-hint").textContent = "保存中…";
  try {
    const saveVersion = state.globalEditVersion || 0;
    const result = await callApi("POST", "config/save", {
      values: collectGlobalValues(),
    });
    if ((state.globalEditVersion || 0) !== saveVersion) {
      // 保存期间又有输入：保留当前表单和 dirty 状态，不能重绘覆盖。
      $("global-hint").textContent = "已保存；保存期间的新修改尚未提交";
      toast("已保存，但有新修改尚未提交");
      return;
    }
    state.globalDirty = false;
    state.globalDirtyPaths = new Set();
    $("global-hint").textContent = result.warning
      ? `已保存，但写盘失败：${result.warning}`
      : "已保存，立即生效";
    toast(result.warning ? "已保存（写盘失败，仅本次运行生效）" : "全局配置已保存");
    await loadGlobal();
    await loadOverview();
  } catch (error) {
    $("global-hint").textContent = "";
    toast(error.message || "保存失败", true);
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* 记忆 RAG                                                            */
/* ------------------------------------------------------------------ */
/** 渲染「当前遗忘规则」摘要：TTL、衰减、归档阈值与巩固周期（r3.3l）。 */
function renderRagRules(rules) {
  const box = $("rag-rules");
  if (!box) return;
  if (!rules || typeof rules !== "object") {
    box.textContent = "";
    return;
  }
  const base = rules.ttl_base || {};
  const days = (value) => (Number(value) > 0 ? `${Number(value)}天` : "永久");
  const ttlText = rules.ttl_adaptive
    ? `TTL 计划${days(base.plan)} / 事件${days(base.episode)} / 偏好${days(
        base.preference
      )} / 关系${days(base.relation)} / 事实${days(base.fact)}（自适应）`
    : "TTL 使用旧版按类型硬过期";
  const decayText = rules.decay_enable
    ? `重要性每天衰减 ${(Number(rules.decay_rate || 0) * 100).toFixed(
        2
      )}%（下限 ${Number(rules.decay_floor || 0).toFixed(2)}）`
    : "重要性不随时间衰减";
  const archiveText = rules.archive_enable
    ? `低价值（有效重要性 ≤ ${Number(rules.archive_importance || 0).toFixed(
        2
      )}）且 ${Number(rules.archive_days)} 天未访问归档`
    : "低价值归档关闭";
  const retention = Number(rules.archive_retention_days || 0);
  const retentionText =
    retention > 0 ? `归档保留 ${retention} 天后物理清理` : "归档永久保留";
  const consolidation = rules.consolidation_enable
    ? `每 ${Number(rules.consolidation_interval_days)} 天巩固一次`
    : "巩固关闭";
  box.innerHTML =
    "<b>遗忘规则：</b>" +
    [ttlText, decayText, archiveText, retentionText, consolidation]
      .map((text) => escapeHtml(text))
      .join(" · ");
}

/** 加载记忆统计；需要筛选器选项时同时更新可用群作用域。 */
async function loadRagStats(updateScopes = false) {
  try {
    const data = await callApi("GET", "rag/stats");
    const stats = data.stats || {};
    if (updateScopes) {
      state.rag.scopes = data.group_scopes || [];
      const select = $("filter-scope");
      if (select) {
        select.innerHTML = [
          `<option value="">全部群范围</option>`,
          // r3.3j：可单独筛选全局层记忆（群作用域为空）。
          `<option value="__global__">全局</option>`,
          ...state.rag.scopes.map((scope) => {
            const scopeLabel = ragScopeLabel(scope) || scope;
            return `<option value="${escapeHtml(scope)}">${escapeHtml(
              scopeLabel
            )}</option>`;
          }),
        ].join("");
      }
    }
    state.rag.rules = data.rules || null;
    renderRagRules(state.rag.rules);
    renderStatsTray(stats);
    renderSourceCard(stats);
  } catch (error) {
    toast(error.message || "加载记忆统计失败", true);
  }
}

/** 渲染「记忆统计」托盘：默认折叠，展开后 4 列小卡片。 */
function renderStatsTray(stats) {
  const card = $("rag-stats-card");
  if (!card) return;
  const sources = stats.source_counts || {};
  const span = stats.oldest_memory_time_iso
    ? `${String(stats.oldest_memory_time_iso).slice(0, 10)} 起`
    : "-";
  const items = [
    ["记忆总数", formatNumber(stats.total_memories), ""],
    ["群范围数", formatNumber(stats.group_scope_count), "不同 scope"],
    ["角色数", formatNumber(stats.role_count), "关联角色 ID"],
    ["今日新增", formatNumber(stats.today), ""],
    ["关键词记忆", formatNumber(sources.keyword), ""],
    ["模型抽取", formatNumber(sources.model), ""],
    // r3.3k：遗忘可见性（即将过期 / 已过期未归档 / 已归档）。
    ["即将过期", formatNumber(stats.expiring_soon), "7 天内"],
    ["已过期未归档", formatNumber(stats.expired_active), "等待归档"],
    ["已归档", formatNumber(stats.archived_count), "不再召回"],
    // r3.3l：最近 7 天新归档（遗忘）数量与原因分布。
    [
      "本周遗忘",
      formatNumber(stats.weekly_forgotten),
      Object.entries(stats.weekly_by_reason || {})
        .map(
          ([reason, count]) =>
            `${ARCHIVE_REASON_LABELS[reason] || escapeHtml(String(reason))} ${formatNumber(
              count
            )}`
        )
        .join(" · "),
    ],
  ];
  const expanded = card.dataset.expanded === "1";
  card.innerHTML = `
    <button id="stats-card-toggle" class="source-card-head" type="button">
      <span><b>记忆统计</b><span class="muted small">总数 / 来源 / 时间跨度</span></span>
      <span class="pill">${formatNumber(stats.total_memories)} 条</span>
      <span class="icon source-caret icon-arrow-right" aria-hidden="true"></span>
    </button>
    <div class="source-tray ${expanded ? "open" : ""}">
      <div class="stats-grid-mini">
        <div class="mini-card time-span-card">
          <span class="mini-label">时间跨度</span>
          <b class="mini-value">${escapeHtml(span)}</b>
          <span class="mini-hint">${escapeHtml(
            stats.latest_memory_time_iso
              ? `最近 ${String(stats.latest_memory_time_iso).slice(0, 10)}`
              : "暂无记录"
          )}</span>
        </div>
        ${items
          .map(
            ([label, value, hint]) => `
          <div class="mini-card">
            <span class="mini-label">${label}</span>
            <b class="mini-value">${value}</b>
            ${hint ? `<span class="mini-hint">${hint}</span>` : ""}
          </div>`
          )
          .join("")}
      </div>
    </div>`;
  $("stats-card-toggle").addEventListener("click", () => {
    const next = card.dataset.expanded !== "1";
    card.dataset.expanded = next ? "1" : "0";
    card.querySelector(".source-tray").classList.toggle("open", next);
  });
}

/** 渲染「记忆来源」卡片与可展开托盘。 */
function renderSourceCard(stats) {
  const card = $("rag-source-card");
  if (!card) return;
  const types = stats.type_counts || {};
  const rows = [
    ["个人", Number(types.personal || 0), "personal"],
    ["群聊", Number(types.group || 0), "group"],
    ["私聊", Number(types.private || 0), "private"],
    ["未知", Number(types.unknown || 0), "unknown"],
  ];
  // 徽标展示真实合计；除零兜底只能用在进度条宽度上，否则 0 条会显示成 1 条。
  const total = rows.reduce((sum, row) => sum + row[1], 0);
  const barScale = total || 1;
  const expanded = card.dataset.expanded === "1";
  card.innerHTML = `
    <button id="source-card-toggle" class="source-card-head" type="button">
      <span><b>记忆来源</b><span class="muted small">个人 / 群聊 / 私聊 / 未知</span></span>
      <span class="pill">${formatNumber(total)} 条</span>
      <span class="icon source-caret icon-arrow-right" aria-hidden="true"></span>
    </button>
    <div class="source-tray ${expanded ? "open" : ""}">
      ${rows
        .map(
          ([label, count, kind]) => `
        <div class="source-row${count === 0 ? " is-zero" : ""}">
          <span class="source-dot ${kind}"></span>
          <span class="source-label">${label}</span>
          <span class="source-count">${formatNumber(count)}</span>
          <span class="source-bar"><i class="${kind}" style="width:${count === 0 ? 0 : (count / barScale) * 100}%"></i></span>
        </div>`
        )
        .join("")}
    </div>`;
  $("source-card-toggle").addEventListener("click", () => {
    const next = card.dataset.expanded !== "1";
    card.dataset.expanded = next ? "1" : "0";
    card.querySelector(".source-tray").classList.toggle("open", next);
  });
}

/** 记忆类型白名单：只有已知枚举才能拼进 CSS 类名（F44）。 */
const MEMORY_TYPE_CLASSES = new Set(["personal", "group", "private"]);

/** 归档原因文案（由服务端写入 archive_reason；旧数据为空）。 */
const ARCHIVE_REASON_LABELS = {
  expired: "过期",
  stale: "低价值",
  consolidation: "巩固",
  manual: "手动",
};

/** 角色徽标：值来自记忆元数据，只使用固定文案与固定类名，避免被投毒。 */
const ROLE_BADGES = {
  owner: { label: "所有者", cls: "role-owner" },
  group_owner: { label: "群主", cls: "role-group_owner" },
  group_admin: { label: "群管", cls: "role-group_admin" },
  member: { label: "群成员", cls: "role-member" },
};

/** 稳定参与者标签：昵称#后四位；没有身份记录时用「用户#后四位」。 */
function participantLabel(id, participantIdentities) {
  const target = String(id || "").trim();
  if (!target) return "-";
  const list = Array.isArray(participantIdentities) ? participantIdentities : [];
  const found = list.find(
    (entry) => entry && String(entry.sender_id || "").trim() === target
  );
  if (!found) return `用户#${target.slice(-4)}`;
  if (found.is_bot) return "我";
  const name = String(found.display_name || "").trim();
  if (!name || name === target) return `用户#${target.slice(-4)}`;
  return `${name}#${target.slice(-4)}`;
}

/** 渲染「徽标 + 稳定参与者标签」列表；所有文本都必须转义（F02）。 */
function renderRoleBadges(roleIds, identities, participantIdentities) {
  const ids = Array.isArray(roleIds) ? roleIds : [];
  if (!ids.length) return '<span class="muted">-</span>';
  const map = identities && typeof identities === "object" ? identities : {};
  const badges = ids.map((id) => {
    const role = String(map[id] || "member");
    const meta = ROLE_BADGES[role] || ROLE_BADGES.member;
    const label = participantLabel(id, participantIdentities);
    return (
      '<span class="role-badge">' +
      `<span class="pill ${meta.cls}">${escapeHtml(meta.label)}</span>` +
      `<span class="mono role-id">${escapeHtml(label)}</span>` +
      "</span>"
    );
  });
  return `<span class="mem-role-badges">${badges.join("")}</span>`;
}

/** 记忆 kind 的固定徽标：只允许白名单，避免记忆内容注入类名。 */
const MEMORY_KIND_BADGES = {
  episode: { label: "事件", cls: "kind-episode" },
  plan: { label: "计划", cls: "kind-plan" },
  preference: { label: "偏好", cls: "kind-preference" },
  relation: { label: "关系", cls: "kind-relation" },
  experience: { label: "经验", cls: "kind-experience" },
  fact: { label: "事实", cls: "kind-fact" },
};

function memoryKindBadge(item, isAtom) {
  const kind = String(item.kind || "fact").toLowerCase();
  const meta = MEMORY_KIND_BADGES[kind] || MEMORY_KIND_BADGES.fact;
  const extra = item.extra_metadata || {};
  const confidence = item.kind_confidence ?? extra.kind_confidence;
  const reason = item.kind_reason ?? extra.kind_reason;
  const reviewSuggested =
    item.kind_review_suggested ?? extra.kind_review_suggested;
  const reviewReason = item.kind_review_reason ?? extra.kind_review_reason;
  const needsReview = Boolean(
    item.kind_needs_review ?? extra.kind_needs_review
  );
  const details = [];
  if (confidence !== undefined && confidence !== null) {
    details.push(`置信度 ${Number(confidence).toFixed(2)}`);
  }
  if (reason) details.push(String(reason));
  if (reviewSuggested) details.push(`建议 ${reviewSuggested}`);
  if (reviewReason) details.push(String(reviewReason));
  const title = details.length ? ` title="${escapeHtml(details.join(" · "))}"` : "";
  const atom = isAtom ? '<span class="pill atom-mark">原子</span>' : "";
  const review = needsReview
    ? '<span class="pill kind-review">待复核</span>'
    : "";
  return `<span class="pill ${meta.cls}"${title}>${meta.label}</span>${atom}${review}`;
}

function memoryEpisodeId(extraMetadata) {
  const raw = extraMetadata && extraMetadata.episode_id;
  const id = Number(raw);
  return Number.isInteger(id) && id > 0 ? id : null;
}

/** 卡片时间：2026-11-07T08:50:56 -> 2026-11-7 08:50:56。 */
function formatMemoryCardTime(value) {
  const text = String(value || "").replace("T", " ");
  const match = text.match(
    /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/
  );
  if (!match) return text.slice(0, 19) || "-";
  const [, year, month, day, hour, minute, second] = match;
  return `${year}-${Number(month)}-${Number(day)} ${hour}:${minute}:${second}`;
}

/** 渲染分页记忆记录，并为详情、归档/恢复与删除按钮绑定对应操作。 */
function renderRagRows(items) {
  const body = $("rag-body");
  const typeLabels = {
    personal: "个人",
    group: "群聊",
    private: "私聊",
  };
  if (!Array.isArray(items) || !items.length) {
    body.closest("table")?.classList.toggle("is-selecting", state.rag.selectMode);
    body.innerHTML =
      '<tr><td colspan="4" class="muted center pad">当前筛选条件下没有记忆</td></tr>';
    body.classList.remove("rag-fade-in");
    void body.offsetWidth;
    if (state.rag.loadedOnce) body.classList.add("rag-fade-in");
    syncRagBulkBar();
    return;
  }
  // 同页内按 episode_id 分组：episode 作为父卡片，原子记忆折叠进父卡片。
  const episodeIds = new Set(
    items
      .filter((item) => String(item.kind || "").toLowerCase() === "episode")
      .map((item) => String(item.memory_id))
  );
  const atomsByEpisode = new Map();
  for (const item of items) {
    const extra = item.extra_metadata && typeof item.extra_metadata === "object"
      ? item.extra_metadata
      : {};
    const episodeId = memoryEpisodeId(extra);
    if (episodeId && episodeIds.has(String(episodeId))) {
      const key = String(episodeId);
      if (!atomsByEpisode.has(key)) atomsByEpisode.set(key, []);
      atomsByEpisode.get(key).push(item);
    }
  }
  const groupedAtomIds = new Set(
    [...atomsByEpisode.values()].flat().map((item) => String(item.memory_id))
  );
  body.innerHTML = items
    .filter((item) => !groupedAtomIds.has(String(item.memory_id)))
    .map((item) => {
      const role = item.sender_role || "member";
      // 记忆类型可能来自被投毒的记忆行：类名只允许已知枚举（F44）。
      const rawType = String(item.memory_type || "group").toLowerCase();
      const type = MEMORY_TYPE_CLASSES.has(rawType) ? rawType : "unknown";
      const superseded = item.superseded_by
        ? '<span class="pill type-invalid">已失效</span>'
        : "";
      // 已归档记忆复用灰色 pill 样式；列表行与展开详情都能看到标记。
      const isArchived = Boolean(item.archived);
      const archiveReason = String(item.archive_reason || "");
      const archived = isArchived
        ? `<span class="pill type-invalid">已归档${
            ARCHIVE_REASON_LABELS[archiveReason]
              ? `·${ARCHIVE_REASON_LABELS[archiveReason]}`
              : ""
          }</span>`
        : "";
      // r3.3k：遗忘可见性（即将过期 / 已过期未归档）。
      const daysLeft =
        typeof item.days_left === "number" ? item.days_left : null;
      let forgettingPill = "";
      if (!isArchived && item.expired) {
        forgettingPill =
          '<span class="pill pill-danger" title="有效期已过，等待维护任务归档">已过期待归档</span>';
      } else if (!isArchived && daysLeft !== null && daysLeft <= 7) {
        forgettingPill = `<span class="pill pill-warn" title="有效期剩余 ${Math.max(
          0,
          daysLeft
        )} 天">剩 ${Math.max(0, daysLeft)} 天</span>`;
      }
      const time = escapeHtml(formatMemoryCardTime(item.memory_time_iso || ""));
      const cardGroupId = String(item.group_id || "").trim();
      const cardGroupName = String(item.group_name || "").trim();
      const cardScope = cardGroupId
        ? `${cardGroupName || "群聊"} ${cardGroupId}`
        : "全局";
      const extraMetadata =
        item.extra_metadata && typeof item.extra_metadata === "object"
          ? item.extra_metadata
          : {};
      const identityMap =
        extraMetadata.role_identities &&
        typeof extraMetadata.role_identities === "object"
          ? extraMetadata.role_identities
          : {};
      const identities = { ...identityMap };
      if (item.sender_id && item.sender_role) {
        identities[item.sender_id] = item.sender_role;
      }
      // 角色与 QQ 号分列展示：徽标用固定文案，ID 统一转义（F02）。
      const episodeId = memoryEpisodeId(extraMetadata);
      const isAtom = episodeId !== null;
      const isEpisode = String(item.kind || "").toLowerCase() === "episode";
      const children = isEpisode
        ? atomsByEpisode.get(String(item.memory_id)) || []
        : [];
      const kindBadge = memoryKindBadge(item, isAtom && !isEpisode);
      const atomCount = children.length
        ? `<span class="pill atom-count">+${children.length} 原子</span>`
        : "";
      const orphanAtom =
        isAtom && !episodeIds.has(String(episodeId))
          ? `<span class="muted small mono">↳ #${escapeHtml(episodeId)}</span>`
          : "";
      const childrenHtml = children.length
        ? `<div class="mem-atoms"><div class="mem-atoms-title">原子记忆 · ${children.length} 条</div>${children
            .map((child) => {
              const childMeta =
                child.extra_metadata && typeof child.extra_metadata === "object"
                  ? child.extra_metadata
                  : {};
              return `<div class="mem-atom-card">
                <div class="mem-atom-head">${memoryKindBadge(child, true)}<span class="mono muted">#${escapeHtml(child.memory_id)}</span>
                <span class="muted small">置信度 ${Number(
                  childMeta.confidence ?? 0
                ).toFixed(2)}</span></div>
                <p class="mem-atom-content">${escapeHtml(child.content)}</p>
                <div class="mem-atom-actions">
                  <button class="link-btn icon-action" title="详情" aria-label="详情" data-detail="${escapeHtml(
                    child.memory_id
                  )}">${icon("eye")}</button>
                  <button class="link-btn icon-action" title="原始消息" aria-label="原始消息" data-source="${escapeHtml(
                    child.memory_id
                  )}">${icon("message")}</button>
                  ${
                    child.archived
                      ? `<button class="link-btn icon-action" title="恢复" aria-label="恢复" data-unarchive="${escapeHtml(
                          child.memory_id
                        )}">${icon("reset")}</button>
                         <button class="link-btn icon-action danger" title="彻底删除" aria-label="彻底删除" data-purge="${escapeHtml(
                           child.memory_id
                         )}">${icon("trash")}</button>`
                      : `<button class="link-btn icon-action" title="归档" aria-label="归档" data-archive="${escapeHtml(
                          child.memory_id
                        )}">${icon("archive")}</button>
                         <button class="link-btn icon-action danger" title="删除" aria-label="删除" data-delete="${escapeHtml(
                           child.memory_id
                         )}">${icon("trash")}</button>`
                  }
                </div>
              </div>`;
            })
            .join("")}</div>`
        : "";
      const roleBadges = renderRoleBadges(
        item.related_role_ids,
        identities,
        extraMetadata.participant_identities
      );
      const confidence =
        extraMetadata.confidence !== undefined && extraMetadata.confidence !== null
          ? Number(extraMetadata.confidence).toFixed(2)
          : "-";
      const kindClass =
        MEMORY_KIND_BADGES[String(item.kind || "fact").toLowerCase()]?.cls ||
        MEMORY_KIND_BADGES.fact.cls;
      const selected = state.rag.selected.has(String(item.memory_id));
      const selectionClass = state.rag.selectMode
        ? ` is-selecting${selected ? " selected" : ""}`
        : "";
      return `
      <tr class="mem-row ${kindClass}${selectionClass}" data-mem-row="${escapeHtml(item.memory_id)}">
        <td class="mono" data-label="ID">${
          state.rag.selectMode
            ? `<button class="mem-check${selected ? " checked" : ""}" type="button" data-select="${escapeHtml(
                item.memory_id
              )}" aria-pressed="${selected ? "true" : "false"}" aria-label="选择记忆 ${escapeHtml(
                item.memory_id
              )}"><span class="icon icon-select" aria-hidden="true"></span></button>`
            : ""
        }<span class="mem-id-prefix">ID：</span><span class="mem-id-value">${escapeHtml(
          item.memory_id
        )}</span>${kindBadge}${atomCount}${archived}${forgettingPill}${orphanAtom}</td>
        <td class="mono small" data-label="时间">${time}</td>
        <td class="mono" data-label="群范围">${escapeHtml(cardScope)}</td>
        <td class="mem-toggle-cell">
          <button class="icon-btn mem-toggle" type="button" data-toggle="${escapeHtml(
            item.memory_id
          )}" aria-label="展开记忆"><span class="icon icon-arrow-right" aria-hidden="true"></span></button>
        </td>
      </tr>
      <tr class="mem-detail" data-mem-detail="${escapeHtml(item.memory_id)}">
        <td colspan="4">
          <div class="mem-detail-body">
            <div class="mem-detail-meta">
              <span class="pill type-${type}">${escapeHtml(
                typeLabels[type] || "未知"
              )}</span>
              ${kindBadge}
              ${superseded}
              ${archived}
              <span class="muted small">置信度：${confidence}</span>
              ${
                item.effective_importance !== undefined
                  ? `<span class="muted small" title="按时间衰减后的有效重要性">有效重要性：${Number(
                      item.effective_importance
                    ).toFixed(2)}</span>`
                  : ""
              }
              ${
                item.valid_until
                  ? `<span class="muted small">有效期：${escapeHtml(
                      String(item.valid_until_iso || "").slice(0, 10) || "-"
                    )}${
                      item.expired
                        ? "（已过期）"
                        : daysLeft !== null
                          ? `（剩 ${Math.max(0, daysLeft)} 天）`
                          : ""
                    }</span>`
                  : '<span class="muted small">有效期：永久</span>'
              }
              <span class="mem-role"><span class="muted small">角色：</span>${roleBadges}</span>
            </div>
            <p class="mem-detail-content">${escapeHtml(item.content)}</p>
            <div class="mem-detail-actions">
              <div class="mem-action-group" role="group" aria-label="记忆操作">
                <button class="link-btn action-segment" title="详情" aria-label="详情" data-detail="${escapeHtml(
                  item.memory_id
                )}">${icon("eye")}</button>
                <button class="link-btn action-segment" title="原始消息" aria-label="原始消息" data-source="${escapeHtml(
                  item.memory_id
                )}">${icon("message")}</button>
                ${
                  isArchived
                    ? `<button class="link-btn action-segment" title="恢复" aria-label="恢复" data-unarchive="${escapeHtml(
                        item.memory_id
                      )}">${icon("reset")}</button>
                <button class="link-btn action-segment danger" title="彻底删除" aria-label="彻底删除" data-purge="${escapeHtml(
                  item.memory_id
                )}">${icon("trash")}</button>`
                    : `<button class="link-btn action-segment" title="归档" aria-label="归档" data-archive="${escapeHtml(
                        item.memory_id
                      )}">${icon("archive")}</button>
                <button class="link-btn action-segment danger" title="删除" aria-label="删除" data-delete="${escapeHtml(
                  item.memory_id
                )}">${icon("trash")}</button>`
                }
              </div>
            </div>
            ${childrenHtml}
          </div>
        </td>
      </tr>`;
    })
    .join("");

  body.classList.remove("rag-fade-in");
  void body.offsetWidth;
  if (state.rag.loadedOnce) body.classList.add("rag-fade-in");
  body.closest("table")?.classList.toggle("is-selecting", state.rag.selectMode);
  syncRagBulkBar();
  body.querySelectorAll(".mem-row").forEach((row) => {
    row.addEventListener("click", () => {
      if (state.rag.selectMode) {
        toggleRagSelected(row.dataset.memRow);
        return;
      }
      const detail = body.querySelector(
        `[data-mem-detail="${CSS.escape(row.dataset.memRow)}"]`
      );
      if (!detail) return;
      const open = !detail.classList.contains("open");
      detail.classList.toggle("open", open);
      row.querySelector(".mem-toggle")?.classList.toggle("open", open);
    });
  });
  body.querySelectorAll("[data-detail]").forEach((btn) => {
    btn.addEventListener("click", () => void showMemory(btn.dataset.detail));
  });
  body.querySelectorAll("[data-source]").forEach((btn) => {
    btn.addEventListener("click", () => void showSource(btn.dataset.source));
  });
  body.querySelectorAll("[data-delete]").forEach((btn) => {
    btn.addEventListener("click", () => void deleteMemory(btn.dataset.delete));
  });
  // r3.3：归档记忆可恢复或彻底删除，未归档记忆可归档（删除仍走回收站 + 撤销）。
  body.querySelectorAll("[data-archive]").forEach((btn) => {
    btn.addEventListener("click", () => void archiveMemory(btn.dataset.archive));
  });
  body.querySelectorAll("[data-unarchive]").forEach((btn) => {
    btn.addEventListener("click", () =>
      void unarchiveMemory(btn.dataset.unarchive)
    );
  });
  body.querySelectorAll("[data-purge]").forEach((btn) => {
    btn.addEventListener("click", () => void purgeMemory(btn.dataset.purge));
  });
}

/** 把一条原始记录渲染成聊天气泡。 */
function sourceMessageHtml(raw) {
  const text = String(raw || "").trim();
  const match = text.match(
    /^\[([^\]]+)\]\s*(?:\(([^)]*)\))?\s*(#[^\s:]+)?\s*:?\s*([\s\S]*)$/
  );
  if (!match) {
    return `<div class="source-msg source-msg-plain"><p>${escapeHtml(
      text || "（空消息）"
    )}</p></div>`;
  }
  const identity = String(match[1] || "").split("/");
  const time = identity.length >= 3 ? identity.pop() : "";
  const uid = identity.length >= 2 ? identity.pop() : "";
  const name = identity.join("/") || "未知";
  const role = String(match[2] || "").trim();
  const msgId = String(match[3] || "").trim();
  const body = String(match[4] || "").trim();
  return `<article class="source-msg">
    <header class="source-msg-head">
      <span class="source-name">${escapeHtml(name)}</span>
      ${role ? `<span class="source-role">${escapeHtml(role)}</span>` : ""}
      ${time ? `<time class="source-time">${escapeHtml(time)}</time>` : ""}
    </header>
    <p class="source-text">${escapeHtml(body || "（空消息）")}</p>
    <footer class="source-msg-foot">
      ${uid ? `<span>UID ${escapeHtml(uid)}</span>` : ""}
      ${msgId ? `<span>${escapeHtml(msgId)}</span>` : ""}
    </footer>
  </article>`;
}

/** JSON 轻量高亮；输入已转义，避免破坏 HTML。 */
function highlightJson(json) {
  const escaped = escapeHtml(json);
  return escaped
    .replace(/(&quot;.*?&quot;)(\s*:)/g, '<span class="json-key">$1</span>$2')
    .replace(/(:\s*)(&quot;.*?&quot;)/g, '$1<span class="json-string">$2</span>')
    .replace(/(:\s*)(-?\d+(?:\.\d+)?)([,}])/g, '$1<span class="json-number">$2</span>$3')
    .replace(/(:\s*)(true|false)([,}])/g, '$1<span class="json-bool">$2</span>$3')
    .replace(/(:\s*)(null)([,}])/g, '$1<span class="json-null">$2</span>$3');
}

/** 展示记忆关联的原始消息证据。 */
let detailRequestSeq = 0;

async function showSource(memoryId) {
  const requestId = ++detailRequestSeq;
  try {
    const item = await callApi("GET", `rag/${encodeURIComponent(memoryId)}`);
    if (requestId !== detailRequestSeq) return;
    const meta = item.extra_metadata || {};
    const quotes = meta.quotes || (meta.evidence ? [meta.evidence] : []);
    const sourceMessages = Array.isArray(meta.source_messages)
      ? meta.source_messages
      : [];
    const extraction = meta.extraction || {};
    const extractionLine = extraction.window
      ? `抽取窗口：${extraction.window} 条 → 有效 ${extraction.filtered} 条 → 送入 ${extraction.model_lines} 条（字数上限 ${extraction.char_limit}${extraction.truncated ? "，已截断" : ""}）`
      : "";
    const selfLine = meta.self_participated ? "机器人参与：是" : "";
    const records = sourceMessages.length ? sourceMessages : quotes;
    const threadHtml = records.length
      ? records.map((record) => sourceMessageHtml(record)).join("")
      : `<div class="source-empty">未记录原文</div>`;
    const summary = [
      ["记忆 ID", `#${escapeHtml(item.memory_id)}`],
      ["来源消息 ID", escapeHtml(item.source_msg_id || "-")],
      ["发送者", escapeHtml(item.sender_id || "-")],
      ["记忆时间", escapeHtml(item.memory_time_iso || "-")],
    ]
      .map(
        ([label, value]) =>
          `<div class="source-summary-item"><span>${label}</span><strong class="mono">${value}</strong></div>`
      )
      .join("");
    $("detail-title").textContent = "原始消息";
    $("detail-body").classList.add("detail-rich");
    $("detail-body").innerHTML = `
      <div class="source-sheet">
        <div class="source-summary">${summary}</div>
        ${extractionLine ? `<div class="source-window">${escapeHtml(extractionLine)}</div>` : ""}
        ${selfLine ? `<div class="source-window">${escapeHtml(selfLine)}</div>` : ""}
        <div class="source-section-title">原始消息记录 <em>${records.length} 条</em></div>
        <div class="source-thread">${threadHtml}</div>
      </div>`;
    openDetailPanel();
  } catch (error) {
    toast(error.message || "读取原始消息失败", true);
  }
}

const AUTO_SCOPE_LABELS = {
  group_first: "自动",
  global: "仅全局",
  group: "仅本群",
};

/** 获取自动记忆的当前开关、参数和来源统计。 */
/* ------------------------------------------------------------------ */
/* 用户屏蔽                                                             */
/* ------------------------------------------------------------------ */
function renderBlockList(container, entries, scope, groupId, emptyText) {
  if (!container) return;
  if (!entries || !entries.length) {
    container.innerHTML = `<p class="muted small pad">${escapeHtml(
      emptyText || "暂无屏蔽记录"
    )}</p>`;
    return;
  }
  container.innerHTML = entries
    .map(
      (entry) => `
      <div class="block-row">
        <div class="block-row-main">
          <span class="block-user mono">${escapeHtml(entry.user_id)}</span>
          <span class="pill">${escapeHtml(entry.scope_text || (entry.scope === "all" ? "全部会话" : "仅本群"))}</span>
          <span class="block-until">${escapeHtml(entry.until_text || "永久")}</span>
        </div>
        <div class="block-row-sub">原因：${escapeHtml(entry.reason || "未说明")}</div>
        <button class="link-btn danger" type="button"
          data-block-remove="${escapeHtml(entry.user_id)}"
          data-block-scope="${escapeHtml(scope)}"
          data-block-group="${escapeHtml(groupId || "")}">解除</button>
      </div>`
    )
    .join("");
}

function filterBlocks(entries, keyword) {
  const text = String(keyword || "").trim().toLowerCase();
  if (!text) return entries || [];
  return (entries || []).filter(
    (entry) =>
      String(entry.user_id || "").toLowerCase().includes(text) ||
      String(entry.reason || "").toLowerCase().includes(text)
  );
}

function renderGlobalBlocks() {
  const view = state.blockState.global;
  const entries = filterBlocks((state.blocks || {}).global, view.search);
  renderBlockList(
    $("block-global-list"),
    entries,
    "all",
    "",
    view.search ? "没有匹配的屏蔽记录" : "暂无屏蔽记录"
  );
  const list = $("block-global-list");
  if (list) list.classList.toggle("collapsed", !view.expanded);
  const toggle = $("block-toggle");
  if (toggle) {
    toggle.textContent = view.expanded ? "折叠" : `展开（${entries.length}）`;
  }
}

function renderGroupBlocks() {
  const view = state.blockState.group;
  const entries = filterBlocks(state.groupBlocks, view.search);
  renderBlockList(
    $("group-block-list"),
    entries,
    "group",
    (state.currentGroup || {}).group_id || "",
    view.search ? "没有匹配的屏蔽记录" : "本群暂无单独屏蔽记录"
  );
  const list = $("group-block-list");
  if (list) list.classList.toggle("collapsed", !view.expanded);
  const toggle = $("group-block-toggle");
  if (toggle) {
    toggle.textContent = view.expanded ? "折叠" : `展开（${entries.length}）`;
  }
}

async function loadBlocks() {
  const requestId = (state.blocksSeq || 0) + 1;
  state.blocksSeq = requestId;
  try {
    const data = await callApi("GET", "blocks");
    if (requestId !== state.blocksSeq) return;
    state.blocks = data;
    const status = $("block-status");
    status.textContent = data.enabled ? "已启用" : "已关闭";
    status.className = `pill ${data.enabled ? "" : "muted-pill"}`;
    renderGlobalBlocks();
  } catch (error) {
    if (requestId !== state.blocksSeq) return;
    $("block-global-list").innerHTML = `<p class="muted small pad">加载失败：${escapeHtml(
      error.message || "未知错误"
    )}</p>`;
  }
}

async function loadGroupBlocks() {
  const data = state.currentGroup;
  if (!data) return;
  const groupId = String(data.group_id);
  const requestId = (state.groupBlocksSeq || 0) + 1;
  state.groupBlocksSeq = requestId;
  try {
    const result = await callApi("GET", "blocks", { group_id: groupId });
    if (
      requestId !== state.groupBlocksSeq ||
      String(state.currentGroup?.group_id || "") !== groupId
    ) {
      return;
    }
    state.groupBlocks = result.group || [];
    renderGroupBlocks();
  } catch (error) {
    if (requestId !== state.groupBlocksSeq) return;
    const box = $("group-block-list");
    if (box) {
      box.innerHTML = `<p class="muted small pad">加载失败：${escapeHtml(
        error.message || "未知错误"
      )}</p>`;
    }
  }
}

async function submitBlock(payload) {
  return callApi("POST", "blocks/add", payload);
}

async function onBlockAddGlobal() {
  const userId = $("block-user").value.trim();
  const reason = $("block-reason").value.trim();
  const scope = $("block-scope").value;
  const groupId = $("block-group").value.trim();
  const duration = $("block-duration").value.trim() || "0";
  if (!/^[A-Za-z0-9]+$/.test(userId)) {
    toast("用户 ID 只能包含英文字母和数字", true);
    return;
  }
  try {
    await submitBlock({
      user_id: userId,
      reason,
      scope,
      group_id: scope === "group" ? groupId : "",
      duration,
    });
    toast("已添加屏蔽");
    $("block-user").value = "";
    $("block-reason").value = "";
    $("block-duration").value = "0";
    // 无论加的是全局还是「仅本群」，都重新拉取屏蔽列表：此前只有 scope=all
    // 才刷新，按群添加后页面看起来像没生效（F41）。
    void loadBlocks();
    if (scope === "group") {
      toast(`已加入群 ${groupId} 的屏蔽列表`, false);
      if ((state.currentGroup || {}).group_id === groupId) {
        void loadGroupBlocks();
      }
    }
  } catch (error) {
    toast(error.message || "添加失败", true);
  }
}

async function onBlockAddGroup() {
  const data = state.currentGroup;
  if (!data) return;
  const userId = $("group-block-user").value.trim();
  const reason = $("group-block-reason").value.trim();
  const duration = $("group-block-duration").value.trim() || "0";
  if (!/^[A-Za-z0-9]+$/.test(userId)) {
    toast("用户 ID 只能包含英文字母和数字", true);
    return;
  }
  try {
    await submitBlock({
      user_id: userId,
      reason,
      scope: "group",
      group_id: data.group_id,
      duration,
    });
    toast("已在本群屏蔽该用户");
    $("group-block-user").value = "";
    $("group-block-reason").value = "";
    $("group-block-duration").value = "0";
    void loadGroupBlocks();
  } catch (error) {
    toast(error.message || "添加失败", true);
  }
}

async function removeBlockEntry(button) {
  const userId = button.dataset.blockRemove;
  const scope = button.dataset.blockScope || "all";
  const groupId = button.dataset.blockGroup || "";
  button.disabled = true;
  try {
    await callApi("POST", "blocks/remove", {
      user_id: userId,
      scope,
      group_id: groupId,
    });
    toast("已解除屏蔽");
    if (scope === "group") {
      void loadGroupBlocks();
    } else {
      void loadBlocks();
    }
  } catch (error) {
    toast(error.message || "解除失败", true);
    button.disabled = false;
  }
}

async function loadAutoMemory() {
  try {
    const data = await callApi("GET", "memory/auto");
    state.autoMemory = data;
    renderAutoMemory(data);
  } catch (error) {
    $("auto-memory-body").innerHTML = `<p class="muted small pad">加载失败：${escapeHtml(
      error.message || "未知错误"
    )}</p>`;
  }
}

/** 展示自动记忆参数、开关状态和关键词/模型的记忆数量统计。 */
function renderAutoMemory(data) {
  if (!data) return;
  const memoryEnabled = data.memory_enabled !== false;
  const enabled = Boolean(data.enabled);
  const toggle = $("auto-memory-toggle");
  toggle.checked = Boolean(enabled && memoryEnabled);
  // 总开关关闭时也要允许从自动记忆页打开，由后端联动两个开关。
  toggle.disabled = false;
  const provider = data.provider_id ? data.provider_id : "跟随当前会话模型";
  const rows = [
    ["抽取模型", provider],
    ["累计消息记忆", `冷却后总结最近 ${data.batch_messages} 条`],
    ["抽取间隔", `${data.interval_minutes} 分钟`],
    ["私聊抽取频率", `每 ${data.private_every_n} 条`],
    ["最低置信度", Number(data.min_confidence).toFixed(2)],
    ["记忆作用域", AUTO_SCOPE_LABELS[data.scope] || data.scope || "-"],
  ];
  $("auto-memory-body").innerHTML = `
    <div class="auto-memory-grid">
      ${rows
        .map(
          ([label, value]) => `
        <div class="auto-memory-item">
          <span class="auto-memory-label">${escapeHtml(label)}</span>
          <span class="auto-memory-value">${escapeHtml(String(value))}</span>
        </div>`
        )
        .join("")}
    </div>
    `;
}

/** 切换自动记忆开关并刷新面板及记忆统计。 */
async function toggleAutoMemory() {
  const current = state.autoMemory;
  if (!current) return;
  const next = !current.enabled;
  const toggle = $("auto-memory-toggle");
  toggle.disabled = true;
  try {
    const data = await callApi("POST", "memory/auto/toggle", { enable: next });
    state.autoMemory = data;
    renderAutoMemory(data);
    toast(next ? "已开启自动记忆" : "已关闭自动记忆");
    void loadRagStats(true);
    if (state.view === "settings") void loadGlobal();
  } catch (error) {
    toast(error.message || "切换失败", true);
  } finally {
    toggle.disabled = false;
    toggle.checked = Boolean(state.autoMemory?.enabled);
  }
}

/** 记忆来源筛选值的中文名（与筛选弹窗选项一致）。 */
const RAG_SOURCE_LABELS = {
  personal: "个人",
  group: "群聊",
  private: "私聊",
  unknown: "未知",
};

const RAG_KIND_LABELS = {
  fact: "事实",
  episode: "事件",
  plan: "计划",
  preference: "偏好",
  relation: "关系",
  experience: "经验",
};

/** 参与者身份筛选值的中文名（与筛选弹窗选项一致）。 */
const RAG_IDENTITY_LABELS = {
  owner: "所有者",
  group_owner: "群主",
  group_admin: "群管理",
  member: "群成员",
};

/** 生命周期筛选值的中文名（与筛选弹窗选项一致）。 */
const RAG_LIFECYCLE_LABELS = {
  expiring: "即将过期",
  expired: "已过期",
  stale: "低价值将归档",
  cold: "长期未访问",
};

/**
 * 把规范作用域显示成无平台前缀的群号：
 *   ``SnowLuma:996392557`` → ``996392557``；
 *   ``SnowLuma:private:10001`` → ``私聊 10001``；空作用域返回空串。
 */
function ragScopeLabel(scope) {
  const text = String(scope || "").trim();
  if (!text) return "";
  if (text === "__global__") return "全局";
  const parts = text.split(":").filter((part) => part);
  if (!parts.length) return text;
  const tail = parts[parts.length - 1];
  if (parts.some((part) => String(part).toLowerCase() === "private")) {
    return `私聊 ${tail}`;
  }
  return tail;
}

/**
 * 汇总「筛选弹窗」里生效的条件，供按钮显示与 title 提示。
 * 关键词/角色输入框本身已经展示内容，不重复放进按钮文案。
 * 输出顺序：参与者身份 → 记忆来源 → 群号，例如「群主 群聊 996392557」。
 */
function ragFilterSummary() {
  const parts = [];
  if (state.rag.identity) {
    parts.push(RAG_IDENTITY_LABELS[state.rag.identity] || state.rag.identity);
  }
  if (state.rag.memoryType) {
    parts.push(RAG_SOURCE_LABELS[state.rag.memoryType] || state.rag.memoryType);
  }
  if (state.rag.kind) {
    parts.push(RAG_KIND_LABELS[state.rag.kind] || state.rag.kind);
  }
  if (state.rag.needsReview) parts.push("待复核");
  if (state.rag.scope) {
    const scopeLabel = ragScopeLabel(state.rag.scope);
    if (scopeLabel) parts.push(scopeLabel);
  }
  if (state.rag.lifecycle) {
    parts.push(
      RAG_LIFECYCLE_LABELS[state.rag.lifecycle] || state.rag.lifecycle
    );
  }
  return parts;
}

/** 把当前筛选内容同步到「筛选」按钮（无筛选时恢复为「筛选」）。 */
function syncRagFilterButton() {
  const button = $("rag-search");
  const label = $("rag-filter-label");
  if (!button || !label) return;
  const parts = ragFilterSummary();
  const active = parts.length > 0;
  const summary = parts.join(" ");
  button.classList.toggle("has-filters", active);
  label.textContent = active ? summary : "筛选";
  button.title = active ? `当前筛选：${summary}（点击修改）` : "筛选记忆";
  button.setAttribute(
    "aria-label",
    active ? `筛选，当前条件：${summary}` : "筛选记忆"
  );
}

/** 以当前页码、关键词、群范围和角色筛选条件加载记忆列表。 */
/** 记忆列表骨架：尺寸/行距与真实三行卡片一致。 */
function ragSkeletonRows(count = 6) {
  return Array.from(
    { length: count },
    () => `<tr class="mem-row skeleton-mem-row" aria-hidden="true">
      <td colspan="4" class="skeleton-mem-cell">
        <div class="skeleton-mem-card">
          <span class="skeleton skeleton-line skeleton-line-id"></span>
          <span class="skeleton skeleton-line skeleton-line-time"></span>
          <span class="skeleton skeleton-line skeleton-line-scope"></span>
          <span class="skeleton skeleton-arrow"></span>
        </div>
      </td>
    </tr>`
  ).join("");
}

function showRagSkeleton() {
  const body = $("rag-body");
  if (body) body.innerHTML = ragSkeletonRows(6);
}

/** 抽屉群列表骨架：行高与真实群条目一致。 */
function drawerSkeletonRows(count = 5) {
  return Array.from(
    { length: count },
    () => `<div class="drawer-group skeleton-drawer-row" aria-hidden="true">
      <span class="skeleton skeleton-avatar"></span>
      <span class="skeleton skeleton-name"></span>
      <span class="skeleton skeleton-dot"></span>
    </div>`
  ).join("");
}

function showGroupsSkeleton() {
  const drawer = $("drawer-groups");
  if (drawer) drawer.innerHTML = drawerSkeletonRows(5);
  const list = $("group-items");
  if (list) {
    list.innerHTML = `<div class="skeleton-group-list">${Array.from(
      { length: 5 },
      () => `<div class="group-item skeleton-group-item" aria-hidden="true">
        <span class="skeleton skeleton-avatar"></span>
        <span class="skeleton-group-text">
          <span class="skeleton skeleton-name"></span>
          <span class="skeleton skeleton-gid"></span>
        </span>
        <span class="skeleton skeleton-dot"></span>
      </div>`
    ).join("")}</div>`;
  }
}

async function loadRagList() {
  const requestId = (state.rag.requestId || 0) + 1;
  state.rag.requestId = requestId;
  // 仅首次加载显示骨架；筛选、翻页、刷新保留当前列表。
  if (!state.rag.loadedOnce) showRagSkeleton();
  if (state.rag.selectMode) {
    state.rag.selected.clear();
    syncRagBulkBar();
  }
  syncRagFilterButton();
  const params = {
    page: state.rag.page,
    page_size: state.rag.pageSize,
  };
  if (state.rag.keyword) params.keyword = state.rag.keyword;
  if (state.rag.scope) params.group_scope = state.rag.scope;
  if (state.rag.role) params.role_id = state.rag.role;
  if (state.rag.memoryType) params.memory_type = state.rag.memoryType;
  if (state.rag.kind) params.kind = state.rag.kind;
  if (state.rag.needsReview) params.needs_review = 1;
  // r3.3m：「已归档」视图下生命周期筛选无意义（后端也会忽略，这里同步隐藏）。
  if (state.rag.lifecycle && !state.rag.archivedOnly) {
    params.lifecycle = state.rag.lifecycle;
  }
  if (state.rag.identity) params.identity = state.rag.identity;
  // r3.3：只查看已归档记录时传 archived_only，不再使用旧的 include_archived。
  if (state.rag.archivedOnly) params.archived_only = 1;
  try {
    const data = await callApi("GET", "rag/list", params);
    // 慢请求不能覆盖后来的筛选/翻页结果。
    if (requestId !== state.rag.requestId) return;
    state.rag.total = Number(data.total || 0);
    state.rag.hasMore = Boolean(data.has_more);
    const totalPages = Math.max(1, Math.ceil(state.rag.total / state.rag.pageSize));
    // r3.3m：删除/归档最后一页的最后一条后，页码自动回收到有效范围。
    if (state.rag.page > totalPages && !state.rag.clamping) {
      state.rag.clamping = true;
      state.rag.page = totalPages;
      try {
        await loadRagList();
      } finally {
        state.rag.clamping = false;
      }
      return;
    }
    state.rag.items = Array.isArray(data.items) ? data.items : [];
    state.rag.loadedOnce = true;
    renderRagRows(state.rag.items);
    $("rag-page-info").textContent = `第 ${state.rag.page} / ${totalPages} 页 · 共 ${formatNumber(
      state.rag.total
    )} 条`;
    $("rag-prev").disabled = state.rag.page <= 1;
    $("rag-next").disabled = !state.rag.hasMore;
  } catch (error) {
    if (requestId === state.rag.requestId) {
      toast(error.message || "加载记忆列表失败", true);
    }
  }
}

/** 从关键词/角色输入框同步筛选条件，并从第一页重查（r3.3 快捷搜索）。 */
function quickSearchRag() {
  const keyword = $("rag-keyword");
  const role = $("rag-role");
  state.rag.keyword = keyword ? keyword.value.trim() : state.rag.keyword;
  state.rag.role = role ? role.value.trim() : state.rag.role;
  state.rag.page = 1;
  void loadRagList();
}

/** 同步「已归档」按钮的高亮态与 aria-pressed。 */
function syncRagArchivedButton() {
  const button = $("rag-archived-btn");
  if (!button) return;
  const active = Boolean(state.rag.archivedOnly);
  button.classList.toggle("active", active);
  button.setAttribute("aria-pressed", active ? "true" : "false");
}

/** 切换「已归档」过滤：只看归档记录或回到正常列表（r3.3）。 */
function toggleRagArchived() {
  if (state.rag.selectMode) {
    state.rag.selectMode = false;
    state.rag.selected.clear();
  }
  state.rag.archivedOnly = !state.rag.archivedOnly;
  if (state.rag.archivedOnly) {
    // r3.3m：已归档视图与生命周期筛选互斥，避免组合出恒空结果。
    state.rag.lifecycle = "";
    const select = $("filter-lifecycle");
    if (select) select.value = "";
  }
  state.rag.page = 1;
  syncRagArchivedButton();
  // 退出批量选择时同步刷新选择按钮/批量栏，避免按钮残留 active 高亮。
  syncRagBulkBar();
  void loadRagList();
}

/** 同步批量选择按钮与批量操作栏的状态。 */
function syncRagSelectButton() {
  const button = $("rag-select-btn");
  if (!button) return;
  const active = Boolean(state.rag.selectMode);
  button.classList.toggle("active", active);
  button.setAttribute("aria-pressed", active ? "true" : "false");
  button.title = active ? "退出批量选择" : "批量选择";
}

function syncRagBulkBar() {
  const bar = $("rag-bulk-bar");
  if (bar) bar.hidden = !state.rag.selectMode;
  const count = state.rag.selected.size;
  const countLabel = $("rag-bulk-count");
  if (countLabel) countLabel.textContent = `已选 ${count} 条`;
  const archiveBtn = $("rag-bulk-archive");
  if (archiveBtn) {
    archiveBtn.textContent = state.rag.archivedOnly ? "取消归档" : "归档";
    archiveBtn.disabled = count === 0;
  }
  const deleteBtn = $("rag-bulk-delete");
  if (deleteBtn) deleteBtn.disabled = count === 0;
  // 无论批量栏是否存在都要同步选择按钮，避免按钮状态卡死。
  syncRagSelectButton();
}

/** 进入/退出批量选择模式；进入时清空旧选择。 */
function toggleRagSelectMode(force) {
  const next = typeof force === "boolean" ? force : !state.rag.selectMode;
  if (next === state.rag.selectMode) {
    // 即使状态没变也重新同步，修复历史遗留的 active 高亮卡住。
    syncRagBulkBar();
    return;
  }
  state.rag.selectMode = next;
  state.rag.selected.clear();
  // 先同步按钮，再重绘；即使渲染异常也不会把按钮高亮卡住。
  syncRagSelectButton();
  try {
    renderRagRows(state.rag.items);
  } finally {
    syncRagBulkBar();
    syncRagSelectButton();
  }
}

/** 切换单条记忆的勾选状态。 */
function toggleRagSelected(memoryId) {
  const id = String(memoryId);
  if (!id) return;
  if (state.rag.selected.has(id)) state.rag.selected.delete(id);
  else state.rag.selected.add(id);
  const row = document.querySelector(
    `.mem-row[data-mem-row="${CSS.escape(id)}"]`
  );
  const selected = state.rag.selected.has(id);
  row?.classList.toggle("selected", selected);
  const check = row?.querySelector("[data-select]");
  if (check) {
    check.classList.toggle("checked", selected);
    check.setAttribute("aria-pressed", selected ? "true" : "false");
  }
  syncRagBulkBar();
}

/** 批量归档/取消归档已选记忆。 */
async function bulkArchiveSelected() {
  const ids = [...state.rag.selected];
  if (!ids.length) return;
  const archivedMode = state.rag.archivedOnly;
  const verb = archivedMode ? "取消归档" : "归档";
  const confirmed = await askConfirm(
    `${verb}已选的 ${ids.length} 条记忆？`,
    archivedMode ? {} : { danger: false }
  );
  if (!confirmed) return;
  let ok = 0;
  let failed = 0;
  for (const id of ids) {
    try {
      await callApi(
        "POST",
        `rag/${encodeURIComponent(id)}/${archivedMode ? "unarchive" : "archive"}`,
        {}
      );
      ok += 1;
    } catch {
      failed += 1;
    }
  }
  state.rag.selected.clear();
  state.rag.selectMode = false;
  renderRagRows(state.rag.items);
  syncRagBulkBar();
  await loadRagStats(false);
  await loadRagList();
  toast(
    failed
      ? `${verb}完成 ${ok} 条，失败 ${failed} 条`
      : `已${verb} ${ok} 条`
  );
}

/** 批量删除已选记忆（进入回收站，可单条撤销）。 */
async function bulkDeleteSelected() {
  const ids = [...state.rag.selected];
  if (!ids.length) return;
  const confirmed = await askConfirm(
    `删除已选的 ${ids.length} 条记忆？删除后进入回收站。`,
    { danger: true }
  );
  if (!confirmed) return;
  let ok = 0;
  let failed = 0;
  const trashIds = [];
  for (const id of ids) {
    try {
      const data = await callApi(
        "POST",
        `rag/${encodeURIComponent(id)}/delete`,
        {}
      );
      if (data && data.trash_id) trashIds.push(data.trash_id);
      ok += 1;
    } catch {
      failed += 1;
    }
  }
  state.rag.selected.clear();
  state.rag.selectMode = false;
  renderRagRows(state.rag.items);
  syncRagBulkBar();
  await loadRagStats(false);
  await loadRagList();
  showSnackbar(
    failed ? `已删除 ${ok} 条，失败 ${failed} 条` : `已删除 ${ok} 条`,
    trashIds
  );
}

/** 查询并在详情抽屉中展示指定记忆的完整 JSON 内容。 */
async function showMemory(memoryId) {
  const requestId = ++detailRequestSeq;
  try {
    const data = await callApi("GET", `rag/${encodeURIComponent(memoryId)}`);
    if (requestId !== detailRequestSeq) return;
    const rules = state.rag.rules || {};
    const meta =
      data.extra_metadata && typeof data.extra_metadata === "object"
        ? data.extra_metadata
        : {};
    const reason = String(data.archive_reason || "");
    const archiveText = data.archived
      ? `${data.archived_at_iso || "-"}（${ARCHIVE_REASON_LABELS[reason] || "已归档"}）`
      : data.expired
        ? "已过期，等待维护任务（每小时）归档"
        : data.valid_until
          ? `有效期 ${String(data.valid_until_iso || "").slice(0, 19)} 到期后 1 小时内`
          : `低价值且 ${Number(rules.archive_days ?? 30)} 天未访问时（维护任务每小时）`;
    const validityText = data.valid_until
      ? `${String(data.valid_until_iso || "").slice(0, 19)}${
          data.expired
            ? "（已过期）"
            : typeof data.days_left === "number"
              ? `（剩 ${Math.max(0, data.days_left)} 天）`
              : ""
        }`
      : "永久（该类型不写有效期）";
    const kind = String(data.kind || "fact").toLowerCase();
    const needsReview = Boolean(
      data.kind_needs_review ?? meta.kind_needs_review
    );
    const atoms = Array.isArray(data.atoms) ? data.atoms : [];
    const topics = Array.isArray(meta.topics)
      ? meta.topics.map(String).filter(Boolean)
      : [];
    const entities = Array.isArray(meta.entities)
      ? meta.entities.map(String).filter(Boolean)
      : [];
    const childTags = atoms.flatMap((atom) => {
      const childMeta =
        atom.extra_metadata && typeof atom.extra_metadata === "object"
          ? atom.extra_metadata
          : {};
      const values = [];
      if (Array.isArray(childMeta.topics)) values.push(...childMeta.topics);
      if (Array.isArray(childMeta.entities)) values.push(...childMeta.entities);
      return values.map(String).filter(Boolean);
    });
    const tags = [...new Set([...topics, ...entities, ...childTags])].slice(0, 12);
    const isAtomDetail = Boolean(meta.episode_id);
    const kindBadge = memoryKindBadge(data, isAtomDetail);
    const createdAt = String(data.created_at_iso || "-").replace("T", " ").slice(0, 19);
    const updatedAt = String(
      data.updated_at_iso || data.memory_time_iso || data.created_at_iso || "-"
    )
      .replace("T", " ")
      .slice(0, 19);
    const persona = String(meta.persona_name || meta.persona_id || "").trim();
    const origin = String(meta.origin || "").trim();
    const facts = atoms.length
      ? atoms
      : kind === "episode"
        ? []
        : [{ memory_id: data.memory_id, kind, content: data.content, extra_metadata: meta }];
    const factsHtml = facts.length
      ? `<ul class="detail-facts">${facts
          .map((fact) => {
            const factMeta =
              fact.extra_metadata && typeof fact.extra_metadata === "object"
                ? fact.extra_metadata
                : {};
            return `<li class="detail-fact">
              <span class="mono muted">#${escapeHtml(fact.memory_id)}</span>
              ${memoryKindBadge(fact, true)}
              <span>${escapeHtml(fact.content)}</span>
            </li>`;
          })
          .join("")}</ul>`
      : '<p class="muted">暂无原子事实</p>';
    const tagsHtml = tags.length
      ? `<div class="detail-tags">${tags
          .map((tag) => `<span class="detail-tag">${escapeHtml(tag)}</span>`)
          .join("")}</div>`
      : '<p class="muted">无主题标签</p>';
    const episodeHtml = data.episode
      ? `<section class="detail-section"><h4>所属事件</h4>
          <a class="detail-link" href="#" data-detail="${escapeHtml(
            data.episode.memory_id
          )}">#${escapeHtml(data.episode.memory_id)} ${escapeHtml(
            data.episode.content || ""
          )}</a></section>`
      : "";
    const timeline = [
      `写入：${data.created_at_iso || "-"}${
        data.memory_time_iso
          ? `（发生 ${String(data.memory_time_iso).replace("T", " ").slice(0, 19)}）`
          : ""
      }`,
      `重要性：原始 ${Number(data.importance ?? 0).toFixed(2)} → 当前有效 ${Number(
        data.effective_importance ?? data.importance ?? 0
      ).toFixed(2)}`,
      `召回：${formatNumber(data.access_count || 0)} 次${
        data.last_access_iso
          ? `（最近 ${String(data.last_access_iso).replace("T", " ").slice(0, 19)}）`
          : ""
      }`,
      `有效期：${validityText}`,
      `预计归档：${archiveText}`,
      data.archived
        ? `归档时间：${data.archived_at_iso || "-"}（原因：${
            ARCHIVE_REASON_LABELS[reason] || "已归档"
          }）`
        : "",
    ].filter(Boolean);

    $("detail-title").textContent = `记忆详情 #${data.memory_id || memoryId}`;
    $("detail-body").classList.add("detail-rich");
    $("detail-body").innerHTML = `
      <div class="detail-hero">
        <div class="detail-hero-id mono">#${escapeHtml(data.memory_id || memoryId)}</div>
        <div class="detail-hero-badges">${kindBadge}
          <span class="pill type-${escapeHtml(
            MEMORY_TYPE_CLASSES.has(String(data.memory_type || "").toLowerCase())
              ? String(data.memory_type).toLowerCase()
              : "unknown"
          )}">${escapeHtml(data.memory_type || "group")}</span>
        </div>
      </div>
      <section class="detail-section">
        <h4>基本信息</h4>
        <div class="detail-meta-grid">
          <div><span>会话 ID</span><strong class="mono">${escapeHtml(origin || "-")}</strong></div>
          <div><span>人格 ID</span><strong>${escapeHtml(persona || "未记录")}</strong></div>
          <div><span>创建时间</span><strong class="mono">${escapeHtml(createdAt)}</strong></div>
          <div><span>更新时间</span><strong class="mono">${escapeHtml(updatedAt)}</strong></div>
          <div><span>置信度</span><strong>${Number(meta.confidence ?? 0).toFixed(2)}</strong></div>
          <div><span>群范围</span><strong class="mono">${escapeHtml(
            String(data.group_scope || "").trim() || "全局"
          )}</strong></div>
        </div>
      </section>
      <section class="detail-section">
        <h4>分类复核</h4>
        <div class="detail-kind-editor">
          <select id="detail-kind-select" class="input" aria-label="记忆类型">
            ${Object.entries(MEMORY_KIND_BADGES)
              .map(
                ([value, badge]) =>
                  `<option value="${value}" ${value === kind ? "selected" : ""}>${badge.label}</option>`
              )
              .join("")}
          </select>
          <div class="detail-kind-actions">
            <button id="detail-kind-save" class="primary small-btn" type="button">保存类型</button>
            <button id="detail-kind-reviewed" class="ghost small-btn" type="button">标记已复核</button>
          </div>
        </div>
        <p class="muted small">
          分类置信度：${
            meta.kind_confidence !== undefined
              ? Number(meta.kind_confidence).toFixed(2)
              : "未记录"
          }${meta.kind_reason ? ` · ${escapeHtml(String(meta.kind_reason))}` : ""}${
            meta.kind_review_suggested
              ? ` · 建议 ${escapeHtml(String(meta.kind_review_suggested))}`
              : ""
          }${needsReview ? " · 当前待复核" : ""}
        </p>
      </section>
      <section class="detail-section">
        <h4>关键事实</h4>
        ${factsHtml}
      </section>
      ${episodeHtml}
      <section class="detail-section">
        <h4>主题</h4>
        ${tagsHtml}
      </section>
      <section class="detail-section">
        <h4>正文</h4>
        <p class="detail-content">${escapeHtml(data.content || "")}</p>
      </section>
      <section class="detail-section">
        <h4>遗忘时间线</h4>
        <pre class="detail-timeline mono">${escapeHtml(timeline.join("\n"))}</pre>
      </section>
      <details class="detail-raw">
        <summary>原始记录</summary>
        <div class="raw-json-card">
          <div class="raw-json-head"><span>JSON</span><button type="button" class="raw-copy" data-copy-raw>复制</button></div>
          <pre class="raw-json mono">${highlightJson(JSON.stringify(data, null, 2))}</pre>
        </div>
      </details>`;
    $("detail-body").querySelectorAll("[data-detail]").forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        void showMemory(link.dataset.detail);
      });
    });
    $("detail-body").querySelectorAll("[data-copy-raw]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await copyTextToClipboard(JSON.stringify(data, null, 2));
          toast("原始 JSON 已复制");
        } catch {
          toast("复制失败，请长按选择", true);
        }
      });
    });
    const saveKind = async (markReviewed, useSelected = true) => {
      const select = $("detail-kind-select");
      const nextKind = useSelected && select ? select.value : kind;
      if (!nextKind) return;
      try {
        await callApi(
          "POST",
          `rag/${encodeURIComponent(data.memory_id || memoryId)}/kind`,
          {
            kind: nextKind,
            expected_kind: kind,
            mark_reviewed: markReviewed,
          }
        );
        toast(useSelected ? "类型已保存并标记已复核" : "已标记复核");
        await Promise.all([loadRagStats(false), loadRagList()]);
        await showMemory(data.memory_id || memoryId);
      } catch (error) {
        toast(error.message || "修改类型失败", true);
      }
    };
    $("detail-kind-save")?.addEventListener("click", () => void saveKind(true, true));
    $("detail-kind-reviewed")?.addEventListener("click", () => void saveKind(true, false));
    openDetailPanel();
  } catch (error) {
    toast(error.message || "加载详情失败", true);
  }
}

/** 打开记忆详情/自检抽屉：和汉堡菜单一样从左侧挤出主页面。 */
function openDetailPanel() {
  if (document.activeElement && document.activeElement !== document.body) {
    document.activeElement.blur?.();
  }
  const detail = $("detail");
  if (!isDesktopLayout() && isDrawerOpen()) closeDrawer();
  if (!detail.classList.contains("open")) {
    pushScrollY = window.scrollY || 0;
  }
  window.scrollTo(0, 0);
  updateScrolledClass();
  $("main-pane").classList.add("is-pushed");
  setContentScrollOffset(pushScrollY);
  document.body.classList.add("detail-open");
  detail.classList.add("open");
  detail.setAttribute("aria-hidden", "false");
  $("scrim").classList.add("show");
  // 详情侧栏没有自己的 ×：汉堡按钮变成 ×，并用它关闭
  setMenuMorph(true);
  $("menu-btn").setAttribute("aria-expanded", "true");
  $("menu-btn").setAttribute("aria-label", "关闭详情");
}

/** 关闭记忆详情/自检抽屉，页面滚动位置保持不变。 */
function closeDetail() {
  const wasOpen = $("detail").classList.contains("open");
  beginPanelClose();
  $("detail").classList.remove("open");
  $("detail").setAttribute("aria-hidden", "true");
  $("main-pane").classList.remove("is-pushed");
  setContentScrollOffset(0);
  document.body.classList.remove("detail-open");
  $("scrim").classList.remove("show");
  setMenuMorph(false);
  $("menu-btn").setAttribute("aria-expanded", "false");
  $("menu-btn").setAttribute("aria-label", "打开菜单");
  if (wasOpen) window.scrollTo(0, pushScrollY);
  updateScrolledClass();
}

let snackbarTimer = null;
let snackbarTrash = null;

/** 显示带撤销按钮的 Snackbar，并用 4 秒进度条提示剩余时间。 */
function showSnackbar(text, trashId = null) {
  const bar = $("snackbar-bar");
  $("snackbar-text").textContent = text;
  $("snackbar").classList.add("show");
  if (Array.isArray(trashId)) {
    const ids = trashId.map((id) => Number(id)).filter((id) => id > 0);
    snackbarTrash = ids.length ? ids : null;
  } else {
    snackbarTrash = trashId;
  }
  if (bar) {
    bar.classList.remove("countdown");
    bar.style.width = "";
    void bar.offsetWidth;
    bar.classList.add("countdown");
  }
  clearTimeout(snackbarTimer);
  snackbarTimer = setTimeout(hideSnackbar, 4000);
}

function hideSnackbar() {
  $("snackbar").classList.remove("show");
  const bar = $("snackbar-bar");
  if (bar) bar.classList.remove("countdown");
  snackbarTrash = null;
}

async function undoDelete() {
  if (!snackbarTrash) {
    hideSnackbar();
    return;
  }
  const trashIds = Array.isArray(snackbarTrash)
    ? snackbarTrash
    : [snackbarTrash];
  hideSnackbar();
  let ok = 0;
  let failed = 0;
  for (const trashId of trashIds) {
    try {
      await callApi("POST", "rag/restore", { trash_id: trashId });
      ok += 1;
    } catch {
      failed += 1;
    }
  }
  toast(
    failed ? `已撤销 ${ok} 条，失败 ${failed} 条` : `已撤销删除 ${ok} 条`
  );
  await loadRagStats(false);
  await loadRagList();
}

/** 确认删除指定记忆，成功后显示可撤销 Snackbar。 */
async function deleteMemory(memoryId) {
  const confirmed = await askConfirm(`确定删除记忆 ${memoryId}？`, {
    danger: true,
  });
  if (!confirmed) return;
  try {
    const data = await callApi(
      "POST",
      `rag/${encodeURIComponent(memoryId)}/delete`,
      {}
    );
    showSnackbar(`已删除记忆 ${memoryId}`, data.trash_id);
    await loadRagStats(false);
    await loadRagList();
  } catch (error) {
    toast(error.message || "删除失败", true);
  }
}

/** 归档指定记忆（r3.3）：归档后不再参与召回，可在「已归档」中恢复或彻底删除。 */
async function archiveMemory(memoryId) {
  const confirmed = await askConfirm(
    `归档记忆 ${memoryId}？归档后不再参与召回，可在「已归档」中取消归档恢复。`
  );
  if (!confirmed) return;
  try {
    await callApi("POST", `rag/${encodeURIComponent(memoryId)}/archive`, {});
    toast("已归档");
    await loadRagStats(false);
    await loadRagList();
  } catch (error) {
    toast(error.message || "归档失败", true);
  }
}

/** 取消归档并恢复指定记忆（r3.3）：后端会按 TTL 自适应重算有效期。 */
async function unarchiveMemory(memoryId) {
  try {
    await callApi("POST", `rag/${encodeURIComponent(memoryId)}/unarchive`, {});
    toast("已取消归档");
    await loadRagStats(false);
    await loadRagList();
  } catch (error) {
    toast(error.message || "取消归档失败", true);
  }
}

/** 彻底删除已归档记忆（r3.3）：物理删除、不进回收站，需二次确认。 */
async function purgeMemory(memoryId) {
  const confirmed = await askConfirm(
    `彻底删除记忆 ${memoryId}？将直接从数据库移除，不进入回收站，之后只能靠备份恢复。`,
    { danger: true }
  );
  if (!confirmed) return;
  try {
    await callApi("POST", `rag/${encodeURIComponent(memoryId)}/purge`, {
      confirm: 1,
    });
    toast("已彻底删除");
    await loadRagStats(false);
    await loadRagList();
  } catch (error) {
    toast(error.message || "彻底删除失败", true);
  }
}

/** 打开筛选弹窗，回填当前筛选条件。 */
function openFilterModal() {
  $("filter-type").value = state.rag.memoryType;
  if ($("filter-kind")) $("filter-kind").value = state.rag.kind;
  $("filter-scope").value = state.rag.scope;
  $("filter-identity").value = state.rag.identity;
  if ($("filter-lifecycle")) $("filter-lifecycle").value = state.rag.lifecycle;
  const modal = $("filter-modal");
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
}

function closeFilterModal() {
  const modal = $("filter-modal");
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
}

/* ------------------------------------------------------------------ */
/* 开发者选项（r3.3）                                                  */
/* ------------------------------------------------------------------ */
/** 切换开发者页「密钥门槛」与「工具面板」的显隐。 */
function setDevAuthed(authed) {
  const gate = $("dev-gate");
  const tools = $("dev-tools");
  if (gate) gate.hidden = Boolean(authed);
  if (tools) tools.hidden = !authed;
}

/** 清空开发者令牌与密钥输入，回到「等待输入密钥」的门槛状态。 */
function resetDevGate() {
  state.dev.token = "";
  state.dev.authed = false;
  const input = $("dev-key");
  if (input) input.value = "";
  ["dev-prompt-group", "dev-prompt-private"].forEach((id) => {
    const area = $(id);
    if (area) area.value = "";
  });
  setDevAuthed(false);
}

/** 开发者接口报「密钥无效/过期」时清令牌并回到门槛；返回是否已处理。 */
function handleDevAuthError(error) {
  const message = String((error && error.message) || "");
  if (!message.includes("开发者密钥")) return false;
  resetDevGate();
  toast("开发者密钥无效或已过期，请重新输入", true);
  const input = $("dev-key");
  if (input) input.focus();
  return true;
}

/** 打开记忆修复设置：从当前全局配置回填。 */
async function openRepairSettings() {
  // 开发者页不会自动加载全局设置；先确认值存在，避免用页面默认值
  // 覆盖用户已保存的修复阈值/反向开关。
  let confidence = getPath(
    state.globalValues,
    "memory_rag.memory_generation.memory_kind_repair_confidence"
  );
  let reverse = getPath(
    state.globalValues,
    "memory_rag.memory_generation.memory_kind_reverse_repair"
  );
  if (confidence === undefined || reverse === undefined) {
    try {
      const data = await callApi("GET", "config");
      state.globalValues = data.values || {};
      if (data.schema) state.schema = data.schema;
      confidence = getPath(
        state.globalValues,
        "memory_rag.memory_generation.memory_kind_repair_confidence"
      );
      reverse = getPath(
        state.globalValues,
        "memory_rag.memory_generation.memory_kind_reverse_repair"
      );
    } catch (error) {
      toast(error.message || "读取记忆修复设置失败", true);
      return;
    }
  }
  const input = $("repair-confidence");
  if (input) input.value = String(confidence ?? 0.8);
  if ($("repair-confidence-value")) {
    $("repair-confidence-value").textContent = Number(input?.value || 0.8).toFixed(2);
  }
  if ($("repair-reverse")) $("repair-reverse").checked = Boolean(reverse);
  const modal = $("repair-settings-modal");
  if (modal) {
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
  }
}

function closeRepairSettings() {
  const modal = $("repair-settings-modal");
  if (!modal) return;
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
}

async function saveRepairSettings() {
  const button = $("repair-settings-save");
  if (!button || button.disabled) return;
  const confidence = Number($("repair-confidence")?.value || 0.8);
  const reverse = Boolean($("repair-reverse")?.checked);
  button.disabled = true;
  try {
    await callApi("POST", "config/save", {
      values: {
        "memory_rag.memory_generation.memory_kind_repair_confidence": confidence,
        "memory_rag.memory_generation.memory_kind_reverse_repair": reverse,
      },
    });
    setPath(
      state.globalValues,
      "memory_rag.memory_generation.memory_kind_repair_confidence",
      confidence
    );
    setPath(
      state.globalValues,
      "memory_rag.memory_generation.memory_kind_reverse_repair",
      reverse
    );
    closeRepairSettings();
    toast("记忆修复设置已保存");
  } catch (error) {
    toast(error.message || "保存记忆修复设置失败", true);
  } finally {
    button.disabled = false;
  }
}

/** 记忆修复：先预览候选，确认后再写库；无需开发者密钥。 */
async function repairMemoryKinds() {
  const button = $("dev-memory-repair");
  if (!button || button.disabled) return;
  button.disabled = true;
  try {
    const preview = await callApi("POST", "rag/repair_kinds", {
      apply: false,
      use_model: true,
    });
    const count = Number(preview.repairable || 0);
    const reviewCount = Number(preview.review_count || 0);
    if (!count) {
      toast(
        reviewCount
          ? `没有高置信度可修复项；${reviewCount} 条置信度不足，尚未处理`
          : "没有需要修复的记忆"
      );
      return;
    }
    const samples = (Array.isArray(preview.items) ? preview.items : [])
      .slice(0, 3)
      .map((item) => `#${item.memory_id} ${item.old_kind} → ${item.new_kind}`)
      .join("；");
    const ok = await askConfirm(
      `发现 ${count} 条可修复记忆${samples ? `：${samples}` : ""}。${reviewCount ? `另有 ${reviewCount} 条低置信度记录待复核。` : ""}\n\n先用规则、再用模型复核后改判类型、重算有效期并修复群身份，是否应用？`,
      { danger: true }
    );
    if (!ok) return;
    const result = await callApi("POST", "rag/repair_kinds", {
      apply: true,
      use_model: true,
    });
    const marked = Number(result.review_marked || 0);
    toast(
      `已修复 ${Number(result.applied || 0)} 条记忆` +
        (marked ? `，标记待复核 ${marked} 条` : "")
    );
    await Promise.all([loadRagStats(true), loadRagList()]);
  } catch (error) {
    toast(error.message || "记忆修复失败", true);
  } finally {
    button.disabled = false;
  }
}

/** 开发者登录：用密钥换取限时令牌，成功后展示提示词/自检工具。 */
async function devLogin() {
  const input = $("dev-key");
  const button = $("dev-login");
  // 回车与点击可能连续触发，上一次登录未结束时忽略重复提交。
  if (button && button.disabled) return;
  const key = input ? input.value.trim() : "";
  if (!key) {
    toast("请输入开发者密钥", true);
    if (input) input.focus();
    return;
  }
  if (button) button.disabled = true;
  try {
    const data = await callApi("POST", "dev/login", { key });
    const token = String(data.token || "");
    if (!token) throw new Error("登录失败：未返回开发者令牌");
    state.dev.token = token;
    state.dev.authed = true;
    // 登录成功后立即清空密钥输入框，避免明文停留在 DOM 中。
    if (input) input.value = "";
    setDevAuthed(true);
    toast("已进入开发者选项");
    void loadDevPrompts();
  } catch (error) {
    // 失败时清空输入，避免密钥明文停留在页面上。
    resetDevGate();
    toast(error.message || "开发者密钥错误", true);
    if (input) input.focus();
  } finally {
    if (button) button.disabled = false;
  }
}

/** 退出开发者选项：注销服务端令牌并回到密钥门槛。 */
async function devLogout() {
  const token = state.dev.token;
  resetDevGate();
  try {
    await callApi("POST", "dev/logout", { dev_token: token });
  } catch (error) {
    // 令牌已失效时本地照常退出，不打扰用户。
    console.warn("[ActiveReply] 注销开发者令牌失败", error);
  }
  toast("已退出开发者选项");
}

/** 读取开发者选项里的抽取提示词补充层（r3.3n）。 */
async function loadDevPrompts() {
  if (!state.dev.token) return;
  try {
    // POST 走请求体带令牌，令牌不再出现在 URL/query 中。
    const data = await callApi("POST", "dev/prompts", {
      dev_token: state.dev.token,
    });
    const group = $("dev-prompt-group");
    const priv = $("dev-prompt-private");
    if (group) group.value = String(data.group_prompt || "");
    if (priv) priv.value = String(data.private_prompt || "");
  } catch (error) {
    if (!handleDevAuthError(error)) {
      toast(error.message || "读取提示词失败", true);
    }
  }
}

/** 保存开发者选项里的抽取提示词补充层（r3.3n）。 */
async function saveDevPrompts() {
  if (!state.dev.token) {
    resetDevGate();
    toast("请先输入开发者密钥", true);
    return;
  }
  const button = $("dev-prompt-save");
  if (button) button.disabled = true;
  try {
    await callApi("POST", "dev/prompts/save", {
      dev_token: state.dev.token,
      group_prompt: $("dev-prompt-group")?.value || "",
      private_prompt: $("dev-prompt-private")?.value || "",
    });
    toast("提示词已保存");
  } catch (error) {
    if (!handleDevAuthError(error)) {
      toast(error.message || "保存提示词失败", true);
    }
  } finally {
    if (button) button.disabled = false;
  }
}

/**
 * 执行记忆自检（仅开发者可见）。
 *
 * 默认只读：不向生产库写临时记忆、不调用抽取模型；按住 Shift 点击才运行
 * 深度自检（写删临时记忆 + 调用一次抽取模型），避免日常排障消耗 token（F23）。
 */
async function devSelftest(deep = false) {
  const button = $("dev-selftest");
  if (!state.dev.token) {
    resetDevGate();
    toast("请先输入开发者密钥", true);
    return;
  }
  if (button) button.disabled = true;
  try {
    const data = await callApi("POST", "rag/selftest", {
      deep: deep ? 1 : 0,
      dev_token: state.dev.token,
    });
    const steps = data.steps || [];
    const prompt = data.prompt || null;
    const isDeep = data.mode === "deep";
    const lines = [
      data.passed ? "✅ 存储自检通过" : "❌ 存储自检未通过",
      `模式：${isDeep ? "深度（写临时记忆 + 调用抽取模型）" : "只读（不写库、不调模型）"}`,
      "",
      ...steps.map(
        (step) =>
          `${step.ok ? "✅" : "❌"} ${step.name}${step.detail ? `：${step.detail}` : ""}`
      ),
    ];
    if (prompt) {
      const promptItems = prompt.items || [];
      lines.push(
        "",
        "── 提示词自检 ──",
        prompt.ok
          ? `✅ 解析 ${promptItems.length} 条，证据通过 ${prompt.evidence_ok} 条，${prompt.elapsed_ms}ms`
          : `❌ ${prompt.error || "未解析出记忆"}`,
        `模型：${prompt.model || "-"}｜契约版本 v${prompt.version ?? "-"}`,
        ...promptItems.map(
          (item, index) =>
            `${index + 1}. [${item.kind}] ${item.content}（置信 ${Number(
              item.confidence
            ).toFixed(2)}）`
        )
      );
      if (prompt.raw) lines.push("", "原始输出：", prompt.raw);
    } else {
      lines.push(
        "",
        "── 提示词自检 ──",
        "本次为只读自检，未调用抽取模型；按住 Shift 点击「自检」可运行深度自检。"
      );
    }
    $("detail-title").textContent = "记忆自检";
    $("detail-body").classList.remove("detail-rich");
    $("detail-body").textContent = lines.join("\n");
    openDetailPanel();
    const allOk = Boolean(data.passed) && (!prompt || Boolean(prompt.ok));
    toast(allOk ? "记忆自检通过" : "记忆自检未全部通过", !allOk);
  } catch (error) {
    if (!handleDevAuthError(error)) {
      toast(error.message || "自检失败", true);
    }
  } finally {
    if (button) button.disabled = false;
  }
}

/** 提示词对照（仅开发者可见）：展示设置里的补充要求与实际发送的提示词。 */
async function devPromptPreview() {
  const button = $("dev-prompt");
  if (!state.dev.token) {
    resetDevGate();
    toast("请先输入开发者密钥", true);
    return;
  }
  if (button) button.disabled = true;
  try {
    const data = await callApi("POST", "config/prompt-preview", {
      kind: "group",
      dev_token: state.dev.token,
    });
    const lines = [
      `版本 v${data.version}｜设置 ${data.configured_chars} 字｜实际发送 ${data.chars} 字`,
      `${data.is_default ? "默认补充" : "自定义补充"}｜多原子：${
        data.multi_atom ? "开" : "关"
      }｜输出契约：${data.contract ? "多条原子" : "单条摘要"}`,
      "",
      "── 你的补充要求（可编辑）──",
      data.supplement || "(空)",
      "",
      "── 系统固定规则（身份/格式/时间/评分/输出契约，不可编辑）──",
      data.system_rules || "(空)",
      "",
      "── 实际发送（补充 + 固定规则，与试跑/抽取一致）──",
      data.prompt || "(空)",
    ];
    $("detail-title").textContent = "提示词对照";
    $("detail-body").classList.remove("detail-rich");
    $("detail-body").textContent = lines.join("\n");
    openDetailPanel();
  } catch (error) {
    if (!handleDevAuthError(error)) {
      toast(error.message || "获取提示词失败", true);
    }
  } finally {
    if (button) button.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* 事件绑定                                                            */
/* ------------------------------------------------------------------ */
/** 按钮加载锁最短展示时长；实际解锁还要等接口请求结束（F42）。 */
const BUTTON_LOCK_MIN_MS = 1000;
/** 兜底上限：极端情况下请求迟迟不结算，也不能让按钮永久卡住。 */
const BUTTON_LOCK_MAX_MS = 6000;
/** 刷新图标转一圈的时长，必须与 CSS 的 areply-icon-spin 保持一致。 */
const BUTTON_SPIN_MS = 1000;
/** 形变回原图标的收尾时长。 */
const BUTTON_MORPH_BACK_MS = 500;
/** 当前处于加载锁状态的按钮：button -> { startedAt, startedAtCount, spinner }。 */
const lockedButtons = new Map();
let lockMonitorTimer = 0;

/** 把按钮图标替换成可被 Morphicons 操作的 SVG path。 */
function startButtonMorph(button, entry) {
  if (!button || !lockedButtons.has(button) || entry.morphStarted) return;
  const refreshD = BUTTON_MORPH_PATHS.refresh;
  if (!refreshD) {
    entry.morphStarted = false;
    void preloadButtonMorphPaths().then(() => {
      if (lockedButtons.has(button)) startButtonMorph(button, entry);
    });
    return;
  }
  if (!morphicons) {
    entry.morphStarted = false;
    window.setTimeout(() => {
      if (lockedButtons.has(button)) startButtonMorph(button, entry);
    }, 120);
    return;
  }
  entry.morphStarted = true;
  const originalIcon = button.querySelector(".icon");
  let iconName = "";
  if (originalIcon) {
    iconName = [...originalIcon.classList]
      .find((name) => name.startsWith("icon-") && name !== "icon")
      ?.slice(5) || "";
  }
  const initialD = BUTTON_MORPH_PATHS[iconName] || refreshD;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "icon morph-icon btn-morph-icon");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const pathEl = document.createElementNS("http://www.w3.org/2000/svg", "path");
  pathEl.setAttribute("d", initialD);
  svg.appendChild(pathEl);
  if (originalIcon) {
    entry.originalIcon = originalIcon;
    originalIcon.replaceWith(svg);
  } else {
    entry.createdIcon = true;
    entry.morphSvg = svg;
    svg.classList.add("btn-morph-overlay");
    button.classList.add("btn-morph-textless");
    button.appendChild(svg);
  }
  entry.morphSvg = svg;
  entry.originalPathD = initialD;
  if (!entry.spinStartedAt) entry.spinStartedAt = performance.now();
  try {
    const morph = morphicons.createMorph(pathEl, initialD, {
      strokeWidth: 1.8,
      reducedMotion: "user",
    });
    entry.morph = morph;
    morph.morphTo(refreshD);
  } catch (error) {
    // 形变失败时保留刷新图标，由 CSS 继续旋转。
    pathEl.setAttribute("d", refreshD);
  }
}

/** 给按钮加加载锁；同一次点击只登记一次，由 unlockButton 释放。 */
function lockButton(button) {
  if (!button || lockedButtons.has(button)) return;
  const entry = {
    startedAt:
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now(),
    startedAtCount: apiStartedCount,
    morphStarted: false,
    spinStartedAt: 0,
    unlockScheduled: false,
    finishing: false,
  };
  button.classList.add("btn-loading");
  button.setAttribute("aria-busy", "true");
  button.dataset.locking = "1";
  lockedButtons.set(button, entry);
  startButtonMorph(button, entry);
  ensureLockMonitor();
}

/** 解锁单个按钮并清理加载态标记。 */
function finishButtonUnlock(button, entry) {
  entry.finishing = true;
  button.classList.add("btn-morph-back");
  const backD = entry.originalPathD;
  const refreshD = BUTTON_MORPH_PATHS.refresh;
  if (entry.morph && backD && refreshD && backD !== refreshD) {
    try {
      entry.morph.morphTo(backD);
    } catch (error) {
      // 形变回退失败时直接恢复原图标。
    }
  }
  window.setTimeout(() => {
    if (entry.morph) entry.morph.destroy();
    if (entry.morphSvg) {
      if (entry.originalIcon) entry.morphSvg.replaceWith(entry.originalIcon);
      else entry.morphSvg.remove();
    }
    button.classList.remove("btn-loading", "btn-morph-back", "btn-morph-textless");
    button.removeAttribute("aria-busy");
    delete button.dataset.locking;
    lockedButtons.delete(button);
  }, BUTTON_MORPH_BACK_MS);
}

/** 解锁单个按钮；若图标还在转，先转完完整一圈再形变回原图标。 */
function unlockButton(button) {
  const entry = lockedButtons.get(button);
  if (!entry) {
    button.classList.remove("btn-loading", "btn-morph-back", "btn-morph-textless");
    button.removeAttribute("aria-busy");
    delete button.dataset.locking;
    return;
  }
  if (entry.finishing || entry.unlockScheduled) return;
  if (entry.morphSvg) {
    const now =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    const started = entry.spinStartedAt || now;
    const elapsed = Math.max(0, now - started);
    let wait = 0;
    if (elapsed < BUTTON_SPIN_MS) {
      wait = BUTTON_SPIN_MS - elapsed;
    } else {
      const remainder = elapsed % BUTTON_SPIN_MS;
      wait = remainder < 40 ? 0 : BUTTON_SPIN_MS - remainder;
    }
    entry.unlockScheduled = true;
    if (wait > 0) {
      window.setTimeout(() => finishButtonUnlock(button, entry), wait + 10);
      return;
    }
  }
  finishButtonUnlock(button, entry);
}

/**
 * 释放满足条件的按钮锁：至少展示 1 秒，且本按钮触发的接口请求已经结束。
 * 若点击没有发出请求（如前端校验失败），只受最短时长约束。
 */
function releaseFinishedButtonLocks() {
  const now =
    typeof performance !== "undefined" && performance.now
      ? performance.now()
      : Date.now();
  lockedButtons.forEach((entry, button) => {
    const elapsed = now - entry.startedAt;
    if (elapsed < BUTTON_LOCK_MIN_MS) return;
    const requestStarted = apiStartedCount > entry.startedAtCount;
    if (requestStarted && apiInFlight > 0 && elapsed < BUTTON_LOCK_MAX_MS) {
      return;
    }
    unlockButton(button);
  });
  if (!lockedButtons.size && lockMonitorTimer) {
    window.clearInterval(lockMonitorTimer);
    lockMonitorTimer = 0;
  }
}

/** 有按钮处于加载锁时启动轮询，避免每个按钮各自挂一个计时器。 */
function ensureLockMonitor() {
  if (lockMonitorTimer) return;
  lockMonitorTimer = window.setInterval(releaseFinishedButtonLocks, 120);
}

/**
 * 键盘视口恢复（r3.3v）：Android WebView 在收起软键盘后偶尔不重算可视视口，
 * 表现为点击偏移或下半屏点不动。任何 input/textarea 失焦后，若焦点已回落到
 * body，则分两次（键盘动画中/结束后）触发 resize 并原地重设滚动位置，强制重算。
 */
/** 键盘视口恢复：键盘收起后恢复滚动位置并强制重算命中区。 */
/** 键盘视口恢复：键盘收起后恢复滚动位置并强制重算命中区。 */
function installViewportRecovery() {
  const viewport = window.visualViewport;
  let keyboardOpen = false;
  let stabilizeTimer = 0;

  if (viewport) {
    const onViewportChange = () => {
      if (isKeyboardViewportOpen()) {
        keyboardOpen = true;
        clearTimeout(stabilizeTimer);
        return;
      }
      if (!keyboardOpen) {
        scheduleBaselineRefresh();
        return;
      }
      clearTimeout(stabilizeTimer);
      stabilizeTimer = window.setTimeout(() => {
        keyboardOpen = false;
        scheduleBaselineRefresh();
      }, 160);
    };
    viewport.addEventListener("resize", onViewportChange, { passive: true });
  }

  // r3.4s：所有输入框只唤起键盘、不滚动页面。
  // 1) 记录按下时的滚动位置；2) 拦截默认聚焦并手动 focus({preventScroll:true})；
  // 3) 若浏览器仍因聚焦/键盘调整了滚动，随后恢复回按下前位置。
  let lastPointerScrollY = null;
  let suppressFocusRestore = false;
  const focusableSelector = 'input, textarea, [contenteditable="true"]';

  const restoreFocusScroll = (targetScrollY) => {
    if (targetScrollY === null) return;
    requestAnimationFrame(() => {
      if (Math.abs(window.scrollY - targetScrollY) > 1) {
        window.scrollTo(0, targetScrollY);
      }
    });
  };

  document.addEventListener(
    "pointerdown",
    (event) => {
      lastPointerScrollY = Math.max(0, Number(window.scrollY) || 0);
      if (event.pointerType === "mouse") return;
      const target =
        event.target instanceof Element
          ? event.target.closest(focusableSelector)
          : null;
      if (!target || target.disabled) return;
      if (target.tagName !== "INPUT" && target.tagName !== "TEXTAREA") return;
      if (target.readOnly) return;
      if (document.activeElement === target) return; // 已聚焦：保留默认光标定位
      event.preventDefault();
      suppressFocusRestore = true;
      try {
        target.focus({ preventScroll: true });
      } catch (error) {
        target.focus();
      }
      suppressFocusRestore = false;
      if (typeof target.setSelectionRange === "function") {
        const length = String(target.value || "").length;
        try {
          target.setSelectionRange(length, length);
        } catch (error) {
          /* 某些 input 类型不支持选区，忽略 */
        }
      }
      restoreFocusScroll(lastPointerScrollY);
    },
    { capture: true, passive: false }
  );

  // 记录键盘状态与诊断历史（不做滚动位置恢复/视图纠正）。
  document.addEventListener(
    "focusin",
    (event) => {
      const target = event.target;
      if (!target) return;
      const tag = String(target.tagName || "").toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable) {
        keyboardOpen = true;
        recordUiSample("focusin");
        // 兜底：用键盘/无障碍等方式聚焦时，若页面被浏览器滚动了，恢复按下前位置。
        if (!suppressFocusRestore && lastPointerScrollY !== null) {
          restoreFocusScroll(lastPointerScrollY);
          lastPointerScrollY = null;
        }
      }
    },
    true
  );
  document.addEventListener(
    "focusout",
    (event) => {
      const target = event.target;
      if (!target) return;
      const tag = String(target.tagName || "").toUpperCase();
      if (tag !== "INPUT" && tag !== "TEXTAREA") return;
      keyboardOpen = false;
      recordUiSample("focusout");
    },
    true
  );

  // 诊断历史：定期采样 + 关键事件采样（聚焦/失焦/滚动/resize）。
  startUiHistorySampler();
  window.addEventListener("scroll", () => scheduleUiSample("scroll"), {
    passive: true,
  });
  window.addEventListener("resize", () => scheduleUiSample("resize"), {
    passive: true,
  });

  // 键盘基线：初始、真实 resize（含旋转/地址栏变化）后重算。
  scheduleBaselineRefresh();
  window.addEventListener(
    "resize",
    () => {
      scheduleScrolledClass();
      scheduleBaselineRefresh();
    },
    { passive: true }
  );

  window.addEventListener("orientationchange", () => {
    keyboardBaselineHeight = 0;
    window.setTimeout(() => {
      scheduleBaselineRefresh();
      refreshTopbarHeight();
    }, 300);
  });
}
const FEEDBACK_SELECTOR = [
  "#save-group",
  "#save-global",
  "#reset-group",
  "#reset-global",
  "#overview-refresh",
  "#groups-refresh",
  "#rag-search",
  "#rag-prev",
  "#rag-next",
  "#auto-memory-toggle",
  "#block-add",
  "#group-block-add",
  "#modal-ok",
  "[data-delete]",
  "[data-archive]",
  "[data-unarchive]",
  "[data-purge]",
  "[data-block-remove]",
  "[data-feedback]",
].join(",");

let uiDiagTimer = 0;
const uiHistory = [];
let uiHistoryTimer = 0;

/** 记录一条视口历史（诊断用，保留最近 20 秒）。 */
function recordUiSample(reason = "") {
  if (document.hidden) return;
  const vv = window.visualViewport;
  const active = document.activeElement;
  uiHistory.push({
    t: Date.now(),
    y: Math.round(window.scrollY || 0),
    inner: Math.round(window.innerHeight || 0),
    vh: vv ? Math.round(vv.height) : 0,
    top: vv ? Math.round(vv.offsetTop) : 0,
    pageTop: vv ? Math.round(vv.pageTop) : 0,
    active: active
      ? `${active.tagName}${active.id ? `#${active.id}` : ""}`
      : "body",
    reason,
  });
  if (uiHistory.length > 80) uiHistory.shift();
}

let uiSampleRaf = 0;

function scheduleUiSample(reason = "") {
  if (uiSampleRaf) return;
  uiSampleRaf = requestAnimationFrame(() => {
    uiSampleRaf = 0;
    recordUiSample(reason);
  });
}

function startUiHistorySampler() {
  if (uiHistoryTimer) return;
  // 诊断采样只需要低频趋势；400ms 会持续读取布局属性，提升到 1s 降低开销。
  uiHistoryTimer = window.setInterval(() => recordUiSample("tick"), 1000);
}

function formatUiHistory() {
  const now = Date.now();
  const rows = uiHistory.slice(-10).map((item) => {
    const dt = ((item.t - now) / 1000).toFixed(1);
    const reason = item.reason && item.reason !== "tick" ? ` ${item.reason}` : "";
    return `t${dt}s y=${item.y} vh=${item.vh}/${item.inner} top=${item.top} pt=${item.pageTop} ${item.active}${reason}`;
  });
  return rows.length ? rows.join("\n") : "（无历史）";
}

/** 采集当前视口/滚动/iframe/键盘状态，用于定位点击偏移根因（r3.3x）。 */
function collectUiDiagnostics() {
  const lines = [];
  const vv = window.visualViewport;
  const focused = document.activeElement;
  const frame = (() => {
    try {
      return window.frameElement;
    } catch (error) {
      return null;
    }
  })();
  const parentInfo = (() => {
    try {
      if (!window.parent || window.parent === window) return "无（顶层页面）";
      const ph = window.parent.innerHeight;
      const pv = window.parent.visualViewport;
      return `parent.innerHeight=${ph} parent.vv=${pv ? Math.round(pv.height) : "-"}`;
    } catch (error) {
      return "跨域不可读";
    }
  })();
  lines.push(`viewport = ${window.innerWidth} × ${window.innerHeight} @${window.devicePixelRatio}x`);
  lines.push(
    `visualViewport = ${vv ? `${Math.round(vv.width)} × ${Math.round(vv.height)}` : "不支持"}`
  );
  if (vv) {
    lines.push(
      `  offsetTop=${Math.round(vv.offsetTop)} pageTop=${Math.round(vv.pageTop)} scale=${vv.scale.toFixed(3)}`
    );
  }
  lines.push(`scrollY=${Math.round(window.scrollY)} scrollX=${Math.round(window.scrollX)}`);
  lines.push(
    `docEl.clientHeight=${document.documentElement.clientHeight} body.scrollHeight=${document.body.scrollHeight}`
  );
  lines.push(`iframe = ${parentInfo}`);
  if (frame) {
    try {
      const rect = frame.getBoundingClientRect();
      lines.push(
        `frameRect = x${Math.round(rect.left)} y${Math.round(rect.top)} ${Math.round(rect.width)} × ${Math.round(rect.height)}`
      );
    } catch (error) {
      lines.push("frameRect = 不可读");
    }
  }
  lines.push(
    `activeElement = ${focused ? `${focused.tagName}${focused.id ? `#${focused.id}` : ""}` : "body"}`
  );
  lines.push(`isKeyboardViewportOpen = ${isKeyboardViewportOpen()}`);
  lines.push(`keyboardBaseline = ${keyboardBaselineHeight}`);
  lines.push(`body.class = ${Array.from(document.body.classList).join(" ") || "-"}`);
  const drawer = $("drawer");
  const detail = $("detail");
  const scrim = $("scrim");
  lines.push(
    `drawer=${drawer?.classList.contains("open")} detail=${detail?.classList.contains("open")} scrim=${scrim?.classList.contains("show")}`
  );
  lines.push(
    `pane.is-pushed=${$("main-pane")?.classList.contains("is-pushed")} panel-closing=${document.body.classList.contains("panel-closing")}`
  );
  const topbar = getComputedStyle(document.documentElement).getPropertyValue("--topbar-h");
  const tabsFade = getComputedStyle(document.documentElement).getPropertyValue("--tabs-fade-h");
  lines.push(`--topbar-h=${topbar || "-"} --tabs-fade-h=${tabsFade || "-"}`);
  lines.push("");
  lines.push("最近视口历史（y=scrollY vh/inner=视口高 top/pageTop=visualViewport偏移）:");
  lines.push(formatUiHistory());
  return lines.join("\n");
}

function renderUiDiagnostics() {
  const pre = $("ui-diag-text");
  if (pre) pre.textContent = collectUiDiagnostics();
}

function showUiDiagnostics() {
  const modal = $("ui-diag");
  if (!modal) return;
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
  renderUiDiagnostics();
  clearInterval(uiDiagTimer);
  uiDiagTimer = window.setInterval(renderUiDiagnostics, 500);
}

function hideUiDiagnostics() {
  clearInterval(uiDiagTimer);
  uiDiagTimer = 0;
  const modal = $("ui-diag");
  if (!modal) return;
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
}

let menuRecoverTriggered = false;
let menuLongPressTimer = 0;
let menuLongPressStart = null;
let uiWatchdogTimer = 0;

/** 长按菜单胶囊触发界面自愈；点击事件会被吞掉，避免同时开关抽屉。 */
function bindMenuLongPressRecovery() {
  const button = $("menu-btn");
  if (!button) return;
  const cancel = () => {
    clearTimeout(menuLongPressTimer);
    menuLongPressTimer = 0;
    menuLongPressStart = null;
  };
  button.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    menuLongPressStart = { x: event.clientX, y: event.clientY };
    clearTimeout(menuLongPressTimer);
    menuLongPressTimer = window.setTimeout(() => {
      if (!menuLongPressStart) return;
      menuRecoverTriggered = true;
      recoverUiState("long-press");
      toast("界面已恢复");
    }, 800);
  });
  button.addEventListener("pointermove", (event) => {
    if (!menuLongPressStart) return;
    if (
      Math.abs(event.clientX - menuLongPressStart.x) > 12 ||
      Math.abs(event.clientY - menuLongPressStart.y) > 12
    ) {
      cancel();
    }
  });
  button.addEventListener("pointerup", cancel);
  button.addEventListener("pointercancel", cancel);
  button.addEventListener("pointerleave", cancel);
  // 阻止 Android 长按弹出的文本选择菜单，确保长按手势归我们处理。
  button.addEventListener("contextmenu", (event) => event.preventDefault());
  button.addEventListener("selectstart", (event) => event.preventDefault());
}

/**
 * 界面自愈（r3.3r）：清理卡死时残留的遮罩 / body 状态 / 面板变换，
 * 以及按住不放的按钮锁。每 3 秒巡检一次，页面重新可见时也执行。
 */
function recoverUiState(reason = "") {
  // 手势拖动中（边缘推开）不要清理内联变换，避免下一次 pointermove 前跳变。
  if (typeof dragState !== "undefined" && dragState) return;
  // 卡死的按钮锁：没有请求在途时全部释放（输入聚焦时也允许释放按钮）。
  if (apiInFlight === 0 && lockedButtons.size) {
    Array.from(lockedButtons.keys()).forEach((button) => unlockButton(button));
  }
  // r3.3w：输入框聚焦（软键盘可能展开）或数字键盘弹窗打开时不要改布局，
  // 否则会和 WebView 的视口重算互相打架，造成点击偏移。
  if (editingRangeValue) return;
  const focused = document.activeElement;
  if (focused && focused !== document.body) {
    const focusedTag = String(focused.tagName || "").toUpperCase();
    if (focusedTag === "INPUT" || focusedTag === "TEXTAREA") return;
  }
  const drawer = $("drawer");
  const detail = $("detail");
  const scrim = $("scrim");
  const pane = $("main-pane");
  if (!drawer || !detail || !scrim || !pane) return;
  const drawerOpen = drawer.classList.contains("open");
  const detailOpen = detail.classList.contains("open");
  if (drawerOpen && detailOpen) {
    // 手机端两个面板同时打开是非法状态；PC 端保留侧栏，由汉堡按钮控制。
    if (!isDesktopLayout()) {
      closeDrawer();
      return;
    }
  }
  if (!drawerOpen) {
    document.body.classList.remove("drawer-open");
    drawer.setAttribute("aria-hidden", "true");
  }
  if (!detailOpen) {
    document.body.classList.remove("detail-open");
    detail.setAttribute("aria-hidden", "true");
  }
  if (!drawerOpen && !detailOpen) {
    scrim.classList.remove("show");
    pane.classList.remove("is-pushed");
    pane.style.transition = "";
    pane.style.transform = "";
    pane.style.borderRadius = "";
    pane.style.boxShadow = "";
    // 拖拽会给 drawer 写 inline transform/transition；自愈时一并清掉，
    // 否则关闭后抽屉仍可能停在半推开位置。
    drawer.style.transition = "";
    drawer.style.transform = "";
    detail.style.transition = "";
    detail.style.transform = "";
    const veil = document.querySelector(".pane-veil");
    if (veil) {
      veil.style.transition = "";
      veil.style.opacity = "";
    }
    clearFloatingControlsInline();
    setContentScrollOffset(0);
    document.body.classList.remove("panel-closing");
    if (!$("modal").classList.contains("open")) {
      $("menu-btn").setAttribute("aria-expanded", "false");
    }
  }
  if (reason) {
    console.debug("[ActiveReply] 界面自愈", reason);
  }
}

/** 启动界面巡检：正常时几乎零开销（只读 class 状态）。 */
function startUiWatchdog() {
  if (uiWatchdogTimer) return;
  uiWatchdogTimer = window.setInterval(() => {
    try {
      recoverUiState("watchdog");
    } catch (error) {
      console.warn("[ActiveReply] 界面自愈失败", error);
    }
  }, 3000);
}

function installButtonFeedback() {
  document.addEventListener(
    "click",
    (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      if (button.classList.contains("no-lock")) return;
      if (!button.matches(FEEDBACK_SELECTOR)) return;
      if (button.classList.contains("btn-loading") || button.dataset.locking === "1") {
        // 上一次请求尚未结束：在捕获阶段拦下重复提交，避免重复写配置/删记忆。
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
      if (button.disabled) return;
      lockButton(button);
    },
    true
  );
}

/** 给每个主页内容套一层宽度视口，侧栏开合时只动画外层宽度。 */
function wrapContentViews() {
  document.querySelectorAll("#main-pane > .content").forEach((content) => {
    if (content.parentElement?.classList.contains("content-viewport")) return;
    const viewport = document.createElement("div");
    viewport.className = "content-viewport";
    content.parentNode.insertBefore(viewport, content);
    viewport.appendChild(content);
  });
}

function bindEvents() {
  $("menu-btn").addEventListener("click", () => {
    if (menuRecoverTriggered) {
      menuRecoverTriggered = false;
      return;
    }
    if ($("detail").classList.contains("open")) {
      closeDetail();
      return;
    }
    const opened = $("drawer").classList.contains("open");
    opened ? closeDrawer() : openDrawer();
  });
  state.menuBound = true;
  // r3.3r：长按左上角胶囊 800ms = 强制恢复界面（遮罩/面板/按钮锁卡死时的逃生口）。
  bindMenuLongPressRecovery();
  bindPushGesture();
  bindCloseGesture();
  window.addEventListener("scroll", scheduleScrolledClass, { passive: true });
  // PC 端主页面自身是滚动容器，必须监听它的 scroll 才能切换 Tab 浮标。
  $("main-pane")?.addEventListener("scroll", scheduleScrolledClass, {
    passive: true,
  });
  // resize 处理统一放在 installViewportRecovery（含键盘基线重算）。
  updateScrolledClass();
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      // PC 端侧栏只允许通过汉堡按钮开关。
      if (!isDesktopLayout()) closeDrawer();
      closeDetail();
    }
  });

  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => switchView(item.dataset.view));
  });

  $("drawer-manage-groups").addEventListener("click", () => {
    state.currentGroupId = "";
    switchView("groups");
  });
  $("drawer-settings").addEventListener("click", () => switchView("settings"));

  $("overview-refresh").addEventListener("click", () => void loadOverview(true));
  $("groups-refresh").addEventListener("click", () => void loadGroups(true));
  $("group-search").addEventListener("input", (event) => {
    state.groupSearch = event.target.value;
    renderGroupList();
  });

  $("manual-group-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const groupId = $("manual-group-id").value.trim();
    if (!/^\d{4,20}$/.test(groupId)) {
      toast("请输入合法的群号（纯数字）", true);
      return;
    }
    if (!state.groups.some((item) => item.group_id === groupId)) {
      state.groups.push({
        group_id: groupId,
        name: "",
        independent: false,
        active_reply: false,
      });
    }
    renderGroupList();
    void openGroup(groupId);
  });

  // 未开启独立配置时，群面板里的按钮（分区 Tab / 保存 / 恢复默认）点击都会抖动提示
  $("group-panel").addEventListener(
    "click",
    (event) => {
      if (!state.currentGroup) return;
      if ($("independent-toggle").checked) return;
      const button = event.target.closest("button");
      if (!button) return;
      // “用户屏蔽”不是群配置覆盖项，不受「先开独立配置」限制
      if (
        button.closest('[data-panel="user_block"]') ||
        (button.classList.contains("tab") && button.dataset.tab === "user_block")
      ) {
        return;
      }
      shakeIndependent();
      if (!button.classList.contains("tab")) {
        // 保存 / 恢复默认：直接拦下，避免重复提示
        event.preventDefault();
        event.stopPropagation();
      }
    },
    true
  );

  $("group-tabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (tab) {
      switchTab(tab.dataset.tab);
      playTabSwitch($("group-form"));
      centerTab(tab);
    }
  });

  $("independent-toggle").addEventListener("change", (event) => {
    const data = state.currentGroup;
    if (!data) return;

    if (!event.target.checked) {
      // 关闭独立配置 = 删除该群配置、回到跟随全局；先还原勾选，等用户确认
      event.target.checked = true;
      void followGlobal(data.group_id);
      return;
    }

    data.independent = true;
    $("save-hint").textContent = "当前为本群独立配置，会覆盖全局对应项";
    syncGroupTopActions();
  });

  $("save-group").addEventListener("click", () => void saveGroup());
  $("reset-group").addEventListener("click", () => void resetGroup());

  $("group-mini-independent")?.addEventListener("click", () => {
    const data = state.currentGroup;
    const toggle = $("independent-toggle");
    if (!data || !toggle) return;
    if (toggle.checked) {
      void followGlobal(data.group_id);
      return;
    }
    toggle.checked = true;
    data.independent = true;
    $("save-hint").textContent = "当前为本群独立配置，会覆盖全局对应项";
    syncGroupTopActions();
  });
  $("group-mini-reset")?.addEventListener("click", resetGroup);
  $("group-mini-save")?.addEventListener("click", () => void saveGroup());

  // 全局设置：兜底用 document 级事件委托，避免个别环境里逐元素绑定失效
  document.addEventListener(
    "change",
    (event) => {
      const target = event.target;
      if (!target || !target.closest) return;
      if (target.closest("#global-form")) {
        if (target.matches("select, input[type=checkbox]")) {
          rerenderActiveGlobalPanel();
        } else {
          refreshGlobalConditions();
        }
        return;
      }
      if (target.closest("#group-form")) {
        refreshConditionalRows();
      }
    },
    true
  );

  $("global-tabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (tab) {
      switchGlobalTab(tab.dataset.gtab);
      playTabSwitch($("global-form"));
      centerTab(tab);
    }
  });

  $("save-global").addEventListener("click", () => void saveGlobal());
  $("reset-global").addEventListener("click", () => resetGlobalDefaults());

  $("rag-search").addEventListener("click", () => openFilterModal());
  $("filter-apply").addEventListener("click", () => {
    state.rag.memoryType = $("filter-type").value;
    state.rag.kind = $("filter-kind")?.value || "";
    state.rag.needsReview = Boolean($("filter-review")?.checked);
    state.rag.scope = $("filter-scope").value;
    state.rag.identity = $("filter-identity").value;
    state.rag.lifecycle = $("filter-lifecycle")?.value || "";
    state.rag.keyword = $("rag-keyword").value.trim();
    state.rag.role = $("rag-role").value.trim();
    state.rag.page = 1;
    closeFilterModal();
    void loadRagList();
  });
  $("filter-cancel").addEventListener("click", closeFilterModal);
  // r3.3g：筛选弹窗新增「清空」；清空筛选条件但保留「已归档」开关状态。
  $("filter-clear").addEventListener("click", () => {
    $("filter-type").value = "";
    if ($("filter-kind")) $("filter-kind").value = "";
    if ($("filter-review")) $("filter-review").checked = false;
    $("filter-scope").value = "";
    $("filter-identity").value = "";
    if ($("filter-lifecycle")) $("filter-lifecycle").value = "";
    $("rag-keyword").value = "";
    $("rag-role").value = "";
    state.rag.memoryType = "";
    state.rag.kind = "";
    state.rag.needsReview = false;
    state.rag.scope = "";
    state.rag.identity = "";
    state.rag.lifecycle = "";
    state.rag.keyword = "";
    state.rag.role = "";
    state.rag.page = 1;
    closeFilterModal();
    syncRagFilterButton();
    void loadRagList();
  });
  $("snackbar-undo").addEventListener("click", () => void undoDelete());
  // r3.3：关键词/角色输入框右侧的快捷搜索按钮与两输入框回车，
  // 都按当前输入重查第一页；筛选弹窗仍由「筛选」按钮打开。
  $("rag-quick-search")?.addEventListener("click", quickSearchRag);
  ["rag-keyword", "rag-role"].forEach((id) => {
    const input = $(id);
    if (!input) return;
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      quickSearchRag();
    });
  });
  // r3.3：「已归档」开关切换为只显示归档记录（请求参数 archived_only=1）。
  $("rag-archived-btn")?.addEventListener("click", toggleRagArchived);
  $("rag-select-btn")?.addEventListener("click", () => toggleRagSelectMode());
  $("rag-bulk-cancel")?.addEventListener("click", () => toggleRagSelectMode(false));
  $("rag-bulk-archive")?.addEventListener("click", () => void bulkArchiveSelected());
  $("rag-bulk-delete")?.addEventListener("click", () => void bulkDeleteSelected());
  $("theme-btn").addEventListener("click", () => cycleTheme());
  $("block-add").addEventListener("click", () => void onBlockAddGlobal());
  $("block-search").addEventListener("input", (event) => {
    state.blockState.global.search = event.target.value;
    if (event.target.value.trim()) state.blockState.global.expanded = true;
    renderGlobalBlocks();
  });
  $("block-toggle").addEventListener("click", () => {
    state.blockState.global.expanded = !state.blockState.global.expanded;
    renderGlobalBlocks();
  });
  document.addEventListener("click", (event) => {
    const rangeValue = event.target.closest(".range-value");
    if (rangeValue) {
      startRangeValueEdit(rangeValue);
      return;
    }
    const button = event.target.closest("[data-block-remove]");
    if (button) void removeBlockEntry(button);
  });
  $("auto-memory-toggle").addEventListener("change", () => void toggleAutoMemory());
  $("auto-memory-settings").addEventListener("click", () => {
    state.globalTab = "memory_rag";
    switchView("settings");
  });
  $("rag-prev").addEventListener("click", () => {
    if (state.rag.page <= 1) return;
    state.rag.page -= 1;
    void loadRagList();
  });
  $("rag-next").addEventListener("click", () => {
    if (!state.rag.hasMore) return;
    state.rag.page += 1;
    void loadRagList();
  });
  $("rag-page-size").addEventListener("change", (event) => {
    state.rag.pageSize = Number(event.target.value) || 20;
    state.rag.page = 1;
    void loadRagList();
  });
  // 开发者选项（r3.3）：侧栏扳手按钮进入，密钥校验通过后才展示提示词/自检。
  $("drawer-dev")?.addEventListener("click", () => switchView("dev"));
  $("dev-memory-repair")?.addEventListener("click", () => void repairMemoryKinds());
  $("dev-memory-repair-settings")?.addEventListener("click", () => void openRepairSettings());
  $("repair-settings-cancel")?.addEventListener("click", closeRepairSettings);
  $("repair-settings-save")?.addEventListener("click", () => void saveRepairSettings());
  $("repair-confidence")?.addEventListener("input", () => {
    if ($("repair-confidence-value")) {
      $("repair-confidence-value").textContent = Number(
        $("repair-confidence").value || 0.8
      ).toFixed(2);
    }
  });
  $("dev-tour-replay")?.addEventListener("click", () => startTour(true, "beta.6"));
  $("dev-tour-replay-beta5")?.addEventListener("click", () => startTour(true, "beta.5"));
  $("tour-skip")?.addEventListener("click", () => closeTour(true));
  $("tour-next")?.addEventListener("click", () => {
    if (tourStepIndex + 1 < TOUR_STEPS.length) {
      void renderTourStep(tourStepIndex + 1);
    } else {
      closeTour(true);
    }
  });
  $("tour-hit")?.addEventListener("click", () => {
    const step = TOUR_STEPS[tourStepIndex];
    const target = step ? document.querySelector(step.selector) : null;
    if (!target) return;
    const last = tourStepIndex + 1 >= TOUR_STEPS.length;
    if (last) closeTour(true);
    target.click();
    if (!last) {
      window.setTimeout(() => {
        if ($("tour-root") && !$("tour-root").hidden) {
          void renderTourStep(tourStepIndex + 1);
        }
      }, 120);
    }
  });
  $("dev-login")?.addEventListener("click", () => void devLogin());
  $("dev-key")?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    void devLogin();
  });
  $("dev-prompt")?.addEventListener("click", () => void devPromptPreview());
  $("dev-selftest")?.addEventListener("click", (event) =>
    void devSelftest(Boolean(event.shiftKey))
  );
  $("dev-logout")?.addEventListener("click", () => void devLogout());
  $("dev-prompt-save")?.addEventListener("click", () => void saveDevPrompts());
  // 开发者选项保留「界面诊断」入口；注意它会切页，抓现场请用 Tab 连点 5 次。
  $("dev-diag")?.addEventListener("click", () => showUiDiagnostics());
  $("ui-diag-close")?.addEventListener("click", hideUiDiagnostics);
  $("ui-diag-recover")?.addEventListener("click", () => {
    hideUiDiagnostics();
    recoverUiState("diagnostics");
  });
  installButtonFeedback();
}

/* ------------------------------------------------------------------ */
/* 启动                                                                */
/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* 漫游式引导                                                          */
/* ------------------------------------------------------------------ */
const TOUR_VERSION = "beta.6";
let activeTourVersion = TOUR_VERSION;
let TOUR_SEEN_KEY = `activeReply.tour.${activeTourVersion}`;
let tourStepIndex = 0;

// 历史版本的引导步骤保留在代码里，服务端 tour_seen 也按版本分别记录；
// 升级只切换当前 TOUR_VERSION，不覆盖旧版本引导数据。
const TOUR_STEPS_BETA5 = [
  {
    selector: "#menu-btn",
    title: "从侧边栏开始",
    text: "先打开侧边栏，开发者选项的入口在这里。",
    before: () => closeDrawer(),
  },
  {
    selector: "#drawer-dev",
    title: "开发者选项",
    text: "侧边栏底部的扳手按钮就是开发者选项；记忆修复不需要开发者密钥。",
    before: () => openDrawer(),
  },
  {
    selector: "#dev-memory-repair",
    title: "记忆修复",
    text: "在这里扫描并修复被误标为「事实」的事件、计划、偏好和关系，以及群主/群管身份，先预览再确认，不会直接改库。",
    before: () => switchView("dev"),
  },
];

/** 等记忆列表的首条真实记录（排除骨架）；空库超时返回 null。 */
async function waitForTourMemoryRow(timeoutMs = 3000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const row = document.querySelector(
      "#rag-body tr.mem-row:not(.skeleton-mem-row)"
    );
    if (row) return row;
    await new Promise((resolve) => setTimeout(resolve, 120));
  }
  return null;
}

const TOUR_STEPS_BETA6 = [
  {
    selector: "#rag-body",
    title: "待复核记录",
    text: "带「待复核」的记忆先在这里找。顶部「筛选」里还有「只看待复核」开关，可只显示需要人工确认的记录。",
    before: () => {
      closeDrawer();
      closeDetail();
      closeFilterModal();
      switchView("rag");
    },
  },
  {
    selector: "#rag-body tr.mem-row:not(.skeleton-mem-row)",
    title: "展开记忆",
    text: "点这条记忆（或行尾箭头）展开记录。",
    before: async () => {
      closeDrawer();
      closeDetail();
      closeFilterModal();
      switchView("rag");
      const row = await waitForTourMemoryRow();
      if (!row) {
        toast("暂无可复核记忆，已跳过本次引导", true);
        closeTour(true);
        return;
      }
      const detail = document.querySelector(
        `[data-mem-detail="${CSS.escape(row.dataset.memRow)}"]`
      );
      if (detail?.classList.contains("open")) row.click();
    },
  },
  {
    selector: "#rag-body [data-detail]",
    title: "打开记忆详情",
    text: "展开后点眼睛图标，进入「记忆详情」。",
    before: async () => {
      closeDrawer();
      closeDetail();
      switchView("rag");
      // switchView 会触发列表重绘；等重绘完成再展开，否则 click 命中的
      // 是即将被替换的旧行，下一步眼睛图标仍然不可见。
      const row = await waitForTourMemoryRow();
      if (!row) {
        toast("暂无可复核记忆，已跳过本次引导", true);
        closeTour(true);
        return;
      }
      const detail = document.querySelector(
        `[data-mem-detail="${CSS.escape(row.dataset.memRow)}"]`
      );
      if (detail && !detail.classList.contains("open")) row.click();
      await new Promise((resolve) => setTimeout(resolve, 260));
    },
  },
  {
    selector: "#detail-kind-select",
    title: "分类复核",
    text: "详情底部的「分类复核」：选类型 → 保存类型；确认无误后点「标记已复核」。",
    before: async () => {
      closeDrawer();
      closeFilterModal();
      switchView("rag");
      const button = document.querySelector("#rag-body [data-detail]");
      if (button) await showMemory(button.dataset.detail);
      await new Promise((resolve) => setTimeout(resolve, 120));
    },
  },
];

const TOURS = {
  "beta.5": TOUR_STEPS_BETA5,
  "beta.6": TOUR_STEPS_BETA6,
};
let TOUR_STEPS = TOURS[activeTourVersion];

let tourSeenOnServer = false;

function tourSeen() {
  try {
    if (localStorage.getItem(TOUR_SEEN_KEY) === "1") return true;
  } catch {
    // fall through to server-side state
  }
  return tourSeenOnServer;
}

async function loadTourSeen() {
  if (tourSeen()) return true;
  try {
    const data = await callApi("GET", "tour/state", { version: TOUR_VERSION });
    tourSeenOnServer = Boolean(data && data.seen);
  } catch {
    tourSeenOnServer = false;
  }
  if (tourSeenOnServer) {
    try {
      localStorage.setItem(TOUR_SEEN_KEY, "1");
    } catch {
      // 服务端已经记住，localStorage 不可用不影响结果。
    }
  }
  return tourSeenOnServer;
}

function markTourSeen() {
  tourSeenOnServer = true;
  try {
    localStorage.setItem(TOUR_SEEN_KEY, "1");
  } catch {
    // 服务端文件才是跨重进插件的真正持久化来源。
  }
  void callApi("POST", "tour/seen", { version: activeTourVersion }).catch(
    () => {}
  );
}

function positionTourPopover(target, animate = false) {
  const pop = $("tour-pop");
  if (!pop || !target) return;
  const spotNode = $("tour-spot");
  if (spotNode) spotNode.style.transition = animate ? "" : "none";
  pop.style.transition = animate ? "" : "none";
  const rect = target.getBoundingClientRect();
  const pad = 6;
  const spot = $("tour-spot");
  if (spot) {
    spot.style.top = `${Math.max(4, rect.top - pad)}px`;
    spot.style.left = `${Math.max(4, rect.left - pad)}px`;
    spot.style.width = `${Math.min(window.innerWidth - 8, rect.width + pad * 2)}px`;
    spot.style.height = `${Math.min(window.innerHeight - 8, rect.height + pad * 2)}px`;
  }
  const hit = $("tour-hit");
  if (hit) {
    hit.style.top = `${Math.max(4, rect.top - pad)}px`;
    hit.style.left = `${Math.max(4, rect.left - pad)}px`;
    hit.style.width = `${Math.min(window.innerWidth - 8, rect.width + pad * 2)}px`;
    hit.style.height = `${Math.min(window.innerHeight - 8, rect.height + pad * 2)}px`;
  }
  const popRect = pop.getBoundingClientRect();
  const maxLeft = Math.max(12, window.innerWidth - popRect.width - 12);
  const left = Math.min(maxLeft, Math.max(12, rect.left + rect.width / 2 - popRect.width / 2));
  let top = rect.bottom + 12;
  if (top + popRect.height > window.innerHeight - 12) {
    top = Math.max(12, rect.top - popRect.height - 12);
  }
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
}

function closeTour(markSeen = true) {
  const root = $("tour-root");
  if (root) {
    root.hidden = true;
    root.setAttribute("aria-hidden", "true");
  }
  if (markSeen) markTourSeen();
}

async function renderTourStep(index) {
  const root = $("tour-root");
  if (!root || root.hidden) return;
  const step = TOUR_STEPS[index];
  if (!step) {
    closeTour(true);
    return;
  }
  tourStepIndex = index;
  try {
    await step.before?.();
  } catch (error) {
    console.warn("[ActiveReply] 引导步骤切换失败", error);
  }
  await new Promise((resolve) => setTimeout(resolve, 320));
  const target = document.querySelector(step.selector);
  if (!target) {
    if (index + 1 < TOUR_STEPS.length) void renderTourStep(index + 1);
    else closeTour(true);
    return;
  }
  // 引导定位必须等页面/抽屉滚动完成；平滑滚动会让 rect 在动画中过期，
  // 导致聚光灯落在按钮原来的位置。
  target.scrollIntoView({ block: "center", behavior: "auto" });
  await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  await new Promise((resolve) => setTimeout(resolve, 60));
  $("tour-step").textContent = `${index + 1} / ${TOUR_STEPS.length}`;
  $("tour-title").textContent = step.title;
  $("tour-text").textContent = step.text;
  $("tour-next").textContent = index + 1 < TOUR_STEPS.length ? "下一步" : "完成";
  root.hidden = false;
  root.setAttribute("aria-hidden", "false");
  positionTourPopover(target, true);
}

function startTour(force = false, version = TOUR_VERSION) {
  const steps = TOURS[version] || TOURS[TOUR_VERSION];
  if (!steps) return;
  activeTourVersion = version;
  TOUR_STEPS = steps;
  TOUR_SEEN_KEY = `activeReply.tour.${version}`;
  if (!force && tourSeen()) return;
  const root = $("tour-root");
  if (!root) return;
  markTourSeen();
  tourStepIndex = 0;
  root.hidden = false;
  root.setAttribute("aria-hidden", "false");
  void renderTourStep(0);
}

async function maybeStartTour() {
  if (await loadTourSeen()) return;
  window.setTimeout(() => startTour(false), 700);
}

let tourRepositionRaf = 0;

function repositionTour() {
  const root = $("tour-root");
  if (!root || root.hidden) return;
  const step = TOUR_STEPS[tourStepIndex];
  const target = step ? document.querySelector(step.selector) : null;
  if (target) positionTourPopover(target, false);
}

function scheduleTourReposition() {
  if (tourRepositionRaf) return;
  tourRepositionRaf = requestAnimationFrame(() => {
    tourRepositionRaf = 0;
    repositionTour();
  });
}
window.addEventListener("scroll", scheduleTourReposition, {
  passive: true,
  capture: true,
});
window.addEventListener("resize", scheduleTourReposition, { passive: true });

/** 等待 AstrBot bridge、注册主题变化监听，再进入状态概览页。 */
async function init() {
  refreshTopbarHeight();
  try {
    wrapContentViews();
    bindEvents();
  } catch (error) {
    // r3.3r：单个绑定失败不能让整页事件全部失效。
    console.error("[ActiveReply] 事件绑定失败，保留已绑定部分", error);
    bindMenuLongPressRecovery();
    if (!state.menuBound) {
      state.menuBound = true;
      $("menu-btn")?.addEventListener("click", () => {
        const opened = $("drawer")?.classList.contains("open");
        opened ? closeDrawer() : openDrawer();
      });
    }
    $("scrim")?.addEventListener("click", () => {
      if (isDesktopLayout()) return;
      if ($("detail")?.classList.contains("open")) closeDetail();
      else closeDrawer();
    });
    $("theme-btn")?.addEventListener("click", () => cycleTheme());
  }
  installViewportRecovery();
  startUiWatchdog();
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) recoverUiState("visible");
  });
  window.addEventListener("pageshow", () => recoverUiState("pageshow"));
  state.themeMode = loadThemeMode();
  state.systemDark =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  void initMorphIcons();
  applyThemeMode(state.themeMode);
  $("drawer-version").textContent = `UI${UI_BUILD}`;
  if (!bridge) {
    document.body.insertAdjacentHTML(
      "afterbegin",
      '<div style="padding:14px 20px;background:#fdeceb;color:#b42318;font-size:13px;">' +
        "未检测到 AstrBot 插件视图环境，请从 AstrBot 面板的插件详情页打开本页面</div>"
    );
    return;
  }
  try {
    const context = await bridge.ready();
    if (context) {
      state.systemDark = Boolean(context.isDark);
      applyThemeMode(state.themeMode);
      document.title = context.pageTitle || "主动回复 · 控制台";
    }
    if (typeof bridge.onContext === "function") {
      bridge.onContext((next) => {
        if (next) {
          state.systemDark = Boolean(next.isDark);
          applyThemeMode(state.themeMode);
        }
      });
    }
  } catch (error) {
    console.warn("bridge.ready failed", error);
  }
  switchView("overview");
  maybeStartTour();
}

void init();
