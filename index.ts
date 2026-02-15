import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const MAX_FILE_BYTES = 128 * 1024;
const MAX_PATCH_CHARS = 4096;
const PROGRESS_HEARTBEAT_MS = 1500;
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const LIBRARIAN_SUBAGENT_TOOLS = [
  "read_github",
  "search_github",
  "commit_search",
  "diff",
  "list_directory_github",
  "list_repositories",
  "glob_github",
];

type LibrarianPhase = "booting" | "exploring" | "writing";

type LibrarianProgressState = {
  startedAt: number;
  phase: LibrarianPhase;
  startedTools: number;
  completedTools: number;
  failedTools: number;
  currentAction?: string;
  recentActions: string[];
};

function formatDuration(ms: number): string {
  const sec = Math.max(0, Math.floor(ms / 1000));
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  const rem = sec % 60;
  return `${min}m ${rem}s`;
}

function truncateInline(text: string, max = 88): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1)}…`;
}

function summarizeListRepositoriesCall(args: any): string {
  const filters: string[] = [];

  if (typeof args?.pattern === "string" && args.pattern.trim()) {
    filters.push(`name~"${truncateInline(args.pattern.trim(), 24)}"`);
  }

  if (typeof args?.organization === "string" && args.organization.trim()) {
    filters.push(`org:${args.organization.trim()}`);
  }

  if (typeof args?.language === "string" && args.language.trim()) {
    filters.push(`lang:${args.language.trim()}`);
  }

  const limit = Number.isFinite(args?.limit) ? Number(args.limit) : 30;
  const offset = Number.isFinite(args?.offset) ? Number(args.offset) : 0;

  const scope = filters.length > 0 ? ` (${filters.join(", ")})` : "";
  const page = offset > 0 ? ` [offset ${offset}, limit ${limit}]` : "";
  return `Discovering repositories${scope}${page}`;
}

function summarizeToolCall(toolName: string, args: any): string {
  const repo = typeof args?.repository === "string" ? args.repository : undefined;

  switch (toolName) {
    case "read_github": {
      const p = typeof args?.path === "string" ? args.path : "(unknown path)";
      return `Reading ${repo ?? "repo"}:${p}`;
    }
    case "search_github": {
      const pattern = typeof args?.pattern === "string" ? args.pattern : "query";
      return `Searching code for “${truncateInline(pattern, 52)}”${repo ? ` in ${repo}` : ""}`;
    }
    case "glob_github": {
      const pattern = typeof args?.filePattern === "string" ? args.filePattern : "pattern";
      return `Globbing ${truncateInline(pattern, 52)}${repo ? ` in ${repo}` : ""}`;
    }
    case "list_directory_github": {
      const p = typeof args?.path === "string" ? args.path || "/" : "/";
      return `Listing directory ${p}${repo ? ` in ${repo}` : ""}`;
    }
    case "commit_search": {
      const q = typeof args?.query === "string" ? ` for “${truncateInline(args.query, 48)}”` : "";
      return `Scanning commits${q}${repo ? ` in ${repo}` : ""}`;
    }
    case "diff": {
      const base = typeof args?.base === "string" ? args.base : "base";
      const head = typeof args?.head === "string" ? args.head : "head";
      return `Comparing ${base}...${head}${repo ? ` in ${repo}` : ""}`;
    }
    case "list_repositories":
      return summarizeListRepositoriesCall(args);
    case "librarian":
      return "Coordinating repository analysis";
    default:
      return `Running ${toolName}`;
  }
}

function renderProgress(state: LibrarianProgressState): string {
  const elapsed = Date.now() - state.startedAt;
  const frame = SPINNER_FRAMES[Math.floor(elapsed / 120) % SPINNER_FRAMES.length];

  const header =
    state.phase === "writing"
      ? `${frame} Librarian is drafting the final answer (${formatDuration(elapsed)})`
      : state.phase === "booting"
        ? `${frame} Librarian is starting up (${formatDuration(elapsed)})`
        : `${frame} Librarian is exploring repositories (${formatDuration(elapsed)})`;

  const counts =
    state.failedTools > 0
      ? `Tools: ${state.completedTools}/${state.startedTools} completed (${state.failedTools} failed)`
      : `Tools: ${state.completedTools}/${state.startedTools} completed`;

  const lines = [header, counts];
  if (state.currentAction) lines.push(`Current: ${truncateInline(state.currentAction)}`);
  if (state.recentActions.length > 0) {
    lines.push(`Recent: ${state.recentActions.map((a) => truncateInline(a, 42)).join(" • ")}`);
  }

  return lines.join("\n");
}

type GitHubRepo = {
  owner: string;
  repo: string;
  fullName: string;
};

function parseRepository(repository: string): GitHubRepo {
  let raw = repository.trim();
  if (!raw) throw new Error("Repository is required");

  if (raw.includes("://")) {
    const u = new URL(raw);
    if (u.hostname !== "github.com") {
      throw new Error(`Only github.com repositories are supported, got ${u.hostname}`);
    }
    raw = u.pathname;
  }

  raw = raw.replace(/\.git$/, "").replace(/^\/+|\/+$/g, "");
  const [owner, repo] = raw.split("/");
  if (!owner || !repo) {
    throw new Error(`Invalid repository: expected owner/repo, got "${repository}"`);
  }

  return { owner, repo, fullName: `${owner}/${repo}` };
}

function normalizePath(input: string): string {
  let p = input;
  if (p.startsWith("file://")) p = p.slice(7);
  return p.replace(/^\/+/, "");
}

function decodeBase64Utf8(data: string): string {
  return Buffer.from(data.replace(/\n/g, ""), "base64").toString("utf8");
}

function globMatches(pattern: string, filePath: string): boolean {
  let regex = "";
  let i = 0;

  while (i < pattern.length) {
    const ch = pattern[i];

    if (ch === "*") {
      if (pattern[i + 1] === "*") {
        if (pattern[i + 2] === "/") {
          regex += "(?:.+/)?";
          i += 3;
        } else {
          regex += ".*";
          i += 2;
        }
      } else {
        regex += "[^/]*";
        i += 1;
      }
      continue;
    }

    if (ch === "?") {
      regex += "[^/]";
      i += 1;
      continue;
    }

    if (ch === "{") {
      const close = pattern.indexOf("}", i);
      if (close !== -1) {
        const items = pattern
          .slice(i + 1, close)
          .split(",")
          .map((s) => escapeRegex(s));
        regex += `(?:${items.join("|")})`;
        i = close + 1;
        continue;
      }
    }

    if (ch === "[") {
      const close = pattern.indexOf("]", i);
      if (close !== -1) {
        regex += pattern.slice(i, close + 1);
        i = close + 1;
        continue;
      }
    }

    regex += escapeRegex(ch);
    i += 1;
  }

  return new RegExp(`^${regex}$`).test(filePath);
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function readRangeSlice(content: string, range?: number[]): { content: string; startLine: number } {
  if (!range || range.length !== 2) {
    return { content, startLine: 1 };
  }

  const [start, end] = range;
  const startSafe = Math.max(1, start || 1);
  const endSafe = Math.max(startSafe, end || startSafe);
  const lines = content.split("\n").slice(startSafe - 1, endSafe);
  return { content: lines.join("\n"), startLine: startSafe };
}

function validateSearchPattern(pattern: string) {
  if (pattern.length > 256) {
    throw new Error("pattern exceeds 256 characters");
  }

  const operators = pattern.match(/\b(AND|OR|NOT)\b/gi) ?? [];
  if (operators.length > 5) {
    throw new Error("pattern exceeds max 5 boolean operators (AND/OR/NOT)");
  }

  const stripped = pattern.replace(/\b(AND|OR|NOT)\b/gi, " ").trim();
  if (!stripped) {
    throw new Error("pattern must include at least one search term");
  }
}

async function ghApi(
  pi: ExtensionAPI,
  endpoint: string,
  options?: {
    method?: string;
    params?: Record<string, string | number | boolean | undefined>;
    headers?: string[];
    signal?: AbortSignal;
  },
): Promise<any> {
  const method = options?.method ?? "GET";
  const params = options?.params ?? {};
  const headers = options?.headers ?? [];

  const args: string[] = ["api", endpoint, "-X", method];

  for (const header of headers) {
    args.push("-H", header);
  }

  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) continue;
    args.push("-f", `${key}=${String(value)}`);
  }

  const result = await pi.exec("gh", args, { signal: options?.signal, timeout: 90_000 });
  if (result.code !== 0) {
    throw new Error((result.stderr || result.stdout || "gh api failed").trim());
  }

  const out = result.stdout.trim();
  if (!out) return null;
  try {
    return JSON.parse(out);
  } catch {
    throw new Error(`Failed to parse gh api output as JSON for ${endpoint}`);
  }
}

function asTextResult(data: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }],
    details: data,
  };
}

async function runLibrarianSubagent(
  cwd: string,
  prompt: string,
  signal: AbortSignal | undefined,
  onUpdate:
    | ((partial: { content?: Array<{ type: "text"; text: string }>; details?: Record<string, unknown> }) => void)
    | undefined,
): Promise<{ finalText: string; stderr: string }> {
  const systemPrompt = `You are the Librarian, a specialized codebase understanding agent that helps answer questions about large, complex codebases across repositories.

Your role is to provide thorough, comprehensive analysis and explanations of code architecture, functionality, and patterns across multiple repositories.

You are running inside pi as a subagent. Use the available GitHub tools extensively before answering.

Guidelines:
- Use all available tools to explore thoroughly before answering.
- Execute tools in parallel whenever possible for efficiency.
- Read files deeply and trace implementations end-to-end.
- Use commit history and diffs when historical context matters.
- Return a comprehensive answer in Markdown.
- Include concrete file paths and line references where possible.

Repository provider: GitHub only.
Use read_github, list_directory_github, list_repositories, search_github, glob_github, commit_search, diff.
`; 

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-librarian-"));
  const promptPath = path.join(tmpDir, "system-prompt.md");
  fs.writeFileSync(promptPath, systemPrompt, { encoding: "utf8", mode: 0o600 });

  let lastAssistantText = "";
  let resultText = "";
  let stderr = "";

  const progress: LibrarianProgressState = {
    startedAt: Date.now(),
    phase: "booting",
    startedTools: 0,
    completedTools: 0,
    failedTools: 0,
    recentActions: [],
  };

  let lastProgressText = "";
  const emitProgress = (force = false) => {
    if (!onUpdate) return;

    const text = renderProgress(progress);
    if (!force && text === lastProgressText) return;

    lastProgressText = text;
    onUpdate({
      content: [{ type: "text", text }],
      details: {
        phase: progress.phase,
        startedTools: progress.startedTools,
        completedTools: progress.completedTools,
        failedTools: progress.failedTools,
        currentAction: progress.currentAction,
      },
    });
  };

  try {
    const args = [
      "--mode",
      "json",
      "-p",
      "--no-session",
      "--tools",
      LIBRARIAN_SUBAGENT_TOOLS.join(","),
      "--append-system-prompt",
      promptPath,
      prompt,
    ];

    const exitCode = await new Promise<number>((resolve) => {
      const proc = spawn("pi", args, { cwd, stdio: ["ignore", "pipe", "pipe"], shell: false });
      let stdoutBuffer = "";
      let aborted = false;
      const heartbeat = setInterval(() => emitProgress(false), PROGRESS_HEARTBEAT_MS);
      (heartbeat as any).unref?.();

      const addRecent = (item: string) => {
        const first = progress.recentActions[0];
        if (first) {
          if (first === item) {
            progress.recentActions[0] = `${item} ×2`;
            return;
          }

          const aggregated = first.match(/^(.*) ×(\d+)$/);
          if (aggregated && aggregated[1] === item) {
            const count = Number.parseInt(aggregated[2], 10);
            progress.recentActions[0] = `${item} ×${Number.isFinite(count) ? count + 1 : 2}`;
            return;
          }
        }

        progress.recentActions.unshift(item);
        if (progress.recentActions.length > 4) progress.recentActions.length = 4;
      };
      const activeActions = new Map<string, string>();

      const processLine = (line: string) => {
        if (!line.trim()) return;

        let event: any;
        try {
          event = JSON.parse(line);
        } catch {
          return;
        }

        if (event.type === "tool_execution_start") {
          progress.phase = "exploring";
          progress.startedTools += 1;

          const action = summarizeToolCall(String(event.toolName ?? "tool"), event.args);
          const toolCallId = String(event.toolCallId ?? `tool-${progress.startedTools}`);
          activeActions.set(toolCallId, action);

          progress.currentAction = action;
          emitProgress(true);
          return;
        }

        if (event.type === "tool_execution_end") {
          progress.phase = "exploring";
          progress.completedTools += 1;
          if (event.isError) progress.failedTools += 1;

          const toolCallId = String(event.toolCallId ?? "");
          const action = activeActions.get(toolCallId) ?? summarizeToolCall(String(event.toolName ?? "tool"), event.args);
          if (toolCallId) activeActions.delete(toolCallId);

          addRecent(`${event.isError ? "✗" : "✓"} ${action}`);
          progress.currentAction = undefined;
          emitProgress(true);
          return;
        }

        if (
          event.type === "message_update" &&
          (event.assistantMessageEvent?.type === "text_start" || event.assistantMessageEvent?.type === "text_delta")
        ) {
          if (progress.phase !== "writing") {
            progress.phase = "writing";
            progress.currentAction = "Synthesizing findings";
            emitProgress(true);
          }
          return;
        }

        if (event.type === "message_end" && event.message?.role === "assistant") {
          const text = (event.message.content ?? [])
            .filter((p: any) => p?.type === "text")
            .map((p: any) => p.text)
            .join("\n")
            .trim();

          if (text) {
            lastAssistantText = text;
            progress.phase = "writing";
            progress.currentAction = undefined;
            onUpdate?.({
              content: [{ type: "text", text }],
              details: {
                phase: "assistant",
                stopReason: event.message.stopReason,
                startedTools: progress.startedTools,
                completedTools: progress.completedTools,
                failedTools: progress.failedTools,
              },
            });
          }
          return;
        }

        if (event.type === "result" && typeof event.result === "string") {
          resultText = event.result;
        }
      };

      emitProgress(true);

      proc.stdout.on("data", (chunk) => {
        stdoutBuffer += chunk.toString();
        const lines = stdoutBuffer.split("\n");
        stdoutBuffer = lines.pop() ?? "";
        for (const line of lines) processLine(line);
      });

      proc.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });

      proc.on("close", (code) => {
        clearInterval(heartbeat);
        if (stdoutBuffer.trim()) processLine(stdoutBuffer);
        resolve(code ?? 0);
      });

      proc.on("error", () => {
        clearInterval(heartbeat);
        resolve(1);
      });

      if (signal) {
        const abort = () => {
          aborted = true;
          proc.kill("SIGTERM");
          setTimeout(() => {
            if (!proc.killed) proc.kill("SIGKILL");
          }, 5_000);
        };

        if (signal.aborted) abort();
        else signal.addEventListener("abort", abort, { once: true });
      }

      if (aborted) resolve(1);
    });

    if (exitCode !== 0) {
      throw new Error(stderr.trim() || `subagent exited with code ${exitCode}`);
    }

    const finalText = resultText.trim() || lastAssistantText.trim();
    if (!finalText) {
      throw new Error("librarian returned no output");
    }

    return { finalText, stderr };
  } finally {
    try {
      fs.unlinkSync(promptPath);
    } catch {
      // ignore
    }
    try {
      fs.rmdirSync(tmpDir);
    } catch {
      // ignore
    }
  }
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "read_github",
    label: "Read GitHub File",
    description: "Read a file from a GitHub repository with optional line range.",
    parameters: Type.Object({
      path: Type.String({ description: "The path to the file to read" }),
      read_range: Type.Optional(
        Type.Array(Type.Number(), {
          minItems: 2,
          maxItems: 2,
          description: "Optional [start_line, end_line] to read only specific lines",
        }),
      ),
      repository: Type.String({ description: "Repository URL or owner/repo (e.g., https://github.com/owner/repo)" }),
      ref: Type.Optional(Type.String({ description: "Optional branch/tag/commit ref" })),
    }),

    async execute(_id, params, signal) {
      try {
        const repo = parseRepository(params.repository);
        const normalizedPath = normalizePath(params.path);
        const endpoint = `repos/${repo.fullName}/contents/${normalizedPath}`;
        const data = await ghApi(pi, endpoint, { params: { ref: params.ref }, signal });

        if (!data || Array.isArray(data)) {
          throw new Error("Path points to a directory or missing file");
        }

        const raw = data.encoding === "base64" ? decodeBase64Utf8(data.content ?? "") : String(data.content ?? "");
        const sliced = readRangeSlice(raw, params.read_range as number[] | undefined);
        const bytes = Buffer.byteLength(sliced.content, "utf8");

        if (bytes > MAX_FILE_BYTES) {
          throw new Error(
            `File is too large (${Math.round(bytes / 1024)}KB). Retry with a smaller read_range (max 128KB per call).`,
          );
        }

        const numbered = sliced.content
          .split("\n")
          .map((line, i) => `${sliced.startLine + i}: ${line}`)
          .join("\n");

        return asTextResult({ absolutePath: normalizedPath, content: numbered });
      } catch (error) {
        return {
          content: [{ type: "text", text: `read_github error: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "list_directory_github",
    label: "List GitHub Directory",
    description: "List files and directories for a path in a GitHub repository.",
    parameters: Type.Object({
      path: Type.String({ description: "Directory path to list (use empty string for root)" }),
      repository: Type.String({ description: "Repository URL or owner/repo" }),
      ref: Type.Optional(Type.String({ description: "Optional branch/tag/commit ref" })),
      limit: Type.Optional(Type.Number({ minimum: 1, maximum: 1000, description: "Max entries to return" })),
    }),

    async execute(_id, params, signal) {
      try {
        const repo = parseRepository(params.repository);
        const normalizedPath = normalizePath(params.path || "");
        const endpoint = `repos/${repo.fullName}/contents/${normalizedPath}`;
        const data = await ghApi(pi, endpoint, { params: { ref: params.ref }, signal });

        if (!Array.isArray(data)) {
          throw new Error("Path is not a directory");
        }

        const entries = data
          .map((entry: any) => (entry.type === "dir" ? `${entry.name}/` : entry.name))
          .sort((a: string, b: string) => {
            const aDir = a.endsWith("/");
            const bDir = b.endsWith("/");
            if (aDir && !bDir) return -1;
            if (!aDir && bDir) return 1;
            return a.localeCompare(b);
          })
          .slice(0, params.limit ?? 100);

        return asTextResult(entries);
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `list_directory_github error: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "glob_github",
    label: "Glob GitHub Files",
    description: "Find repository files matching a glob pattern.",
    parameters: Type.Object({
      filePattern: Type.String({ description: 'Glob pattern (e.g., "**/*.ts")' }),
      repository: Type.String({ description: "Repository URL or owner/repo" }),
      ref: Type.Optional(Type.String({ description: "Optional branch/tag/commit ref" })),
      limit: Type.Optional(Type.Number({ minimum: 1, maximum: 1000, description: "Max files to return" })),
      offset: Type.Optional(Type.Number({ minimum: 0, description: "Pagination offset" })),
    }),

    async execute(_id, params, signal) {
      try {
        const repo = parseRepository(params.repository);
        const ref = params.ref ?? "HEAD";
        const tree = await ghApi(pi, `repos/${repo.fullName}/git/trees/${encodeURIComponent(ref)}`, {
          params: { recursive: 1 },
          signal,
        });

        if (!tree || !Array.isArray(tree.tree)) {
          throw new Error("Failed to fetch repository tree");
        }

        if (tree.truncated) {
          throw new Error("Repository tree is too large. Use search_github or a narrower query.");
        }

        const all = tree.tree
          .filter((node: any) => node.type === "blob")
          .map((node: any) => String(node.path))
          .filter((p: string) => globMatches(params.filePattern, p));

        const offset = params.offset ?? 0;
        const limit = params.limit ?? 100;
        return asTextResult(all.slice(offset, offset + limit));
      } catch (error) {
        return {
          content: [{ type: "text", text: `glob_github error: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "search_github",
    label: "Search GitHub Code",
    description: "Search code in a repository and return grouped contextual snippets.",
    parameters: Type.Object({
      pattern: Type.String({ description: "Search query (supports GitHub operators AND/OR/NOT and qualifiers)" }),
      repository: Type.String({ description: "Repository URL or owner/repo" }),
      path: Type.Optional(Type.String({ description: "Optional path qualifier" })),
      limit: Type.Optional(Type.Number({ minimum: 1, maximum: 100, description: "Max results" })),
      offset: Type.Optional(Type.Number({ minimum: 0, description: "Pagination offset" })),
    }),

    async execute(_id, params, signal) {
      try {
        validateSearchPattern(params.pattern);
        const repo = parseRepository(params.repository);
        const limit = params.limit ?? 30;
        const offset = params.offset ?? 0;
        if (offset % limit !== 0) {
          throw new Error(`offset (${offset}) must be divisible by limit (${limit})`);
        }

        const perPage = Math.min(limit, 100);
        const page = Math.floor(offset / perPage) + 1;
        let q = `${params.pattern} repo:${repo.fullName}`;
        if (params.path && params.path !== ".") q += ` path:${params.path}`;

        const data = await ghApi(pi, "search/code", {
          params: { q, per_page: perPage, page },
          headers: ["Accept: application/vnd.github.v3.text-match+json"],
          signal,
        });

        const items = Array.isArray(data?.items) ? data.items : [];
        const grouped = new Map<string, string[]>();

        for (const item of items) {
          const file = String(item.path ?? "");
          if (!grouped.has(file)) grouped.set(file, []);
          const chunks = grouped.get(file)!;

          const textMatches = Array.isArray(item.text_matches) ? item.text_matches : [];
          for (const match of textMatches) {
            if (match.property !== "content" || !match.fragment) continue;
            const fragment = String(match.fragment).trim();
            chunks.push(fragment.length > 2048 ? `${fragment.slice(0, 2048)}... (truncated)` : fragment);
          }
        }

        return asTextResult({
          results: Array.from(grouped.entries()).map(([file, chunks]) => ({ file, chunks })),
          totalCount: Number(data?.total_count ?? 0),
        });
      } catch (error) {
        return {
          content: [
            { type: "text", text: `search_github error: ${error instanceof Error ? error.message : String(error)}` },
          ],
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "commit_search",
    label: "Search GitHub Commits",
    description: "Search commit history by query, author, date range, and path.",
    parameters: Type.Object({
      repository: Type.String({ description: "Repository URL or owner/repo" }),
      query: Type.Optional(Type.String({ description: "Text query for commit message/author" })),
      author: Type.Optional(Type.String({ description: "Author username or email" })),
      since: Type.Optional(Type.String({ description: "ISO date lower bound" })),
      until: Type.Optional(Type.String({ description: "ISO date upper bound" })),
      path: Type.Optional(Type.String({ description: "Filter to commits touching this path" })),
      limit: Type.Optional(Type.Number({ minimum: 1, maximum: 100, description: "Max commits" })),
      offset: Type.Optional(Type.Number({ minimum: 0, description: "Pagination offset" })),
    }),

    async execute(_id, params, signal) {
      try {
        const repo = parseRepository(params.repository);
        const limit = params.limit ?? 50;
        const offset = params.offset ?? 0;
        if (offset % limit !== 0) {
          throw new Error(`offset (${offset}) must be divisible by limit (${limit})`);
        }

        const perPage = Math.min(limit, 100);
        const page = Math.floor(offset / perPage) + 1;

        let commits: any[] = [];
        let totalCount = 0;

        if (params.path || !params.query) {
          const data = await ghApi(pi, `repos/${repo.fullName}/commits`, {
            params: {
              per_page: perPage,
              page,
              since: params.since,
              until: params.until,
              author: params.author,
              path: params.path,
            },
            signal,
          });

          commits = Array.isArray(data) ? data : [];

          if (params.query) {
            const q = params.query.toLowerCase();
            commits = commits.filter((c) => {
              const msg = String(c?.commit?.message ?? "").toLowerCase();
              const name = String(c?.commit?.author?.name ?? "").toLowerCase();
              const email = String(c?.commit?.author?.email ?? "").toLowerCase();
              return msg.includes(q) || name.includes(q) || email.includes(q);
            });
          }

          totalCount = commits.length;
        } else {
          const terms = [params.query, `repo:${repo.fullName}`].filter(Boolean) as string[];
          if (params.author) terms.push(`author:${params.author}`);
          if (params.since) terms.push(`author-date:>=${params.since}`);
          if (params.until) terms.push(`author-date:<=${params.until}`);

          const data = await ghApi(pi, "search/commits", {
            params: {
              q: terms.join(" "),
              per_page: perPage,
              page,
              sort: "author-date",
              order: "desc",
            },
            headers: ["Accept: application/vnd.github.cloak-preview+json"],
            signal,
          });

          commits = Array.isArray(data?.items) ? data.items : [];
          totalCount = Number(data?.total_count ?? commits.length);
        }

        const mapped = commits.map((c) => {
          const messageRaw = String(c?.commit?.message ?? "").trim();
          return {
            sha: String(c?.sha ?? ""),
            message: messageRaw.length > 1024 ? `${messageRaw.slice(0, 1024)}... (truncated)` : messageRaw,
            author: {
              name: String(c?.commit?.author?.name ?? ""),
              email: String(c?.commit?.author?.email ?? ""),
              date: String(c?.commit?.author?.date ?? ""),
            },
          };
        });

        return asTextResult({ commits: mapped, totalCount });
      } catch (error) {
        return {
          content: [{ type: "text", text: `commit_search error: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "diff",
    label: "GitHub Diff",
    description: "Compare two refs (commit/branch/tag) and optionally include file patches.",
    parameters: Type.Object({
      repository: Type.String({ description: "Repository URL or owner/repo" }),
      base: Type.String({ description: "Base ref (branch/tag/sha)" }),
      head: Type.String({ description: "Head ref (branch/tag/sha)" }),
      includePatches: Type.Optional(
        Type.Boolean({ description: "Include patch text (token-heavy, truncated to ~4k chars per file)" }),
      ),
    }),

    async execute(_id, params, signal) {
      try {
        const repo = parseRepository(params.repository);
        const data = await ghApi(pi, `repos/${repo.fullName}/compare/${encodeURIComponent(params.base)}...${encodeURIComponent(params.head)}`, {
          signal,
        });

        const files = (Array.isArray(data?.files) ? data.files : []).map((f: any) => ({
          filename: f.filename,
          status: f.status,
          additions: f.additions,
          deletions: f.deletions,
          changes: f.changes,
          patch:
            params.includePatches && typeof f.patch === "string"
              ? f.patch.length > MAX_PATCH_CHARS
                ? `${f.patch.slice(0, MAX_PATCH_CHARS)}\n... [truncated]`
                : f.patch
              : undefined,
          previous_filename: f.previous_filename,
          sha: f.sha,
          blob_url: f.blob_url,
        }));

        const commits = Array.isArray(data?.commits) ? data.commits : [];
        const headCommit = commits.length > 0 ? commits[commits.length - 1] : undefined;

        return asTextResult({
          files,
          base_commit: {
            sha: data?.base_commit?.sha ?? params.base,
            message: String(data?.base_commit?.commit?.message ?? "").trim(),
          },
          head_commit: {
            sha: headCommit?.sha ?? params.head,
            message: String(headCommit?.commit?.message ?? "").trim(),
          },
          ahead_by: Number(data?.ahead_by ?? 0),
          behind_by: Number(data?.behind_by ?? 0),
          total_commits: Number(data?.total_commits ?? 0),
        });
      } catch (error) {
        return {
          content: [{ type: "text", text: `diff error: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "list_repositories",
    label: "List Repositories",
    description:
      "List repositories, prioritizing repositories accessible to the authenticated user and supplementing with public search when needed.",
    parameters: Type.Object({
      pattern: Type.Optional(Type.String({ description: "Optional name pattern" })),
      organization: Type.Optional(Type.String({ description: "Optional org filter" })),
      language: Type.Optional(Type.String({ description: "Optional language filter" })),
      limit: Type.Optional(Type.Number({ minimum: 1, maximum: 100, description: "Max repositories" })),
      offset: Type.Optional(Type.Number({ minimum: 0, description: "Pagination offset" })),
    }),

    async execute(_id, params, signal) {
      try {
        const limit = params.limit ?? 30;
        const offset = params.offset ?? 0;
        if (offset % limit !== 0) {
          throw new Error(`offset (${offset}) must be divisible by limit (${limit})`);
        }

        const userPerPage = Math.min(limit * 5, 100);
        const userPage = Math.floor(offset / userPerPage) + 1;

        const userReposRaw = await ghApi(pi, "user/repos", {
          params: {
            per_page: userPerPage,
            page: userPage,
            sort: "updated",
            affiliation: "owner,collaborator,organization_member",
          },
          signal,
        });

        let userRepos = Array.isArray(userReposRaw) ? userReposRaw : [];

        if (params.pattern) {
          const p = params.pattern.toLowerCase();
          userRepos = userRepos.filter((r) => String(r.full_name ?? "").toLowerCase().includes(p));
        }

        if (params.organization) {
          const org = params.organization.toLowerCase();
          userRepos = userRepos.filter((r) => String(r.full_name ?? "").split("/")[0]?.toLowerCase() === org);
        }

        if (params.language) {
          const lang = params.language.toLowerCase();
          userRepos = userRepos.filter((r) => String(r.language ?? "").toLowerCase() === lang);
        }

        userRepos.sort((a, b) => Number(b.stargazers_count ?? 0) - Number(a.stargazers_count ?? 0));

        const merged = [...userRepos];
        let totalCount = userRepos.length;

        if (merged.length < limit) {
          const queryParts: string[] = [];
          if (params.pattern) queryParts.push(`${params.pattern} in:name`);
          if (params.organization) queryParts.push(`org:${params.organization}`);
          if (params.language) queryParts.push(`language:${params.language}`);
          const q = queryParts.length > 0 ? queryParts.join(" ") : "*";

          const remaining = Math.min(limit - merged.length, 100);
          const search = await ghApi(pi, "search/repositories", {
            params: {
              q,
              per_page: remaining,
              sort: "stars",
              order: "desc",
            },
            signal,
          });

          const searchItems = Array.isArray(search?.items) ? search.items : [];
          const seen = new Set(merged.map((r) => String(r.full_name)));
          const deduped = searchItems.filter((r) => !seen.has(String(r.full_name)));
          merged.push(...deduped.slice(0, remaining));
          totalCount += deduped.length;
        }

        return asTextResult({
          repositories: merged.slice(0, limit).map((r: any) => ({
            name: r.full_name,
            description: r.description,
            language: r.language,
            stargazersCount: r.stargazers_count,
            forksCount: r.forks_count,
            private: r.private,
          })),
          totalCount,
        });
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `list_repositories error: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "librarian",
    label: "Librarian",
    description:
      "Specialized multi-repository codebase understanding agent for GitHub. Delegates to an isolated subagent with repository analysis tools.",
    parameters: Type.Object({
      query: Type.String({ description: "Your question about the codebase" }),
      context: Type.Optional(Type.String({ description: "Optional context on what you're trying to achieve" })),
    }),

    async execute(_id, params, signal, onUpdate, ctx) {
      try {
        const authStatus = await pi.exec("gh", ["auth", "status", "-h", "github.com"], {
          signal,
          timeout: 15_000,
        });

        if (authStatus.code !== 0) {
          throw new Error(
            `GitHub authentication required. Run: gh auth login\nDetails: ${(authStatus.stderr || authStatus.stdout).trim()}`,
          );
        }

        const prompt = params.context ? `Context: ${params.context}\n\nQuery: ${params.query}` : params.query;

        onUpdate?.({
          content: [{ type: "text", text: "Starting Librarian subagent..." }],
          details: { phase: "booting" },
        });

        const { finalText } = await runLibrarianSubagent(ctx.cwd, prompt, signal, (partial) => {
          onUpdate?.(partial);
        });

        return {
          content: [{ type: "text", text: finalText }],
          details: {
            subagentTools: LIBRARIAN_SUBAGENT_TOOLS,
          },
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `librarian error: ${error instanceof Error ? error.message : String(error)}` }],
          isError: true,
        };
      }
    },
  });
}
