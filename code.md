# File: deepseek-api/package-lock.json
{
  "name": "betazero-browser-automation",
  "lockfileVersion": 3,
  "requires": true,
  "packages": {
    "": {
      "name": "betazero-browser-automation",
      "dependencies": {
        "playwright-core": "^1.52.0"
      }
    },
    "node_modules/playwright-core": {
      "version": "1.59.1",
      "resolved": "https://registry.npmjs.org/playwright-core/-/playwright-core-1.59.1.tgz",
      "integrity": "sha512-HBV/RJg81z5BiiZ9yPzIiClYV/QMsDCKUyogwH9p3MCP6IYjUFu/MActgYAvK0oWyV9NlwM3GLBjADyWgydVyg==",
      "license": "Apache-2.0",
      "bin": {
        "playwright-core": "cli.js"
      },
      "engines": {
        "node": ">=18"
      }
    }
  }
}


# File: deepseek-api/package.json
{
  "name": "betazero-browser-automation",
  "private": true,
  "type": "module",
  "scripts": {
    "deepseek": "node scripts/deepseek-thorium.mjs",
    "deepseek-api": "node scripts/deepseek-api.mjs"
  },
  "dependencies": {
    "playwright-core": "^1.52.0"
  }
}

# File: deepseek-api/scripts/deepseek-api.mjs
import { createServer } from "node:http";
import { DeepSeekSession, buildSessionOptions } from "./deepseek-session.mjs";

const DEFAULT_PORT = 8787;
let session = null;
let sessionKey = "";
let requestQueue = Promise.resolve();

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error("Request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
  });
}

function writeJson(res, statusCode, payload) {
  res.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(payload, null, 2));
}

function getSessionConfig(payload = {}) {
  const options = buildSessionOptions({
    chat: typeof payload.chat === "string" ? payload.chat : "",
    mode: typeof payload.mode === "string" ? payload.mode : "Expert",
    profileDirectory: typeof payload.profileDirectory === "string" ? payload.profileDirectory : "Default",
    userDataDir: typeof payload.userDataDir === "string" ? payload.userDataDir : undefined,
    executablePath: typeof payload.executablePath === "string" ? payload.executablePath : undefined,
    pauseForLoginMs: typeof payload.pauseForLoginMs === "number" ? payload.pauseForLoginMs : undefined,
    turnDelayMs: typeof payload.turnDelayMs === "number" ? payload.turnDelayMs : undefined,
    timeoutMs: typeof payload.timeoutMs === "number" ? payload.timeoutMs : undefined,
    debugDir: typeof payload.debugDir === "string" ? payload.debugDir : "",
    cloneUserDataDir: payload.cloneUserDataDir !== false,
    keepTempProfile: payload.keepTempProfile === true
  });
  return options;
}

function stableStringify(value) {
  return JSON.stringify(value, Object.keys(value).sort());
}

async function ensureSession(payload = {}) {
  const config = getSessionConfig(payload);
  const nextKey = stableStringify({
    chat: config.chat,
    mode: config.mode,
    executablePath: config.executablePath,
    userDataDir: config.userDataDir,
    profileDirectory: config.profileDirectory,
    cloneUserDataDir: config.cloneUserDataDir,
    keepTempProfile: config.keepTempProfile
  });

  if (session && sessionKey === nextKey) {
    return session;
  }

  if (session) {
    await session.close();
  }

  session = new DeepSeekSession(config);
  sessionKey = nextKey;
  await session.init();
  return session;
}

function enqueue(task) {
  const next = requestQueue.then(task, task);
  requestQueue = next.catch(() => {});
  return next;
}

const server = createServer(async (req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    const snapshot = session ? await session.snapshot() : { initialized: false };
    writeJson(res, 200, { ok: true, session: snapshot });
    return;
  }

  if (req.method === "POST" && req.url === "/chat") {
    try {
      const payload = await readJsonBody(req);
      const prompts = Array.isArray(payload.prompts) ? payload.prompts.filter(Boolean) : [];
      if (!prompts.length) {
        writeJson(res, 400, { ok: false, error: "prompts must be a non-empty array" });
        return;
      }

      const result = await enqueue(async () => {
        const activeSession = await ensureSession(payload);
        return activeSession.chat(prompts, payload);
      });
      writeJson(res, 200, { ok: true, stateful: true, result });
    } catch (error) {
      writeJson(res, 500, {
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      });
    }
    return;
  }

  if (req.method === "POST" && req.url === "/reset") {
    try {
      await enqueue(async () => {
        if (session) {
          await session.close();
        }
        session = null;
        sessionKey = "";
      });
      writeJson(res, 200, { ok: true, reset: true });
    } catch (error) {
      writeJson(res, 500, {
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      });
    }
    return;
  }

  writeJson(res, 404, { ok: false, error: "Not found" });
});

server.listen(DEFAULT_PORT, "127.0.0.1", () => {
  console.log(`DeepSeek API listening on http://127.0.0.1:${DEFAULT_PORT}`);
});


# File: deepseek-api/scripts/deepseek_batch_client.py
#!/usr/bin/env python3
import json
import sys
import urllib.request
from pathlib import Path


API_URL = "http://127.0.0.1:8787/chat"
OUTPUT_PATH = Path("automation-output/deepseek-batch-results.json")
PROMPTS = [
    "Reply with exactly: ONE",
    "Reply with exactly: TWO",
    "Reply with exactly: THREE",
    "Reply with exactly: FOUR",
    "Reply with exactly: FIVE",
]


def send_prompt(prompt: str) -> dict:
    payload = {
        "mode": "Expert",
        "pauseForLoginMs": 30000,
        "turnDelayMs": 15000,
        "timeoutMs": 240000,
        "prompts": [prompt],
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def extract_output(result: dict) -> str | None:
    turns = ((result or {}).get("result") or {}).get("turns") or []
    if not turns:
      return None
    return turns[-1].get("response")


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"[{index}/{len(PROMPTS)}] Sending prompt: {prompt}", flush=True)
        result = send_prompt(prompt)
        output = extract_output(result)
        pairs.append(
            {
                "prompt": prompt,
                "output": output,
                "raw_result": result,
            }
        )
        print(f"[{index}/{len(PROMPTS)}] Output: {output}", flush=True)

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "api_url": API_URL,
                "results": [{"prompt": item["prompt"], "output": item["output"]} for item in pairs],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote results to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# File: deepseek-api/scripts/deepseek_api_client.py
#!/usr/bin/env python3
import json
import urllib.request


API_URL = "http://127.0.0.1:8787/chat"


def main():
    payload = {
        "mode": "Expert",
        "pauseForLoginMs": 30000,
        "turnDelayMs": 15000,
        "timeoutMs": 240000,
        "prompts": [
            "Reply with exactly: ONE",
            "Reply with exactly: TWO",
        ],
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8")
        result = json.loads(body)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# File: deepseek-api/scripts/deepseek-thorium.mjs
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import readline from "node:readline/promises";
import { chromium } from "playwright-core";

const DEFAULT_USER_DATA_DIR = path.join(os.homedir(), ".config", "thorium");
const DEFAULT_EXECUTABLE = "/usr/bin/thorium-browser";
const DEFAULT_URL = "https://chat.deepseek.com/";

function parseArgs(argv) {
  const options = {
    chat: "",
    prompt: "",
    prompts: [],
    mode: "",
    executablePath: DEFAULT_EXECUTABLE,
    userDataDir: DEFAULT_USER_DATA_DIR,
    profileDirectory: "Default",
    cloneUserDataDir: true,
    keepTempProfile: false,
    interactive: false,
    debugDir: "",
    output: "",
    timeoutMs: 120000,
    pauseForLoginMs: 30000,
    turnDelayMs: 12000
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === "--chat" && next) {
      options.chat = next;
      i += 1;
    } else if (arg === "--prompt" && next) {
      options.prompt = next;
      options.prompts.push(next);
      i += 1;
    } else if (arg === "--mode" && next) {
      options.mode = next;
      i += 1;
    } else if (arg === "--profile-directory" && next) {
      options.profileDirectory = next;
      i += 1;
    } else if (arg === "--user-data-dir" && next) {
      options.userDataDir = next;
      i += 1;
    } else if (arg === "--executable-path" && next) {
      options.executablePath = next;
      i += 1;
    } else if (arg === "--output" && next) {
      options.output = next;
      i += 1;
    } else if (arg === "--no-clone-user-data-dir") {
      options.cloneUserDataDir = false;
    } else if (arg === "--keep-temp-profile") {
      options.keepTempProfile = true;
    } else if (arg === "--interactive") {
      options.interactive = true;
    } else if (arg === "--debug-dir" && next) {
      options.debugDir = next;
      i += 1;
    } else if (arg === "--turn-delay-ms" && next) {
      options.turnDelayMs = Number(next);
      i += 1;
    } else if (arg === "--timeout-ms" && next) {
      options.timeoutMs = Number(next);
      i += 1;
    } else if (arg === "--pause-for-login-ms" && next) {
      options.pauseForLoginMs = Number(next);
      i += 1;
    }
  }

  return options;
}

function assertPaths(options) {
  if (!fs.existsSync(options.executablePath)) {
    throw new Error(`Browser executable not found: ${options.executablePath}`);
  }
  if (!fs.existsSync(options.userDataDir)) {
    throw new Error(`User data dir not found: ${options.userDataDir}`);
  }
}

function cloneUserDataDir(sourceDir) {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "thorium-playwright-"));
  fs.cpSync(sourceDir, tempRoot, {
    recursive: true,
    filter: (source) => {
      const base = path.basename(source);
      return ![
        "SingletonCookie",
        "SingletonLock",
        "SingletonSocket",
        "LOCK",
        "lockfile"
      ].includes(base);
    }
  });
  return tempRoot;
}

async function maybeSelectChat(page, chatTitle, timeoutMs) {
  if (!chatTitle) {
    return false;
  }

  const exactLocator = page.getByText(chatTitle, { exact: true }).first();
  if (await exactLocator.count()) {
    await exactLocator.click({ timeout: 5000 });
    await page.waitForTimeout(1500);
    return true;
  }

  const partialLocator = page.getByText(chatTitle, { exact: false }).first();
  if (await partialLocator.count()) {
    await partialLocator.click({ timeout: 5000 });
    await page.waitForTimeout(1500);
    return true;
  }

  const clicked = await page.evaluate((target) => {
    const elements = Array.from(document.querySelectorAll("a, button, div, li"));
    const normalized = target.trim().toLowerCase();
    const candidate = elements.find((element) => {
      const text = element.textContent?.trim().toLowerCase();
      return text && text.includes(normalized);
    });
    if (candidate) {
      candidate.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      return true;
    }
    return false;
  }, chatTitle);

  if (clicked) {
    await page.waitForLoadState("networkidle", { timeout: Math.min(timeoutMs, 15000) }).catch(() => {});
    await page.waitForTimeout(1500);
  }

  return clicked;
}

async function extractMessages(page) {
  return page.evaluate(() => {
    const normalize = (value) => value?.replace(/\s+/g, " ").trim() || "";
    const seen = new Set();
    const messages = [];

    const listItems = document.querySelectorAll(
      ".ds-virtual-list-visible-items > [data-virtual-list-item-key]"
    );

    for (const item of listItems) {
      const assistantNode = item.querySelector(".ds-message .ds-markdown");
      const userNode = item.querySelector(".ds-message .fbb737a4, .ds-message textarea, .ds-message");
      const node = assistantNode || userNode;
      const text = normalize(node?.textContent);
      if (!text || text.length < 1 || seen.has(text)) {
        continue;
      }

      seen.add(text);
      messages.push({
        selector: assistantNode ? ".ds-message .ds-markdown" : ".ds-message",
        role: assistantNode ? "assistant" : "user",
        text
      });
    }

    if (messages.length) {
      return messages;
    }

    const fallbackSelectors = [
      ".ds-message .ds-markdown",
      ".ds-message .ds-markdown-paragraph",
      "[data-role='assistant']",
      "[data-message-author-role='assistant']",
      "[class*='assistant']",
      "[class*='message']",
      "main article",
      "main [role='article']"
    ];

    for (const selector of fallbackSelectors) {
      for (const node of document.querySelectorAll(selector)) {
        const text = normalize(node.textContent);
        if (!text || text.length < 1 || seen.has(text)) {
          continue;
        }
        seen.add(text);
        messages.push({
          selector,
          role: selector.includes("assistant") || selector.includes("markdown") ? "assistant" : "unknown",
          text
        });
      }
    }

    return messages;
  });
}

async function writeDebugArtifacts(page, debugDir, label) {
  if (!debugDir) {
    return;
  }

  fs.mkdirSync(debugDir, { recursive: true });
  const safeLabel = label.replace(/[^a-zA-Z0-9_-]+/g, "-");
  const htmlPath = path.join(debugDir, `${safeLabel}.html`);
  const textPath = path.join(debugDir, `${safeLabel}.txt`);
  const pageHtml = await page.content();
  const pageText = await page.evaluate(() => document.body.innerText || "");
  fs.writeFileSync(htmlPath, pageHtml);
  fs.writeFileSync(textPath, pageText);
}

function normalizeMessageText(value) {
  return value?.replace(/\s+/g, " ").trim() || "";
}

function getMessageSignature(message) {
  return `${message.role}:${normalizeMessageText(message.text)}`;
}

function getNewAssistantMessage(messages, previousSignatures) {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const candidate = messages[i];
    if (candidate.role !== "assistant") {
      continue;
    }

    const text = normalizeMessageText(candidate.text);
    if (!text) {
      continue;
    }

    const signature = getMessageSignature(candidate);
    if (!previousSignatures.has(signature)) {
      return text;
    }
  }

  return null;
}

async function clickByText(page, targetText) {
  const exactLocator = page.getByText(targetText, { exact: true }).first();
  if (await exactLocator.count()) {
    await exactLocator.click({ timeout: 5000 });
    return true;
  }

  const partialLocator = page.getByText(targetText, { exact: false }).first();
  if (await partialLocator.count()) {
    await partialLocator.click({ timeout: 5000 });
    return true;
  }

  return page.evaluate((target) => {
    const normalized = target.trim().toLowerCase();
    const elements = Array.from(
      document.querySelectorAll("button, [role='button'], div, span, li, a")
    );
    const candidate = elements.find((element) => {
      const text = element.textContent?.replace(/\s+/g, " ").trim().toLowerCase();
      if (!text) {
        return false;
      }
      return text === normalized || text.includes(normalized);
    });

    if (!candidate) {
      return false;
    }

    candidate.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    return true;
  }, targetText);
}

async function maybeSelectMode(page, mode) {
  if (!mode) {
    return false;
  }

  const clicked = await clickByText(page, mode);
  if (clicked) {
    await page.waitForTimeout(1000);
  }
  return clicked;
}

async function sendPrompt(page, prompt) {
  if (!prompt) {
    return false;
  }

  const input = page.locator("textarea, div[contenteditable='true']").first();
  await input.waitFor({ timeout: 15000 });
  await input.click({ timeout: 5000 });

  const tagName = await input.evaluate((node) => node.tagName.toLowerCase());
  if (tagName === "textarea") {
    await input.fill(prompt);
  } else {
    await input.evaluate((node, value) => {
      node.textContent = value;
      node.dispatchEvent(new InputEvent("input", { bubbles: true, data: value, inputType: "insertText" }));
    }, prompt);
  }

  await page.waitForTimeout(500);

  const sendSelectors = [
    "button[type='submit']",
    "button[aria-label*='Send']",
    "button[aria-label*='send']"
  ];

  for (const selector of sendSelectors) {
    const button = page.locator(selector).first();
    if (await button.count()) {
      await button.click({ timeout: 5000 });
      return true;
    }
  }

  await input.press("Enter");
  return true;
}

async function waitForTurnCompletion(page, prompt, previousMessages, timeoutMs, turnDelayMs) {
  const deadline = Date.now() + timeoutMs;
  const previousSignatures = new Set(previousMessages.map(getMessageSignature));
  const normalizedPrompt = normalizeMessageText(prompt);
  const pollIntervalMs = 1000;
  const retryCount = 3;
  const retryDelayMs = Math.max(1000, Math.min(turnDelayMs, 3000));
  let lastCandidate = null;
  let stableCandidateCount = 0;

  const inspect = async () => {
    const messages = await extractMessages(page);
    const promptSeen = messages.some((message) => {
      return message.role === "user" && normalizeMessageText(message.text) === normalizedPrompt;
    });
    const response = getNewAssistantMessage(messages, previousSignatures);
    return { messages, promptSeen, response };
  };

  while (Date.now() < deadline) {
    const snapshot = await inspect();
    if (snapshot.promptSeen && snapshot.response) {
      if (snapshot.response === lastCandidate) {
        stableCandidateCount += 1;
      } else {
        lastCandidate = snapshot.response;
        stableCandidateCount = 1;
      }

      if (stableCandidateCount >= 2) {
        return {
          completed: true,
          promptSeen: true,
          response: snapshot.response,
          messages: snapshot.messages,
          status: "completed"
        };
      }
    } else {
      lastCandidate = null;
      stableCandidateCount = 0;
    }

    await page.waitForTimeout(pollIntervalMs);
  }

  for (let attempt = 1; attempt <= retryCount; attempt += 1) {
    await page.waitForTimeout(retryDelayMs);
    const snapshot = await inspect();
    if (snapshot.promptSeen && snapshot.response) {
      return {
        completed: true,
        promptSeen: true,
        response: snapshot.response,
        messages: snapshot.messages,
        status: attempt === 1 ? "completed_after_timeout" : "completed_after_retry"
      };
    }
  }

  const finalSnapshot = await inspect();
  return {
    completed: false,
    promptSeen: finalSnapshot.promptSeen,
    response: finalSnapshot.response,
    messages: finalSnapshot.messages,
    status: finalSnapshot.promptSeen ? "assistant_response_not_detected" : "prompt_not_detected"
  };
}

async function runTurns(page, prompts, timeoutMs, turnDelayMs, debugDir) {
  const turns = [];

  for (const [index, prompt] of prompts.entries()) {
    const messagesBeforePrompt = await extractMessages(page);
    const promptSent = await sendPrompt(page, prompt);
    if (!promptSent) {
      turns.push({
        prompt,
        promptSent: false,
        response: null
      });
      continue;
    }

    await page.waitForTimeout(turnDelayMs);
    const turnResult = await waitForTurnCompletion(
      page,
      prompt,
      messagesBeforePrompt,
      Math.min(timeoutMs, 15000),
      turnDelayMs
    );
    await writeDebugArtifacts(page, debugDir, `turn-${index + 1}`);
    turns.push({
      prompt,
      promptSent: true,
      response: turnResult.response,
      promptSeenAfterSend: turnResult.promptSeen,
      waitStatus: turnResult.status,
      messageCountAfterTurn: turnResult.messages.length
    });
  }

  return turns;
}

async function waitForUserReady(page, options) {
  if (!options.interactive) {
    console.error("If DeepSeek shows a verification/login gate, finish it in the opened Thorium window.");
    await page.waitForTimeout(options.pauseForLoginMs);
    return;
  }

  console.error("Interactive mode enabled.");
  console.error("Use the opened Thorium window to pass verification, log in if needed, and open the target chat.");
  console.error("When the page is ready to scrape, return to this terminal and press Enter.");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stderr
  });

  try {
    await rl.question("");
  } finally {
    rl.close();
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  assertPaths(options);

  const launchUserDataDir = options.cloneUserDataDir
    ? cloneUserDataDir(options.userDataDir)
    : options.userDataDir;

  const context = await chromium.launchPersistentContext(launchUserDataDir, {
    channel: undefined,
    executablePath: options.executablePath,
    headless: false,
    viewport: null,
    args: [
      `--profile-directory=${options.profileDirectory}`,
      "--disable-blink-features=AutomationControlled"
    ]
  });

  const page = context.pages()[0] ?? (await context.newPage());

  try {
    await page.goto(DEFAULT_URL, { waitUntil: "domcontentloaded", timeout: options.timeoutMs });
    await page.waitForLoadState("networkidle", { timeout: 15000 }).catch(() => {});

    const title = await page.title();
    console.error(`Opened page: ${title}`);
    await waitForUserReady(page, options);

    const selected = await maybeSelectChat(page, options.chat, options.timeoutMs);
    if (options.chat) {
      console.error(selected ? `Selected chat: ${options.chat}` : `Chat not found: ${options.chat}`);
    }

    if (options.mode) {
      const modeSelected = await maybeSelectMode(page, options.mode);
      console.error(modeSelected ? `Selected mode: ${options.mode}` : `Mode not found: ${options.mode}`);
    }

    const prompts = options.prompts.length ? options.prompts : (options.prompt ? [options.prompt] : []);
    let turns = [];
    if (prompts.length) {
      turns = await runTurns(page, prompts, options.timeoutMs, options.turnDelayMs, options.debugDir);
      console.error(`Completed turns: ${turns.length}`);
    }

    await page.waitForTimeout(2000);
    const messages = await extractMessages(page);
    const payload = {
      url: page.url(),
      title: await page.title(),
      selectedChat: options.chat || null,
      selectedMode: options.mode || null,
      prompt: options.prompt || null,
      prompts,
      turns,
      messageCount: messages.length,
      messages
    };

    const serialized = JSON.stringify(payload, null, 2);
    if (options.output) {
      fs.mkdirSync(path.dirname(options.output), { recursive: true });
      fs.writeFileSync(options.output, serialized);
      console.error(`Wrote output to ${options.output}`);
    } else {
      process.stdout.write(`${serialized}\n`);
    }
  } finally {
    await context.close();
    if (options.cloneUserDataDir && !options.keepTempProfile) {
      fs.rmSync(launchUserDataDir, { recursive: true, force: true });
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});


# File: deepseek-api/scripts/deepseek-session.mjs
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright-core";

export const DEFAULT_USER_DATA_DIR = path.join(os.homedir(), ".config", "thorium");
export const DEFAULT_EXECUTABLE = "/usr/bin/thorium-browser";
export const DEFAULT_URL = "https://chat.deepseek.com/";

export function buildSessionOptions(overrides = {}) {
  const options = {
    chat: "",
    mode: "Expert",
    executablePath: DEFAULT_EXECUTABLE,
    userDataDir: DEFAULT_USER_DATA_DIR,
    profileDirectory: "Default",
    cloneUserDataDir: true,
    keepTempProfile: false,
    debugDir: "",
    timeoutMs: 120000,
    pauseForLoginMs: 30000,
    turnDelayMs: 12000
  };

  for (const [key, value] of Object.entries(overrides)) {
    if (value !== undefined) {
      options[key] = value;
    }
  }

  return options;
}

function assertPaths(options) {
  if (!fs.existsSync(options.executablePath)) {
    throw new Error(`Browser executable not found: ${options.executablePath}`);
  }
  if (!fs.existsSync(options.userDataDir)) {
    throw new Error(`User data dir not found: ${options.userDataDir}`);
  }
}

function cloneUserDataDir(sourceDir) {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "thorium-playwright-"));
  fs.cpSync(sourceDir, tempRoot, {
    recursive: true,
    filter: (source) => {
      const base = path.basename(source);
      return !["SingletonCookie", "SingletonLock", "SingletonSocket", "LOCK", "lockfile"].includes(base);
    }
  });
  return tempRoot;
}

async function clickByText(page, targetText) {
  const exactLocator = page.getByText(targetText, { exact: true }).first();
  if (await exactLocator.count()) {
    await exactLocator.click({ timeout: 5000 });
    return true;
  }

  const partialLocator = page.getByText(targetText, { exact: false }).first();
  if (await partialLocator.count()) {
    await partialLocator.click({ timeout: 5000 });
    return true;
  }

  return page.evaluate((target) => {
    const normalized = target.trim().toLowerCase();
    const elements = Array.from(
      document.querySelectorAll("button, [role='button'], div, span, li, a")
    );
    const candidate = elements.find((element) => {
      const text = element.textContent?.replace(/\s+/g, " ").trim().toLowerCase();
      return text && (text === normalized || text.includes(normalized));
    });
    if (!candidate) {
      return false;
    }
    candidate.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    return true;
  }, targetText);
}

async function maybeSelectChat(page, chatTitle, timeoutMs) {
  if (!chatTitle) {
    return false;
  }

  const clicked = await clickByText(page, chatTitle);
  if (clicked) {
    await page.waitForLoadState("networkidle", { timeout: Math.min(timeoutMs, 15000) }).catch(() => {});
    await page.waitForTimeout(1500);
  }
  return clicked;
}

async function maybeSelectMode(page, mode) {
  if (!mode) {
    return false;
  }
  const clicked = await clickByText(page, mode);
  if (clicked) {
    await page.waitForTimeout(1000);
  }
  return clicked;
}

async function sendPrompt(page, prompt) {
  const input = page.locator("textarea, div[contenteditable='true']").first();
  await input.waitFor({ timeout: 15000 });
  await input.click({ timeout: 5000 });

  const tagName = await input.evaluate((node) => node.tagName.toLowerCase());
  if (tagName === "textarea") {
    await input.fill(prompt);
  } else {
    await input.evaluate((node, value) => {
      node.textContent = value;
      node.dispatchEvent(new InputEvent("input", { bubbles: true, data: value, inputType: "insertText" }));
    }, prompt);
  }

  await page.waitForTimeout(500);

  const sendSelectors = [
    "button[type='submit']",
    "button[aria-label*='Send']",
    "button[aria-label*='send']"
  ];

  for (const selector of sendSelectors) {
    const button = page.locator(selector).first();
    if (await button.count()) {
      await button.click({ timeout: 5000 });
      return true;
    }
  }

  await input.press("Enter");
  return true;
}

async function extractMessages(page) {
  return page.evaluate(() => {
    const normalize = (value) => value?.replace(/\s+/g, " ").trim() || "";
    const seen = new Set();
    const messages = [];

    const listItems = document.querySelectorAll(
      ".ds-virtual-list-visible-items > [data-virtual-list-item-key]"
    );

    for (const item of listItems) {
      const assistantNode = item.querySelector(".ds-message .ds-markdown");
      const userNode = item.querySelector(".ds-message .fbb737a4, .ds-message textarea, .ds-message");
      const node = assistantNode || userNode;
      const text = normalize(node?.textContent);
      if (!text || seen.has(text)) {
        continue;
      }
      seen.add(text);
      messages.push({
        selector: assistantNode ? ".ds-message .ds-markdown" : ".ds-message",
        role: assistantNode ? "assistant" : "user",
        text
      });
    }

    return messages;
  });
}

function normalizeMessageText(value) {
  return value?.replace(/\s+/g, " ").trim() || "";
}

function getMessageSignature(message) {
  return `${message.role}:${normalizeMessageText(message.text)}`;
}

function getNewAssistantMessage(messages, previousSignatures) {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const candidate = messages[i];
    if (candidate.role !== "assistant") {
      continue;
    }

    const text = normalizeMessageText(candidate.text);
    if (!text) {
      continue;
    }

    const signature = getMessageSignature(candidate);
    if (!previousSignatures.has(signature)) {
      return text;
    }
  }

  return null;
}

async function writeDebugArtifacts(page, debugDir, label) {
  if (!debugDir) {
    return;
  }
  fs.mkdirSync(debugDir, { recursive: true });
  const safeLabel = label.replace(/[^a-zA-Z0-9_-]+/g, "-");
  fs.writeFileSync(path.join(debugDir, `${safeLabel}.html`), await page.content());
  fs.writeFileSync(
    path.join(debugDir, `${safeLabel}.txt`),
    await page.evaluate(() => document.body.innerText || "")
  );
}

async function waitForTurnCompletion(page, prompt, previousMessages, timeoutMs, turnDelayMs) {
  const deadline = Date.now() + timeoutMs;
  const previousSignatures = new Set(previousMessages.map(getMessageSignature));
  const normalizedPrompt = normalizeMessageText(prompt);
  const pollIntervalMs = 1000;
  const retryCount = 3;
  const retryDelayMs = Math.max(1000, Math.min(turnDelayMs, 3000));
  let lastCandidate = null;
  let stableCandidateCount = 0;

  const inspect = async () => {
    const messages = await extractMessages(page);
    const promptSeen = messages.some((message) => {
      return message.role === "user" && normalizeMessageText(message.text) === normalizedPrompt;
    });
    const response = getNewAssistantMessage(messages, previousSignatures);
    return { messages, promptSeen, response };
  };

  while (Date.now() < deadline) {
    const snapshot = await inspect();
    if (snapshot.promptSeen && snapshot.response) {
      if (snapshot.response === lastCandidate) {
        stableCandidateCount += 1;
      } else {
        lastCandidate = snapshot.response;
        stableCandidateCount = 1;
      }

      if (stableCandidateCount >= 2) {
        return {
          completed: true,
          promptSeen: true,
          response: snapshot.response,
          messages: snapshot.messages,
          status: "completed"
        };
      }
    } else {
      lastCandidate = null;
      stableCandidateCount = 0;
    }

    await page.waitForTimeout(pollIntervalMs);
  }

  for (let attempt = 1; attempt <= retryCount; attempt += 1) {
    await page.waitForTimeout(retryDelayMs);
    const snapshot = await inspect();
    if (snapshot.promptSeen && snapshot.response) {
      return {
        completed: true,
        promptSeen: true,
        response: snapshot.response,
        messages: snapshot.messages,
        status: attempt === 1 ? "completed_after_timeout" : "completed_after_retry"
      };
    }
  }

  const finalSnapshot = await inspect();
  return {
    completed: false,
    promptSeen: finalSnapshot.promptSeen,
    response: finalSnapshot.response,
    messages: finalSnapshot.messages,
    status: finalSnapshot.promptSeen ? "assistant_response_not_detected" : "prompt_not_detected"
  };
}

export class DeepSeekSession {
  constructor(options = {}) {
    this.options = buildSessionOptions(options);
    this.context = null;
    this.page = null;
    this.launchUserDataDir = null;
    this.startedAt = null;
    this.initialized = false;
    this.chatTitle = this.options.chat || null;
    this.mode = this.options.mode || null;
  }

  async init() {
    if (this.initialized) {
      return;
    }

    assertPaths(this.options);
    this.launchUserDataDir = this.options.cloneUserDataDir
      ? cloneUserDataDir(this.options.userDataDir)
      : this.options.userDataDir;

    this.context = await chromium.launchPersistentContext(this.launchUserDataDir, {
      executablePath: this.options.executablePath,
      headless: false,
      viewport: null,
      args: [
        `--profile-directory=${this.options.profileDirectory}`,
        "--disable-blink-features=AutomationControlled"
      ]
    });

    this.page = this.context.pages()[0] ?? (await this.context.newPage());
    await this.page.goto(DEFAULT_URL, {
      waitUntil: "domcontentloaded",
      timeout: this.options.timeoutMs
    });
    await this.page.waitForLoadState("networkidle", { timeout: 15000 }).catch(() => {});
    await this.page.waitForTimeout(this.options.pauseForLoginMs);

    if (this.chatTitle) {
      await maybeSelectChat(this.page, this.chatTitle, this.options.timeoutMs);
    }
    if (this.mode) {
      await maybeSelectMode(this.page, this.mode);
    }

    this.startedAt = new Date().toISOString();
    this.initialized = true;
  }

  async ensureRoute({ chat, mode } = {}) {
    if (chat && chat !== this.chatTitle) {
      await maybeSelectChat(this.page, chat, this.options.timeoutMs);
      this.chatTitle = chat;
    }
    if (mode && mode !== this.mode) {
      await maybeSelectMode(this.page, mode);
      this.mode = mode;
    }
  }

  async chat(prompts, overrides = {}) {
    await this.init();
    await this.ensureRoute(overrides);

    const turns = [];
    for (const [index, prompt] of prompts.entries()) {
      const messagesBeforePrompt = await extractMessages(this.page);
      const promptSent = await sendPrompt(this.page, prompt);
      if (!promptSent) {
        turns.push({ prompt, promptSent: false, response: null });
        continue;
      }

      const turnDelayMs = overrides.turnDelayMs ?? this.options.turnDelayMs;
      const timeoutMs = overrides.timeoutMs ?? this.options.timeoutMs;
      await this.page.waitForTimeout(turnDelayMs);
      const turnResult = await waitForTurnCompletion(
        this.page,
        prompt,
        messagesBeforePrompt,
        Math.min(timeoutMs, 15000),
        turnDelayMs
      );
      await writeDebugArtifacts(this.page, overrides.debugDir ?? this.options.debugDir, `turn-${index + 1}`);
      turns.push({
        prompt,
        promptSent: true,
        response: turnResult.response,
        promptSeenAfterSend: turnResult.promptSeen,
        waitStatus: turnResult.status,
        messageCountAfterTurn: turnResult.messages.length
      });
    }

    const messages = await extractMessages(this.page);
    return {
      url: this.page.url(),
      title: await this.page.title(),
      selectedChat: this.chatTitle,
      selectedMode: this.mode,
      prompts,
      turns,
      messageCount: messages.length,
      messages,
      session: {
        startedAt: this.startedAt,
        initialized: this.initialized
      }
    };
  }

  async snapshot() {
    if (!this.initialized) {
      return {
        initialized: false
      };
    }
    const messages = await extractMessages(this.page);
    return {
      initialized: true,
      startedAt: this.startedAt,
      url: this.page.url(),
      title: await this.page.title(),
      selectedChat: this.chatTitle,
      selectedMode: this.mode,
      messageCount: messages.length,
      messages
    };
  }

  async close() {
    if (this.context) {
      await this.context.close();
    }
    if (this.options.cloneUserDataDir && this.launchUserDataDir && !this.options.keepTempProfile) {
      fs.rmSync(this.launchUserDataDir, { recursive: true, force: true });
    }
    this.context = null;
    this.page = null;
    this.launchUserDataDir = null;
    this.initialized = false;
    this.startedAt = null;
  }
}


