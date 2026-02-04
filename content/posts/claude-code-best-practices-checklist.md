---
title: "Claude Code CLI Best Practices Checklist"
date: 2026-02-02
---

A comprehensive guide to tips, tricks, and best practices for maximizing productivity with Claude Code.

---

## Table of Contents
1. [Initial Setup & Configuration](#1-initial-setup--configuration)
2. [CLAUDE.md Configuration](#2-claudemd-configuration)
3. [Permissions & Security](#3-permissions--security)
4. [Context Management](#4-context-management)
5. [Custom Commands & Slash Commands](#5-custom-commands--slash-commands)
6. [Hooks for Automation](#6-hooks-for-automation)
7. [Parallel Development with Git Worktrees](#7-parallel-development-with-git-worktrees)
8. [CLI Usage Patterns](#8-cli-usage-patterns)
9. [MCP Server Integration](#9-mcp-server-integration)
10. [Session Management](#10-session-management)
11. [Keyboard Shortcuts & Navigation](#11-keyboard-shortcuts--navigation)
12. [Advanced Tips & Tricks](#12-advanced-tips--tricks)
13. [Common Pitfalls to Avoid](#13-common-pitfalls-to-avoid)

---

## 1. Initial Setup & Configuration

### Getting Started
- [ ] Install Claude Code: `npm install -g @anthropic-ai/claude-code`
- [ ] Run `claude` in your project directory to start
- [ ] Run `/init` to generate a starter CLAUDE.md based on your project structure
- [ ] Install GitHub CLI (`gh`) for seamless GitHub integration
- [ ] Set up your preferred terminal (iTerm2, VS Code terminal recommended)

### Settings Hierarchy
Claude Code uses hierarchical settings stored in JSON files:
- [ ] **User settings**: `~/.claude/settings.json` (applies to all projects)
- [ ] **Project settings**: `.claude/settings.json` (shared with team, committed to git)
- [ ] **Local project settings**: `.claude/settings.local.json` (personal, git-ignored)

### Example settings.json
```json
{
  "model": "claude-sonnet-4-20250514",
  "maxTokens": 4096,
  "permissions": {
    "allowedTools": ["Read", "Write", "Bash(git *)"],
    "deny": [
      "Read(./.env)",
      "Read(./.env.*)",
      "Write(./production.config.*)"
    ]
  }
}
```

---

## 2. CLAUDE.md Configuration

### Purpose
CLAUDE.md is your project's persistent memory—loaded automatically at the start of every session.

### Best Practices
- [ ] Keep it **concise** and human-readable
- [ ] Delete what you don't need from `/init` output (less is more)
- [ ] Focus on **universally applicable** instructions
- [ ] Let linters/formatters handle code style (not CLAUDE.md)
- [ ] The filename is case-sensitive: must be exactly `CLAUDE.md`

### What to Include
- [ ] **Project context**: One-liner describing the project
- [ ] **Key directories**: Architecture overview and important paths
- [ ] **Common commands**: Build, test, lint, deploy commands
- [ ] **Gotchas**: Project-specific warnings and quirks
- [ ] **MCP server instructions**: How to use configured integrations

### What NOT to Include
- [ ] Exhaustive code style guidelines (use linters instead)
- [ ] Non-universal "hotfix" instructions
- [ ] Redundant information Claude can infer from code

### File Locations
| Location | Scope |
|----------|-------|
| `~/.claude/CLAUDE.md` | Global (all projects) |
| `./CLAUDE.md` | Project root (shared with team) |
| `./subdirectory/CLAUDE.md` | Subdirectory-specific |

### Example CLAUDE.md
```markdown
# Project Context
FastAPI REST API for user authentication. Uses SQLAlchemy + Pydantic.

## Key Directories
- `app/models/` - database models
- `app/api/` - route handlers
- `app/core/` - configuration and utilities

## Common Commands
```bash
uvicorn app.main:app --reload  # dev server
pytest tests/                   # run tests
```

## Gotchas
- Always use `--break-system-packages` flag with pip
- Run migrations before testing locally
```

---

## 3. Permissions & Security

### Managing Permissions
- [ ] Use `/permissions` command to configure allowlist
- [ ] Select "Always allow" when prompted for trusted actions
- [ ] Consider `--dangerously-skip-permissions` for trusted workflows (use with caution)
- [ ] Use `/sandbox` for OS-level isolation

### Permission Configuration
```json
{
  "permissions": {
    "allow": [
      "Read",
      "Grep",
      "LS",
      "Bash(npm run test:*)",
      "Bash(git commit:*)",
      "Edit"
    ],
    "deny": [
      "WebFetch",
      "Bash(curl:*)",
      "Read(./.env)",
      "Read(./secrets/**)"
    ]
  }
}
```

### Security Best Practices
- [ ] Never store secrets in CLAUDE.md
- [ ] Deny access to `.env` files and secrets directories
- [ ] Start with minimal permissions, expand as needed
- [ ] Use `--allowedTools` flag for one-time permission sets

---

## 4. Context Management

### Monitor Context Usage
- [ ] Watch the context indicator (bottom right) for token usage
- [ ] Use `/clear` often—start fresh for new tasks
- [ ] Use `/compact` to manually trigger context compaction
- [ ] Be aware that auto-compaction may cause context loss

### Context Optimization Tips
- [ ] Keep conversations focused on single tasks
- [ ] Clear context before switching to unrelated work
- [ ] Use worktrees for parallel tasks instead of context switching
- [ ] Don't let old conversation history consume tokens

### Custom Status Line (Advanced)
You can customize the status line to show:
- Current model
- Git branch and uncommitted file count
- Token usage progress bar
- Last message preview

---

## 5. Custom Commands & Slash Commands

### Creating Project Commands
Store prompt templates in `.claude/commands/` directory:

```bash
mkdir -p .claude/commands
```

### Example: Fix GitHub Issue Command
Create `.claude/commands/fix-issue.md`:
```markdown
Please analyze and fix the GitHub issue: $ARGUMENTS.

Follow these steps:
1. Use `gh issue view` to get the issue details
2. Understand the problem described in the issue
3. Search the codebase for relevant files
4. Implement the necessary changes to fix the issue
5. Write and run tests to verify the fix
6. Ensure code passes linting and type checking
7. Create a descriptive commit message
8. Push and create a PR

Remember to use the GitHub CLI (`gh`) for all GitHub-related tasks.
```

Usage: `/project:fix-issue 1234`

### Personal Commands
Store in `~/.claude/commands/` for commands available across all projects.

### Command Features
- [ ] Use `$ARGUMENTS` variable for parameters
- [ ] Add front matter for allowed tools:
```markdown
---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*)
description: Create a git commit with context
---
```
- [ ] Include dynamic context with `!` prefix: `!`git status``

---

## 6. Hooks for Automation

### Hook Events
| Event | When It Fires |
|-------|--------------|
| `PreToolUse` | Before Claude executes an action |
| `PostToolUse` | After Claude completes an action |
| `UserPromptSubmit` | When you submit a prompt |
| `SessionStart` | When Claude starts |
| `SessionEnd` | When session terminates |
| `Stop` | When Claude finishes responding |
| `Notification` | When Claude sends an alert |
| `PermissionRequest` | When Claude requests permission (v2.0.45+) |

### Example: Auto-Format on File Edit
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
          }
        ]
      }
    ]
  }
}
```

### Example: Block Sensitive Files
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c \"import json, sys; d=json.load(sys.stdin); p=d.get('tool_input',{}).get('file_path',''); sys.exit(2 if any(x in p for x in ['.env','package-lock.json','.git/']) else 0)\""
          }
        ]
      }
    ]
  }
}
```

### Example: Desktop Notifications
```json
{
  "hooks": {
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' 'Awaiting your input'"
          }
        ]
      }
    ]
  }
}
```

### Hook Exit Codes
| Code | Meaning |
|------|---------|
| 0 | Success, continue |
| 1 | Non-blocking error |
| 2 | Block action (PreToolUse/PermissionRequest) |
| 3 | Deferred execution |

---

## 7. Parallel Development with Git Worktrees

### Why Worktrees?
- Run multiple Claude instances simultaneously
- Each worktree has isolated file state
- No context switching—maintain deep understanding per task
- All worktrees share Git history and remotes

### Setup Worktrees
```bash
# Create worktree for a feature
git worktree add ../my-project-feature-auth feature/auth

# Create worktree from main
git worktree add ../my-project-refactor -b refactor/cleanup

# List worktrees
git worktree list

# Remove worktree
git worktree remove ../my-project-feature-auth
```

### Parallel Workflow
1. [ ] Create worktrees for each task/feature
2. [ ] Open each worktree in separate terminal/IDE window
3. [ ] Run `/init` in each worktree session
4. [ ] Assign distinct tasks to each Claude instance
5. [ ] Cycle through to check progress and approve permissions
6. [ ] Cherry-pick completed work into main branch

### Tools for Worktree Management
- `claude-wt` script: Auto-creates worktrees and launches Claude Code
- GitButler: Manages commits/branches across parallel sessions
- `/worktree` custom command: Automates worktree creation

### Best Practices
- [ ] Limit to 2-3 active worktrees (review bandwidth is the bottleneck)
- [ ] Give sessions descriptive names
- [ ] Use draft PRs for Claude-generated changes
- [ ] Clean up worktrees after merging

---

## 8. CLI Usage Patterns

### Basic Usage
```bash
# Start interactive session
claude

# Start with initial prompt
claude "How does turnManager.js work?"

# Headless mode (non-interactive)
claude -p "How many files are in this project?"

# Resume previous session
claude --resume
```

### Piping & Chaining
```bash
# Pipe input to Claude
cat data.csv | claude -p "Who won the most games?"

# Chain with other CLIs
git diff | claude -p "Explain these changes"

# Output to file
claude -p "Generate API docs" > api-docs.md
```

### Useful Flags
| Flag | Purpose |
|------|---------|
| `-p` | Print mode (headless, non-interactive) |
| `--resume` | Resume previous session |
| `--from-pr 123` | Resume session linked to PR |
| `--verbose` | Debug mode (shows detailed output) |
| `--mcp-debug` | Debug MCP connections |
| `--allowedTools` | Specify allowed tools for this run |
| `--output-format json` | JSON output for scripting |
| `--dangerously-skip-permissions` | Skip all permission prompts |

### Shell Integration
```bash
# Run shell commands in Claude session with ! prefix
! git status

# Bypass conversational mode for direct execution
!npm run test
```

---

## 9. MCP Server Integration

### Adding MCP Servers
```bash
# Add MCP server globally
claude mcp add -s user playwright npx @playwright/mcp@latest

# Add to project
claude mcp add puppeteer npx @anthropic/mcp-puppeteer
```

### Configuration in .mcp.json
Check into git for team-wide access:
```json
{
  "servers": {
    "puppeteer": {
      "command": "npx",
      "args": ["@anthropic/mcp-puppeteer"]
    },
    "sentry": {
      "command": "npx",
      "args": ["@sentry/mcp-server"]
    }
  }
}
```

### Popular MCP Servers
- [ ] Puppeteer (browser automation)
- [ ] Sentry (error monitoring)
- [ ] GitHub (issues, PRs)
- [ ] Slack (notifications)
- [ ] Database connectors
- [ ] Figma (design integration)

### Document MCP Usage in CLAUDE.md
```markdown
### Slack MCP
- Posts to #dev-notifications channel only
- Use for deployment notifications and build failures
- Do not use for individual PR updates
```

---

## 10. Session Management

### Session Commands
| Command | Action |
|---------|--------|
| `/resume` | Open session picker |
| `/clear` | Clear current context |
| `/compact` | Manually compact context |
| `/help` | Show all commands |

### Session Picker Features
- Navigate with arrow keys
- Press `R` to rename session
- Sessions grouped by git branch
- Forked sessions grouped under root

### Best Practices
- [ ] Name sessions descriptively
- [ ] Use `/clear` when starting new tasks
- [ ] Sessions are stored per project directory
- [ ] Use `--fork-session` for experimental branches

---

## 11. Keyboard Shortcuts & Navigation

### Essential Shortcuts
| Shortcut | Action |
|----------|--------|
| `Escape` | Stop Claude (not Ctrl+C!) |
| `Escape` (twice) | Show previous messages |
| `Up/Down` | Navigate command history |
| `Ctrl+R` | Reverse search history |
| `Ctrl+V` | Paste images (not Cmd+V) |
| `Shift+Drag` | Reference files properly |

### Terminal Setup
- [ ] Run `/terminal-setup` to install Shift+Enter binding
- [ ] Configure line break behavior in terminal settings
- [ ] Enable Vim mode with `/vim` if preferred

---

## 12. Advanced Tips & Tricks

### Prompting Techniques
- [ ] Use "think hard" or "think carefully" for complex problems
- [ ] Ask Claude to "explore, plan, code, commit" for features
- [ ] Request plans before implementation for large changes
- [ ] Use `/plan` or plan mode for complex tasks

### Using External Tools
- [ ] Tell Claude to use CLI tools (`gh`, `aws`, `gcloud`, `sentry-cli`)
- [ ] CLI tools are more context-efficient than APIs
- [ ] Let Claude learn new tools with `--help`: "Use 'foo-cli --help' to learn foo, then solve X"

### Voice Input
- [ ] Use voice messages for faster input
- [ ] Whisper into earphones for quiet environments
- [ ] Speaking is often faster than typing long prompts

### Container Isolation
Run Claude Code in containers for:
- Long-running experimental tasks
- Risky operations without affecting host
- Orchestrating multiple AI CLIs

### Debugging
- [ ] Use `--verbose` flag for debugging
- [ ] Use `HTTPS_PROXY` to inspect raw traffic
- [ ] Check `~/.claude/projects/` for session transcripts
- [ ] Review logs in JSONL format

### Token Efficiency
- [ ] Use Skills (loaded on-demand) instead of stuffing CLAUDE.md
- [ ] Keep CLAUDE.md focused and minimal
- [ ] Clear context frequently
- [ ] Use specific, detailed prompts

---

## 13. Common Pitfalls to Avoid

### ❌ Don't Do This
- [ ] Don't use `Ctrl+C` to stop Claude (exits entirely)—use `Escape`
- [ ] Don't expect `Cmd+V` to paste images—use `Ctrl+V`
- [ ] Don't drag files without `Shift`—they'll open in new tab
- [ ] Don't let context grow unbounded—use `/clear`
- [ ] Don't run 5+ parallel agents—review bandwidth is the bottleneck
- [ ] Don't stuff CLAUDE.md with code style rules—use linters
- [ ] Don't forget to run `/init` in new worktrees
- [ ] Don't commit `.claude/settings.local.json`

### ✅ Do This Instead
- [ ] Revert often with `git revert` if Claude makes unwanted changes
- [ ] Create draft PRs for Claude-generated code
- [ ] Use deterministic tools (linters, formatters) over LLM suggestions
- [ ] Script worktree bootstrap early in adoption
- [ ] Start with 2 parallel agents, not 5
- [ ] Monitor token usage and API costs

---

## Quick Reference Card

### Daily Workflow
```bash
# Start session
claude

# Initialize project memory
/init

# Clear context for new task
/clear

# Resume previous session
/resume

# Check permissions
/permissions
```

### Key Files
```
~/.claude/
├── settings.json          # Global settings
├── CLAUDE.md              # Global memory
└── commands/              # Personal commands

.claude/
├── settings.json          # Project settings (committed)
├── settings.local.json    # Local settings (git-ignored)
└── commands/              # Project commands

./CLAUDE.md                # Project memory (committed)
.mcp.json                  # MCP server config (committed)
```

### Emergency Commands
```bash
# Skip all permissions (trusted environments only)
claude --dangerously-skip-permissions

# Update Claude Code
claude update

# Check version
claude --version
```

---

*Last updated: February 2026*
*Based on community best practices and official Anthropic documentation*
