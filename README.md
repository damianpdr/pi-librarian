# pi-librarian (pi extension)

A GitHub-focused librarian subagent for pi. Inspired by Amp's librarian.

## Demo

[🎬 Watch demo](https://github.com/damianpdr/pi-librarian/releases/download/v0.1.0/demo.mp4)

## What it adds

- `librarian` tool (GitHub-focused subagent)
- Supporting repo analysis tools:
  - `read_github`
  - `search_github`
  - `commit_search`
  - `diff`
  - `list_directory_github`
  - `list_repositories`
  - `glob_github`

## How it works

`librarian` spawns an isolated `pi` subprocess (`--mode json --no-session`) with only the GitHub tools above enabled, then returns the subagent's final answer.

## Requirements

- GitHub CLI (`gh`) installed
- Authenticated GitHub session:

```bash
gh auth login
```

## Location

This extension is project-local and auto-discovered from:

- `.pi/extensions/pi-librarian/index.ts`

Use `/reload` in pi after edits.

## Notes

- Current port targets GitHub only (no Bitbucket Enterprise path yet).
- Tool output is JSON-formatted text for reliable downstream parsing.
