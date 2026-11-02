# MCP servers (MESA, DuckLake, iRODS, Formation)

The repository ships a project-scope [`.mcp.json`](../.mcp.json) so that Claude Code, opened in
this directory, gets the [MESA](https://idss-mesa.github.io/docs/) servers for the CyVerse Data
Store and Discovery Environment. It holds no secrets: every credential comes from your own
machine.

| Name | What it is | Runs | Sign-in |
|---|---|---|---|
| `mesa-mcp` | iRODS Data Store tools (`ds_*`), OBO/OLS ontology AVUs, DataCite, and DuckLake metadata history (`mesa_ducklake_*`) | locally, stdio | anonymous, or your `~/.irods` |
| `irods` | the reference Go server for the Data Store (`ds_*`) | locally, stdio | anonymous, or `config-stdio.yaml` |
| `formation` | CyVerse's hosted Discovery Environment server: launch apps, follow analyses, read and write Data Store files | hosted, `https://de.cyverse.org/formation/mcp` | browser sign-in with your CyVerse account |

**MESA DuckLake is not a separate server.** `mesa-ducklake` is a library that `mesa-mcp`
imports, and its tools appear as `mesa_ducklake_*` inside `mesa-mcp`. It records the history of
AVU (metadata) changes; it is separate from this project's own DuckLake of results.

## 1. Install the local servers

The two stdio entries point at the MESA installer's default layout under `~/.mesa`. Install
once per machine (needs `git`, `curl`, and Go 1.25 or newer for `irods`; `uv` is installed if
missing):

```bash
curl -fsSL https://raw.githubusercontent.com/idss-mesa/docs/main/install.sh | bash -s -- --for claude
```

The installer also registers the same three servers at Claude Code **user** scope. Either keep
those and ignore this file, or remove the user-scope copies so `claude mcp list` does not warn
about a name defined in two scopes:

```bash
claude mcp remove mesa-mcp -s user
claude mcp remove irods -s user
```

Leave the user-scope `formation` if you have already signed in to it; removing it also removes
the sign-in. Without Go, add `--no-go` and the `irods` entry will fail to start, which is
harmless.

Claude Code asks once to approve the servers in `.mcp.json` the first time you open the project.

## 2. Credentials (never in this repository)

- **mesa-mcp and irods** read public collections anonymously. To act as yourself, run `iinit`
  once (CyVerse iCommands); `mesa-mcp` then reads `~/.irods/irods_environment.json` and
  `~/.irods/.irodsA`. For `irods`, put `irods_user_name` and `irods_user_password` in
  `~/.mesa/repos/irods-mcp-server/config-stdio.yaml`, which is outside this repository.
- **DuckLake metadata history** is off unless you set a catalog before starting Claude Code,
  for example `export MESA_MCP_DUCKLAKE__CATALOG_DSN=duckdb://$HOME/.mesa/avu-history.duckdb`.
  `.mcp.json` passes the variable through and leaves it blank otherwise.
- **formation** has no password or token. Run `/mcp` in Claude Code, choose `formation`, then
  **Authenticate** (or `claude mcp login formation`, with `--no-browser` on a remote machine such
  as a VICE app). Sign in with your CyVerse user account.

Do not add `MESA_MCP_IRODS__PASSWORD` or any other secret to `.mcp.json`; it is committed.

## 3. Check

```bash
claude mcp list     # mesa-mcp, irods, formation; formation needs authentication until step 2
```

Then ask: *"Ping the CyVerse Data Store and list /iplant/home/tswetnam/neon-beetle-seg."*

## claude.ai, Claude Desktop, and cloud sessions

Connectors on claude.ai run in Anthropic's cloud, so only the hosted servers can be added
there; `mesa-mcp` and `irods` above run on your own machine. Add them under
**Customize > Connectors > + Add > Add custom connector**:

| Connector | URL | Authentication | OAuth client |
|---|---|---|---|
| CyVerse Formation | `https://de.cyverse.org/formation/mcp` (exactly, no trailing slash) | Sign in now | **Register automatically** |
| CyVerse Data Store (public, read-only) | `https://mcp-public.cyverse.ai/mcp` | No sign in | — |

A connector added on claude.ai is also available in Claude Code when you are signed in with a
claude.ai subscription; a `formation` entry in `.mcp.json` with the same URL takes precedence and
hides the connector.

Claude Code cloud sessions also need `de.cyverse.org`, `mcp-public.cyverse.ai`, and
`data.cyverse.org` allowed in the cloud environment's network access.

Source: <https://idss-mesa.github.io/docs/> (quickstart, credentials, Claude Code, and
claude.ai pages, read 2026-11-02).
