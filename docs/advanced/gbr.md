# Pair a phone with Build Remote Agent

`mini` can use **Build Remote Agent** as a pairing device: the paid iOS/Android
app spectates (and can inject into) this desktop agent through the free MIT
`gbr-agent`. Phone and PC never open ports to each other.

Do **not** thicken the 100-line agent class for this. Run `mini` as usual; pair
in a second terminal.

Website: https://grokbuildremote.com/
Agent: https://github.com/LinespottingOrg/GrokBuildRemote-Agents (MIT)
Protocol: `gbr/1` · need agent **v0.6.0+**

Not affiliated with xAI or SpaceX.

## Install + pair

```bash
# macOS / Linux
curl -fsSL https://grokbuildremote.com/install.sh | bash
gbr-agent version          # must print v0.6.0 or newer
gbr-agent pair             # QR in browser + printed 8-char code
gbr-agent run              # leave running
```

```powershell
# Windows
irm https://grokbuildremote.com/install.ps1 | iex
gbr-agent version
gbr-agent pair
gbr-agent run
```

Phone: open Build Remote Agent → **Scan QR from computer** (or type the 8-char
code). Sessions appear in the app. **Unpair** in Settings before changing PCs.
Force-close is not enough.

## Attach while mini runs

Terminal 1: `mini` (or `mini-e i` for the Textual inspector).

Terminal 2: `gbr-agent pair && gbr-agent run`.

After `gbr-agent run`:

- HTTP Bot API: `http://127.0.0.1:8788`
- MCP stdio: clone the agent repo and run `node mcp/gbr-mcp/bin/gbr-mcp.js`

```bash
curl -sS http://127.0.0.1:8788/health
curl -sS http://127.0.0.1:8788/v1/sessions
```

Phone is spectator. Orchestration stays on `mini` (or a Grok bot / Claude Cowork
talking to the same Bot API). The Textual inspector is not a pair surface.

Do not commit mailbox keys. Phone **Settings → Bot API** is the only place the
relay key is copied.
