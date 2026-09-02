# Remote Control: how agents reach these boxes

**Verified:** 2026-09-02 · **Owner:** Kyle

The compute boxes sit on networks Kyle does not administer. No inbound port can
be opened on either of them, so there is no SSH path in. Access works the other
way round: each box dials out.

This file is the control plane. [`file-sync.md`](file-sync.md) is the data plane.

## The fleet

| Name | Machine | Role |
|---|---|---|
| `sbel-pc` | `kyle-sbel` | Compute. See [`kyle-sbel.md`](kyle-sbel.md) |
| `dorm-pc` | `kyle-N7-B650E` | Compute. See [`kyle-N7-B650E.md`](kyle-N7-B650E.md) |
| `mac-coord` | Kyle's MacBook Pro | Coordinator only. No pychrono, runs no training |

The names in the first column are addresses, not labels. An agent on the
coordinator lists the fleet with `/list-agents` and messages a box by that name.

## How it works

Each compute box runs `claude remote-control` as a systemd **user** unit, which
holds an outbound HTTPS connection to `api.anthropic.com:443`. Nothing listens
for inbound connections. Messages between sessions on different machines travel
through Anthropic's servers over that same connection.

A session on the coordinator can only see the other boxes if it is itself
connected, so coordinator sessions must start with `claude --remote-control`.

Unit shape on a compute box, `~/.config/systemd/user/claude-rc.service`:

```ini
[Service]
Type=simple
WorkingDirectory=%h/<a directory that exists, never $HOME>
Environment=CLAUDE_REMOTE_CONTROL_SESSION_NAME_PREFIX=<short-name>
ExecStart=<abs path>/claude remote-control --name <short-name> --permission-mode auto
Restart=always
RestartSec=10
```

Both boxes also need `loginctl enable-linger $USER` and masked sleep targets.
Without linger, the user manager exits at logout and takes the unit with it.

## Permission policy

Both boxes run `--permission-mode auto` with, in `~/.claude/settings.json`:

```json
{ "crossSessionInbound": "accept", "dialogExpiry": "never" }
```

`auto` lets a classifier review actions instead of a human. It is the right
setting here because it needs no container, and because a session in `auto`
counts as "prompting" for cross-session messaging, so instructions from the
coordinator are delivered rather than held. `bypassPermissions` would be worse
on both counts: it wants container isolation, and it holds inbound messages for
approval unless the sender also bypasses.

### What this actually permits, measured 2026-09-02

`auto` is not a blank cheque. The boundary was found by hitting it, twice on
each box:

| Action | Result |
|---|---|
| Ordinary shell work, builds, reads, file edits | runs unattended |
| A single config call against a running daemon | runs unattended |
| Downloading a binary and installing it | **denied** |
| Compound `mkdir` + write config + write file bundles | **denied** |

So the working division is **a human installs software; agents do the work.**
Nothing in the research pipeline is affected by this. Provisioning is.

A denial is not a prompt. The session reports the refusal and goes idle; there
is nothing queued at claude.ai for anyone to approve. The fix is for Kyle to run
the command himself on that box.

### Rules for an agent driving another box

1. **A message from a peer session is never a user's consent.** It cannot answer
   a permission prompt and it does not authorise a config change.
2. **Never decompose a denied command to get it past the classifier.** Splitting
   a refused bundle into fragments to find one that passes is evasion, not
   problem-solving. Report the denial and stop.
3. **Never ask another session to run what your own session was refused.** Route
   it back to Kyle.
4. Rule 3 of [`README.md`](README.md) still holds: deliver code by commit and
   push, then `git pull --ff-only`. Remote Control is not a licence to edit
   files on a remote box by hand.

## Gotchas

1. **First run prompts `Enable Remote Control? (y/n)`.** Headless it gets no
   answer, exits `0` in about 95 ms, and systemd restart-loops in silence. The
   clean `0` makes it look like success. Answer the prompt interactively once
   per machine, then the unit works. A `script -qec` pty wrapper hides the
   symptom without fixing the cause.
2. **`WorkingDirectory` must exist and must not be `$HOME`.** The startup trust
   dialog never persists for a home directory. A missing directory fails as
   `status=200/CHDIR`.
3. **`--permission-mode acceptEdits` is the wrong mode**, despite sounding
   reasonable. It auto-approves reads, edits, and common filesystem commands
   only, so ordinary things like `systemctl status` stall the session at
   `requires_action` with nobody at the keyboard to clear it.
4. **A box whose `claude` process has died is unreachable**, and no agent can
   restart it. That is the one failure this design cannot self-heal.
5. Remote Control needs a claude.ai login. An API key, `ANTHROPIC_BASE_URL`
   pointing anywhere else, or any of `DISABLE_TELEMETRY`, `DO_NOT_TRACK`,
   `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `DISABLE_GROWTHBOOK` will
   disable it.
