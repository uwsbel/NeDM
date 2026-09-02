# File sync between the boxes

**Verified:** 2026-09-02 · **Owner:** Kyle

[`remote-control.md`](remote-control.md) is how agents reach the boxes. This is
how bytes move between them. The two are separate systems and neither implies
the other: Remote Control carries text between agent sessions and has no
machine-to-machine file transport at all.

## What carries what

| Content | Mechanism | Why |
|---|---|---|
| This repo | **git** | Already the project rule. Reviewable, atomic, has history |
| Checkpoints, datasets | `rsync`, on demand | Too large, and per-machine by design |
| Notes, scratch, configs, small shared files | **Syncthing** | Continuous, no ceremony |

**Do not put the repo inside the Syncthing folder.** A `.git` directory synced
between machines with work happening on both sides produces conflict files and
can corrupt index state. Rule 3 of [`README.md`](README.md) stands unchanged:
commit, push, `git pull --ff-only` on the far side.

## The Syncthing setup

Syncthing was chosen for the same reason as Remote Control: it needs no inbound
port. It does its own NAT traversal and falls back to relays.

| | |
|---|---|
| Version | v2.1.3 on all three machines |
| Folder ID | `sbel-shared` (must match exactly on every device, or they will not link) |
| Path | `~/sync/sbel` on all three |
| Type | `sendreceive` |
| Install | Homebrew + launchd on the Mac; `~/.local/bin` + systemd user unit on the boxes |

The install on the compute boxes needs no `sudo`: Syncthing is a single static
binary. This matters, because a `sudo` password prompt on a headless box hangs
forever.

Device IDs are recorded in Kyle's local notes, not here. Recover one with
`syncthing device-id` (a subcommand in v2, **not** `--device-id`, which errors).

The folder was deliberately started **empty** rather than pointed at existing
directories on each box, so that nothing was merged by inference. Content gets
moved in deliberately.

### What must stay out

`~/sync/sbel/.stignore` excludes build artifacts, Python caches, OS cruft, and
large media. Keep it that way. Anything in `artifacts/`, any raw frame store,
and any rendered video belongs to `rsync`, not here. See rule 2 of
[`README.md`](README.md) for the data-movement convention.

## Topology, and why it matters for throughput

Measured 2026-09-02, with the Mac on `192.168.1.66`:

| Leg | State |
|---|---|
| Mac ↔ `dorm-pc` | direct, LAN |
| Mac ↔ `sbel-pc` | relayed |
| `dorm-pc` ↔ `sbel-pc` | **relayed, always** |

Which of the Mac's legs is direct depends on which network the laptop is on that
day, so measure it rather than assuming: `syncthing cli show connections`, and
read `isLocal`.

The fixed constraint is the last row. The two compute boxes are on different
networks and can only reach each other through a public Syncthing relay. Relay
traffic is end-to-end encrypted, so the operator cannot read it, but throughput
is limited and the metadata is visible. **Do not plan to move anything large
between the two boxes over this path.** If that becomes necessary, the answer is
a mesh VPN such as Tailscale, which would give both boxes a direct WireGuard
route and let Syncthing and `rsync` use it.

## Gotchas

1. **A file written just after a folder is created will not sync, for up to an
   hour.** The filesystem watcher is not live yet and `rescanIntervalS` is 3600,
   so the file never enters the index: Syncthing reports `localFiles=0` and
   cheerfully says every peer is 100% complete, because the shared set really is
   empty. This looks exactly like a slow or broken link and is neither. Force a
   scan:

   ```bash
   KEY=$(sed -n 's:.*<apikey>\(.*\)</apikey>.*:\1:p' <syncthing config.xml>)
   curl -s -X POST -H "X-API-Key: $KEY" \
     "http://127.0.0.1:8384/rest/db/scan?folder=sbel-shared"
   ```

   Cost when found: about 50 minutes of believing the relay was slow.
2. **`completion=100%` is not proof that sync works.** Check `localFiles` and
   `globalFiles` in `/rest/db/status` too. Three machines agreeing perfectly on
   an empty set reports as fully synced.
3. **The GUI binds `127.0.0.1:8384` only.** On the headless boxes there is no
   browser, so use `syncthing cli config ...` or the REST API.
4. Syncthing services are systemd **user** units like `claude-rc`, so they share
   the same `loginctl enable-linger` dependency. Without it, both stop at logout.
