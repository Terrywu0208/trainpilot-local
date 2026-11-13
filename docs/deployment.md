# Deployment — Putting TrainPilot-Local on a Public URL

The professor's presentation checklist requires the demo to be **hosted
on the cloud**: a public URL that anyone in the classroom can open, with
questions typed into the live demo during Q&A. A laptop-only
`localhost:8000` does not satisfy this. There are two supported routes;
pick one **before the Nov 20 UI freeze** and run the acceptance check at
the bottom of this document.

How the pieces sit: the FastAPI webapp imports the agent in-process, and
the agent talks to the Kind/KubeRay cluster on the same host. Whatever
serves the public URL must therefore be the machine that runs the whole
stack (webapp + agent + Kind), or a tunnel that forwards to it.

## Route A — Cloud VM, full stack (recommended for presentation day)

The whole stack runs on one cloud VM; the laptop is not involved.

1. Provision a VM with at least 4 vCPU and 8 GB RAM (the Kind cluster
   runs 3 nodes plus the KubeRay operator and one RayJob at a time; see
   the resource sizing in the Section B/C design). Any provider's free
   tier or trial credit works — the project itself spends nothing: the
   default LLM provider is the offline stub.
2. Install Docker, `kind`, `kubectl`, and Helm on the VM; clone this
   repository.
3. Create the cluster and install the operator from the checked-in
   manifests: `kind create cluster --config infra/kind/cluster.yaml`,
   then apply the namespace/RBAC/quota manifests under `infra/` and
   install the KubeRay operator Helm chart (chart 1.3.0, Ray 2.41.0)
   with the checked-in values.
4. Build and load the training image (digest-pinned, model and dataset
   pre-downloaded so no run needs the network).
5. Start the webapp bound to localhost on the VM:
   `.venv/bin/uvicorn webapp.main:app --port 8000` (under `systemd` or
   `tmux` so it survives the SSH session).
6. Publish it: either point a cloud load balancer / reverse proxy with
   HTTPS (for example Caddy with automatic Let's Encrypt certificates)
   at `127.0.0.1:8000`, or open port 8000 in the VM firewall and use the
   provider's public DNS name. HTTPS is preferred; the approval buttons
   carry no credentials, but the URL will be projected in a classroom.
7. Run the acceptance check below from a phone on mobile data.

Trade-offs: self-contained and immune to laptop sleep/network problems;
costs setup time (image and scenarios must be moved and re-verified on
the VM) and the VM must stay up through the presentation.

## Route B — Tunnel to a local stack (zero setup, one weak point)

The stack stays on the presenter's laptop; a tunnel exposes
`localhost:8000` at a public URL.

1. Run the full local stack (Kind cluster + operator + webapp on
   port 8000) exactly as for local development.
2. Expose it with one of:
   - Cloudflare Tunnel: `cloudflared tunnel --url http://localhost:8000`
     (free, gives a `*.trycloudflare.com` URL), or
   - ngrok: `ngrok http 8000`, or
   - Tailscale Funnel: `tailscale funnel 8000`.
3. Keep the laptop plugged in, awake, and on a network that will not
   change mid-class; pin the tunnel URL in the slide deck.
4. Run the acceptance check below, including one full Replay.

Trade-offs: zero migration and the exact machine the demo was rehearsed
on; but the laptop is a single point of failure — sleep, a Wi-Fi drop,
or an out-of-memory kill during class takes the live demo down with it.
The pre-recorded video and the Replay of an archived run are the backup
either way.

## Acceptance check (both routes, non-negotiable)

From a device that has never touched this project, on a different
network (for example a phone on mobile data):

1. Open the public URL. The Run Console loads with the request form.
2. Start a run (or play a full Replay of an archived run). Events stream
   in, attempt cards appear.
3. Type a follow-up request in the same session ("Same settings, but
   epochs = 3") and confirm the previous run's summary is shown as
   session context.
4. Trigger Gate 1 and Gate 2 once; both must pause the run until a
   button is clicked, and both decisions must appear in the run's
   `audit_log.jsonl`.

If any step fails on the foreign device, the deployment is not done —
fix it before the recording freeze (Nov 23, 12:00), not in the classroom.

## Notes

- **Zero cost:** both routes work on free tiers / free tunnel plans, and
  the agent's default LLM provider is the deterministic offline stub, so
  a public demo run spends no API money.
- The public instance exposes the approval gates to anyone with the
  URL. For a classroom demo this is acceptable for one session; stop the
  tunnel / VM after the presentation, and never expose a deployment that
  is connected to a real cluster with real credentials.
