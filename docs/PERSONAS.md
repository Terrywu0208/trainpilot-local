# Who TrainPilot-Local is for — four users, one promise

**The promise: training a model should feel like asking for one.** You
describe what you want in plain language; the agent plans it, shows its
work, asks before anything irreversible, repairs what it safely can,
and only calls a run a success after an independent check has verified
the checkpoint. Everything runs locally on CPU — no cloud account, no
GPU, no API key, no cost, and your text never leaves the machine.

This document names the four kinds of users the platform is designed
around, what hurts for each of them today, and the exact journey each
one takes here. Every journey below is executed end-to-end in the
project's verification drives (see the pretraining flow in
OPERATIONS.md).

---

## 1. The first-time learner — "I want to see a language model learn"

**Who.** A student or self-learner meeting language models for the
first time. Laptop only, no GPU, no budget, and no desire to learn
YAML, distributed training, or cluster tooling before seeing a single
result.

**What hurts today.** Every tutorial starts with infrastructure:
environments, CUDA, config files. The first "success" is hours away,
and when something fails the error message assumes knowledge they do
not have yet.

**Their journey here.**
1. Open the console, click the preset **"Pretrain a tiny LM from
   scratch"** — one click, nothing to configure.
2. Approve the first-launch card (Gate 1) — the only decision they
   must make, explained in plain words.
3. Watch the left-hand diagram walk the agent's steps while the chat
   narrates: training started, loss falling, and finally a verdict
   card with the perplexity number — *before training: 431.98 (the
   untrained model is guessing); after: 211.59, roughly half*.
4. Read the model's own generated sample sentence in the verdict. The
   payoff is not a metric; it is *"the model wrote that?"*

**Success looks like:** a verified checkpoint, a perplexity that
visibly dropped, and one generated sentence they can show a friend —
inside three minutes, on a laptop, at zero cost.

**Carried by:** the pretrain preset, the live process diagram, the
plain-language chat, real from-scratch training on the bundled
`tinytales` corpus.

## 2. The indie builder — "I have my own text; give it a voice"

**Who.** A developer or hobbyist with a pile of their own writing —
notes, documentation, a niche community's posts — who wants a small
domain-flavored language model to experiment with, long before any
question of serving or scale.

**What hurts today.** The data plumbing: tokenizers, splits, "is my
corpus even big enough?", silent tooling failures. The worst tools
smile and train on the wrong data.

**Their journey here.**
1. Type what they mean: *"Pretrain a language model from scratch on
   the tinytales corpus with 2 CPU workers."* The corpus is named in
   the sentence, not in a config file.
2. If they name a corpus the local demo does not have, the agent says
   so — plainly, in the chat — and stops. It never quietly substitutes
   different data (the same honesty rule that makes an unknown
   fine-tuning dataset fail instead of silently becoming SST-2).
3. If a corpus is too small to learn from, the run fails with the
   reason stated ("corpus too small … below the 1500-token minimum")
   instead of producing a confident-looking useless model.
4. On success, the generated samples use the corpus's own vocabulary —
   the proof that *their* text is what got trained.

**Success looks like:** their words in, a verified model out, and
honest refusals — never silent substitution — when the data is not
there or not enough.

**Carried by:** corpus naming in plain language, `CORPUS_NOT_AVAILABLE`
/ `CORPUS_TOO_SMALL` honesty paths, sample generation from the loaded
checkpoint.

## 3. The ML engineer — "Stop making me babysit runs"

**Who.** A practitioner who runs many small experiments and has been
burned by silent config drift, unreproducible results, and jobs that
die at 3 a.m. for a reason nobody wrote down.

**What hurts today.** Babysitting. A diverged run, a bad split, a
wrong flag — each costs a human round-trip, and the fix is rarely
recorded anywhere durable.

**Their journey here.**
1. Submit a run and walk away. Every run carries a spec hash; the
   exact resolved config is inside the checkpoint (`training_args.json`),
   so any result can be reproduced bit-for-bit.
2. Something goes wrong — say the request asked for a learning rate
   of 5 and the tiny LM's loss goes NaN on attempt 1. The critic
   diagnoses it from the logs ("learning rate far too high for the
   tiny LM"), the repair router applies the fix (learning rate 0.1)
   as a new spec, and attempt 2 succeeds — all inside the attempt cap,
   all in the audit log.
3. If a failure has no safe repair, the agent gives up *and says why*
   (reason codes like `no_safe_repair`, `attempts_exhausted`) instead
   of looping or papering over it. The failure — diagnosis, fix tried,
   outcome — is written to failure memory, and the next run's planner
   is warned not to repeat it.
4. The verdict is never self-reported: an independent verifier checks
   the checkpoint exists, loads, honors the original request, and
   actually improved (perplexity below 90% of the untrained baseline
   for pretraining; the verified accuracy for fine-tuning).

**Success looks like:** unattended runs that either finish verified or
fail with a recorded, specific reason — and a failure memory that
makes the next run smarter.

**Carried by:** the diagnosis → repair loop with an attempt cap,
spec-hash reproducibility, failure memory, the independent verifier,
the audit log.

## 4. The teacher / presenter — "Three minutes, live, no surprises"

**Who.** An instructor demoing agentic training to a class, or a
candidate demoing this project in an interview. The audience asks
questions the script did not anticipate.

**What hurts today.** Live-demo risk: venue Wi-Fi, a cloud console
that needs a login, a job that takes twenty minutes, a failure with
no explanation. Any of these ends the demo.

**Their journey here.**
1. Everything is local and offline — the demo cannot be killed by a
   network. The stub model is deterministic, so the canonical run
   produces the same verified result every time (fine-tune: accuracy
   0.8771; pretrain: the same perplexity drop).
2. The audience watches two panes: the diagram shows *how* the agent
   works; the chat shows *what it decided* — including the approval
   gates, which make the safety story visible instead of asserted.
3. Follow-up questions become follow-up runs in the same session:
   *"Same corpus, but train for 6 epochs this time"* — the agent
   carries the session context, changes exactly what was asked, and
   the perplexity improves again, live.
4. If time runs out mid-run, the Replay view re-plays any finished
   run step by step, with the watermark making clear it is a replay.

**Success looks like:** a three-minute, network-independent demo where
every number on screen was verified by the system itself, and
audience follow-ups are answered by doing them, not describing them.

**Carried by:** full offline operation, deterministic results, session
follow-ups, the two-pane console, replay.

---

## The five painless rules (all four users)

1. **Plain language in.** No YAML, no flags, no cluster vocabulary is
   required to start. The structured fields exist, but they are
   pre-filled from your sentence.
2. **One click for the canonical journeys.** Presets exist for the
   fine-tune, the follow-up, and the from-scratch pretrain.
3. **Repairs are automatic — and explained.** What the agent can fix
   safely, it fixes; the chat says what was wrong and what changed.
4. **Honest failure beats fake success.** Missing corpus, impossible
   hardware (a GPU request on a CPU demo), too-small data: the answer
   is a clear no, never a silent substitution.
5. **Verified before claimed.** A run is a success only when an
   independent check — checkpoint exists, loads, honors the request,
   metric actually improved — has passed. Otherwise the UI says
   UNVERIFIED, in those words.
