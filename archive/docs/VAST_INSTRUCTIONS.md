# Vast.ai GPU Instance — Complete Setup & Operations Guide

This document covers everything needed to rent, configure, and work with GPU instances on vast.ai. It is written to be generic and reusable across projects — specific examples (like GRPO training) are clearly marked as examples.

---

## 0. Pre-Flight Checklist

**Before renting anything, answer these questions.** They determine what GPU you need, how much disk space, and which API keys to set up. Copy this section into your notes and fill it out each time.

- **What is the project?** (e.g., "Fine-tune Qwen2.5-1.5B on creative writing data using GRPO")
- **What model(s) will you load?** List model names and approximate sizes. A 7B parameter model in fp16 needs ~14GB VRAM. A 1.5B model needs ~3GB. If you're doing LoRA, you need less. If full fine-tuning, you need more.
- **How many GPUs do you need?** Single GPU is fine for models under ~13B with LoRA. For full fine-tuning of 7B+ or running multiple models simultaneously, consider 2+ GPUs.
- **What GPU type?** H100 SXM is the fastest and best for training. A100 80GB is a good cheaper alternative. For inference-only or small models, even an A6000 or 4090 works.
- **How much disk space?** Model checkpoints can be 5-10GB each. Training logs, datasets, and outputs add up. Budget at least 100GB, preferably 200GB+ for serious training runs.
- **What API keys do you need?** Common ones:
  - **Anthropic** — for LLM-as-judge evaluation, Claude Code
  - **Hugging Face** — for downloading gated models (Llama, Mistral, etc.)
  - **OpenAI** — if using GPT as evaluator or for other API calls
  - **Weights & Biases** — for experiment tracking (optional but recommended)
- **How long will this run?** Estimate hours. This determines your budget. Check `$/hr` on vast.ai before renting.
- **Do you have a repo to clone?** If yes, have the URL and any PATs ready.

---

## 1. Renting a GPU Instance

### Step-by-step on vast.ai

1. Go to [cloud.vast.ai](https://cloud.vast.ai) and log in.
2. Go to the team space called **Parsed** (top-left dropdown if you have multiple teams).
3. Click **Search** (or **Create**) in the left sidebar to browse available GPUs.
4. Set your filters:
   - **GPU Type:** Select your target (e.g., H100 SXM, A100 80GB).
   - **GPU Count:** How many GPUs per machine (e.g., 2x for multi-GPU training).
   - **Image:** Use a PyTorch image (e.g., `vastai/pytorch` or `pytorch/pytorch:2.x-cuda12.x`). These come with CUDA, PyTorch, and common ML libraries pre-installed.
   - **Disk Space:** Set minimum disk to your requirement (100GB+).
   - **Jupyter:** Make sure Jupyter is **enabled** — this gives you a web terminal that doesn't depend on SSH.
5. Review each offer:
   - **$/hr** — hourly cost. H100 SXM is typically $2-4/hr per GPU.
   - **DLP (Data Loss Protection)** — whether the host guarantees your data if the machine restarts. Prefer DLP-enabled hosts for multi-day runs.
   - **Reliability** — uptime percentage. Higher is better. Avoid hosts below 95%.
   - **Network speed** — matters for downloading large models. Look for 1Gbps+.
   - **Verification status** — "Verified" hosts are more reliable.
6. Click **RENT** on your chosen offer.

The instance will boot in 1-5 minutes. You'll see it appear under **Instances** in the left sidebar with a status indicator.

### Important: After renting

- Note the **Instance ID** (you'll need it for SSH and management).
- Note the **IP address and port** shown in the instance details.
- The Jupyter URL will appear as a clickable link — open it immediately to verify the instance is working.

---

## 2. Connecting with Cursor (SSH Setup)

Cursor is the primary IDE for this workflow. Setting up SSH properly is critical because you'll be editing code, browsing files, and running commands through it.

### 2.1 Get your SSH connection details

1. On vast.ai, go to **Instances** and find your running instance.
2. Click the **SSH** button/icon — it will show you a connection command like:
   ```
   ssh -p 12345 root@203.0.113.50
   ```
3. Note the **port** (e.g., 12345) and **IP address** (e.g., 203.0.113.50).

### 2.2 Set up your SSH config file

Open `~/.ssh/config` on your local Mac and add an entry for the instance:

```
Host vastai
    HostName 203.0.113.50
    Port 12345
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
```

**What each setting does:**
- `StrictHostKeyChecking no` — vast.ai instances have different host keys each time. Without this, SSH will refuse to connect after you rent a new instance.
- `UserKnownHostsFile /dev/null` — prevents stale host key errors from piling up.
- `ServerAliveInterval 60` — sends a keepalive packet every 60 seconds. This helps prevent the connection from dropping during idle periods.
- `ServerAliveCountMax 3` — if 3 keepalives fail in a row (3 minutes with no response), SSH gives up. This is better than hanging forever.
- `TCPKeepAlive yes` — enables TCP-level keepalives in addition to SSH-level ones.

**You must update HostName and Port every time you rent a new instance.** The IP and port change each time.

### 2.3 Add your SSH key to vast.ai

1. If you don't have an SSH key: `ssh-keygen -t ed25519 -C "your@email.com"`
2. Copy your public key: `cat ~/.ssh/id_ed25519.pub | pbcopy`
3. On vast.ai, go to **Settings** → **SSH Keys** and paste your public key.
4. New instances will automatically have your key installed.

### 2.4 Connect from Cursor

1. Open Cursor.
2. Open the Command Palette (`Cmd+Shift+P`).
3. Type "Remote-SSH: Connect to Host" and select it.
4. Select `vastai` (the name from your SSH config) or type the full connection string.
5. Cursor will connect and open a new window with the remote filesystem.
6. Open the folder where your project is (e.g., `/root/your-project`).

### 2.5 What to do when SSH won't reconnect

This is a known issue — **when you switch Wi-Fi networks (e.g., cafe to home), Cursor's SSH connection drops and often won't reconnect without manual intervention.**

**Fix procedure:**
1. Close the remote Cursor window entirely (not just disconnect — close it).
2. Wait 5-10 seconds.
3. Open a fresh Cursor window and reconnect via Remote-SSH.
4. If that fails, open a local terminal and test: `ssh vastai` — if this works, the SSH config is fine and it's a Cursor issue. Restart Cursor.
5. If the terminal SSH also fails, the instance may have changed IP (rare) or your network is blocking the port. Check vast.ai dashboard for current connection details.

**Pro tip:** If you're about to switch networks, disconnect the remote session in Cursor first (`Cmd+Shift+P` → "Remote-SSH: Close Remote Connection"). A clean disconnect makes reconnecting smoother than a dropped connection.

---

## 3. First-Time Instance Setup

Once you're connected (via SSH or Jupyter terminal), run these steps to set up your environment.

### 3.1 Clone your project repository

```bash
# Example: cloning a project repo (replace with your actual repo URL)
# If the repo is private, use a Personal Access Token (PAT):
git clone https://YOUR_USERNAME:YOUR_PAT@github.com/YOUR_USERNAME/your-repo.git
cd your-repo

# Example with a real project (THIS IS JUST AN EXAMPLE):
# git clone https://jaehoonlee0829:ghp_xxxxx@github.com/jaehoonlee0829/grpo-lite.git
# cd grpo-lite
```

**Note:** The PAT goes directly in the URL for convenience on ephemeral instances. This is acceptable because vast.ai instances are temporary and single-user. Do NOT do this on shared or persistent machines.

### 3.2 Set up API keys

Set all the API keys you need as environment variables. Add them to `~/.bashrc` so they persist across terminal sessions:

```bash
# Add to ~/.bashrc (or run these in tmux before starting work)
echo 'export ANTHROPIC_API_KEY="sk-ant-xxxxx"' >> ~/.bashrc
echo 'export HF_TOKEN="hf_xxxxx"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="sk-xxxxx"' >> ~/.bashrc
echo 'export WANDB_API_KEY="xxxxx"' >> ~/.bashrc

# Reload
source ~/.bashrc

# Verify they're set
echo $ANTHROPIC_API_KEY | head -c 10  # Should show first 10 chars
```

**API key sources:**
- **Anthropic:** [console.anthropic.com](https://console.anthropic.com) → API Keys
- **Hugging Face:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **W&B:** [wandb.ai/authorize](https://wandb.ai/authorize)

### 3.3 Install project dependencies

```bash
# If the project uses uv (fast Python package manager):
pip install uv --break-system-packages
uv sync

# If the project uses pip:
pip install -r requirements.txt --break-system-packages

# If the project uses conda:
conda env create -f environment.yml
conda activate your-env
```

### 3.4 Install Claude Code (NON-NEGOTIABLE)

**Every single vast.ai instance must have Claude Code installed with `--dangerously-skip-permissions`. No exceptions. Zero tolerance.**

```bash
# Install Claude Code
curl -fsSL https://claude.ai/install.sh | sh

# Run Claude Code with dangerously-skip-permissions — ALWAYS
claude --dangerously-skip-permissions
```

**Why `--dangerously-skip-permissions`?** On a temporary, single-user GPU instance, the permission prompts just slow you down. The instance is ephemeral — there's nothing to protect. You want Claude Code to be able to read/write/execute freely so it can help you debug, run scripts, and iterate fast without clicking "allow" 500 times.

**First-time setup inside Claude Code:**
- It will ask for your Anthropic API key — paste it.
- Set the model if needed (e.g., `claude-sonnet-4-20250514`).
- You can then use Claude Code to help set up the rest of your environment, debug issues, run experiments, etc.

### 3.5 Verify GPU access

```bash
# Check GPU is visible
nvidia-smi

# Check PyTorch can see GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.version.cuda}')"
```

---

## 4. Running Long Jobs (Surviving Wi-Fi Drops & Disconnects)

**This is the most important operational section.** Your training run, eval, or any long-running process must survive SSH disconnects, Wi-Fi switches, laptop sleep, and commutes.

### 4.1 Always use tmux

**tmux** is a terminal multiplexer. It runs on the server. When you start a tmux session and run a command inside it, that command keeps running even if your SSH connection drops, your laptop sleeps, or you switch Wi-Fi networks. The compute happens on the GPU instance — it doesn't need your local machine at all.

```bash
# Start a new tmux session (name it something useful)
tmux new -s training

# You're now inside tmux. Run your training:
cd ~/your-repo
uv run python train.py  # or whatever your training command is

# DETACH from tmux (process keeps running!):
# Press: Ctrl+B, then D

# Now you can close SSH, switch Wi-Fi, commute, whatever.
# The training keeps running on the GPU.

# When you reconnect later:
tmux attach -t training
# You'll see exactly where the training is at.
```

**Essential tmux commands:**
- `tmux new -s NAME` — create a new named session
- `tmux attach -t NAME` — reattach to an existing session
- `tmux ls` — list all active sessions
- `Ctrl+B, D` — detach (leave session running in background)
- `Ctrl+B, [` — scroll mode (use arrow keys, press `q` to exit)
- `Ctrl+B, %` — split pane vertically
- `Ctrl+B, "` — split pane horizontally
- `Ctrl+B, arrow keys` — switch between panes

### 4.2 The Jupyter web terminal as backup

The Jupyter web terminal (accessible via the vast.ai instance URL) also persists server-side. If you run a command in the Jupyter terminal and close your browser, the command keeps running. This is a good backup for when you don't want to deal with tmux.

**When to use Jupyter terminal:** Quick commands, checking status, fire-and-forget runs.
**When to use tmux:** Anything you want organized sessions for, multiple parallel processes, or when you need to scroll back through lots of output.

### 4.3 Recommended workflow for long runs

1. SSH into the instance from Cursor for code editing and browsing.
2. Open a terminal in Cursor (or a separate terminal) and start a tmux session.
3. Inside tmux, kick off your training/eval.
4. Detach from tmux (`Ctrl+B, D`).
5. Continue editing code in Cursor, or close your laptop and go.
6. When you want to check progress: re-SSH and `tmux attach -t training`.
7. Alternative: open the Jupyter web terminal in Chrome and run `tmux attach -t training` there — no SSH needed.

### 4.4 What happens when Wi-Fi drops?

- **Your SSH connection dies.** This is normal and expected.
- **Processes inside tmux keep running.** They don't care about your SSH.
- **Processes NOT inside tmux die.** This is why you always use tmux.
- **Cursor's remote window becomes unresponsive.** Close it and reconnect (see Section 2.5).
- **The GPU instance itself is fine.** It has its own internet connection and doesn't depend on yours.
- **Claude Code running inside tmux also keeps running.** It talks to Anthropic's API from the GPU instance's internet, not yours.

---

## 5. Project-Specific Setup (Example)

**This section is an example using the GRPO creative writing project. Replace with your actual project setup.**

```bash
# Example: GRPO-Lite project setup
cd ~/grpo-lite

# Install dependencies with uv
uv sync

# Set up API keys for RLAIF (Claude as judge)
export ANTHROPIC_API_KEY="sk-ant-xxxxx"

# Run training on GPU 0
CUDA_VISIBLE_DEVICES=0 uv run python main_creative.py

# Run eval on GPU 1 (if multi-GPU)
CUDA_VISIBLE_DEVICES=1 uv run python eval_base_model.py
```

**Note:** When your project uses `uv`, always prefix Python commands with `uv run` to ensure the correct virtual environment is used.

---

## 6. Troubleshooting

### SSH connection drops when switching Wi-Fi

**Symptom:** You move from cafe Wi-Fi to home Wi-Fi (or vice versa), and your SSH/Cursor connection dies and won't reconnect.

**Why:** Your local IP address changes when you switch networks. The existing TCP connection becomes invalid, and SSH can't recover it.

**Fix:**
1. Close the Cursor remote window completely.
2. Wait 5-10 seconds for stale connections to time out.
3. Reconnect from a fresh Cursor window.
4. If it still fails, test from terminal: `ssh vastai`
5. If terminal SSH works but Cursor doesn't, restart Cursor entirely.

**Prevention:** Before switching networks, cleanly disconnect from Cursor first. Also, always run long processes inside tmux so you don't lose work.

### SSH connection refused / timeout

**Symptom:** `ssh: connect to host X port Y: Connection refused` or timeout.

**Possible causes:**
- Instance is still booting — wait 1-2 minutes and retry.
- Instance crashed or was stopped — check vast.ai dashboard.
- Wrong port or IP — vast.ai changes these with each new instance. Update your SSH config.
- Your network blocks the port — try from a different network, or use the Jupyter web terminal instead.

### Cursor SSH hangs on "Setting up SSH Host"

**Symptom:** Cursor shows a progress bar that never completes when connecting.

**Fix:**
1. Cancel the connection.
2. Open `~/.ssh/known_hosts` and delete any lines with the vast.ai IP.
3. Try again.
4. If still stuck, connect from terminal first (`ssh vastai`), accept the host key, then try Cursor.

### CUDA out of memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Fixes:**
- Reduce batch size.
- Enable gradient checkpointing.
- Use mixed precision (fp16/bf16).
- Check if another process is using the GPU: `nvidia-smi`
- Kill orphan processes: `kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)`

### Disk space full

**Symptom:** `No space left on device`

**Diagnosis:**
```bash
df -h              # Check overall disk usage
du -sh ~/* | sort -h   # Find largest directories
```

**Common culprits:** Model checkpoints (5-10GB each), Hugging Face cache (`~/.cache/huggingface`), pip cache.

**Fixes:**
- Delete old checkpoints you don't need.
- Clear HF cache: `rm -rf ~/.cache/huggingface/hub/models--OLD_MODEL`
- Clear pip cache: `pip cache purge`

### API credits depleted mid-training

**Symptom:** API calls return 400/402 errors with "credit balance too low."

**Impact:** If your training uses an API-based judge (like Claude Haiku for RLAIF), the judge scores will fall back to defaults, making all subsequent training data garbage.

**What to do:**
1. Stop training immediately — data after this point is unreliable.
2. Note the last step with clean data.
3. Top up API credits.
4. Resume from the last clean checkpoint, or use the last clean checkpoint as your final result.

**Prevention:** Check your API credit balance before starting a long run. Budget approximately: (number of training steps) × (samples per step) × (cost per API call).

### Claude Code won't install or authenticate

**Symptom:** Claude Code installation fails or can't authenticate.

**Fixes:**
- Make sure `curl` is available: `apt-get update && apt-get install -y curl`
- Check internet connectivity from the instance: `curl -s https://api.anthropic.com`
- If authentication fails, re-export your Anthropic API key: `export ANTHROPIC_API_KEY="sk-ant-xxxxx"`
- Always use `--dangerously-skip-permissions` on vast.ai instances.

---

## 7. Shutting Down

**When you're done, destroy the instance to stop charges.**

1. Make sure all important data is committed and pushed to git.
2. Double-check: `git status` and `git log` to verify everything is pushed.
3. On vast.ai, go to **Instances**.
4. Find your instance and click the **trash/delete icon** (the trash can).
5. Confirm destruction.

**Data is permanently lost when you destroy an instance.** There is no undo. If you didn't push it to git or upload it somewhere, it's gone forever.

### Pre-shutdown checklist

- [ ] All code changes committed and pushed to git
- [ ] All eval metrics / results saved and pushed
- [ ] All plots / analysis files saved and pushed
- [ ] Model checkpoints uploaded to HuggingFace (if needed)
- [ ] Training logs backed up (if needed beyond what's in CSVs)
- [ ] Verified the git push actually went through (`git log origin/BRANCH..HEAD` should be empty)
