"""
Environment bootstrap — call load_env() at the top of every entry-point script.

What it does:
  1. Reads .env into os.environ via python-dotenv
  2. Calls huggingface_hub.login() with HF_TOKEN so that all subsequent
     HF API calls (datasets, hub, transformers) are authenticated.
     Without this step, setting HF_TOKEN in .env is not enough — the
     datasets library checks the token at request time via huggingface_hub,
     which requires an explicit login() call or the token to be in the
     HF-managed credential store (~/.cache/huggingface/token).
  3. Logs in to W&B using WANDB_API_KEY if present.
"""

import os
import logging

logger = logging.getLogger(__name__)


def load_env(dotenv_path: str = ".env") -> None:
    """Load .env and authenticate with HuggingFace and W&B."""
    # ── 1. Load .env into os.environ ─────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(dotenv_path=dotenv_path, override=False)
        if loaded:
            logger.info(f"Loaded environment from {dotenv_path}")
        else:
            logger.debug(f".env not found at {dotenv_path} — relying on shell environment")
    except ImportError:
        logger.debug("python-dotenv not installed — relying on shell environment")

    # ── 2. HuggingFace login ──────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("HuggingFace: authenticated via HF_TOKEN")
        except Exception as exc:
            logger.warning(f"HuggingFace login failed: {exc}")
    else:
        logger.warning(
            "HF_TOKEN not set — HuggingFace requests will be rate-limited (unauthenticated). "
            "Set HF_TOKEN in your .env file."
        )

    # ── 3. W&B login ──────────────────────────────────────────────────────────
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key, relogin=False)
            logger.info("W&B: authenticated via WANDB_API_KEY")
        except Exception as exc:
            logger.warning(f"W&B login failed: {exc}")
