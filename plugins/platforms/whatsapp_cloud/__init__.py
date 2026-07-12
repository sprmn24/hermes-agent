"""WhatsApp Cloud platform plugin — registers the adapter with the Hermes plugin system."""
from __future__ import annotations
import os
from gateway.platforms.whatsapp_cloud import (
    WhatsAppCloudAdapter,
    check_whatsapp_cloud_requirements,
)


def _is_connected(config) -> bool:
    """WhatsApp Cloud is considered connected when credentials are available
    from either environment variables OR explicit config.extra fields."""
    extra = getattr(config, "extra", {}) or {}

    # Prefer config.extra credentials (set via config.yaml) over env vars.
    phone_id = (
        extra.get("phone_number_id")
        or os.getenv("WHATSAPP_CLOUD_PHONE_NUMBER_ID", "")
    )
    token = (
        extra.get("access_token")
        or os.getenv("WHATSAPP_CLOUD_ACCESS_TOKEN", "")
    )
    return bool(phone_id and token)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="whatsapp_cloud",
        label="WhatsApp Cloud",
        adapter_factory=lambda cfg: WhatsAppCloudAdapter(cfg),
        check_fn=check_whatsapp_cloud_requirements,
        is_connected=_is_connected,
        required_env=["WHATSAPP_CLOUD_PHONE_NUMBER_ID", "WHATSAPP_CLOUD_ACCESS_TOKEN"],
        install_hint="pip install aiohttp httpx",
        allowed_users_env="WHATSAPP_CLOUD_ALLOWED_USERS",
        allow_all_env="WHATSAPP_CLOUD_ALLOW_ALL_USERS",
        cron_deliver_env_var="WHATSAPP_CLOUD_HOME_CHANNEL",
        emoji="📱",
        platform_hint="You are on WhatsApp. Keep responses concise. Avoid heavy markdown.",
    )
