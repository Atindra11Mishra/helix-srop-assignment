"""
Account tools used by AccountAgent.

Mock data is acceptable for the take-home; the important part is that these
tools are deterministic and easy for the ADK agent layer to call.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

PLAN_LIMITS = {
    "free": {
        "concurrent_builds_limit": 1,
        "storage_limit_gb": 1.0,
    },
    "pro": {
        "concurrent_builds_limit": 5,
        "storage_limit_gb": 10.0,
    },
    "enterprise": {
        "concurrent_builds_limit": 999,
        "storage_limit_gb": 9999.0,
    },
}


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: datetime
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


async def get_recent_builds(user_id: str, limit: int = 5) -> list[BuildSummary]:
    """
    Return recent builds for a user, newest first.

    The data is mocked but stable enough for tests and agent traces.
    """
    safe_limit = max(1, min(limit, 20))
    return _mock_builds_for_user(user_id)[:safe_limit]


async def get_account_status(user_id: str) -> AccountStatus:
    """Return current mocked account status and plan limits."""
    plan_tier = _infer_plan_tier(user_id)
    limits = PLAN_LIMITS[plan_tier]
    return AccountStatus(
        user_id=user_id,
        plan_tier=plan_tier,
        concurrent_builds_used=1 if plan_tier == "free" else 2,
        concurrent_builds_limit=int(limits["concurrent_builds_limit"]),
        storage_used_gb=0.7 if plan_tier == "free" else 4.2,
        storage_limit_gb=float(limits["storage_limit_gb"]),
    )


def _mock_builds_for_user(user_id: str) -> list[BuildSummary]:
    base_time = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)
    suffix = _stable_user_suffix(user_id)
    specs = [
        ("deploy", "failed", "main", 342),
        ("test", "failed", "feature/api-auth", 188),
        ("lint", "passed", "feature/api-auth", 74),
        ("deploy", "failed", "release/2026-05", 913),
        ("test", "passed", "main", 221),
        ("security-scan", "cancelled", "main", 45),
    ]
    return [
        BuildSummary(
            build_id=f"bld_{suffix:03d}_{index + 1:03d}",
            pipeline=pipeline,
            status=status,
            branch=branch,
            started_at=base_time - timedelta(minutes=index * 37),
            duration_seconds=duration,
        )
        for index, (pipeline, status, branch, duration) in enumerate(specs)
    ]


def _infer_plan_tier(user_id: str) -> str:
    lowered = user_id.lower()
    if "enterprise" in lowered or lowered.startswith("ent_"):
        return "enterprise"
    if "pro" in lowered or lowered.startswith("u_test"):
        return "pro"
    return "free"


def _stable_user_suffix(user_id: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(user_id)) % 1000
