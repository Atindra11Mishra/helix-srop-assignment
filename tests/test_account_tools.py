import pytest

from app.agents.tools.account_tools import get_account_status, get_recent_builds


@pytest.mark.asyncio
async def test_get_recent_builds_returns_newest_first_with_limit():
    builds = await get_recent_builds("u_test_001", limit=3)

    assert len(builds) == 3
    assert [build.status for build in builds].count("failed") == 2
    assert builds[0].started_at > builds[1].started_at > builds[2].started_at
    assert all(build.build_id.startswith("bld_") for build in builds)


@pytest.mark.asyncio
async def test_get_account_status_returns_plan_limits():
    status = await get_account_status("pro_user_001")

    assert status.plan_tier == "pro"
    assert status.concurrent_builds_limit == 5
    assert status.storage_limit_gb == 10.0
