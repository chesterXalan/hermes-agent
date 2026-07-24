"""Tests for Mattermost session-banner threads + auto-title rename.

``/new`` in a Mattermost DM posts a flat bot-owned banner that becomes the
root of the next conversation thread; the auto-title callback later rewrites
that banner's text.  Mattermost threads have no title field — the root
post's text is what the Threads list shows, so rewriting the banner is the
platform's equivalent of Discord's thread rename / Telegram's topic rename.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import EphemeralReply, MessageEvent, MessageType
from gateway.session import SessionSource

BANNER_PROP = "hermes_session_banner"


def _make_adapter(*, bot_user_id: str = "bot_xyz"):
    """Build a MattermostAdapter wired up enough to invoke banner methods."""
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"url": "https://mm.example.com"},
    )
    adapter = MattermostAdapter(config)
    adapter._bot_user_id = bot_user_id
    adapter._session = MagicMock()  # truthy session so guards pass
    adapter._api_get = AsyncMock(return_value={})
    adapter._api_post = AsyncMock(return_value={"id": "post_1"})
    adapter._api_put = AsyncMock(return_value={"id": "post_1"})
    return adapter


def _banner_post(
    *,
    post_id: str = "root_1",
    user_id: str = "bot_xyz",
    root_id: str = "",
    message: str = "banner text",
    auto_title: bool = True,
    manual_title: str = "",
    with_prop: bool = True,
):
    post = {
        "id": post_id,
        "user_id": user_id,
        "root_id": root_id,
        "message": message,
        "props": {},
    }
    if with_prop:
        meta = {"auto_title": auto_title}
        if manual_title:
            meta["manual_title"] = manual_title
        post["props"][BANNER_PROP] = meta
    return post


class TestSanitizeThreadTitle:
    def test_collapses_whitespace(self):
        adapter = _make_adapter()
        assert adapter._sanitize_thread_title("  fix\n\nCI   cache ") == "fix CI cache"

    def test_caps_length(self):
        adapter = _make_adapter()
        result = adapter._sanitize_thread_title("x" * 200)
        assert len(result) == 80
        assert result.endswith("...")

    def test_empty_becomes_empty(self):
        adapter = _make_adapter()
        assert adapter._sanitize_thread_title("   ") == ""


class TestSendSessionBanner:
    @pytest.mark.asyncio
    async def test_posts_flat_with_banner_prop(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={"id": "banner_9"})

        post_id = await adapter.send_session_banner("chan_1", "Session reset!")

        assert post_id == "banner_9"
        adapter._api_post.assert_awaited_once()
        path, payload = adapter._api_post.await_args.args
        assert path == "posts"
        assert payload["channel_id"] == "chan_1"
        assert payload["message"] == "Session reset!"
        assert "root_id" not in payload  # flat post — it must BE the thread root
        assert payload["props"][BANNER_PROP] == {"auto_title": True}

    @pytest.mark.asyncio
    async def test_manual_title_stored_in_props(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()

        await adapter.send_session_banner("chan_1", "titled banner", manual_title="My topic")

        _, payload = adapter._api_post.await_args.args
        assert payload["props"][BANNER_PROP] == {
            "auto_title": False,
            "manual_title": "My topic",
        }

    @pytest.mark.asyncio
    async def test_env_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_AUTO_TITLE", "false")
        adapter = _make_adapter()

        assert await adapter.send_session_banner("chan_1", "text") is None
        adapter._api_post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_api_failure_returns_none(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={})

        assert await adapter.send_session_banner("chan_1", "text") is None

    @pytest.mark.asyncio
    async def test_missing_args_return_none(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()

        assert await adapter.send_session_banner("", "text") is None
        assert await adapter.send_session_banner("chan_1", "") is None
        adapter._api_post.assert_not_awaited()


class TestRenameThread:
    @pytest.mark.asyncio
    async def test_collapses_banner_to_title_line(self, monkeypatch):
        """The reset notice + thread hint are dropped once the title lands."""
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(return_value=_banner_post(message="Session reset!"))

        assert await adapter.rename_thread("root_1", "Fix CI cache") is True

        adapter._api_put.assert_awaited_once()
        path, payload = adapter._api_put.await_args.args
        assert path == "posts/root_1/patch"
        assert payload["message"] == "💬 Fix CI cache"
        assert payload["props"][BANNER_PROP]["titled"] is True

    @pytest.mark.asyncio
    async def test_second_rename_replaces_title(self, monkeypatch):
        """A titled banner renames cleanly — new title replaces, never stacks."""
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        already_titled = _banner_post(message="💬 Old title")
        already_titled["props"][BANNER_PROP]["titled"] = True
        adapter._api_get = AsyncMock(return_value=already_titled)

        await adapter.rename_thread("root_1", "New topic")

        _, payload = adapter._api_put.await_args.args
        assert payload["message"] == "💬 New topic"

    @pytest.mark.asyncio
    async def test_never_rewrites_user_posts(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(
            return_value=_banner_post(user_id="human_1", with_prop=False)
        )

        assert await adapter.rename_thread("root_1", "title") is False
        adapter._api_put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_ordinary_bot_posts_without_prop(self, monkeypatch):
        """A user opening a thread on a normal bot reply must not rewrite it."""
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(return_value=_banner_post(with_prop=False))

        assert await adapter.rename_thread("root_1", "title") is False
        adapter._api_put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_manual_banner_collapses_to_manual_title(self, monkeypatch):
        """/new <title> banners collapse to the USER's title, never the generated one."""
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(
            return_value=_banner_post(auto_title=False, manual_title="Say hello")
        )

        assert await adapter.rename_thread("root_1", "AI generated title") is True

        _, payload = adapter._api_put.await_args.args
        assert payload["message"] == "💬 Say hello"

    @pytest.mark.asyncio
    async def test_legacy_manual_banner_without_stored_title_untouched(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(return_value=_banner_post(auto_title=False))

        assert await adapter.rename_thread("root_1", "title") is False
        adapter._api_put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_replies(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()
        adapter._api_get = AsyncMock(return_value=_banner_post(root_id="other_root"))

        assert await adapter.rename_thread("root_1", "title") is False
        adapter._api_put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_env_disabled_skips_everything(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_AUTO_TITLE", "0")
        adapter = _make_adapter()

        assert await adapter.rename_thread("root_1", "title") is False
        adapter._api_get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_title_or_fetch_failure(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = _make_adapter()

        assert await adapter.rename_thread("root_1", "   ") is False

        adapter._api_get = AsyncMock(return_value={})
        assert await adapter.rename_thread("root_1", "title") is False
        adapter._api_put.assert_not_awaited()


def _mm_source(thread_id=None, chat_type="dm"):
    return SessionSource(
        platform=Platform.MATTERMOST,
        chat_id="chan_1",
        chat_type=chat_type,
        user_id="user_1",
        thread_id=thread_id,
        message_id="msg_1",
    )


class TestMattermostBannerThreadLane:
    def _runner(self):
        return object.__new__(gateway_run.GatewayRunner)

    def test_thread_session_is_lane(self):
        assert self._runner()._is_mattermost_banner_thread_lane(_mm_source(thread_id="root_1")) is True

    def test_flat_dm_is_not_lane(self):
        assert self._runner()._is_mattermost_banner_thread_lane(_mm_source()) is False

    def test_other_platform_is_not_lane(self):
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="c", chat_type="thread", thread_id="t"
        )
        assert self._runner()._is_mattermost_banner_thread_lane(source) is False


class TestRenameMattermostThreadForSessionTitle:
    def _runner(self, adapter):
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {Platform.MATTERMOST: adapter}
        runner._adapter_for_source = MagicMock(return_value=adapter)
        return runner

    @pytest.mark.asyncio
    async def test_calls_adapter_rename(self):
        adapter = MagicMock()
        adapter.rename_thread = AsyncMock(return_value=True)
        runner = self._runner(adapter)

        await runner._rename_mattermost_thread_for_session_title(
            _mm_source(thread_id="root_1"), "sid_1", "Fix CI cache"
        )

        adapter.rename_thread.assert_awaited_once_with("root_1", "Fix CI cache")

    @pytest.mark.asyncio
    async def test_non_lane_source_is_ignored(self):
        adapter = MagicMock()
        adapter.rename_thread = AsyncMock()
        runner = self._runner(adapter)

        await runner._rename_mattermost_thread_for_session_title(
            _mm_source(thread_id=None), "sid_1", "title"
        )

        adapter.rename_thread.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_adapter_without_rename_thread_is_ignored(self):
        adapter = object()  # no rename_thread attribute at all
        runner = self._runner(adapter)

        # Must not raise.
        await runner._rename_mattermost_thread_for_session_title(
            _mm_source(thread_id="root_1"), "sid_1", "title"
        )


def _make_reset_runner(adapter=None, session_db=None):
    """GatewayRunner with just enough wiring to run _handle_reset_command."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_key_for_source = MagicMock(return_value="agent:main:mattermost:dm:chan_1")
    runner._invalidate_session_run_generation = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner.session_store = MagicMock()
    runner.session_store._entries = {}
    new_entry = MagicMock()
    new_entry.session_id = "sid_new"
    runner.session_store.get_or_create_session = MagicMock(return_value=new_entry)
    # async_session_store is a read-only property; seed its backing attribute
    # with a facade whose _store matches so the property returns it as-is.
    async_store = MagicMock()
    async_store._store = runner.session_store
    async_store.reset_session = AsyncMock(return_value=new_entry)
    runner._async_session_store = async_store
    runner._agent_cache_lock = None
    runner._evict_cached_agent = MagicMock()
    runner._queued_events = None
    runner._session_model_overrides = {}
    runner._set_session_reasoning_override = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._reset_notice_session_info = MagicMock(return_value="")
    runner._telegram_topic_new_header = MagicMock(return_value=None)
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    runner._session_db = session_db
    runner.adapters = {Platform.MATTERMOST: adapter} if adapter is not None else {}
    return runner


def _reset_event(text="/new", source=None):
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=source or _mm_source(),
        message_id="msg_1",
    )


@pytest.fixture
def _isolated_side_effects(monkeypatch):
    """Neutralize module-level side effects the reset handler triggers."""
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session", MagicMock(), raising=False
    )
    monkeypatch.setattr(
        "tools.env_passthrough.clear_env_passthrough", MagicMock(), raising=False
    )
    monkeypatch.setattr(
        "tools.credential_files.clear_credential_files", MagicMock(), raising=False
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", MagicMock(), raising=False)
    monkeypatch.setattr(
        "hermes_cli.tips.get_random_tip", MagicMock(return_value="tip"), raising=False
    )


class TestResetCommandMattermostBanner:
    @pytest.mark.asyncio
    async def test_flat_dm_new_posts_banner_and_suppresses_reply(
        self, monkeypatch, _isolated_side_effects
    ):
        monkeypatch.delenv("MATTERMOST_AUTO_TITLE", raising=False)
        adapter = MagicMock()
        adapter.send_session_banner = AsyncMock(return_value="banner_1")
        runner = _make_reset_runner(adapter)

        result = await runner._handle_reset_command(_reset_event())

        adapter.send_session_banner.assert_awaited_once()
        args, kwargs = adapter.send_session_banner.await_args
        assert args[0] == "chan_1"
        assert "Reply in this thread" in args[1]
        assert "titled automatically" in args[1]
        assert kwargs.get("manual_title") is None
        assert isinstance(result, EphemeralReply)
        assert str(result) == ""  # normal reply suppressed — banner carries it

    @pytest.mark.asyncio
    async def test_manual_title_passed_with_short_hint(
        self, monkeypatch, _isolated_side_effects
    ):
        session_db = MagicMock()
        session_db.set_session_title = AsyncMock(return_value=True)
        adapter = MagicMock()
        adapter.send_session_banner = AsyncMock(return_value="banner_1")
        runner = _make_reset_runner(adapter, session_db=session_db)

        result = await runner._handle_reset_command(_reset_event(text="/new My topic"))

        args, kwargs = adapter.send_session_banner.await_args
        assert kwargs.get("manual_title") == "My topic"
        assert "Reply in this thread" in args[1]
        assert "titled automatically" not in args[1]  # manual banner: short hint
        assert str(result) == ""

    @pytest.mark.asyncio
    async def test_banner_failure_falls_back_to_reply_path(
        self, monkeypatch, _isolated_side_effects
    ):
        adapter = MagicMock()
        adapter.send_session_banner = AsyncMock(return_value=None)
        runner = _make_reset_runner(adapter)

        result = await runner._handle_reset_command(_reset_event())

        assert isinstance(result, EphemeralReply)
        assert str(result) != ""  # reset notice still reaches the user

    @pytest.mark.asyncio
    async def test_new_inside_thread_keeps_reply_path(
        self, monkeypatch, _isolated_side_effects
    ):
        """/new inside an existing thread resets in place — no new banner."""
        adapter = MagicMock()
        adapter.send_session_banner = AsyncMock(return_value="banner_1")
        runner = _make_reset_runner(adapter)

        result = await runner._handle_reset_command(
            _reset_event(source=_mm_source(thread_id="root_1"))
        )

        adapter.send_session_banner.assert_not_awaited()
        assert str(result) != ""

    @pytest.mark.asyncio
    async def test_non_mattermost_platform_untouched(
        self, monkeypatch, _isolated_side_effects
    ):
        adapter = MagicMock()
        adapter.send_session_banner = AsyncMock(return_value="banner_1")
        runner = _make_reset_runner(adapter)
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="tg_1", chat_type="dm", user_id="u1"
        )

        result = await runner._handle_reset_command(_reset_event(source=source))

        adapter.send_session_banner.assert_not_awaited()
        assert str(result) != ""

    @pytest.mark.asyncio
    async def test_adapter_without_banner_support_keeps_reply_path(
        self, monkeypatch, _isolated_side_effects
    ):
        adapter = object()  # e.g. older adapter build without send_session_banner
        runner = _make_reset_runner(adapter)

        result = await runner._handle_reset_command(_reset_event())

        assert isinstance(result, EphemeralReply)
        assert str(result) != ""
