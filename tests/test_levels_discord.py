import io
import json
import urllib.error

import pandas as pd
import pytest

from bsnn import levels_cli

from test_levels import TUESDAY_0915, passing_levels, sample_zero_dte

WEBHOOK = "https://discord.com/api/webhooks/1/token"


@pytest.fixture
def discord(monkeypatch, tmp_path):
    """bsnn-levels at 09:15 New York on Tue 06 Oct with passing cards; Discord calls are recorded."""
    calls, ids = [], iter(["m1", "m2", "m3"])
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK)
    monkeypatch.setattr(levels_cli.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(levels_cli, "now_utc", lambda: TUESDAY_0915)
    monkeypatch.setattr(levels_cli, "levels_for", lambda symbol, before, prefer, **kw: passing_levels(symbol))
    monkeypatch.setattr(levels_cli, "live_zero_dte", lambda symbol, prefer: sample_zero_dte(symbol=symbol))
    monkeypatch.setattr(levels_cli.market_data, "latest_price", lambda ticker: 31_400.0)
    monkeypatch.setattr(levels_cli, "due_slot", lambda now, slots, marker: (levels_cli.DAILY_SLOT, ""))

    def post(url, payload):
        calls.append(("post", None, payload))
        return next(ids)

    monkeypatch.setattr(levels_cli, "post", post)
    monkeypatch.setattr(levels_cli, "edit", lambda url, message_id, payload: calls.append(("edit", message_id, payload)))
    return calls, tmp_path


def records(tmp_path):
    return [json.loads(line) for line in (tmp_path / "levels" / "2026-10-06.jsonl").read_text().splitlines()]


def test_manual_post_edits_the_days_card(discord):
    calls, tmp_path = discord
    assert levels_cli.main(["--markets", "NQ", "--post", "--scheduled"]) == 0
    assert levels_cli.main(["--markets", "NQ", "--post"]) == 0
    assert [(kind, message_id) for kind, message_id, _ in calls] == [("post", None), ("edit", "m1")]
    assert calls[1][2]["content"] == "**Options levels for Tuesday 06 October 2026** · as of 09:15 New York"
    assert json.loads((tmp_path / "levels_messages.json").read_text()) == {"2026-10-06": "m1"}
    assert [(r["kind"], r["message_id"]) for r in records(tmp_path)] == [("scheduled", "m1"), ("manual-edit", "m1")]


def test_each_scheduled_post_and_new_message_is_a_new_card(discord):
    calls, tmp_path = discord
    levels_cli.main(["--markets", "NQ", "--post", "--scheduled"])
    levels_cli.main(["--markets", "NQ", "--post", "--new-message"])
    assert [kind for kind, _, _ in calls] == ["post", "post"]
    assert json.loads((tmp_path / "levels_messages.json").read_text()) == {"2026-10-06": "m2"}  # later edits go here


def test_manual_post_without_a_card_yet_posts_one(discord):
    calls, tmp_path = discord
    levels_cli.main(["--markets", "NQ", "--post"])
    assert [kind for kind, _, _ in calls] == ["post"] and records(tmp_path)[0]["kind"] == "manual"


def test_a_card_deleted_in_discord_is_posted_again(discord, monkeypatch):
    calls, tmp_path = discord
    levels_cli.save_message_id(tmp_path / "levels_messages.json", "2026-10-06", "gone")

    def deleted(url, message_id, payload):
        raise urllib.error.HTTPError(url, 404, "Unknown Message", {}, None)

    monkeypatch.setattr(levels_cli, "edit", deleted)
    assert levels_cli.main(["--markets", "NQ", "--post"]) == 0
    assert [kind for kind, _, _ in calls] == ["post"]
    assert json.loads((tmp_path / "levels_messages.json").read_text()) == {"2026-10-06": "m1"}


def test_zero_dte_updates_stay_separate(discord):
    calls, tmp_path = discord
    levels_cli.main(["--markets", "NQ", "--post", "--scheduled"])
    levels_cli.main(["--markets", "NQ", "--post", "--zero-dte"])
    assert [kind for kind, _, _ in calls] == ["post", "post"]
    assert json.loads((tmp_path / "levels_messages.json").read_text()) == {"2026-10-06": "m1"}


def test_only_two_weeks_of_message_ids_are_kept(tmp_path):
    store = tmp_path / "ids.json"
    for day in pd.bdate_range("2026-09-01", periods=20):
        levels_cli.save_message_id(store, day, f"id-{day:%d}")
    ids = levels_cli.load_message_ids(store)
    assert len(ids) == levels_cli.MESSAGES_KEPT and max(ids) == "2026-09-28"


def test_webhook_endpoint_keeps_the_thread_query():
    url = WEBHOOK + "?thread_id=9"
    assert levels_cli.webhook_endpoint(url, wait="true") == WEBHOOK + "?thread_id=9&wait=true"
    assert levels_cli.webhook_endpoint(url, "/messages/5") == WEBHOOK + "/messages/5?thread_id=9"


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_post_waits_for_the_message_id_and_edit_patches_it(monkeypatch):
    sent = []

    def urlopen(request, timeout):
        sent.append((request.get_method(), request.full_url, json.loads(request.data)))
        return FakeResponse(b'{"id": "123"}')

    monkeypatch.setattr(levels_cli.urllib.request, "urlopen", urlopen)
    payload = {"username": "Options levels", "content": "x", "embeds": []}
    assert levels_cli.post(WEBHOOK, payload) == "123"
    levels_cli.edit(WEBHOOK, "123", payload)
    (method, url, body), (edit_method, edit_url, edit_body) = sent
    assert (method, url, body) == ("POST", WEBHOOK + "?wait=true", payload)
    assert (edit_method, edit_url) == ("PATCH", WEBHOOK + "/messages/123")
    assert "username" not in edit_body and edit_body["content"] == "x"
