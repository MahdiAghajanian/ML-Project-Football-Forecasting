from __future__ import annotations

from fastapi.testclient import TestClient

from app import MATCH_GROUPS, app

client = TestClient(app)


def main() -> None:
    health = client.get("/health")
    health.raise_for_status()
    health_payload = health.json()
    assert health_payload["status"] == "ok"
    assert health_payload["matches_named"] == health_payload["matches_loaded"]

    test_matches = client.get("/matches", params={"split": "test"})
    test_matches.raise_for_status()
    matches = test_matches.json()
    assert matches, "No held-out matches available"
    assert matches[0]["home_team"] != "Home"
    assert matches[0]["away_team"] != "Away"

    match_id = int(matches[0]["match_id"])
    pre = client.get(f"/prematch/{match_id}")
    pre.raise_for_status()
    assert abs(sum(pre.json()["outcome_probabilities"].values()) - 1.0) < 1e-6

    # A minute request must resolve to a snapshot at or before the requested time.
    requested_minute = 37.0
    live = client.get(f"/predict/{match_id}", params={"minute": requested_minute})
    live.raise_for_status()
    payload = live.json()
    assert float(payload["snapshot_minute"]) <= requested_minute
    assert abs(sum(payload["outcome_probabilities"].values()) - 1.0) < 1e-6

    replay = client.get(f"/replay/{match_id}")
    replay.raise_for_status()
    assert len(replay.json()) == len(MATCH_GROUPS[match_id])

    print("bonus service smoke test: PASS")


if __name__ == "__main__":
    main()
