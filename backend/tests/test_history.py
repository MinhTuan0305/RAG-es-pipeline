from app.services.chat_service import SHORTENED_NOTE, build_history, estimate_tokens


def turn(role, chars, tag=""):
    return {"role": role, "content": (tag + " " + "x" * chars).strip()}


def total_tokens(history):
    return sum(estimate_tokens(m["content"]) for m in history)


def test_keeps_newest_turns_within_budget():
    # 10 pairs of ~300 tokens each (900 chars); budget 1300 -> the last 4 messages fit
    messages = [turn("user" if i % 2 == 0 else "assistant", 900, f"m{i}") for i in range(20)]
    history = build_history(messages, token_budget=1300)
    assert [m["content"].split()[0] for m in history] == ["m16", "m17", "m18", "m19"]
    assert total_tokens(history) <= 1300
    assert history[0]["role"] == "user"


def test_long_answers_count_more_than_short_ones():
    short = [turn("user", 30), turn("assistant", 60)] * 10  # 20 tiny messages
    assert len(build_history(short, token_budget=600)) == 20
    long = [turn("user", 30), turn("assistant", 3000)] * 10  # each answer ~1000 tokens
    assert len(build_history(long, token_budget=2500)) == 4


def test_latest_pair_is_kept_and_shortened_when_too_long():
    messages = [turn("user", 90, "old-q"), turn("assistant", 90, "old-a"),
                turn("user", 300, "question"), turn("assistant", 30000, "answer")]
    history = build_history(messages, token_budget=1000)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"].startswith("question")  # the question is intact...
    assert len(history[0]["content"]) == len(messages[2]["content"])
    assert history[1]["content"].startswith("answer")  # ...the answer is cut, with a note
    assert history[1]["content"].endswith(SHORTENED_NOTE)
    assert total_tokens(history) <= 1000


def test_failed_turns_are_dropped_with_their_question():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "**Lỗi**", "error": True},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    assert [m["content"] for m in build_history(messages, token_budget=1000)] == ["q1", "a1", "q3", "a3"]


def test_starts_with_a_user_turn():
    messages = [turn("user", 600, "q1"), turn("assistant", 30, "a1"), turn("user", 30, "q2"), turn("assistant", 30, "a2")]
    # budget fits a1, q2, a2 but not q1: the orphan answer a1 must not lead the history
    history = build_history(messages, token_budget=60)
    assert history[0]["role"] == "user" and history[0]["content"].startswith("q2")


def test_zero_budget_and_empty():
    messages = [turn("user", 30), turn("assistant", 30)]
    assert build_history(messages, token_budget=0) == []
    assert build_history([], token_budget=3000) == []
