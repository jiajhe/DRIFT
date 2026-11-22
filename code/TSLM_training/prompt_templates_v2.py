"""User text and short, checkable evidence statements for the four v2 tasks."""

import json
import numpy as np

TASKS = (
    "ATTRIBUTION_COT",
    "COUNTERFACTUAL_COT",
    "REGIME_ROLE_COT",
    "PAIRWISE_FORECAST_COT",
)
SYSTEM = (
    "You are DRIFT-TSLM, a cross-sectional financial time-series analyst. Read the observed embeddings next to their stock names. "
    "Respond with <reasoning>containing two to four short evidence-grounded sentences</reasoning> and <answer>containing compact JSON</answer>."
)
QUESTIONS = (
    "Classify the signal driver (LOCAL_DOMINANT, FLOW_REINFORCED, FLOW_OFFSET) and flow mode (RETENTION_LED, NEW_TRANSMISSION_LED, MULTI_DAY_RELAY). Identify the strongest absolute directed incoming edge among the five candidates. Return driver, flow_mode, top_sender.",
    "Mask the directed edge from {query} into the target, holding the other edges and previous state fixed. Classify the model score shift (INCREASE, DECREASE, NEGLIGIBLE) and sensitivity (LOW, MODERATE, HIGH). Return queried_sender, score_shift, sensitivity.",
    "Classify net_role (NET_RECEIVER, BALANCED, NET_SENDER), role_stability (STABLE, SHIFTING), and market_regime (IDIOSYNCRATIC, MODERATE, HIGH_PROPAGATION). Return these three fields.",
    "Compare A with B for the return from the next close to the following close. Return winner (A or B) and confidence (HIGH or MODERATE).",
)
DRIVER = ("LOCAL_DOMINANT", "FLOW_REINFORCED", "FLOW_OFFSET")
FLOW = ("RETENTION_LED", "NEW_TRANSMISSION_LED", "MULTI_DAY_RELAY")
SHIFT = ("NEGLIGIBLE", "INCREASE", "DECREASE")
SENS = ("LOW", "MODERATE", "HIGH")
ROLE = ("NET_RECEIVER", "BALANCED", "NET_SENDER")
STABILITY = ("STABLE", "SHIFTING")
REGIME = ("IDIOSYNCRATIC", "MODERATE", "HIGH_PROPAGATION")


def user_parts(row):
    """Each (header, block index) places soft tokens directly after its stock name."""
    prefix = f"S&P 500. Observation date {row['date']}.\n"
    if row["task"] == 3:
        headers = [
            prefix + "Candidate A: " + row["ticker"],
            "\nCandidate B: " + row["other"],
        ]
    else:
        headers = [prefix + "Target: " + row["ticker"]] + [
            "\nUpstream candidate: " + name for name in row["senders"]
        ]
    return headers, "\n" + QUESTIONS[row["task"]].format(query=row.get("query", ""))


def evidence(task, answer, d, i, b, variant):
    # Each slot is a statement of observed evidence (or an actual frozen-model intervention).
    # The pairwise explanation never receives the realized winner as an input.
    variant = int(variant) % 9
    a, c = divmod(variant, 3)
    ticker = str(d["instruments"][i])
    stance = (
        "positive"
        if d["p_local"][i] > 0
        else "negative"
        if d["p_local"][i] < 0
        else "flat"
    )
    if task == 0:
        driver = {
            "LOCAL_DOMINANT": "the flow share falls in its lower historical band",
            "FLOW_REINFORCED": "flow reinforces the local signal",
            "FLOW_OFFSET": "flow offsets the local signal",
        }[answer["driver"]]
        mode = {
            "RETENTION_LED": "the retention share is in the upper half of non-relay updates",
            "NEW_TRANSMISSION_LED": "the retention share is in the lower half of non-relay updates",
            "MULTI_DAY_RELAY": "historical retransmission has an elevated share",
        }[answer["flow_mode"]]
        sender = answer["top_sender"]
        s = np.argmax(abs(d["edges"][i]))
        sign = "positive" if d["edges"][i, s] > 0 else "negative"
        return (
            [
                f"{ticker} has a {stance} local signal and {driver}.",
                f"With a {stance} local score, {driver}.",
                f"The local score is {stance}; {driver}.",
            ][a]
            + " "
            + [
                f"In the update, {mode}. The strongest channel is {sender}'s {sign} edge.",
                f"The flow components show that {mode}. {sender} has the largest absolute edge, with {sign} sign.",
                f"Today's update shows that {mode}. The largest inbound coupling is {sign} and comes from {sender}.",
            ][c]
        )
    if task == 1:
        sender = answer["queried_sender"]
        sign = (
            "positive"
            if d["edges"][i, list(d["instruments"][d["senders"][i]]).index(sender)] > 0
            else "negative"
        )
        shift = {
            "NEGLIGIBLE": "changes little",
            "INCREASE": "moves upward",
            "DECREASE": "moves downward",
        }[answer["score_shift"]]
        band = answer["sensitivity"].lower()
        return (
            [
                f"The selected edge carries {sign} coupling from {sender}.",
                f"{sender} enters the target through a {sign} directed channel.",
                f"The incoming {sender} relation has a {sign} operator sign.",
            ][a]
            + " "
            + [
                f"With this edge removed, the model score {shift} and sensitivity is {band}.",
                f"When that path is masked, the recomputed score {shift}. Its response falls in the {band} sensitivity band.",
                f"The frozen-model intervention has {band} sensitivity. Under the mask the target score {shift}.",
            ][c]
        )
    if task == 2:
        net = {
            "NET_SENDER": "sender-tilted",
            "NET_RECEIVER": "receiver-tilted",
            "BALANCED": "balanced",
        }[answer["net_role"]]
        stable = {"STABLE": "relatively stable", "SHIFTING": "shifting"}[
            answer["role_stability"]
        ]
        regime = {
            "IDIOSYNCRATIC": "low",
            "MODERATE": "intermediate",
            "HIGH_PROPAGATION": "high",
        }[answer["market_regime"]]
        return (
            [
                f"{ticker}'s relative role is {net}.",
                f"The two activity gates place {ticker} in the {net} group.",
                f"The sender-minus-receiver activity places the stock in the {net} band.",
            ][a]
            + " "
            + [
                f"Role membership is {stable} over the lookback. Market propagation intensity is {regime}.",
                f"The role memberships are {stable} across the two endpoints. The cross-sectional operator has {regime} propagation intensity.",
                f"The market-wide transmission measure is {regime}. Over twenty sessions the stock's role mix is {stable}.",
            ][c]
        )
    local = (
        "A"
        if d["p_local"][i] > d["p_local"][b]
        else "B"
        if d["p_local"][i] < d["p_local"][b]
        else "neither candidate"
    )
    flow = (
        "A"
        if d["flow_delta"][i] > d["flow_delta"][b]
        else "B"
        if d["flow_delta"][i] < d["flow_delta"][b]
        else "neither candidate"
    )
    return (
        [
            f"The observed local score favors {local}.",
            f"At the observation close, {local} has the stronger local reading."
            if local.startswith(("A", "B"))
            else "The local readings are tied.",
            f"The local-factor comparison favors {local}.",
        ][a]
        + " "
        + [
            f"The relative flow adjustment favors {flow}. The realized ordering remains uncertain.",
            f"Comparing flow adjustments separately favors {flow}. Both signals are available at the observation close.",
            f"The flow adjustment favors {flow} across these two names. The ordering ahead is a forecast.",
        ][c]
    )


def response(row):
    return (
        "<reasoning>"
        + row["evidence"]
        + "</reasoning>\n<answer>"
        + json.dumps(row["answer"], separators=(",", ":"))
        + "</answer>"
    )


def legal_answer(row, answer):
    """Validate a generation using only the question's permitted output vocabulary."""
    if not isinstance(answer, dict):
        return False
    t = row["task"]
    if t == 0:
        values = {"driver": DRIVER, "flow_mode": FLOW, "top_sender": row["senders"]}
    elif t == 1:
        values = {
            "queried_sender": (row["query"],),
            "score_shift": SHIFT,
            "sensitivity": SENS,
        }
    elif t == 2:
        values = {
            "net_role": ROLE,
            "role_stability": STABILITY,
            "market_regime": REGIME,
        }
    else:
        values = {"winner": ("A", "B"), "confidence": ("HIGH", "MODERATE")}
    return set(answer) == set(values) and all(
        isinstance(answer[k], str) and answer[k] in choices
        for k, choices in values.items()
    )
