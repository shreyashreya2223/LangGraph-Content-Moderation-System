from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph

# Define state schema
class ModerationState(TypedDict, total=False):
    content: str
    toxicity: Optional[float]
    spam: Optional[bool]
    violations: Optional[List[str]]
    severity: Optional[float]
    action: Optional[str]
    reason: Optional[str]
    appeal: Optional[str]

# Define moderation functions
def detect_toxicity(state: ModerationState) -> ModerationState:
    text = state["content"]
    toxicity_score = 0.95 if "idiot" in text.lower() else 0.1
    state["toxicity"] = toxicity_score
    if toxicity_score > 0.85:
        state.setdefault("violations", []).append("Toxic language")
    return state

def detect_spam(state: ModerationState) -> ModerationState:
    text = state["content"]
    spam_keywords = ["buy now", "click here"]
    is_spam = any(keyword in text.lower() for keyword in spam_keywords)
    state["spam"] = is_spam
    if is_spam:
        state.setdefault("violations", []).append("Spam content")
    return state

def check_policy(state: ModerationState) -> ModerationState:
    text = state["content"]
    if "hate" in text.lower():
        state.setdefault("violations", []).append("Hate speech")
    return state

def compute_severity(violations: List[str]) -> float:
    weights = {
        "Toxic language": 0.9,
        "Spam content": 0.5,
        "Hate speech": 1.0
    }
    return min(1.0, sum(weights.get(v, 0.3) for v in violations))

def make_decision(state: ModerationState) -> ModerationState:
    violations = state.get("violations", [])
    if not violations:
        state["action"] = "Allow"
        state["reason"] = "No violations detected"
    else:
        severity = compute_severity(violations)
        state["severity"] = severity
        if severity >= 0.9:
            state["action"] = "Block"
        elif severity >= 0.5:
            state["action"] = "Warn"
        else:
            state["action"] = "Flag for Review"
        state["reason"] = f"Detected violations: {', '.join(violations)} (Severity: {severity})"
    return state

def handle_appeal(state: ModerationState) -> ModerationState:
    if state.get("action") in ["Block", "Warn"]:
        state["appeal"] = "Pending human review"
    return state

# ✅ Initialize the graph with state schema
graph = StateGraph(ModerationState)

# Add nodes
graph.add_node("Detect Toxicity", detect_toxicity)
graph.add_node("Detect Spam", detect_spam)
graph.add_node("Policy Check", check_policy)
graph.add_node("Decision", make_decision)
graph.add_node("Appeal", handle_appeal)

# Define flow
graph.set_entry_point("Detect Toxicity")
graph.add_edge("Detect Toxicity", "Detect Spam")
graph.add_edge("Detect Spam", "Policy Check")
graph.add_edge("Policy Check", "Decision")
graph.add_edge("Decision", "Appeal")
graph.set_finish_point("Appeal")

# Compile the workflow
workflow = graph.compile()

# Sample test cases
test_cases = [
    {"content": "You are an idiot!"},
    {"content": "Click here to buy now!"},
    {"content": "I hate everyone."},
    {"content": "Hello, how are you today?"}
]

# Run test cases
if __name__ == "__main__":
    for idx, case in enumerate(test_cases):
        print(f"\n--- Test Case {idx + 1} ---")
        result = workflow.invoke(case)
        for key, value in result.items():
            print(f"{key}: {value}")
