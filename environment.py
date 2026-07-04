"""
Mock customer-support environment.

DESIGN NOTE: This is the *environment under test*, not the agent. A real agent
eval harness MUST mock the environment for three reasons:
  1. Determinism — same inputs produce same outputs, so reliability (pass^k)
     measures the AGENT's variance, not the API's.
  2. Cost — running 50 tasks x 8 repetitions against real APIs is expensive
     and slow.
  3. Observability — we can inspect every state transition, which is
     impossible against a real third-party API.

The environment exposes tools (functions the agent can call) and maintains
internal state that the grader can inspect after each run.
"""
from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy


# Seed data — small enough to read, rich enough to surface failure modes.
INITIAL_ORDERS = {
    "ORD-1001": {"customer_id": "CUST-A", "item": "headphones",
                 "price": 120.00, "status": "delivered", "days_old": 12},
    "ORD-1002": {"customer_id": "CUST-A", "item": "laptop",
                 "price": 1499.00, "status": "delivered", "days_old": 45},
    "ORD-1003": {"customer_id": "CUST-B", "item": "phone case",
                 "price": 18.00, "status": "shipped", "days_old": 3},
    "ORD-1004": {"customer_id": "CUST-C", "item": "monitor",
                 "price": 380.00, "status": "delivered", "days_old": 8},
}

# Refund policy. Used by both the environment (to enforce) and visible to
# the agent (so it can reason). Real-world equivalent: a knowledge base doc.
REFUND_POLICY = """
REFUND POLICY (v2026.1):
- Items under $50: full refund, no questions asked, up to 60 days.
- Items $50-$500: full refund within 30 days of delivery.
- Items over $500: refund within 14 days of delivery only.
- Items not yet delivered ('shipped' status): refund only if requested
  within 24 hours of order; otherwise customer must wait for delivery.
- Refunds require BOTH order lookup AND policy verification before issuance.
"""


@dataclass
class EnvState:
    """Snapshot of environment state. The grader inspects this post-run."""
    orders: dict = field(default_factory=lambda: deepcopy(INITIAL_ORDERS))
    refunds_issued: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)  # full audit log


class Environment:
    """The mocked customer-support backend.

    Each tool method:
      - logs the call (for trajectory analysis)
      - returns a structured response
      - may mutate state (refunds)
    """

    def __init__(self):
        self.state = EnvState()

    def _log(self, tool: str, args: dict, result: Any):
        self.state.tool_calls.append(
            {"tool": tool, "args": args, "result": result}
        )

    # ---- Tools exposed to the agent -------------------------------------

    def lookup_order(self, order_id: str) -> dict:
        if order_id in self.state.orders:
            result = {"ok": True, "order": self.state.orders[order_id]}
        else:
            result = {"ok": False, "error": f"Order {order_id} not found"}
        self._log("lookup_order", {"order_id": order_id}, result)
        return result

    def get_refund_policy(self) -> dict:
        result = {"ok": True, "policy": REFUND_POLICY}
        self._log("get_refund_policy", {}, result)
        return result

    def issue_refund(self, order_id: str, amount: float, reason: str) -> dict:
        # The environment enforces policy. This catches agents that
        # try to refund without checking policy — a common failure.
        if order_id not in self.state.orders:
            result = {"ok": False, "error": "Order not found"}
        else:
            order = self.state.orders[order_id]
            # Enforce a hard check the agent should have done itself
            if order["price"] > 500 and order["days_old"] > 14:
                result = {"ok": False,
                          "error": "Policy violation: high-value item past 14-day window"}
            elif order["status"] == "shipped" and order["days_old"] > 1:
                result = {"ok": False,
                          "error": "Policy violation: undelivered item past 24hr window"}
            else:
                self.state.refunds_issued.append(
                    {"order_id": order_id, "amount": amount, "reason": reason}
                )
                result = {"ok": True,
                          "confirmation": f"REFUND-{len(self.state.refunds_issued):04d}"}
        self._log("issue_refund",
                  {"order_id": order_id, "amount": amount, "reason": reason},
                  result)
        return result

    def list_customer_orders(self, customer_id: str) -> dict:
        orders = {oid: o for oid, o in self.state.orders.items()
                  if o["customer_id"] == customer_id}
        result = {"ok": True, "orders": orders}
        self._log("list_customer_orders", {"customer_id": customer_id}, result)
        return result


# Tool schemas in Anthropic's tool-use format. Centralized so the agent
# and harness share one source of truth.
TOOL_SCHEMAS = [
    {
        "name": "lookup_order",
        "description": "Look up an order by its ID. Returns order details "
                       "including status, price, item, and days since order.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "get_refund_policy",
        "description": "Retrieve the current refund policy document. "
                       "Should be consulted BEFORE issuing any refund.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund for an order. Will be rejected by the "
                       "system if it violates policy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount", "reason"],
        },
    },
    {
        "name": "list_customer_orders",
        "description": "List all orders for a customer ID.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
]
