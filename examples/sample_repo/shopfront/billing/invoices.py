"""Invoice generation, proration and tax."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal

CENT = Decimal("0.01")


@dataclass
class LineItem:
    description: str
    amount: Decimal


@dataclass
class Invoice:
    customer_id: str
    period_start: date
    period_end: date
    currency: str = "EUR"
    items: list[LineItem] = field(default_factory=list)
    source_event_id: str | None = None

    @property
    def subtotal(self) -> Decimal:
        return sum((i.amount for i in self.items), Decimal("0"))


def prorate(monthly_price: Decimal, days_used: int, days_in_period: int) -> Decimal:
    """Charge for the used fraction of a billing period.

    Rounds with banker's rounding (ROUND_HALF_EVEN) to the cent. Plain float math and
    ROUND_HALF_UP over-charged customers by one cent on ~0.4% of mid-cycle upgrades (PR #118).
    """
    if not 0 <= days_used <= days_in_period:
        raise ValueError("days_used must be within the billing period")
    raw = monthly_price * Decimal(days_used) / Decimal(days_in_period)
    return raw.quantize(CENT, rounding=ROUND_HALF_EVEN)


def apply_tax(amount: Decimal, rate: Decimal) -> Decimal:
    """VAT is computed on the invoice subtotal, not per line, to avoid accumulated rounding."""
    return (amount * rate).quantize(CENT, rounding=ROUND_HALF_EVEN)


def build_invoice(
    customer_id: str,
    plan_price: Decimal,
    period_start: date,
    period_end: date,
    upgrade_day: int | None = None,
    source_event_id: str | None = None,
) -> Invoice:
    """Create the invoice for one billing period, prorating if the plan changed mid-cycle."""
    days = (period_end - period_start).days
    invoice = Invoice(customer_id, period_start, period_end, source_event_id=source_event_id)
    if upgrade_day is None:
        invoice.items.append(LineItem("Subscription", plan_price))
    else:
        invoice.items.append(
            LineItem("Subscription (prorated)", prorate(plan_price, days - upgrade_day, days))
        )
    return invoice
