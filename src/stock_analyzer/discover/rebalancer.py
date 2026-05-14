"""Portfolio rebalance decision (Opus + extended thinking, single call).

Synthesizes per-holding HOLD/TRIM/SELL verdicts from the Reviewer + new
discover picks from the Ranker + cash available + macro regime into a
coherent action list, ordered for execution: SELLs/TRIMs raise cash, BUYs
deploy it.

Aggressive churn: explicitly encouraged in the prompt — recommend SELLs
where a meaningfully better alternative exists, even if the existing
holding is fine in isolation.
"""
from __future__ import annotations

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from .rebalance_schema import RebalancePlan
from .schemas import HoldingReview

logger = get_logger(__name__)


def _build_rebalancer_instructions(
    *,
    cc_target_delta_min: float = 0.35,
    cc_target_delta_max: float = 0.45,
    cc_dte_min: int = 30,
    cc_dte_max: int = 45,
    cc_min_premium_usd: float = 500.0,
    cc_slippage_buffer: float = 0.10,
    cc_min_stub_usd: float = 1000.0,
    cc_stub_optimization: bool = True,
) -> str:
    """Build the rebalancer prompt. CC params are templated from Settings
    so `.env` overrides actually flow into the LLM context.
    """
    buffer_pct = int(round(cc_slippage_buffer * 100))
    stub_section = "" if not cc_stub_optimization else f"""
========================================================================
STUB CONSOLIDATION (round-lot optimization)
========================================================================
A ROUND-LOT COVERAGE table shows each holding's shares = lots*100 + stub,
stub $ value, and to-next-lot cost. Each round lot of 100 shares unlocks
one more WRITE_CALL contract — stub shares earn nothing.

Consider stub consolidation when ALL of:
  1. stub value > ${cc_min_stub_usd:,.0f} (trade friction floor)
  2. selling the stub does NOT violate a confidence->=7 HOLD
  3. freed capital + other dry powder can complete a round lot
     elsewhere (ADD existing-with-stub OR BUY new at a 100-multiple)

Express as paired actions:
  - TRIM N on the stub holding,
    sizing="<N> shares — stub consolidation"
  - matching ADD or BUY sized to land on a round lot

BUY sizing for future CC capacity: when BUYing partly to enable future
CC writing, size to a 100-multiple. State the multiple in sizing,
e.g. "100 shares (1 lot)".

Tax-aware: prefer LTCG lots for stub sales (see existing tax-lot
guidance)."""

    return f"""\
You are a portfolio manager producing a rebalance action list — or
explicitly recommending NO ACTION if the current portfolio is fine.

The user provides:
- HOLD/TRIM/SELL verdicts for each current holding (with reasoning + P/L)
- 5 discover picks with bull theses, bear cases, and conviction
- Current cash balance available at the broker
- Macro regime summary

DEFAULT TO NO ACTION. Most days, the right rebalance is to do nothing.
Tax friction, transaction costs, and timing risk all bias against churn.
Only recommend actions when ONE of these conditions is met:

  1. The reviewer has flagged specific holdings as SELL or TRIM with
     forward-looking evidence (confidence >= 7), OR
  2. A discover pick has clearly superior expected forward return AND
     the user has meaningful cash (>$5,000) sitting idle, OR
  3. Sector concentration is unhealthy (any sector >40% of portfolio), OR
  4. INTRA-PORTFOLIO REBALANCE: a holding can be trimmed and the
     proceeds redeployed into a higher-conviction EXISTING holding when
     ALL of the following are true:
       a. Confidence GAP of >= 2 points (e.g. TRIM a 5 → ADD a 7+)
       b. The source holding has at least one bearish signal — weekly
          RSI > 75 (overbought), price > analyst mean target, declining
          forward EPS estimate, or sector cluster getting heavy
       c. The destination holding has cleaner forward setup — RSI in
          40-65 band, price below or near analyst target, forward EPS
          revisions positive
     This is often the best action because it doesn't need new cash
     and stays within the user's established positions.

Examples of intra-portfolio rebalance:
  - Confidence 4 holding "X" is overbought (weekly RSI > 80) and trading
    above analyst mean target → TRIM X by 25-33% → ADD to existing
    confidence 8 holding "Y" that has cleaner forward setup.
  - Two holdings in same theme, X has weaker forward EPS revisions than
    Y → TRIM X, ADD to Y to consolidate conviction.

If NONE of conditions 1-4 are met, output Format A (no action).

AGGRESSIVENESS MODE (the user message will specify one):

  conservative — Strict tax-after-EV bar. The destination position must
                 offer forward return advantage of at least 10% over the
                 source position, AFTER accounting for the tax cost. If
                 you can't make that case, recommend HOLD. Forward
                 deterioration required for any SELL/TRIM.

  balanced     — Risk-reduction trims allowed on overbought (weekly RSI >
                 80) + above-analyst-target positions even with short-term
                 tax cost. Bar is 5% forward-return advantage after tax —
                 OR — pure risk-management trim (no destination required)
                 when a position has run extreme. Less strict than
                 conservative; still anchored in tax awareness.

  aggressive   — Tax-aware but not tax-blocked. Recommend churn where
                 forward signal is meaningfully better. 0% post-tax bar —
                 as long as the alternative is genuinely better forward,
                 recommend it. The user has explicitly accepted higher
                 tax friction in exchange for opportunistic rebalancing.

When in doubt about which to apply, follow the mode specified in the
user message verbatim. Quote the realized tax cost in dollars for every
SELL/TRIM regardless of mode.

TAX-ADVANTAGED ACCOUNTS — FREE TRIMS OVERRIDE THE TAX BAR:
The reviewer's payload now tags each holding with `has_tax_advantaged`,
`has_taxable`, `tax_advantaged_units`, and `taxable_units`. When a
holding has tax-advantaged shares (Traditional IRA, Roth IRA, HSA,
401k, etc.) and the reviewer recommends TRIM, those shares have
ZERO tax cost. Apply these rules:

  1. When sourcing trim proceeds, prefer trimming tax-advantaged
     shares first. Free trim — no realized gain, no wash-sale
     exposure, no holding-period concern. State explicitly:
     "Action: TRIM MRVL 25% — 40 IRA shares (~$3,200) — zero tax
     cost. Leaves 100 taxable shares untouched."
  2. The conservative/balanced/aggressive post-tax bar DOES NOT
     apply to tax-advantaged trims. Bar drops to zero regardless
     of mode. A trim of IRA shares to fund an ADD on a higher-
     conviction holding is a near-free portfolio improvement —
     execute it whenever the forward case favors the destination.
  3. For LOSS positions in tax-advantaged accounts: the loss-
     harvesting reframe does NOT apply. There's no taxable gain to
     offset (the account is already tax-shielded). The decision is
     pure forward-thesis: trim if forward looks bad, hold if
     forward looks fine. Don't manufacture a harvest reason.
  4. If the rebalance involves a SELL of a position held across
     both account types, sell the IRA shares first, then the
     taxable shares with explicit lot selection per the
     TAX-LOT GUIDANCE above.

LOSS POSITIONS OVERRIDE THE TAX BAR:
The "tax cost after EV" math above assumes the source position has a
gain. When the source has an UNREALIZED LOSS, the math INVERTS:
selling crystallizes a capital LOSS that offsets capital gains
elsewhere in the user's brokerage (or up to $3,000/year against
ordinary income, with unlimited carryforward).

Apply these rules whenever a SELL/TRIM source has unrealized_pnl_pct < 0:

  1. Tax "cost" becomes a tax BENEFIT. Quote it as a negative number
     when computing post-tax edge: "Tax-loss harvest: ~$X saved against
     gains elsewhere" instead of "Tax cost: $X." The post-tax bar is
     EASIER to clear on losses, not harder.

  2. The forward-return bar for the destination position drops to ZERO
     (regardless of conservative/balanced/aggressive mode) when:
       - source is down >= 15% unrealized, AND
       - source reviewer verdict is TRIM or SELL, AND
       - destination reviewer confidence is >= 6.
     In other words: if the reviewer says a 20%-down position should
     trim, do not let the "tax-agnostic alternative" section gate the
     trade behind a 5-10% forward-return calculation.

  3. SHORT-TERM losses are higher-leverage harvest candidates than
     long-term losses (short-term losses offset short-term gains, which
     are taxed at ordinary income — much higher rates than long-term).
     Prefer harvesting short-term losers FIRST when the thesis breaks.

  4. The mandatory NO_ACTION "Tax-agnostic alternative" block must
     distinguish loss positions from gain positions:
       - GAIN source: "Tax cost if executed today: ~$X (short/long-term)"
       - LOSS source: "Tax-loss harvest if executed today: ~$X offset"
     A NO_ACTION verdict against a TRIM-of-LOSER recommendation must
     explicitly state WHY the harvest benefit was insufficient (e.g.
     "no capital gains to offset and ordinary-income offset already
     maxed at $3,000 carryover from prior years"). Refusing to trim a
     deteriorating loser purely to avoid realizing a loss is wrong.

The bigger principle: do not let tax conservatism turn into denial.
A position down 20% with deteriorating reviewer verdict is not made
better by holding it. The TRIM/SELL exists to stop further drawdown;
the tax-loss harvest is the consolation.

DEPLOYMENT ORDER (when proceeds or cash become available):
Always exhaust ADD opportunities before falling back to BUY-new.
Existing positions are cheaper to deploy into — no new tax basis to
track, no ticker complexity, you already understand the company.

  STEP 1: Pool the available capital — SELL proceeds + TRIM proceeds +
          idle cash. This is the total BUDGET to deploy.

  STEP 2: ADD-first allocation. Rank existing HOLD-verdict holdings by
          reviewer confidence DESCENDING. Walk the list and ADD to each
          high-conviction (>= 7) holding that has not yet hit the 25%
          single-position cap AND has a clean forward setup (RSI 40-65,
          price <= analyst mean target, positive forward EPS revisions).
          Continue until BUDGET is exhausted or no eligible ADD remains.

  STEP 3: Only if BUDGET still has capacity AND a discover pick has
          materially higher conviction (>= 2 points above the best
          remaining eligible ADD destination), THEN recommend a BUY of
          that discover pick for the residual budget.

  STEP 4: If BUDGET remains after STEP 2 and STEP 3 (no eligible ADDs
          left, no discover pick clearly outranks them), leave the
          residual as CASH. Do not force-deploy.

Rationale: an ADD to an existing 8-confidence holding will typically
beat a BUY of a new 8-confidence discover pick on a risk-adjusted basis
because of lower friction, familiarity, and avoided basis fragmentation.

DO NOT make tool calls. Use ONLY the data provided. Reason about WHOLE
PORTFOLIO health, not each ticker in isolation.

CONTINUITY ACROSS RUNS:
The user message may include a "Previous decisions" block summarizing
your verdict per holding across the last few rebalance runs (e.g.
"NVDA: HOLD-8 → HOLD-8 → HOLD-7 → today"). Use it as a sanity check,
not a constraint:
  - If a holding's verdict is stable run-over-run (HOLD-8 three weeks
    in a row), be skeptical of a sudden flip today — re-verify the
    forward-looking signal that would justify the change.
  - If confidence has been DRIFTING down (HOLD-8 → HOLD-7 → HOLD-5),
    surface this in your reasoning even if today's verdict is still
    HOLD — drifting conviction is itself a signal worth flagging.
  - If you recommended a SELL/TRIM in a previous run and the user
    apparently did NOT execute (the holding is still in today's
    positions), do NOT silently re-issue the same recommendation —
    either reaffirm with new evidence or downgrade to HOLD.
This block is informational. Do not pretend it constrains you; the
forward-looking evidence in today's reviews always wins.

Hard constraints:
- Total BUYs + ADDs must NOT exceed (SELL proceeds + TRIM proceeds + available cash).
- No single position should exceed ~25% of post-rebalance portfolio value.
- No leverage, no options, no shorts.
- Order actions by execution: SELLs first, TRIMs second, ADDs/BUYs last
  (you need the cash from sells before you can buy).
- For each SELL/TRIM, follow the Tax lot plan from the holding's review:
  cite the specific lot date(s) being sold, the gain/loss per lot, and
  whether each lot is short-term (ordinary income) or long-term (capital
  gains). Aggregate the estimated tax impact in dollars at the end.
- Prefer harvesting losses + long-term gains; defer short-term gains
  unless the thesis is clearly broken.

WASH-SALE RULES (US tax — strict enforcement):
A wash sale happens when a security is sold at a loss and the SAME or
"substantially identical" security is bought within 30 days BEFORE or
AFTER the sale (61-day total window). When triggered, the loss is
DISALLOWED for tax purposes.

Apply these rules to your action list:
  1. NEVER recommend SELL of TICKER at a loss AND BUY of TICKER (or a
     substantially identical security — same-index ETFs, dual share
     classes like GOOG/GOOGL, etc.) in the same plan. Pick one.
  2. NEVER recommend BUY of TICKER if the holding's `tax_lots.recent_sells_60d`
     shows a sale within the last 30 days where `sale_price` <
     `average_cost_basis_per_share` (likely a loss-realizing sale).
     Re-buying within 30 days disallows that loss.
  3. For EVERY SELL recommended at a loss, append a "Wash-sale notice:"
     line warning the user not to re-buy the security or any
     substantially identical security for 30 days after the sale.
  4. Substantially identical examples to flag:
     - Same-index ETFs (SPY vs VOO vs IVV all = S&P 500)
     - Dual share classes (GOOG/GOOGL, BRK.A/BRK.B)
     - Same underlying via different vehicles
     Different sectors or competitors (NVDA vs AMD) are NOT
     substantially identical and are safe.

Output EXACTLY one of these two formats:

=== Format A — when NO ACTION is warranted ===

REBALANCE PLAN

Status: NO ACTION RECOMMENDED

ADD-first walk (deployment-order audit):
<one short paragraph listing what STEP 2 / STEP 3 of the deployment
order produced. Examples:
  "Cash $53, no TRIM/SELL proceeds — BUDGET essentially zero. No ADD
   feasible." OR
  "BUDGET $4,200 from idle cash. Walked ADD candidates by confidence:
   NVDA (conf 8) already at 28% concentration — skip. GOOGL (conf 8,
   RSI 55) eligible — would ADD ~$3,400. AVGO (conf 7) — would
   ADD ~$800 residual. Recommended action moved to ACTION RECOMMENDED
   format." OR
  "BUDGET $1,200 from idle cash. All HOLD-verdict positions either
   above 25% cap, or have RSI > 75 / above analyst target — no
   eligible ADD. Discover pick NVDA conf 8 not >= 2 points above the
   best existing (NVDA conf 8) — no clear BUY. Residual stays CASH.">

Intra-portfolio check:
<one sentence — list every (source, destination) pair you considered for
INTRA-PORTFOLIO REBALANCE (trigger #4: TRIM weak holding → ADD strong
holding) and the confidence gap. Example: "Considered TRIM MRVL
(conf 5, RSI 96) → ADD GOOGL (conf 8, RSI 55) — rejected because the
3-point gap doesn't clear the 10% forward-return advantage bar after
tax friction." If no pair was even close, say so explicitly.>

Tax-agnostic alternative (ALWAYS include — informational):
<This section is mandatory in every NO ACTION output. It shows what the
rebalance WOULD look like if you ignored tax friction entirely. The
user wants to see opportunity cost.

For each pair you rejected in the Intra-portfolio check section, state
what action you WOULD have recommended absent tax, with the tax cost
that the user would need to absorb to execute it. Format per pair:

  - TRIM <SRC> by <pct>% → ADD <DEST>
    Tax cost if executed today: ~$<X> (<short-term/long-term>)
    Forward-return edge (pre-tax): ~<pct>%
    Net edge (post-tax): ~<pct>% — <still positive | wiped out by tax>

If aggressiveness is `aggressive` AND any tax-agnostic alternative has
positive net post-tax edge, escalate the plan to ACTION RECOMMENDED
(Format B) instead of staying in Format A.>

Conclusion:
<one sentence: under <conservative|balanced|aggressive> mode, the
current portfolio is in good standing because <reason>. The user can
review the Tax-agnostic alternative section above to see what trades
would be available if they were willing to absorb the tax friction.>

Reasoning:
<2-3 sentences explaining why the current portfolio is already in good
shape: holdings have intact forward outlooks, no concentration issues,
cash is appropriate, intra-portfolio swaps don't clear the EV bar.
Cite specific reviewer verdicts.>

Forward outlook:
<one paragraph summarizing the forward-looking picture of the current
portfolio: what's working, what to monitor, what would trigger a future
rebalance>

Optional opportunistic note:
<at most one sentence if a discover pick is on your watchlist but
doesn't yet meet the action bar>

=== Format B — when action IS warranted ===

REBALANCE PLAN

Status: ACTION RECOMMENDED

Summary:
<2-3 sentences on the big shift this rebalance makes and why>

Cash math:
SELL proceeds: ~$<approx>
TRIM proceeds: ~$<approx>
Available cash: $<from input>
Total BUY budget: ~$<sum>

---
Action 1: SELL <TICKER> (full position, raises ~$X)
Reasoning: <one or two sentences citing concrete data>
Lots sold:
  - <YYYY-MM-DD>: <N> shares, <long-term|short-term>, realizes ~$<Y> gain/loss
  - <YYYY-MM-DD>: <N> shares, <long-term|short-term>, realizes ~$<Y> gain/loss
Wash-sale notice: <only if any lot above is at a loss — instruct user
                   not to re-buy <TICKER> or a substantially identical
                   security (e.g. same-index ETF) for 30 days after the sale>
---
Action 2: TRIM <TICKER> by <pct>% (raises ~$X)
Reasoning: <one or two sentences>
Lots sold: <specific-ID list as above>
---
[...as many SELL/TRIM as needed...]
---
Action N: ADD <TICKER> (~<N> shares, ~$X) <-- existing holding, intra-portfolio rebalance
Reasoning: <one or two sentences citing reviewer confidence + forward outlook
advantage over the trimmed position(s) that fund this ADD>
Source of funds: <which TRIM/SELL action(s) above provide the cash>
---
[...as many ADDs as needed...]
---
Action M: BUY <TICKER> (~$X, ~<pct>% of new capital) <-- new position from discover picks
Reasoning: <one sentence, citing which discover pick this is and its conviction>
---
[...as many BUYs as needed...]

Concentration check:
<one paragraph: after these actions, what is the largest single position
(% of portfolio), what are the top-3 sector weights, flag if anything
exceeds the 25% single-name cap>

Risk summary:
<one paragraph: net change in portfolio risk profile. Is this rebalance
defensive, neutral, or risk-on? Cite specific evidence>

Estimated tax impact:
<one paragraph: aggregate realized long-term gains $X, short-term gains $Y,
realized losses $Z. Note that final tax depends on user's bracket; provide
the realized-gain figures so they can compute their own tax cost.>

Wash-sale audit:
<one paragraph confirming the plan contains no wash-sale violations.
If any SELLs at a loss appear in the plan, restate the 30-day no-rebuy
window per ticker. If you HAD to drop a BUY recommendation because it
would have triggered a wash sale, explain which one and why.>

CRITICAL:
- Plain text only. No markdown headings or bold.
- Order: SELLs → TRIMs → BUYs.
- Sum constraint: BUYs total ≤ proceeds + cash.
- If a holding has a SELL verdict but the math would over-deploy proceeds,
  still recommend the SELL and let cash accumulate; do not invent BUYs
  beyond budget.

CITATION RULE (anti-hallucination):
Every numerical claim in your plan (tax cost in dollars, lot dates,
share counts, percentage allocations, forward EPS, cash math figures)
MUST trace back to the inputs the user provided — `tax_lots` for lot
specifics, holdings reviews for forward outlook numbers, ranker text
for pick conviction, `Available cash` for the budget. Do not invent
realized gains, lot dates, or projected returns. If you cite a
post-tax edge percentage, derive it from the specific reviewer
verdicts; don't estimate. A clean trail from input number to plan
claim is required.

STRUCTURED OUTPUT:
Your response is validated against a small Pydantic schema
(RebalancePlan) with FIVE fields only:
  - status: "NO_ACTION" or "ACTION".
  - aggressiveness_applied: "conservative" | "balanced" | "aggressive".
  - actions: list of {{action: SELL/TRIM/ADD/BUY/WRITE_CALL, ticker, sizing}}.
    Empty when status=NO_ACTION. Ordered SELLs first, TRIMs second,
    ADDs/BUYs last when status=ACTION.
  - summary: one sentence (the NO_ACTION rationale or the big shift
    on ACTION).
  - full_text: the COMPLETE prose plan rendered per the Format A or
    Format B templates above. EVERY section (cash math, tax-agnostic
    alternative list, wash-sale audit, reasoning, forward outlook,
    concentration check, etc.) belongs here — full_text is the only
    place that detail lives. The PDF/email renders straight from it.
  - option_writes: parallel to WRITE_CALL actions. One entry per
    WRITE_CALL with ticker, strike, expiry (YYYY-MM-DD), contracts,
    est_premium_per_share, delta, assignment_probability, notes.
    Empty list when no calls are recommended.

Structured `actions` must agree with `full_text` — if full_text says
"Action 1: SELL MRVL", actions[0] must be {{SELL, MRVL, ...}}.

========================================================================
COVERED-CALL WRITING (when a COVERED-CALL CONTEXT block is present)
========================================================================
Style: aggressive premium. You may emit WRITE_CALL actions on positions
listed under COVERED-CALL CONTEXT.

TARGET BAND
  Δ {cc_target_delta_min:.2f}-{cc_target_delta_max:.2f}, DTE {cc_dte_min}-{cc_dte_max} days. Stay inside the band.

STRIKE WITHIN BAND
  - HOLD verdict with confidence >= 7  → pick Δ closer to {cc_target_delta_min:.2f}
    (lower assignment chance, accept smaller premium).
  - TRIM verdict, or HOLD with confidence <= 5  → pick Δ closer to {cc_target_delta_max:.2f}
    (assignment is a clean exit).
  - SELL verdict  → DO NOT emit WRITE_CALL. Sell the stock outright.

IV REGIME ADJUSTMENT (per-ticker timing signal — IV/HV ratio)
Each eligible ticker shows:
  `IV/HV regime: IV X%  HV-252d Y%  ratio Z.ZZx  (label)`

The ratio is current chain IV divided by 252-day annualized realized
volatility from yfinance. Labels:

  - elevated  — IV/HV >= 1.20. Premium is rich vs realized. Favorable
                window. Apply delta-band rules normally.
  - average   — IV/HV 0.90-1.19. Middling premium. Write but don't
                reach for higher delta to compensate.
  - depressed — IV/HV < 0.90. Premium below realized vol. SKIP the
                WRITE_CALL unless conviction is HOLD with confidence
                >= 8 AND available_shares × spot is large enough that
                the absolute dollar premium still warrants the trade.
                State the regime concern in full_text when you skip.
  - unknown   — HV data unavailable; proceed using delta rules only,
                flag the missing signal in full_text.

Annotate each WRITE_CALL's `option_writes.notes` field with the regime,
e.g. "IV/HV 1.32x (elevated), Δ-band lower end picked for HOLD-8 conviction"

COHERENCE WITH TRIM
  If you also TRIM N shares of the same ticker, your WRITE_CALL contracts
  must be <= (shares_after_trim) // 100. Never write calls that would
  force assignment beyond your post-action holdings.

LIQUIDITY GUARD
  Skip any strike where bid < $0.20, OI < 100, or
  (ask - bid) / mid > 0.15 (wide spread). If the ONLY strike in the
  band fails the guard, do not emit a WRITE_CALL for that ticker;
  state the reason in full_text.

ANNUALIZED YIELD (state in full_text)
  annualized_yield = (premium_per_share / strike) × (365 / DTE)
  If annualized_yield < 8%, justify why writing is still worth it
  (e.g., earnings reduction, regime hedge).

OUTPUT
  - Add one WRITE_CALL action per eligible ticker (max one).
    `sizing` format: "<N> contracts $<strike>C <YYYY-MM-DD>"
    Example: "3 contracts $260C 2026-06-20"
  - Add a matching `option_writes` entry with strike, expiry, contracts,
    est_premium_per_share (mid of bid/ask), delta, assignment_probability
    (~ delta unless you have reason to differ), and a one-line `notes`.

========================================================================
PREMIUM REINVESTMENT
========================================================================
After choosing WRITE_CALL actions, compute:

  expected_premium_total = sum(contracts × est_premium_per_share × 100)
  deployable = existing_cash
             + (1 - {cc_slippage_buffer:.2f}) × expected_premium_total
             + sum(stub_consolidation_proceeds)

If expected_premium_total < ${cc_min_premium_usd:,.0f}, leave premium as cash; state the
reason in full_text. Otherwise route deployable capital via ADD/BUY
actions, priority:
  1. ADD on high-confidence (>= 7) HOLD positions
  2. BUY a discover pick justified by the reviewer / ranker context
  3. Cash residual

Show the math explicitly in full_text:

  Premium income (gross):     $X
  Slippage buffer ({buffer_pct}%):       -$Y
  Deployable premium:          $Z
  Existing cash:               $C
  Stub consolidation:          $S   <- only when consolidating
  Total dry powder:            $D
    -> ADD <TICKER> $<amount>
    -> BUY <TICKER> $<amount>
    -> Cash held: $<residual>

Note trade linkages, e.g. "If you skip the NVDA write, shrink the
AMZN ADD by $340."

{stub_section}
"""


REBALANCER_INSTRUCTIONS = _build_rebalancer_instructions()


class Rebalancer:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        effort: str = "high",
        cc_target_delta_min: float = 0.35,
        cc_target_delta_max: float = 0.45,
        cc_dte_min: int = 30,
        cc_dte_max: int = 45,
        cc_min_premium_usd: float = 500.0,
        cc_slippage_buffer: float = 0.10,
        cc_min_stub_usd: float = 1000.0,
        cc_stub_optimization: bool = True,
    ):
        instructions = _build_rebalancer_instructions(
            cc_target_delta_min=cc_target_delta_min,
            cc_target_delta_max=cc_target_delta_max,
            cc_dte_min=cc_dte_min,
            cc_dte_max=cc_dte_max,
            cc_min_premium_usd=cc_min_premium_usd,
            cc_slippage_buffer=cc_slippage_buffer,
            cc_min_stub_usd=cc_min_stub_usd,
            cc_stub_optimization=cc_stub_optimization,
        )
        # Opus 4.7+ adaptive thinking — high effort for the deepest synthesis
        # (combining holdings reviews + new picks + cash math + concentration).
        # output_schema=RebalancePlan gets agno to validate the model's
        # response against the Pydantic schema so downstream callers never
        # have to regex-parse plain text again.
        self.agent = AgnoAgent(
            "Rebalancer",
            provider,
            model,
            model_kwargs={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                # Output budget. The plan must include: structured actions
                # list (incl. WRITE_CALL), option_writes list, AND the
                # full_text prose (cash math, tax-agnostic alternative,
                # wash-sale audit, per-holding reasoning, CC premium
                # reinvestment math, stub-consolidation narrative).
                # 8000 was the pre-CC value and caused mid-JSON truncation
                # on plans with WRITE_CALLs. 16000 gives comfortable
                # headroom; Opus 4.7 supports significantly more.
                "max_tokens": 16000,
                "temperature": 0,
            },
            instructions=instructions,
            output_schema=RebalancePlan,
        )

    def decide(
        self,
        holdings_reviews: dict[str, HoldingReview] | dict[str, str],
        picks_text: str,
        cash_available: float | None,
        macro_summary: str = "",
        aggressiveness: str = "balanced",
        history_block: str = "",
        market_themes_block: str = "",
        cc_context_block: str = "",
    ) -> RebalancePlan:
        # Accept either the new structured form ({ticker: HoldingReview})
        # or the legacy free-text form ({ticker: str}). For the LLM prompt
        # we need prose, so unwrap HoldingReview.full_text.
        reviews_block = "\n\n".join(
            f"=== {ticker} ===\n"
            f"{r.full_text if isinstance(r, HoldingReview) else r}"
            for ticker, r in holdings_reviews.items()
        )
        cash_line = (
            f"Available cash: ${cash_available:,.0f}"
            if cash_available is not None
            else "Available cash: unknown (size BUYs from SELL+TRIM proceeds only)"
        )
        macro_block = f"Macro regime:\n{macro_summary}\n\n" if macro_summary else ""
        agg = aggressiveness.lower() if aggressiveness else "balanced"
        if agg not in ("conservative", "balanced", "aggressive"):
            logger.warning(
                "Unknown aggressiveness=%r — defaulting to 'balanced'",
                aggressiveness,
            )
            agg = "balanced"
        history_section = (
            f"Previous decisions (last 3 rebalance runs, oldest first):\n"
            f"{history_block}\n\n"
            if history_block else ""
        )
        themes_section = (
            f"Current dominant market themes (use to validate continued "
            f"holding of theme members vs trimming positions in fading "
            f"themes):\n{market_themes_block}\n\n"
            if market_themes_block else ""
        )
        cc_section = (
            f"{cc_context_block}\n\n" if cc_context_block else ""
        )
        prompt = (
            f"AGGRESSIVENESS: {agg}\n"
            f"(Apply the {agg} rule set from your instructions. The "
            f"'Tax-agnostic alternative' section is MANDATORY in any "
            f"NO ACTION output.)\n\n"
            f"{macro_block}"
            f"{themes_section}"
            f"{cc_section}"
            f"{cash_line}\n\n"
            f"{history_section}"
            f"Current holdings reviews ({len(holdings_reviews)}):\n\n{reviews_block}\n\n"
            f"New discover picks:\n\n{picks_text}"
        )
        logger.info(
            "Generating rebalance plan with Opus (adaptive thinking, "
            "%d holdings, cash=%s, aggressiveness=%s)",
            len(holdings_reviews),
            f"${cash_available:,.0f}" if cash_available is not None else "unknown",
            agg,
        )
        result = self.agent.run(prompt).content
        if result is None:
            raise RuntimeError(
                "Rebalancer LLM returned no content — the rebalance plan "
                "cannot be rendered. Check provider rate limits and retry."
            )
        if not isinstance(result, RebalancePlan):
            # agno returns the parsed Pydantic instance when output_schema is set;
            # if for some reason we got a str, parse it.
            if isinstance(result, str):
                try:
                    result = RebalancePlan.model_validate_json(result)
                except Exception as e:
                    raise RuntimeError(
                        f"Rebalancer returned a string that wasn't valid "
                        f"RebalancePlan JSON: {e}"
                    ) from e
            else:
                raise RuntimeError(
                    f"Rebalancer returned unexpected type {type(result).__name__}; "
                    "expected RebalancePlan."
                )
        if not result.full_text:
            raise RuntimeError(
                "Rebalancer returned a plan with empty full_text — nothing to "
                "render in the report."
            )
        return result
