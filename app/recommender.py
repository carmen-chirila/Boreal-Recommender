"""Recommendation engine for Boréal Marché.

Pure Python: no Streamlit and NO Apriori. It only reads the rules saved by the notebook
(models/rules.json). Mining is expensive and offline; recommending is instant and online.

Products are identified by the French name used in the transactions (the 'Item' column).
"""
import json
from dataclasses import dataclass, field

TOP_K = 5   # at most 5 suggestions

# Strength label from the lift (documented in the README). The first line that matches wins.
STRENGTH_LEVELS = [(5.0, 'Very strong'), (3.5, 'Strong'), (2.0, 'Moderate'), (0.0, 'Weak')]


@dataclass(frozen=True)
class Rule:
    antecedent: frozenset      # the products on the left ("if the basket contains...")
    consequent: str            # the product on the right ("...then suggest")
    support: float
    confidence: float
    lift: float


@dataclass
class Result:
    status: str                                       # 'empty_basket' | 'no_rule_fires' | 'ok'
    message: str
    suggestions: list = field(default_factory=list)   # best rule per product, by descending lift, at most TOP_K
    fired: list = field(default_factory=list)         # every rule that applied to the basket, by descending lift


def strength(lift):
    """weak / moderate / strong / very strong, from the lift.
    The lift is rounded to 2 decimals first, so the label always agrees with the lift shown on screen
    (a lift displayed as 3.50 must not be labelled as if it were 3.49)."""
    lift = round(lift, 2)
    for threshold, label in STRENGTH_LEVELS:
        if lift >= threshold:
            return label
    return 'Weak'


def load_rules(path):
    """Read models/rules.json (one rule per entry: if, then, support, confidence, lift)."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return [Rule(frozenset(r['if']), r['then'], r['support'], r['confidence'], r['lift']) for r in raw]


def fired_rules(rules, basket):
    """Rules whose left side is entirely inside the basket (subset matching), and whose
    suggested product is NOT already in the basket. Strongest lift first."""
    basket = set(basket)
    out = [r for r in rules if r.antecedent <= basket and r.consequent not in basket]
    return sorted(out, key=lambda r: (-r.lift, -r.confidence, -len(r.antecedent)))


def best_rule_per_product(fired):
    """If two rules suggest the same product, keep the one with the higher lift.
    `fired` is already sorted by lift, so the first rule seen for a product is the strongest."""
    best = {}
    for r in fired:
        best.setdefault(r.consequent, r)
    return list(best.values())


def recommend(rules, basket, k=TOP_K):
    basket = set(basket)
    if not basket:
        return Result('empty_basket',
                      'Your basket is empty. No rule can fire yet, so we show our most popular products.')
    fired = fired_rules(rules, basket)
    if not fired:
        return Result('no_rule_fires',
                      'No rule applies to this basket, so we have no personalised suggestion.')
    return Result('ok', '', best_rule_per_product(fired)[:k], fired)


def rule_status(rule, result):
    """Why a rule that fired is, or is not, displayed (used in the audit table)."""
    if rule in result.suggestions:
        return 'shown'
    if any(s.consequent == rule.consequent for s in result.suggestions):
        return 'not shown: a rule with a higher lift suggests the same product'
    return f'not shown: only the {TOP_K} strongest suggestions are displayed'


def explain(rule, name_of=str):
    """Plain-language explanation of one suggestion."""
    ant = sorted(name_of(i) for i in rule.antecedent)
    listed = ant[0] if len(ant) == 1 else ', '.join(ant[:-1]) + ' and ' + ant[-1]      # "A", "A and B", "A, B and C"
    who = 'that' if len(ant) == 1 else 'these together'
    return (f"Because your basket contains {listed}, customers who buy {who} also buy "
            f"{name_of(rule.consequent)} {rule.confidence:.0%} of the time \u2014 "
            f"that is {rule.lift:.1f}\u00d7 more often than average.")


def popular_products(item_support, exclude=(), n=TOP_K):
    """Most popular products (fallback when no rule can be used)."""
    skip = set(exclude)
    ranked = sorted(item_support.items(), key=lambda kv: -kv[1])
    return [(item, share) for item, share in ranked if item not in skip][:n]


def items_in_rules(rules):
    """Every product that appears in at least one rule (left or right side)."""
    found = set()
    for r in rules:
        found |= r.antecedent
        found.add(r.consequent)
    return found


def no_rule_reason(item, in_rules, item_support, thresholds, n_baskets):
    """None if the product appears in some rule. Otherwise, a sentence that explains why it does not."""
    if item in in_rules:
        return None
    share = item_support.get(item)
    min_support = thresholds['min_support']
    if share is not None and share < min_support:
        return (f"is bought in only {share:.2%} of baskets (about {round(share * n_baskets):,} out of "
                f"{n_baskets:,}). The rules were mined with a minimum support of {min_support:.1%}, so a "
                f"product this rare can never be part of a frequent itemset, and therefore it appears in no rule.")
    return (f"is bought often enough, but no rule that contains it passes the filters "
            f"(confidence of at least {thresholds['min_confidence']:.0%} and lift above {thresholds['min_lift']}).")
