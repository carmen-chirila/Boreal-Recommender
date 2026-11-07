"""Tests of the recommendation engine and of the five edge cases of the brief.

Run with:   python tests/test_edge_cases.py      (or: pytest tests)
Uses a small hand-made rule set, so the results do not depend on the mined data.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))

from recommender import (Rule, TOP_K, fired_rules, items_in_rules, load_rules, no_rule_reason,  # noqa: E402
                         popular_products, recommend, strength)

R = lambda ant, then, conf, lift: Rule(frozenset(ant), then, 0.05, conf, lift)

RULES = [
    R({'charbon'}, 'steak', 0.66, 6.2),
    R({'charbon'}, 'buns', 0.62, 6.1),
    R({'charbon'}, 'ketchup', 0.55, 6.0),          # same product (ketchup) suggested by two rules...
    R({'steak', 'buns'}, 'ketchup', 0.61, 6.7),    # ...this one has the higher lift
    R({'charbon', 'steak'}, 'biere', 0.52, 3.5),
    R({'sushi'}, 'nori', 0.80, 9.5),
]
ITEM_SUPPORT = {'charbon': 0.14, 'steak': 0.11, 'buns': 0.10, 'ketchup': 0.09, 'biere': 0.15,
                'sushi': 0.07, 'nori': 0.08, 'safran': 0.0008, 'sac': 0.90}
THRESHOLDS = {'min_support': 0.005, 'min_confidence': 0.30, 'min_lift': 1.1}
N_BASKETS = 1_000_000


def test_1_empty_basket():
    res = recommend(RULES, [])
    assert res.status == 'empty_basket' and res.suggestions == [] and 'empty' in res.message
    fallback = popular_products(ITEM_SUPPORT, exclude=[], n=TOP_K)
    assert fallback[0][0] == 'sac' and len(fallback) == TOP_K      # most popular first


def test_2_no_rule_fires():
    res = recommend(RULES, ['biere'])              # 'biere' is only on the right side of a rule
    assert res.status == 'no_rule_fires' and res.suggestions == [] and res.message
    fallback = popular_products(ITEM_SUPPORT, exclude=['biere'], n=TOP_K)
    assert all(item != 'biere' for item, _ in fallback)            # fallback never repeats a basket item


def test_3_item_already_in_basket_is_never_suggested():
    basket = ['charbon', 'steak']
    res = recommend(RULES, basket)
    assert res.status == 'ok'
    assert all(r.consequent not in basket for r in res.suggestions)
    assert all(r.consequent not in basket for r in res.fired)
    assert 'steak' not in [r.consequent for r in fired_rules(RULES, basket)]


def test_4_two_rules_same_product_keep_higher_lift():
    res = recommend(RULES, ['charbon', 'steak', 'buns'])
    ketchup = [r for r in res.suggestions if r.consequent == 'ketchup']
    assert len(ketchup) == 1                        # suggested once
    assert ketchup[0].lift == 6.7 and ketchup[0].antecedent == frozenset({'steak', 'buns'})
    assert len([r for r in res.fired if r.consequent == 'ketchup']) == 2   # both rules fired and are auditable


def test_5_very_rare_product_is_explained():
    in_rules = items_in_rules(RULES)
    assert 'safran' not in in_rules
    why = no_rule_reason('safran', in_rules, ITEM_SUPPORT, THRESHOLDS, N_BASKETS)
    assert why and '0.08%' in why and '0.5%' in why and '800' in why
    assert no_rule_reason('charbon', in_rules, ITEM_SUPPORT, THRESHOLDS, N_BASKETS) is None


def test_extra_ranking_limit_and_subset_matching():
    res = recommend(RULES, ['charbon', 'steak', 'buns'])
    lifts = [r.lift for r in res.suggestions]
    assert lifts == sorted(lifts, reverse=True)                    # strongest lift first
    assert len(res.suggestions) <= TOP_K
    # a rule with two items on the left needs BOTH items in the basket
    assert not any(r.antecedent == frozenset({'steak', 'buns'}) for r in fired_rules(RULES, ['steak']))


def test_extra_explanation_sentence():
    from recommender import explain
    one = explain(R({'charbon'}, 'steak', 0.66, 6.2))
    assert one == ('Because your basket contains charbon, customers who buy that also buy steak '
                   '66% of the time \u2014 that is 6.2\u00d7 more often than average.')
    three = explain(R({'a', 'b', 'c'}, 'steak', 0.5, 3.0))
    assert 'contains a, b and c, customers who buy these together' in three


def test_extra_strength_labels():
    assert [strength(x) for x in (1.5, 2.0, 3.4, 3.5, 4.9, 5.0, 12)] == \
        ['Weak', 'Moderate', 'Moderate', 'Strong', 'Strong', 'Very strong', 'Very strong']


def test_extra_real_rules_file_if_present():
    path = ROOT / 'models' / 'rules.json'
    if not path.exists():
        print('   (models/rules.json not found: real-file test skipped)')
        return
    real = load_rules(path)
    assert real and all(r.lift > 1 for r in real)
    assert all(r.consequent not in r.antecedent for r in real)
    assert len({(r.antecedent, r.consequent) for r in real}) == len(real)      # no duplicate rule


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    for n, f in tests:
        f()
        print(f'PASS  {n}')
    print(f'\n{len(tests)} tests passed')
