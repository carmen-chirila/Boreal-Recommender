"""Boréal Marché: recommendation engine (Streamlit application).

This app NEVER mines anything. It only loads models/rules.json and models/metadata.json,
which were produced offline by notebooks/01_mining.ipynb.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so that 'recommender.py' is always found

from recommender import (TOP_K, STRENGTH_LEVELS, explain, items_in_rules, load_rules, no_rule_reason,
                         popular_products, recommend, rule_status, strength)

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title='Boréal Marché: recommendations', page_icon='🛒', layout='wide')


# ----------------------------------------------------------------------------- data (loaded from files)
@st.cache_data
def load_data():
    rules = load_rules(ROOT / 'models' / 'rules.json')
    with open(ROOT / 'models' / 'metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    products = pd.read_csv(ROOT / 'data' / 'products.csv')
    products['Item'] = products['Item'].str.strip()
    return rules, meta, products


try:
    rules, meta, products = load_data()
except FileNotFoundError as e:
    st.error(f'File not found: {e.filename}. Run notebooks/01_mining.ipynb (Phase 5) first, '
             'and make sure data/products.csv is in the repository.')
    st.stop()

INFO = {row.Item: {'en': row.Item_EN, 'category': row.Category, 'price': float(row.Price)}
        for row in products.itertuples()}
IN_RULES = items_in_rules(rules)
ITEM_SUPPORT = meta['item_support']
N_BASKETS = meta['data']['baskets']
THRESHOLDS = meta['thresholds']


def name(item):
    return INFO[item]['en']


def norm(text):
    """lower case, no accents: so that 'cafe' finds 'Café moulu'"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()


# ----------------------------------------------------------------------------- basket (session state)
if 'basket' not in st.session_state:
    st.session_state.basket = []


def add_item(item):
    if item not in st.session_state.basket:
        st.session_state.basket.append(item)


def remove_item(item):
    if item in st.session_state.basket:
        st.session_state.basket.remove(item)


def clear_basket():
    st.session_state.basket = []


basket = st.session_state.basket
total = sum(INFO[i]['price'] for i in basket)

STRENGTH_COLORS = {'Very strong': '#a855f7', 'Strong': '#22c55e', 'Moderate': '#f59e0b', 'Weak': '#94a3b8'}


def pill(text, color):
    return (f'<span style="background:{color}22;color:{color};border:1px solid {color}66;padding:2px 9px;'
            f'border-radius:10px;font-size:0.8rem;margin-right:6px;white-space:nowrap">{text}</span>')


def rules_table(rule_list, result=None):
    rows = []
    for r in rule_list:
        row = {'If the basket contains': ' + '.join(sorted(name(i) for i in r.antecedent)),
               'Then suggest': name(r.consequent),
               'Support': f'{r.support:.2%}', 'Confidence': f'{r.confidence:.1%}', 'Lift': f'{r.lift:.2f}'}
        if result is not None:
            row['Status'] = rule_status(r, result)
        rows.append(row)
    return pd.DataFrame(rows)


def show_popular(exclude):
    st.markdown('**Most popular products**')
    for item, share in popular_products(ITEM_SUPPORT, exclude=exclude, n=TOP_K):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f'**{name(item)}**')
            left.caption(f'{INFO[item]["category"]} · bought in {share:.1%} of baskets')
            right.markdown(f'**${INFO[item]["price"]:.2f}**')


def rare_product_notes():
    notes = []
    for item in basket:
        reason = no_rule_reason(item, IN_RULES, ITEM_SUPPORT, THRESHOLDS, N_BASKETS)
        if reason:
            notes.append(f'**{name(item)}** {reason}')
    return notes


# ----------------------------------------------------------------------------- header
st.markdown('## 🛒 Boréal Marché: Recommendation Engine')
st.caption(f'Association rules · Apriori algorithm · {N_BASKETS:,} baskets · {len(products)} products · '
           'rules are mined offline and loaded from a file')

m1, m2, m3, m4 = st.columns(4)
m1.metric('Rules loaded', f'{len(rules):,}')
m2.metric('Products', len(products))
m3.metric('Items in basket', len(basket))
m4.metric('Basket total', f'${total:,.2f}')

with st.sidebar:
    st.markdown('### About the rules')
    st.write(f"Mined on **{meta['mining_date']}** from {N_BASKETS:,} baskets.")
    st.write(f"Minimum support {THRESHOLDS['min_support']:.1%} · minimum confidence "
             f"{THRESHOLDS['min_confidence']:.0%} · minimum lift {THRESHOLDS['min_lift']}.")
    cov = meta['coverage']
    st.write(f"This system can recommend **{cov['recommendable_products']} of {cov['total_products']}** products.")
    st.markdown('### Strength label (from the lift)')
    levels = sorted(STRENGTH_LEVELS)
    for i, (low, label) in enumerate(levels):
        high = levels[i + 1][0] if i + 1 < len(levels) else None
        st.write(f'**{label}**: lift {low:g} to {high:g}' if high else f'**{label}**: lift {low:g} or more')

col_cat, col_basket, col_reco = st.columns([1.1, 1, 1.4], gap='large')

# ----------------------------------------------------------------------------- zone 1: catalogue
with col_cat:
    st.subheader('1 · Catalogue')
    category = st.selectbox('Category', ['All categories'] + sorted(products['Category'].unique()),
                            label_visibility='collapsed')
    query = st.text_input('Search', placeholder='🔎 Search by name (English or French)',
                          label_visibility='collapsed')
    shown = products if category == 'All categories' else products[products['Category'] == category]
    if query.strip():
        q = norm(query.strip())
        shown = shown[[q in norm(f'{en} {fr}') for en, fr in zip(shown['Item_EN'], shown['Item'])]]
    st.caption(f'{len(shown)} product(s)')

    with st.container(height=620):
        for row in shown.itertuples():
            left, right = st.columns([3, 1])
            left.markdown(f'**{row.Item_EN}**')
            tag = ' · ℹ️ in no rule' if row.Item not in IN_RULES else ''
            left.caption(f'{row.Category} · ${row.Price:.2f} · {row.Item}{tag}')
            right.button('✓' if row.Item in basket else 'Add', key=f'add_{row.Item}',
                         on_click=add_item, args=(row.Item,), disabled=row.Item in basket)

# ----------------------------------------------------------------------------- zone 2: basket
with col_basket:
    st.subheader('2 · My basket')
    if not basket:
        st.info('Your basket is empty. Add a product from the catalogue to see recommendations.')
    for item in basket:
        left, right = st.columns([4, 1])
        left.markdown(f'**{name(item)}**')
        left.caption(f'{INFO[item]["category"]} · ${INFO[item]["price"]:.2f}')
        right.button('✕', key=f'rm_{item}', on_click=remove_item, args=(item,))
    st.button('Clear basket', on_click=clear_basket, disabled=not basket)
    st.markdown(f'**Total: ${total:,.2f}**')

# ----------------------------------------------------------------------------- zone 3: recommendations
with col_reco:
    st.subheader('3 · You may also like')
    result = recommend(rules, basket, k=TOP_K)

    if result.status == 'empty_basket':
        st.info(f'**{result.message}**  \nThis is the "cold start" case: with no product in the basket, '
                'no rule can fire.')
        show_popular(exclude=())

    elif result.status == 'no_rule_fires':
        st.warning(f'**{result.message}**')
        for note in rare_product_notes():
            st.info(f'ℹ️ {note}')
        st.caption('No rule matches these products (every product on the left of a rule must be in the basket). '
                   'As a fallback, here are the most popular products you do not have yet.')
        show_popular(exclude=basket)

    else:
        for rule in result.suggestions:
            item = rule.consequent
            with st.container(border=True):
                left, right = st.columns([4, 1])
                left.markdown(f'**{name(item)}**')
                left.caption(f'{INFO[item]["category"]} · {item}')
                right.markdown(f'**${INFO[item]["price"]:.2f}**')
                st.markdown('💡 **Why this?** ' + explain(rule, name_of=name))
                label = strength(rule.lift)
                st.markdown(
                    pill(f'Lift {rule.lift:.2f}', '#3b82f6') + pill(f'Confidence {rule.confidence:.0%}', '#22c55e')
                    + pill(f'Support {rule.support:.2%}', '#f59e0b')
                    + pill(f'{label} association', STRENGTH_COLORS[label]), unsafe_allow_html=True)

        for note in rare_product_notes():
            st.info(f'ℹ️ {note}')

        with st.expander('🔍 Rules that fired'):
            st.caption('The rules behind the suggestions above. This table is what makes every suggestion auditable.')
            st.dataframe(rules_table(result.suggestions), hide_index=True)
            if st.checkbox(f'Show every rule that matched this basket ({len(result.fired)})'):
                st.dataframe(rules_table(result.fired, result), hide_index=True)
