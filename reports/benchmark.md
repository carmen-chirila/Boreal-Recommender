# Benchmark: timing and memory

Boréal Marché, 1,000,000 baskets, 98 products, Apriori (`mlxtend`).

## Machine

- RAM: 8.2 GB total. Page file (swap): 24.7 GB.
- Free RAM when the runs were done: about 1.8 to 4 GB (other programs were open).
- Memory is measured as the peak RAM used by the mining process (on Windows: peak working set). When RAM is tight, Windows can move part of a process to disk, so the real need can be higher than the number shown.

## Phase 1: building the True/False table

| Item | Value |
|---|---|
| Table size | 1,000,000 baskets × 98 products (True/False) |
| Build time | 16.0 s in the last run (18.8 s and 19.6 s in earlier runs) |
| Memory of the table | 98 MB |
| Peak memory during the build (Python side) | 713 MB |

**Approach:** each basket gets a row number and each product gets a column number, and one numpy step sets `True` at every (row, column). No loops, no `pivot_table`. I did not measure the naive approaches. The brief warns that they can take minutes or run out of memory.

## Phase 2: Apriori runs

Settings used for every run:
- `max_len = 4` (itemsets of at most 4 products). This keeps the number of itemsets bounded and still allows rules with 3 items in the basket.
- Rules are kept when confidence is at least 0.30 (`association_rules`, metric `confidence`).
- Each run was done in its own separate Python process, so a crash or a MemoryError could not destroy the notebook.
- Every run was repeated 7 times. The time column gives the median, with the smallest and largest value in brackets (whole seconds). The memory column gives the median.

| min_support | Mode | Itemsets | Rules | Apriori time (s) | Peak memory (MB) | Outcome |
|---|---|---|---|---|---|---|
| 0.05 | default | 191 | 282 | 82 [76-117] | 4,172 | OK |
| 0.05 | `low_memory=True` | 191 | 282 | 3 [3-5] | 408 | OK |
| 0.02 | default | - | - | - | - | not attempted (needs about 13 GB) |
| 0.02 | `low_memory=True` | 635 | 2,207 | 6 [5-12] | 411 | OK |
| 0.01 | default | - | - | - | - | not attempted (needs about 13 GB) |
| 0.01 | `low_memory=True` | 2,132 | 4,988 | 14 [12-26] | 411 | OK |
| 0.005 | default | - | - | - | - | not attempted (needs about 13 GB) |
| 0.005 | `low_memory=True` | 6,519 | 13,892 | 30 [29-222] | 411 | OK |

The times change from one repeat to the next. The clearest case is `low_memory=True` at 0.005: the median is 30 s, and one repeat took 222 s. I did not investigate this. Other programs were open on the computer, which may have played a part, but I did not check. The memory numbers are very stable.

## Why the default mode runs out of memory

In the default mode, Apriori tests all candidate pairs at once. It builds one block of True/False values with one row per basket and one column per candidate pair (times 2, one per product in the pair). Each value takes 1 byte.

With 1,000,000 baskets, the size is: 1,000,000 × (number of pairs) × 2 bytes.

| min_support | Products that pass | Candidate pairs | Block of pairs |
|---|---|---|---|
| 0.05 | 72 | 2,556 | 5.1 GB |
| 0.02 | 93 | 4,278 | 8.6 GB |
| 0.01 | 93 | 4,278 | 8.6 GB |
| 0.005 | 93 | 4,278 | 8.6 GB |

The block is also copied once, so the real need is about 1.5 times larger (about 13 GB for 93 products). That is more than the 8.2 GB of RAM on this computer. The block gets bigger with the number of products that pass `min_support`, so a lower `min_support` makes it worse. Triples and 4-itemsets then need even more.

At min_support 0.05, the default mode did finish, and its cost was already high: a median of 4.2 GB and 82 s. In the first five runs, which I looked at in detail, the peak ranged from 3.7 to 4.8 GB, a little under the 5.1 GB calculated for the block alone. A likely reason is that Windows had moved part of the memory to disk because RAM was tight (see the note under "Machine"), but I did not check.

## The fix: `low_memory=True`

With `low_memory=True`, Apriori checks the candidates one at a time instead of building the whole block. Memory stays at about 410 MB, which is mostly the table itself (98 MB) and the Python tools, whatever the `min_support`.

At min_support 0.05, both modes found exactly the same 191 itemsets and 282 rules. The result is the same, only the cost changes.

**Surprise:** `low_memory=True` was also faster (a median of 3 s instead of 82 s, about 27 times faster). I did not measure why. A likely reason is that building and moving a 4 to 5 GB block costs more time than checking candidates one at a time. So on this data, the name is misleading: it is not a "slower but lighter" mode.

## What I could not show

The brief says to expect a `MemoryError`. On this computer I did not trigger it. For min_support 0.02 and lower, the calculation above says the default mode needs about 13 GB, while only 1.8 to 4 GB of RAM was free. Running it would have filled the RAM and the page file and made the computer freeze, so the notebook skipped those runs on purpose (a safety check before each run) and recorded the reason.

The evidence I do have:
1. The calculation of the block size (table above).
2. The run at 0.05, where the default mode already used a median of 4.2 GB and was about 27 times slower than `low_memory=True`.

I did not see the error message itself.

## Itemsets grow fast when min_support goes down

From min_support 0.05 to 0.005, the number of itemsets grows from 191 to 6,519 (34 times more), and the number of rules from 282 to 13,892.

## The Reusable Bag and the itemsets

At min_support 0.005, the Reusable Bag (in 90% of baskets) is in 2,805 of the 6,426 itemsets with 2 or more products (44%). My Phase 1 prediction said "almost every combination", which was too strong. It is still much more than a typical product (about 3% if all products were equally common), but more than half of the itemsets do not contain it.

## Phase 4: mining each season

Same method as Phase 2 (`low_memory=True`, `min_support = 0.005`, `max_len = 4`, confidence at least 0.30), one run per season.

| Season | Baskets | Itemsets | Rules | Apriori time | Peak memory |
|---|---|---|---|---|---|
| Summer (June to August) | 252,474 | 7,667 | 19,920 | 7.0 s | 179 MB |
| Winter (December to February) | 246,965 | 5,767 | 10,915 | 5.6 s | 178 MB |

A season is mined in a few seconds, so the cost of mining is not what limits how often the rules can be refreshed (see the conclusion in the notebook, section 4.4).

## Phase 6: coverage

**The system can recommend 57 of the 98 products (58%).** The 41 others are never suggested:

| Why | Products | Examples |
|---|---|---|
| Too rare: under 0.5% of baskets, the minimum support, so they are in no frequent itemset | 5 | Saffron, Truffle oil, Caviar, Foie gras, Aged balsamic vinegar (each in 0.08% of baskets, about 800) |
| Common enough, but no rule suggests them | 36 | Carrots (3.1% of baskets), Cookies (16.1%), Cheddar cheese, Dish soap, Baguette, Sliced bread, and the Reusable bag (90%) |

(Notebook section 6.1 splits the 36 further: products in no rule at all, and products that only appear on the left side of rules, where they can trigger a suggestion but are never suggested.)

Most of these products have no strong link with another product. Among the 6,719 rules kept before the redundant ones were removed, the smallest lift was 2.1, so a product that is bought independently of the others never gets a useful rule. The Reusable bag is the extreme case: it is in 90% of baskets, so every rule that ends with it has a lift of about 1 and was filtered out.

This is the reinforcement-bias risk of the brief, made visible: suggestions come only from a few strong shopping habits (for example BBQ and raclette), and about 42% of the catalogue (41 of 98 products) is never put in front of a customer by this system.

## Two things this approach cannot do

1. **Recommend a product that is rare or has no strong link with others** (41 of 98 products never, see above). To fix it, I would add a second recommender for the products that rules cannot reach: for example "popular in the same category", or item-to-item similarity computed from the customers' purchase history (collaborative filtering). Business rules (promoting a high-margin or new product) could also be added. The coverage measure above would then be repeated to check the effect.
2. **Adapt to the season or to the customer.** The rules are the same for everybody all year. Phase 4 showed that in winter, 27% of the simple year-round rules state a confidence more than 20% away from the real winter value. The `CustomerID` column is not used at all. To fix it, I would keep one rule set per season, re-mined automatically each month (one season takes about 6 seconds), and add a customer-level component that uses each customer's own purchase history.
