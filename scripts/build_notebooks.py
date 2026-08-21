"""Builds notebooks/01_*.ipynb, 02_*.ipynb, 03_*.ipynb from templates."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ============================================================
# NOTEBOOK 1: Airline
# ============================================================
nb1_cells = [
    md("# Case 1 — Regional Airline × Fuel Price Shock\n"
       "**Winter Consulting Capstone, Consulting & Analytics Club, IIT Guwahati**\n\n"
       "**Business question:** A regional airline connecting Tier-2/3 cities faces a sharp "
       "FY25 ATF (jet fuel) price shock. Leadership must pick one primary strategic path "
       "over the next 18-24 months:\n"
       "- **A.** Network Optimization & Dynamic Pricing\n"
       "- **B.** Government-Backed Route Expansion (UDAN)\n"
       "- **C.** Structural Operating Model Redesign\n\n"
       "**Data note:** the original case is a strategy brief with no dataset attached. "
       "`src/data_generation/generate_airline_data.py` builds a route-economics dataset "
       "(48 routes × 12 months) from first-principles aviation unit economics — "
       "`revenue = seats × load_factor × fare + ancillary`, "
       "`fuel_cost = flights × block_hours × fuel_burn_per_hr × ATF_price` — calibrated so "
       "pre-shock margins land in the case's stated 'thin but positive' range."),
    code("import sys, pathlib\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n"
         "import pandas as pd\n"
         "pd.set_option('display.max_columns', 20)\n"
         "from data_generation.generate_airline_data import generate\n"
         "df = generate()\n"
         "df.head()"),
    md("## 1. Data overview"),
    code("df.describe(include='all').T"),
    md("## 2. Fuel price shock vs network profitability\n"
       "Answers: *how much of current losses are fuel-driven vs structurally embedded?*"),
    code("from case1_airline.analysis import load, fig1_fuel_price_vs_margin, fig2_loss_attribution, fig3_route_viability, fig4_scenario_comparison\n"
         "df = load()\n"
         "fig1_fuel_price_vs_margin(df)\n"
         "from IPython.display import Image, display\n"
         "display(Image('../reports/figures/case1_fuel_price_vs_margin.png'))"),
    code("attribution = fig2_loss_attribution(df)\n"
         "display(Image('../reports/figures/case1_loss_attribution.png'))\n"
         "attribution.head()"),
    md("## 3. Which routes are fundamentally broken vs salvageable?"),
    code("verdicts, pct_unviable = fig3_route_viability(df)\n"
         "display(Image('../reports/figures/case1_route_viability.png'))\n"
         "print(f'{pct_unviable:.0%} of routes structurally unviable post-shock')\n"
         "verdicts['verdict'].value_counts()"),
    md("## 4. Scenario model: Option A vs B vs C"),
    code("fig4_scenario_comparison()\n"
         "display(Image('../reports/figures/case1_scenario_comparison.png'))"),
    md("## 5. Recommendation\n\n"
       "**Primary: Option A (Network Optimization & Dynamic Pricing)**, applied to the ~52% "
       "of routes that remain healthy or recoverable post-shock — fastest time-to-impact "
       "(3-6 months) and directly attacks the largest loss driver (fuel-cost pass-through via "
       "yield management).\n\n"
       "**Supporting: Option B (selective UDAN expansion)**, targeted only at the "
       "structurally-unviable route subset, where guaranteed minimum revenue is the "
       "difference between operating and grounding the route — not applied network-wide, "
       "since the trade-off (capped pricing, mandatory capacity) would cap margin recovery "
       "on routes that don't need it.\n\n"
       "**Deferred: Option C (structural redesign)** — correct long-term direction (fleet mix, "
       "cost base) but 12-24 month time-to-impact is too slow to be the *primary* response to "
       "an 18-24 month decision window; flagged as a parallel workstream, not the headline bet.\n\n"
       "**Explicit trade-off accepted:** short-term margin will stay compressed on UDAN routes "
       "in exchange for network stability; the airline is consciously *not* pursuing aggressive "
       "capacity growth until the core network is fuel-shock-resilient.\n\n"
       "**KPIs to track:** CASK, route-level EBITDA margin, cash burn rate."),
]

# ============================================================
# NOTEBOOK 2: OTT
# ============================================================
nb2_cells = [
    md("# Case 2 — OTT Viewer Retention: Diagnosing Engagement Patterns\n"
       "**Winter Consulting Capstone, Consulting & Analytics Club, IIT Guwahati**\n\n"
       "**Business question:** viewers disengage during Season 1 of original series. "
       "The platform wants to know: (1) why viewers drop off, (2) which factors matter most, "
       "(3) what content/product changes would help.\n\n"
       "**Data note:** the case links a real episode-level engagement dataset that isn't "
       "reachable in this environment. `src/data_generation/generate_ott_data.py` builds a "
       "structurally identical synthetic dataset (same column design: pacing, cognitive load, "
       "episode structure, drop-off outcomes) with a deliberately embedded ground-truth signal, "
       "so the modeling below has real relationships to recover rather than random noise."),
    code("import sys, pathlib\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n"
         "import pandas as pd\n"
         "from data_generation.generate_ott_data import generate\n"
         "df = generate()\n"
         "df.head()"),
    md("## 1. EDA — where in the season do viewers drop off?"),
    code("from case2_ott.eda import load, fig1_dropoff_by_episode, fig2_correlation_heatmap, fig3_pacing_cogload_scatter, fig4_genre_comparison\n"
         "df = load()\n"
         "fig1_dropoff_by_episode(df)\n"
         "from IPython.display import Image, display\n"
         "display(Image('../reports/figures/case2_dropoff_by_episode.png'))"),
    code("fig2_correlation_heatmap(df)\n"
         "display(Image('../reports/figures/case2_correlation_heatmap.png'))"),
    code("fig3_pacing_cogload_scatter(df)\n"
         "display(Image('../reports/figures/case2_pacing_cogload_scatter.png'))"),
    md("## 2. Which factors have the greatest impact on drop-off risk?\n"
       "A gradient-boosted classifier predicts whether an episode is a high-drop-off-risk "
       "episode (above platform median); SHAP explains *why* per-feature."),
    code("from case2_ott.churn_model import train\n"
         "model, auc, mean_abs_shap = train()\n"
         "display(Image('../reports/figures/case2_shap_summary.png'))"),
    md("**Reading the SHAP plot:** high cognitive_load_score (red, right side) pushes "
       "drop-off risk up; high pacing_score (red, but on the *left*) pushes it down — faster "
       "episodes retain viewers better. This directly answers Key Question #2."),
    md("## 3. Can episodes be meaningfully segmented?"),
    code("from case2_ott.segmentation import run as run_segmentation\n"
         "seg_df, profile = run_segmentation()\n"
         "display(Image('../reports/figures/case2_segments_pca.png'))\n"
         "profile"),
    code("display(Image('../reports/figures/case2_segments_dropoff.png'))"),
    md("## 4. Recommendation\n\n"
       "**Primary action: editing/pacing review specifically for episodes 4-5** (the mid-"
       "season slump) — this is where the two dominant SHAP drivers (cognitive load, pacing) "
       "converge with the observed drop-off spike.\n\n"
       "**Segment-level action:** the highest-drop-off cluster is slow-paced + high-cognitive-"
       "load + below-average watch-through-percentage episodes. Any future episode matching "
       "this profile should get a pacing/complexity review *before* release, not after "
       "viewership data confirms the problem.\n\n"
       "**Prioritization:** pacing/cognitive-load edits are lower-cost and faster to ship than "
       "structural episode-count changes, so they're prioritized first; a recap-before-complex-"
       "episodes feature is a secondary, product-level lever (recap_provided shows a "
       "protective effect in the correlation heatmap).\n\n"
       "**Risk:** over-simplifying pacing/complexity risks diluting creative quality — "
       "mitigated by scoping the recommendation to episodes 4-5 specifically rather than a "
       "blanket pacing mandate across the season."),
]

# ============================================================
# NOTEBOOK 3: Telemedicine
# ============================================================
nb3_cells = [
    md("# Case 3 — Reducing Missed Appointments in a Telemedicine Platform\n"
       "**Winter Consulting Capstone, Consulting & Analytics Club, IIT Guwahati**\n\n"
       "**Role:** Product Manager. **Constraint:** no full redesign, no new hardware — "
       "only flow/default/in-app changes.\n\n"
       "**Data note:** no dataset was supplied for this case (pure product-strategy brief). "
       "`src/data_generation/generate_telemedicine_data.py` builds a 12,000-appointment event "
       "log consistent with the case's stated constraints (first-time-user share, variable "
       "connectivity, fixed doctor schedules), with a ground-truth no-show model that rewards "
       "realistic, actionable product levers."),
    code("import sys, pathlib\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n"
         "import pandas as pd\n"
         "from data_generation.generate_telemedicine_data import generate\n"
         "df = generate()\n"
         "df.head()"),
    md("## 1. Problem identification — grounded in system behavior, not surface symptoms"),
    code("from case3_telemedicine.eda import load, fig1_noshow_by_firsttime, fig2_noshow_by_reminder, fig3_noshow_by_leadtime, fig4_noshow_by_connectivity_slot\n"
         "df = load()\n"
         "fig1_noshow_by_firsttime(df)\n"
         "from IPython.display import Image, display\n"
         "display(Image('../reports/figures/case3_noshow_firsttime.png'))"),
    code("fig2_noshow_by_reminder(df)\n"
         "display(Image('../reports/figures/case3_noshow_reminder.png'))"),
    code("fig4_noshow_by_connectivity_slot(df)\n"
         "display(Image('../reports/figures/case3_noshow_heatmap.png'))"),
    md("## 2. No-show risk model + SHAP\n"
       "A model a real product team could run in production to flag high-risk appointments "
       "for an extra nudge or proactive reschedule offer."),
    code("from case3_telemedicine.noshow_model import train\n"
         "pipe, auc, mean_abs_shap = train()\n"
         "display(Image('../reports/figures/case3_shap_importance.png'))"),
    md("**Note on model performance:** AUC ≈ 0.64 and the top-20%-risk threshold approach "
       "(rather than default 0.5) are deliberate — no-shows are only ~16% of appointments, so "
       "a naive classifier that predicts \"will show up\" for everyone would look 84% "
       "accurate while being useless. The realistic framing here is: *rank* appointments by "
       "risk and act on the riskiest slice, which is exactly how a production risk-flagging "
       "feature would be used."),
    md("## 3. Solution design — quantified product levers\n"
       "Three concrete, low-lift interventions (default/flow changes only, per the case's "
       "constraints), each sized using observed before/after no-show rates for comparable "
       "segments."),
    code("from case3_telemedicine.intervention_simulation import run as run_sim\n"
         "impact, baseline_rate, projected_rate = run_sim()\n"
         "display(Image('../reports/figures/case3_intervention_impact.png'))\n"
         "impact"),
    md("## 4. Recommendation & prioritization\n\n"
       "**Primary: default all bookings to SMS+Push reminders** (currently a meaningful share "
       "get no reminder or a single channel) — largest projected impact, zero infrastructure "
       "change, pure default/flow change.\n\n"
       "**Secondary: pre-call connectivity check nudge** for users with a history of poor "
       "connectivity — smaller impact but addresses a distinct failure mode (technical "
       "friction, not forgetting).\n\n"
       "**Tertiary: T-24h re-confirmation nudge** for appointments booked >4 days out — "
       "smallest impact of the three, kept as a low-cost addition rather than a standalone bet.\n\n"
       "**Why not a bigger swing (e.g. requiring pre-payment):** case brief notes most "
       "consults are pay-after-session by design; changing that is a business-model change, "
       "not a flow change, and risks suppressing first-time bookings — outside the PM's "
       "stated mandate.\n\n"
       "**Success metrics:** primary = no-show rate (target: 15.7% → ~11.6%); "
       "supporting = doctor idle-time per week, rescheduling-related support tickets.\n\n"
       "**Risk & mitigation:** notification fatigue from more reminders — mitigated by "
       "scoping the upgrade to users currently under-reminded, not adding a 4th channel "
       "on top of already-well-reminded users."),
]

nbf.write(make_nb(nb1_cells), NB_DIR / "01_case1_airline_fuel_shock.ipynb")
nbf.write(make_nb(nb2_cells), NB_DIR / "02_case2_ott_retention.ipynb")
nbf.write(make_nb(nb3_cells), NB_DIR / "03_case3_telemedicine_noshow.ipynb")
print("Notebooks written.")
