"""
Intervention Simulator for EAPS

Most attrition studies stop at "who is likely to leave and why". This module goes
one step further: it takes an at-risk employee, applies a realistic HR action
(e.g. remove overtime, raise salary, improve work-life balance) to their profile,
re-runs the trained model, and checks whether the predicted risk of leaving
actually drops. That tests whether *fixing* the flagged problem really helps,
instead of just predicting and explaining it.
"""

from typing import Callable, Dict, List


# Each intervention is a small, realistic HR action and a function that edits
# a copy of the employee's raw input dict to reflect that action.
INTERVENTIONS: Dict[str, Callable[[dict], dict]] = {
    "remove_overtime": lambda d: {**d, "OverTime": "No"},
    "improve_work_life_balance": lambda d: {**d, "WorkLifeBalance": min(4, d.get("WorkLifeBalance", 3) + 1)},
    "increase_job_satisfaction": lambda d: {**d, "JobSatisfaction": min(4, d.get("JobSatisfaction", 3) + 1)},
    "raise_salary_15pct": lambda d: {**d, "MonthlyIncome": round(d.get("MonthlyIncome", 5000) * 1.15)},
    "promote_stock_options": lambda d: {**d, "StockOptionLevel": min(3, d.get("StockOptionLevel", 1) + 1)},
}

INTERVENTION_LABELS = {
    "remove_overtime": "Remove Overtime",
    "improve_work_life_balance": "Improve Work-Life Balance (+1)",
    "increase_job_satisfaction": "Increase Job Satisfaction (+1)",
    "raise_salary_15pct": "Raise Monthly Salary by 15%",
    "promote_stock_options": "Increase Stock Option Level (+1)",
}


def simulate_intervention(employee_data: dict, intervention_key: str, predict_fn: Callable[[dict], float]) -> dict:
    """Apply one intervention to an employee and compare risk before vs after.

    predict_fn: a function that takes a raw employee dict and returns the model's
    predicted probability of attrition (0-1). Keeping this as an injected callable
    means the simulator has no dependency on how the model/encoders/scaler are
    loaded — that stays entirely in api/main.py.
    """
    if intervention_key not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {intervention_key}. "
                          f"Choose from: {list(INTERVENTIONS.keys())}")

    before_prob = predict_fn(employee_data)
    modified_data = INTERVENTIONS[intervention_key](employee_data)
    after_prob = predict_fn(modified_data)

    reduction = before_prob - after_prob
    reduction_pct = (reduction / before_prob * 100) if before_prob > 0 else 0.0

    return {
        "intervention": intervention_key,
        "intervention_label": INTERVENTION_LABELS[intervention_key],
        "probability_before": round(before_prob, 4),
        "probability_after": round(after_prob, 4),
        "absolute_reduction": round(reduction, 4),
        "relative_reduction_pct": round(reduction_pct, 2),
        "helped": reduction > 0.001,
        "changed_fields": {
            k: v for k, v in modified_data.items()
            if employee_data.get(k) != v
        },
    }


def simulate_all_interventions(employee_data: dict, predict_fn: Callable[[dict], float]) -> List[dict]:
    """Run every known intervention on the employee and rank by how much it helps."""
    results = [
        simulate_intervention(employee_data, key, predict_fn)
        for key in INTERVENTIONS
    ]
    results.sort(key=lambda r: r["absolute_reduction"], reverse=True)
    return results
