import pandas as pd
import numpy as np
from typing import Callable, Any

class Evaluator:
    def __init__(self):
        self.metrics = {
            'hits': 0,
            'total': 0,
            'satisfaction_scores': [],
            'usability_feedback': []
        }

    def qualitative_survey(self, recs: pd.DataFrame) -> float:
        """
        Stub: collect a 1–5 Likert score. Here we hard-code 4.0.
        """
        score = 4.0
        self.metrics['satisfaction_scores'].append(score)
        return score

    def record_usability(self, feedback: str):
        self.metrics['usability_feedback'].append(feedback)

    def evaluate_hit_rate(
        self,
        sample: pd.DataFrame,
        recommender: Any,
        sample_size: int = 100
    ) -> float:
        """
        For each randomly chosen row, ask the recommender to
        re-find something with the same cuisine+budget+city;
        compute fraction where at least one rec matches city or cuisine.
        """
        # Reset counters
        self.metrics['hits'] = 0
        self.metrics['total'] = 0

        for idx in sample.sample(sample_size, random_state=42).index:
            orig = sample.loc[idx]
            recs = recommender.recommend(
                cuisines=[orig['Primary Cuisine']],
                budget_range=(orig['Cost Category'], orig['Cost Category']),
                location=orig['City'],
                top_n=5
            )
            self.metrics['total'] += 1
            # count a “hit” if any recommended row shares cuisine OR city
            if (recs['Primary Cuisine'] == orig['Primary Cuisine']).any() \
               or (recs['City'] == orig['City']).any():
                self.metrics['hits'] += 1

        return self.metrics['hits'] / max(1, self.metrics['total'])

    def run_ab_test(
        self,
        data: pd.DataFrame,
        rec_a: Callable[..., pd.DataFrame],
        rec_b: Callable[..., pd.DataFrame],
        cohort_split: float = 0.5
    ) -> dict:
        """
        Simulate an A/B test by randomly splitting the data
        and collecting average satisfaction for each recommender.
        """
        n = len(data)
        idxs = data.sample(frac=1, random_state=0).index.to_list()
        split = int(cohort_split * n)
        cohort1 = data.loc[idxs[:split]]
        cohort2 = data.loc[idxs[split:]]

        sat1 = np.mean([
            self.qualitative_survey(rec_a(
                cuisines=[row['Primary Cuisine']],
                budget_range=(row['Cost Category'], row['Cost Category']),
                location=row['City'],
                top_n=5
            ))
            for _, row in cohort1.head(50).iterrows()
        ])
        sat2 = np.mean([
            self.qualitative_survey(rec_b(
                cuisines=[row['Primary Cuisine']],
                budget_range=(row['Cost Category'], row['Cost Category']),
                location=row['City'],
                top_n=5
            ))
            for _, row in cohort2.head(50).iterrows()
        ])

        return {
            'cohort1_satisfaction': sat1,
            'cohort2_satisfaction': sat2,
            'delta': sat2 - sat1
        }
