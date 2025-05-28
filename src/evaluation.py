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
        # Simulate a realistic Likert score with a bit of variation
        score = np.random.normal(4.0, 0.3)
        score = max(1.0, min(5.0, score))  # Clamp to 1–5
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
        compute fraction where at least one rec matches both city and cuisine.
        """
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

            match = (
                (recs['Primary Cuisine'] == orig['Primary Cuisine']) &
                (recs['City'] == orig['City'])
            ).any()

            if match:
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
        Also records simulated usability feedback.
        """
        n = len(data)
        idxs = data.sample(frac=1, random_state=0).index.to_list()
        split = int(cohort_split * n)
        cohort1 = data.loc[idxs[:split]]
        cohort2 = data.loc[idxs[split:]]

        scores1 = []
        for _, row in cohort1.head(50).iterrows():
            recs = rec_a(
                cuisines=[row['Primary Cuisine']],
                budget_range=(row['Cost Category'], row['Cost Category']),
                location=row['City'],
                top_n=5
            )
            score = self.qualitative_survey(recs)
            scores1.append(score)
            # Simulated usability feedback
            if score > 4.5:
                self.record_usability("Cohort A: Excellent suggestions!")
            elif score < 3.5:
                self.record_usability("Cohort A: Recommendations need improvement.")
            else:
                self.record_usability("Cohort A: Fair but could be better.")

        scores2 = []
        for _, row in cohort2.head(50).iterrows():
            recs = rec_b(
                cuisines=[row['Primary Cuisine']],
                budget_range=(row['Cost Category'], row['Cost Category']),
                location=row['City'],
                top_n=5
            )
            score = self.qualitative_survey(recs)
            scores2.append(score)
            # Simulated usability feedback
            if score > 4.5:
                self.record_usability("Cohort B: Excellent suggestions!")
            elif score < 3.5:
                self.record_usability("Cohort B: Recommendations need improvement.")
            else:
                self.record_usability("Cohort B: Fair but could be better.")

        sat1 = np.mean(scores1)
        sat2 = np.mean(scores2)

        return {
            'cohort1_satisfaction': float(sat1),
            'cohort2_satisfaction': float(sat2),
            'delta': float(sat2 - sat1)
        }
