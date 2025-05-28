import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
from typing import List, Optional, Tuple


class ZomatoRecommender:
    def __init__(self, data: pd.DataFrame):
        # Make a copy and normalize text columns for robust matching
        self.data = data.copy()
        # lowercase key text fields
        self.data['City'] = self.data['City'].str.lower().str.strip()
        self.data['Primary Cuisine'] = self.data['Primary Cuisine'].str.lower().str.strip()

        # Precompute normalization constants
        self._min_rating = self.data['Rating'].min()
        self._max_rating = self.data['Rating'].max()
        self._min_votes = self.data['Votes'].min()
        self._max_votes = self.data['Votes'].max()

        # Define budget ordering
        self._budget_order = ['low', 'medium', 'high']

    def _filter(
        self,
        cuisines: Optional[List[str]],
        budget_range: Optional[Tuple[str, str]],
        location: Optional[str]
    ) -> pd.DataFrame:
        df = self.data.copy()  # Fresh copy for each filter operation

        # Cuisine filter
        if cuisines:
            cuisines = [c.strip().lower() for c in cuisines]
            pattern = '|'.join([re.escape(c) for c in cuisines])
            df = df[df['Primary Cuisine'].str.contains(
                r'\b(' + pattern + r')\b',
                case=False,
                na=False,
                regex=False
            )]

        # Budget filter
        if budget_range:
            low, high = [b.strip().lower() for b in budget_range]
            if low not in self._budget_order or high not in self._budget_order:
                raise ValueError(f"Budget categories must be one of {self._budget_order}")
            
            low_idx = self._budget_order.index(low)
            high_idx = self._budget_order.index(high)
            
            if low_idx > high_idx:
                raise ValueError(f"Budget range lower bound '{low}' cannot exceed upper bound '{high}'")
            
            allowable = self._budget_order[low_idx:high_idx+1]
            df = df[df['Cost Category'].str.lower().str.strip().isin(allowable)]

        # Location filter
        if location:
            loc = location.strip().lower()
            # First try exact match
            exact_matches = df[df['City'] == loc]
            if not exact_matches.empty:
                df = exact_matches
            else:
                # Then try fuzzy match
                candidates = get_close_matches(loc, df['City'].unique(), n=3, cutoff=0.5)
                if candidates:
                    df = df[df['City'].isin(candidates)]
                else:
                    # Finally try substring match
                    df = df[df['City'].str.contains(loc, case=False, na=False)]

        return df

    def _score(
        self,
        df: pd.DataFrame,
        w_rating: float = 0.7,
        w_votes: float = 0.3
    ) -> pd.Series:
        # Safely normalize rating and votes
        if (self._max_rating - self._min_rating) < 1e-6:
            nr = 0.5
        else:
            nr = (df['Rating'] - self._min_rating) / (self._max_rating - self._min_rating)

        if (self._max_votes - self._min_votes) < 1e-6:
            nv = 0.5
        else:
            nv = (df['Votes'] - self._min_votes) / (self._max_votes - self._min_votes)

        return w_rating * nr + w_votes * nv

    def recommend(
        self,
        cuisines: Optional[List[str]] = None,
        budget_range: Optional[Tuple[str, str]] = None,
        location: Optional[str] = None,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Enhanced recommendation with proper coordinate handling
        """
        # First make sure we include coordinates in the filtered data
        required_columns = [
            'Restaurant Name', 'City', 'Primary Cuisine',
            'Cost Category', 'Rating', 'Votes', 'Longitude', 'Latitude'
        ]
        
        # Check which columns actually exist in our data
        available_columns = [col for col in required_columns if col in self.data.columns]
        
        filtered = self._filter(cuisines, budget_range, location)
        
        # Fallback logic if no exact matches
        if filtered.empty:
            fallback_filters = [
                (None, budget_range, location),  # Remove cuisine
                (cuisines, None, location),     # Remove budget
                (cuisines, budget_range, None)  # Remove location
            ]
            
            for f_cuisine, f_budget, f_loc in fallback_filters:
                fallback_results = self._filter(f_cuisine, f_budget, f_loc)
                if not fallback_results.empty:
                    filtered = fallback_results
                    break

        if filtered.empty:
            return pd.DataFrame(columns=required_columns + ['Score', 'Explanation'])

        # Score and rank results
        df2 = filtered.copy()
        df2['Score'] = self._score(df2)
        df2 = df2.sort_values('Score', ascending=False).head(top_n)

        # Generate explanations
        def explain(row):
            parts = []
            if cuisines:
                parts.append(f"{len(cuisines)} cuisines matched")
            if budget_range:
                parts.append(f"budget:{budget_range[0]}-{budget_range[1]}")
            if location:
                parts.append(f"near:{location}")
            parts.append(f"rating:{row['Rating']:.1f}★")
            return " | ".join(parts)

        df2['Explanation'] = df2.apply(explain, axis=1)
        
        # Ensure we only return columns that exist
        output_columns = [col for col in required_columns + ['Score', 'Explanation'] 
                         if col in df2.columns]
        
        return df2[output_columns]

    filter_and_rank = recommend