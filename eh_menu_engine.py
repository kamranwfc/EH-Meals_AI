# eh_menu_engine.py
from __future__ import annotations
from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import json
import math
import os

# Import Pydantic Meal models (or define minimal compatible ones here)
class DietDetail(BaseModel):
    uuid: str
    name: str

class MealDiet(BaseModel):
    uuid: str
    diet_id: str
    diet_details: DietDetail

class MealType(BaseModel):
    uuid: str
    title: Optional[str]
    name: str

class Meal(BaseModel):
    id: int
    uuid: str
    title: str
    description: Optional[str]
    meal_type: MealType
    meal_diets: List[MealDiet] = Field(default_factory=list)
    image: Optional[str]
    is_featured: int = 0
    cafeteria: Optional[int]
    order_count: int = 0
    last_ordered: Optional[str] = None

# Engine
class MenuGeneratorEngine:
    """
    Production-focused menu engine that uses predicted popularity when available.
    """

    # Meal-type ratio (production defaults)
    MEAL_TYPE_RATIO = {"breakfast": 0.20, "lunch_dinner": 0.80}
    MAJOR_DIETS_MIN_COUNTS = {
        "Gluten Free": 20,
        "Low Carb": 18,
        "Dairy Free": 15,
        "Diabetic Friendly": 12,
        "Low Sodium": 10,
        "Vegetarian": 6,
    }

    # Weights: note 'pred_popularity' will be used as primary popularity signal.
    WEIGHTS = {
        "pred_popularity": 0.50,
        "recency": 0.15,
        "diversity": 0.10,
        "balance": 0.25
    }

    def __init__(self, meals: List[Meal]):
        self.meals = meals
        self.now = datetime.now(timezone.utc)

    @staticmethod
    def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                if fmt.endswith("%z"):
                    return datetime.strptime(dt_str, fmt)
                else:
                    return datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except Exception:
            return None

    @staticmethod
    def unique_by_uuid(meals: List[Meal]) -> List[Meal]:
        seen: Set[str] = set()
        out = []
        for m in meals:
            if m.uuid not in seen:
                seen.add(m.uuid)
                out.append(m)
        return out

    def _recency_penalty_days(self, last_dt: Optional[datetime]) -> float:
        if not last_dt:
            return 0.0
        diff = self.now - last_dt
        days = diff.total_seconds() / 86400.0
        if days <= 7:
            return 1.0
        if days <= 30:
            return 0.6
        if days <= 60:
            return 0.3
        return 0.0

    @staticmethod
    def _normalize_list(values: List[float]) -> List[float]:
        if not values:
            return []
        mn = min(values)
        mx = max(values)
        if math.isclose(mx, mn):
            return [0.5 for _ in values]
        return [(v - mn) / (mx - mn) for v in values]

    def _compute_base_metrics(self, candidates: List[Meal], predicted_popularity: Dict[str, float]):
        pop_scores = []
        recency_scores = []
        diversity_scores = []
        uuids = []
        meal_type_counts = {}
        for m in candidates:
            uuids.append(m.uuid)
            # Use predicted popularity if available, else fallback to raw order_count
            pop_scores.append(predicted_popularity.get(m.uuid, float(m.order_count or 0)))
            last = self.parse_datetime(m.last_ordered)
            recency_scores.append(self._recency_penalty_days(last))
            diversity_scores.append(len(m.meal_diets) if m.meal_diets else 0)
            meal_type_counts[m.meal_type.name] = meal_type_counts.get(m.meal_type.name, 0) + 1
        return uuids, pop_scores, recency_scores, diversity_scores, meal_type_counts

    def _score_candidates(self, candidates: List[Meal], predicted_popularity: Dict[str, float], target_counts_by_type: Dict[str, int]) -> Dict[str, float]:
        uuids, pop, recency_pen, diversity, meal_type_counts = self._compute_base_metrics(candidates, predicted_popularity)
        # Normalize popularity: if pred values seem 0-1 already, normalization will be neutral
        pop_norm = self._normalize_list(pop)
        recency_inverted = [1.0 - p for p in self._normalize_list(recency_pen)]
        diversity_norm = self._normalize_list([d for d in diversity])

        # Balance boost: if type under target -> give boost
        balance_scores = []
        for m in candidates:
            t = m.meal_type.name
            target = target_counts_by_type.get(t, 0)
            if target <= 0:
                balance_scores.append(0.5)
                continue
            current = meal_type_counts.get(t, 0)
            deficit = max(0.0, (target - current) / max(1, target))
            balance_scores.append(min(1.0, deficit * 1.2 + 0.2))

        scores = {}
        for idx, m in enumerate(candidates):
            score = (
                pop_norm[idx] * self.WEIGHTS['pred_popularity'] +
                recency_inverted[idx] * self.WEIGHTS['recency'] +
                diversity_norm[idx] * self.WEIGHTS['diversity'] +
                balance_scores[idx] * self.WEIGHTS['balance']
            )
            scores[m.uuid] = score
        return scores

    def generate_menu(self, min_meals: int = 50, max_meals: int = 70, dietary_balance: bool = True, include_add_ons: bool = False, popularity_scores: Optional[Dict[str, float]] = None) -> List[Meal]:
        # Filter items
        pool = [m for m in self.meals if include_add_ons or m.meal_type.name != 'add_on_item']
        pool = self.unique_by_uuid(pool)

        # Build meal type targets
        target_by_type = {}
        for tname, ratio in self.MEAL_TYPE_RATIO.items():
            target_by_type[tname] = max(1, int(round(max_meals * ratio)))

        # If popularity_scores not passed, try to load minimal fallback mapping using order_count
        if popularity_scores is None:
            popularity_scores = {}

        if not dietary_balance:
            scores = self._score_candidates(pool, popularity_scores, target_by_type)
            sorted_pool = sorted(pool, key=lambda m: scores.get(m.uuid, 0), reverse=True)
            return sorted_pool[:max_meals]

        # Score and select with diet minima and type targets
        scores = self._score_candidates(pool, popularity_scores, target_by_type)

        # Build diet->candidates mapping
        diet_to_candidates: Dict[str, List[str]] = {}
        uuid_to_meal = {m.uuid: m for m in pool}
        for m in pool:
            diet_names = [d.diet_details.name for d in m.meal_diets] if m.meal_diets else []
            if not diet_names:
                diet_names = ['__none__']
            for dn in diet_names:
                diet_to_candidates.setdefault(dn, []).append(m.uuid)
        for dn, uuids in diet_to_candidates.items():
            uuids.sort(key=lambda u: scores.get(u, 0), reverse=True)

        selected_uuids: List[str] = []
        selected_set: Set[str] = set()

        # 1) Ensure minima for major diets
        for diet_name, min_count in self.MAJOR_DIETS_MIN_COUNTS.items():
            candidates_for_diet = diet_to_candidates.get(diet_name, [])
            take = min(len(candidates_for_diet), min_count)
            for u in candidates_for_diet[:take]:
                if u not in selected_set and len(selected_uuids) < max_meals:
                    selected_uuids.append(u)
                    selected_set.add(u)

        # 2) Enforce meal type targets
        type_counts = {}
        for u in selected_uuids:
            mt = uuid_to_meal[u].meal_type.name
            type_counts[mt] = type_counts.get(mt, 0) + 1
        for mt, target in target_by_type.items():
            need = max(0, target - type_counts.get(mt, 0))
            if need <= 0:
                continue
            candidates_of_type = [m for m in pool if m.meal_type.name == mt and m.uuid not in selected_set]
            candidates_of_type.sort(key=lambda m: scores.get(m.uuid, 0), reverse=True)
            for m in candidates_of_type[:need]:
                if len(selected_uuids) >= max_meals:
                    break
                selected_uuids.append(m.uuid)
                selected_set.add(m.uuid)

        # 3) Fill remaining slots by global score, respecting recency preference
        if len(selected_uuids) < max_meals:
            candidates_sorted = sorted(pool, key=lambda m: (scores.get(m.uuid, 0), -self._recency_penalty_days(self.parse_datetime(m.last_ordered))), reverse=True)
            for m in candidates_sorted:
                if m.uuid in selected_set:
                    continue
                if len(selected_uuids) >= max_meals:
                    break
                selected_uuids.append(m.uuid)
                selected_set.add(m.uuid)

        # 4) Ensure min_meals by relaxing constraints
        if len(selected_uuids) < min_meals:
            all_sorted = sorted(pool, key=lambda m: scores.get(m.uuid, 0), reverse=True)
            for m in all_sorted:
                if m.uuid in selected_set:
                    continue
                if len(selected_uuids) >= min_meals:
                    break
                selected_uuids.append(m.uuid)
                selected_set.add(m.uuid)

        final_meals = [uuid_to_meal[u] for u in selected_uuids]
        return final_meals

    def get_menu_statistics(self, selected_meals: List[Meal]) -> Dict:
        stats = {'total_meals': len(selected_meals), 'meal_types': {}, 'dietary_options': {}, 'avg_order_count': 0}
        for m in selected_meals:
            mt = m.meal_type.name
            stats['meal_types'][mt] = stats['meal_types'].get(mt, 0) + 1
            for d in m.meal_diets:
                dn = d.diet_details.name
                stats['dietary_options'][dn] = stats['dietary_options'].get(dn, 0) + 1
        total_orders = sum(m.order_count or 0 for m in selected_meals)
        stats['avg_order_count'] = (total_orders / len(selected_meals)) if selected_meals else 0
        return stats
