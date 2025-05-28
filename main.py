import pandas as pd
from src.recommender import ZomatoRecommender
from src.evaluation import Evaluator

def main():
    # Load cleaned data
    df = pd.read_csv("data/cleaned_zomato.csv")

    # Initialize recommender
    rec = ZomatoRecommender(df)

    # Example usage
    print(rec.recommend(
        cuisines=["Indian", "Chinese"],
        budget_range=("low", "medium"),
        location="New Delhi",
        top_n=5
    ))

    # Evaluation
    ev = Evaluator()
    hit_rate = ev.evaluate_hit_rate(df, rec, sample_size=200)
    print(f"Hit-rate: {hit_rate:.2%}")

    # A/B test two weightings (identical here as a placeholder)
    ab = ev.run_ab_test(
        df,
        rec_a=lambda **kw: rec.recommend(**{**kw, 'top_n': 3}),
        rec_b=lambda **kw: rec.recommend(**{**kw, 'top_n': 3})
    )
    print("A/B test:", ab)

    # Write a simple Markdown report
    with open("evaluation_report.md", "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"- Hit-rate: **{hit_rate:.2%}**\n")
        f.write(f"- A/B delta in satisfaction: **{float(ab['delta']):.2f}** points\n")
        f.write("\n## Raw metrics\n")
        f.write(f"- Satisfaction scores: {list(map(float, ev.metrics['satisfaction_scores']))}\n")
        f.write(f"- Usability feedback: {ev.metrics['usability_feedback']}\n")

    print("Wrote evaluation_report.md")

if __name__ == "__main__":
    main()
