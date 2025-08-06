'''
PART 3: MAIN SCRIPT
- This is the only file that will be run directly
- It should call all functions from other files
- Keep outputs clear and organized
'''

from src.preprocessing import load_data, process_data
from src.metrics_calculation import calculate_metrics, calculate_sklearn_metrics

def main():
    # Load and process data
    model_pred_df, genres_df = load_data()
    genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts = process_data(model_pred_df, genres_df)

    # Calculate manual metrics
    micro_metrics, macro_precision, macro_recall, macro_f1 = calculate_metrics(
        model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
    )

    # Calculate sklearn metrics
    macro_sk, micro_sk = calculate_sklearn_metrics(model_pred_df, genre_list)

    # Print manual metrics
    print("=== MANUAL METRICS ===")
    print("Micro Precision:", round(micro_metrics[0], 3))
    print("Micro Recall:", round(micro_metrics[1], 3))
    print("Micro F1 Score:", round(micro_metrics[2], 3))
    print("Macro Precision (avg):", round(sum(macro_precision) / len(macro_precision), 3))
    print("Macro Recall (avg):", round(sum(macro_recall) / len(macro_recall), 3))
    print("Macro F1 Score (avg):", round(sum(macro_f1) / len(macro_f1), 3))

    # Print sklearn metrics
    print("\n=== SKLEARN METRICS ===")
    print("Macro Precision:", round(macro_sk[0], 3))
    print("Macro Recall:", round(macro_sk[1], 3))
    print("Macro F1 Score:", round(macro_sk[2], 3))
    print("Micro Precision:", round(micro_sk[0], 3))
    print("Micro Recall:", round(micro_sk[1], 3))
    print("Micro F1 Score:", round(micro_sk[2], 3))


if __name__ == "__main__":
    main()
