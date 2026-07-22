"""
report.py
Outputs baseline_metrics_summary.csv and LaTeX tables for the TRB paper.
"""

import json
import os
import pandas as pd

def generate_csv_report(metrics_data, out_dir):
    summary = metrics_data['summary']
    
    # Create a nice DataFrame for the summary
    df_summary = pd.DataFrame([{
        'Experiment': 'Baseline Calibration',
        'Vehicles Evaluated': summary['total_evaluated'],
        'Median Spatial RMSE (norm)': round(summary['median_rmse'], 4),
        '95th Pctl RMSE (norm)': round(summary['p95_rmse'], 4),
        'Median Timing Deviation': round(summary['median_td'], 4),
        'Lane Agreement Rate': round(summary['lane_agreement_rate'], 4)
    }])
    
    csv_path = os.path.join(out_dir, 'baseline_metrics_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    return csv_path

def generate_latex_table(metrics_data, out_dir):
    summary = metrics_data['summary']
    
    tex = r"""\begin{table}[h!]
\centering
\begin{tabular}{|l|c|}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Vehicles Evaluated & %d \\
Median Spatial RMSE & %.4f \\
95\textsuperscript{th} Percentile RMSE & %.4f \\
Median Timing Deviation & %.4f \\
Lane Agreement Rate & %.1f\%% \\
\hline
\end{tabular}
\caption{Baseline Microscopic Fidelity Metrics}
\label{tab:baseline_metrics}
\end{table}
""" % (
        summary['total_evaluated'],
        summary['median_rmse'],
        summary['p95_rmse'],
        summary['median_td'],
        summary['lane_agreement_rate'] * 100
    )
    
    tex_path = os.path.join(out_dir, 'baseline_metrics_summary.tex')
    with open(tex_path, 'w') as f:
        f.write(tex)
    return tex_path

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metrics_path = os.path.join(base_dir, 'outputs', 'micro_metrics.json')
    results_dir = os.path.join(base_dir, 'results')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    if not os.path.exists(metrics_path):
        print("Required metrics.json missing. Run metrics.py first.")
        return
        
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
        
    csv_path = generate_csv_report(metrics_data, results_dir)
    print(f"Generated CSV report: {csv_path}")
    
    tex_path = generate_latex_table(metrics_data, results_dir)
    print(f"Generated LaTeX table: {tex_path}")

if __name__ == "__main__":
    main()
