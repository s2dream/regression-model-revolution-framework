import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import logging
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Visualizer is responsible for plotting model evaluation metrics, actual vs. predicted values,
    residual analyses, and model comparisons. Saves outputs as elegant images (.png) and metadata (.json).
    """
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Apply premium aesthetic configurations for matplotlib/seaborn
        sns.set_theme(style="darkgrid")
        plt.rcParams.update({
            'figure.dpi': 150,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.titlesize': 15,
            'figure.facecolor': '#0d1117',  # Elegant dark slate background
            'axes.facecolor': '#161b22',    # Lighter container background
            'grid.color': '#30363d',        # Soft gridlines
            'text.color': '#c9d1d9',        # Crisp white/grey text
            'axes.labelcolor': '#8b949e',   # Muted grey labels
            'xtick.color': '#8b949e',
            'ytick.color': '#8b949e',
        })
        # Harmonious modern color palette
        self.palette = ["#58a6ff", "#ff7b72", "#aff5b4", "#d2a8ff", "#e3b341"]

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, turn: int = 1) -> str:
        """
        Creates an elegant Scatter Plot of Actual vs. Predicted values with an identity line.
        
        Returns:
            str: Path to the saved visualization
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Scatter plot with elegant translucency
        ax.scatter(y_true, y_pred, alpha=0.6, color=self.palette[0], edgecolors='none', s=25, label='Predictions')
        
        # Identity line (y=x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color=self.palette[1], linestyle='--', lw=2, label='Perfect Fit')
        
        ax.set_title(f"{model_name}: Actual vs Predicted (Turn {turn})", color='#ffffff', pad=15)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        
        plt.tight_layout()
        
        filename = f"turn_{turn}_{model_name}_actual_vs_pred.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        plt.close()
        logger.info(f"Saved Actual vs Pred plot to {filepath}")
        return filepath

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, turn: int = 1) -> str:
        """
        Plots Residuals vs Predicted values to diagnose variance behavior (heteroscedasticity).
        
        Returns:
            str: Path to the saved visualization
        """
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(7, 6))
        
        ax.scatter(y_pred, residuals, alpha=0.6, color=self.palette[2], edgecolors='none', s=25)
        ax.axhline(0, color=self.palette[1], linestyle='--', lw=2)
        
        ax.set_title(f"{model_name}: Residual Plot (Turn {turn})", color='#ffffff', pad=15)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        
        plt.tight_layout()
        
        filename = f"turn_{turn}_{model_name}_residuals.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        plt.close()
        logger.info(f"Saved Residual plot to {filepath}")
        return filepath

    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]], metric_name: str = "RMSE", turn: int = 1) -> str:
        """
        Compares multiple models in a neat horizontal bar chart for a specified metric (e.g. RMSE, R2).
        
        Returns:
            str: Path to the saved visualization
        """
        models = list(metrics.keys())
        values = [m_data.get(metric_name, 0.0) for m_data in metrics.values()]
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        # Sort in ascending/descending depending on the metric
        reverse_sort = True if metric_name in ["R2"] else False
        sorted_pairs = sorted(zip(values, models), reverse=reverse_sort)
        if sorted_pairs:
            sorted_values, sorted_models = zip(*sorted_pairs)
        else:
            sorted_values, sorted_models = [], []
        
        bars = ax.barh(sorted_models, sorted_values, color=self.palette[3], height=0.5, edgecolor='#30363d')
        
        # Add labels to the ends of the bars
        for bar in bars:
            width = bar.get_width()
            max_val = max(sorted_values) if sorted_values else 1.0
            ax.text(
                width + (max_val * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f"{width:.4f}", 
                va='center', 
                ha='left', 
                color='#c9d1d9',
                fontsize=9
            )
            
        ax.set_title(f"Model Comparison: {metric_name} (Turn {turn})", color='#ffffff', pad=15)
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Models")
        
        plt.tight_layout()
        
        filename = f"turn_{turn}_model_comparison_{metric_name.lower()}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        plt.close()
        logger.info(f"Saved Model Comparison plot to {filepath}")
        return filepath

    def save_json_report(
        self, 
        metrics: Dict[str, Dict[str, float]], 
        turn: int = 1, 
        metadata: Optional[Dict[str, Any]] = None,
        shap_reports: Optional[Dict[str, Dict[str, str]]] = None, 
        learning_curves: Optional[Dict[str, str]] = None,
        markdown_report_path: Optional[str] = None
    ) -> str:
        """
        Saves the turn's execution and performance metrics in a structured JSON report.
        
        Returns:
            str: Path to the saved report
        """
        best_model = max(metrics.keys(), key=lambda k: metrics[k].get("R2", -float('inf'))) if metrics else None
        report_data = {
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "best_model": best_model,
            "metrics": metrics,
            "metadata": metadata or {}
        }
        if shap_reports:
            report_data["shap_reports"] = shap_reports
        if learning_curves:
            report_data["learning_curves"] = learning_curves
        if markdown_report_path:
            report_data["markdown_report_path"] = markdown_report_path
            
        filename = f"turn_{turn}_report.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Saved JSON report to {filepath}")
        return filepath

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Helper to encode image file to base64 data URI for standalone HTML embedding."""
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            try:
                with open(image_path, "rb") as img_f:
                    encoded = base64.b64encode(img_f.read()).decode("utf-8")
                    return f"data:image/png;base64,{encoded}"
            except Exception as e:
                logger.warning(f"Could not encode image {image_path} to base64: {e}")
        return ""

    def save_html_report(
        self, 
        metrics: Dict[str, Dict[str, float]], 
        turn: int = 1, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a standalone, interactive, and beautifully styled HTML benchmark report.
        Self-contained with embedded base64 plots, interactive sortable table, and diagnostic tabs.
        
        Returns:
            str: Path to the saved HTML report
        """
        metadata = metadata or {}
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sorted_models = sorted(
            metrics.keys(), 
            key=lambda m: metrics[m].get("R2", -float('inf')), 
            reverse=True
        ) if metrics else []
        
        best_model = sorted_models[0] if sorted_models else "N/A"
        best_metrics = metrics.get(best_model, {}) if best_model in metrics else {}
        best_r2 = best_metrics.get("R2", 0.0)
        best_rmse = best_metrics.get("RMSE", 0.0)
        best_mae = best_metrics.get("MAE", 0.0)
        
        # Gather base64 images for comparison charts
        r2_img_path = os.path.join(self.output_dir, f"turn_{turn}_model_comparison_r2.png")
        rmse_img_path = os.path.join(self.output_dir, f"turn_{turn}_model_comparison_rmse.png")
        r2_b64 = self._encode_image_to_base64(r2_img_path)
        rmse_b64 = self._encode_image_to_base64(rmse_img_path)
        
        # Build table rows
        table_rows_html = []
        for rank, m_name in enumerate(sorted_models, start=1):
            m_scores = metrics[m_name]
            is_champ = (m_name == best_model)
            rank_badge = "🥇 1st" if rank == 1 else ("🥈 2nd" if rank == 2 else ("🥉 3rd" if rank == 3 else f"#{rank}"))
            champ_tag = '<span class="badge badge-champion">Champion</span>' if is_champ else ''
            
            row = f"""
            <tr class="{'champ-row' if is_champ else ''}">
                <td><span class="rank-badge">{rank_badge}</span></td>
                <td><strong>{m_name}</strong> {champ_tag}</td>
                <td class="metric-val r2-val">{m_scores.get('R2', 0.0):.4f}</td>
                <td class="metric-val">{m_scores.get('RMSE', 0.0):.4f}</td>
                <td class="metric-val">{m_scores.get('MAE', 0.0):.4f}</td>
            </tr>
            """
            table_rows_html.append(row)
            
        # Build diagnostic tab items and content
        tab_buttons_html = []
        tab_contents_html = []
        for idx, m_name in enumerate(sorted_models):
            active_class = "active" if idx == 0 else ""
            tab_id = f"diag-tab-{idx}"
            tab_buttons_html.append(f'<button class="tab-btn {active_class}" onclick="switchTab(\'{tab_id}\', this)">{m_name}</button>')
            
            act_pred_img = os.path.join(self.output_dir, f"turn_{turn}_{m_name}_actual_vs_pred.png")
            res_img = os.path.join(self.output_dir, f"turn_{turn}_{m_name}_residuals.png")
            act_pred_b64 = self._encode_image_to_base64(act_pred_img)
            res_b64 = self._encode_image_to_base64(res_img)
            
            content = f"""
            <div id="{tab_id}" class="tab-pane {active_class}">
                <div class="charts-grid">
                    <div class="chart-card">
                        <h4>{m_name}: Actual vs Predicted</h4>
                        {f'<img src="{act_pred_b64}" alt="Actual vs Predicted" />' if act_pred_b64 else '<p class="no-img">Plot not generated</p>'}
                    </div>
                    <div class="chart-card">
                        <h4>{m_name}: Residuals Analysis</h4>
                        {f'<img src="{res_b64}" alt="Residuals" />' if res_b64 else '<p class="no-img">Plot not generated</p>'}
                    </div>
                </div>
            </div>
            """
            tab_contents_html.append(content)

        target_col = metadata.get("target_column", "Target_Y")
        split_method = metadata.get("split_method", "train_test_split")
        train_samples = metadata.get("train_samples", "N/A")
        test_samples = metadata.get("test_samples", "N/A")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Benchmark Report - Turn {turn}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0d1117;
            --bg-card: #161b22;
            --bg-card-hover: #1c2128;
            --border-color: #30363d;
            --text-main: #c9d1d9;
            --text-bright: #f0f6fc;
            --text-muted: #8b949e;
            --accent-purple: #8b5cf6;
            --accent-blue: #58a6ff;
            --accent-green: #2ea043;
            --accent-gold: #e3b341;
            --accent-coral: #ff7b72;
            --champ-gradient: linear-gradient(135deg, #1e1b4b 0%, #31104b 50%, #161b22 100%);
            --header-gradient: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2.5rem 1.5rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .report-header {{
            background: var(--header-gradient);
            color: #ffffff;
            padding: 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(124, 58, 237, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .report-header h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        .report-header p {{
            font-size: 1rem;
            opacity: 0.9;
            margin-top: 0.4rem;
        }}

        .header-badge {{
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(8px);
            padding: 0.5rem 1.2rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.8rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }}

        .champion-card {{
            background: var(--champ-gradient);
            border: 1px solid rgba(139, 92, 246, 0.4);
            box-shadow: 0 10px 30px rgba(139, 92, 246, 0.15);
        }}

        .champion-title {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent-gold);
            font-weight: 700;
            margin-bottom: 0.8rem;
        }}

        .champion-model-name {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.4rem;
            font-weight: 800;
            color: var(--text-bright);
            margin-bottom: 1.2rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.2rem;
        }}

        .metric-box {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }}

        .metric-box-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }}

        .metric-box-val {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent-blue);
        }}

        .metric-box-val.gold {{ color: var(--accent-gold); }}
        .metric-box-val.green {{ color: var(--accent-green); }}

        .meta-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-bottom: 2rem;
        }}

        .meta-pill {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .meta-pill strong {{
            color: var(--text-bright);
            margin-left: 0.3rem;
        }}

        .section-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-bright);
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th {{
            background: #1f242c;
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 1rem;
            border-bottom: 2px solid var(--border-color);
            cursor: pointer;
            user-select: none;
        }}

        th:hover {{
            color: var(--text-bright);
        }}

        td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.95rem;
        }}

        tr:hover td {{
            background: var(--bg-card-hover);
        }}

        .champ-row td {{
            background: rgba(139, 92, 246, 0.08);
            font-weight: 600;
        }}

        .rank-badge {{
            font-weight: 700;
            color: var(--text-muted);
        }}

        .badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}

        .badge-champion {{
            background: rgba(227, 179, 65, 0.2);
            color: var(--accent-gold);
            border: 1px solid rgba(227, 179, 65, 0.4);
        }}

        .metric-val {{
            font-family: monospace;
            font-size: 1rem;
        }}

        .metric-val.r2-val {{
            color: var(--accent-blue);
            font-weight: 700;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
        }}

        .chart-card {{
            background: #11151c;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }}

        .chart-card h4 {{
            color: var(--text-bright);
            font-size: 1rem;
            margin-bottom: 1rem;
        }}

        .chart-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        .no-img {{
            color: var(--text-muted);
            padding: 3rem;
            font-style: italic;
        }}

        .tabs-header {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.8rem;
        }}

        .tab-btn {{
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .tab-btn:hover {{
            background: var(--bg-card-hover);
            color: var(--text-bright);
        }}

        .tab-btn.active {{
            background: var(--accent-purple);
            color: #ffffff;
            border-color: var(--accent-purple);
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }}

        .tab-pane {{
            display: none;
        }}

        .tab-pane.active {{
            display: block;
        }}

        .report-footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-color);
        }}

        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .report-header {{
                padding: 1.8rem;
            }}
            .report-header h1 {{
                font-size: 1.8rem;
            }}
        }}

        @media print {{
            body {{
                background-color: #ffffff !important;
                color: #000000 !important;
            }}
            .report-header {{
                background: #4f46e5 !important;
                color: #ffffff !important;
            }}
            .card, .chart-card {{
                background: #ffffff !important;
                border: 1px solid #cccccc !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="report-header">
            <div>
                <h1>AutoML Regression Benchmark</h1>
                <p>Automated Tabular Model Evaluation & Diagnostic Report</p>
            </div>
            <div class="header-badge">
                Execution Turn {turn} • {timestamp_str}
            </div>
        </header>

        <!-- Metadata Pills -->
        <div class="meta-container">
            <div class="meta-pill">Target Variable: <strong>{target_col}</strong></div>
            <div class="meta-pill">Split Strategy: <strong>{split_method}</strong></div>
            <div class="meta-pill">Train Samples: <strong>{train_samples}</strong></div>
            <div class="meta-pill">Test Samples: <strong>{test_samples}</strong></div>
            <div class="meta-pill">Evaluated Models: <strong>{len(metrics)}</strong></div>
        </div>

        <!-- Champion Card -->
        <div class="card champion-card">
            <div class="champion-title">🏆 Overall Best Performing Model</div>
            <div class="champion-model-name">{best_model}</div>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-box-label">R² Score (Higher is better)</div>
                    <div class="metric-box-val gold">{best_r2:.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">RMSE (Lower is better)</div>
                    <div class="metric-box-val">{best_rmse:.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">MAE (Lower is better)</div>
                    <div class="metric-box-val green">{best_mae:.4f}</div>
                </div>
            </div>
        </div>

        <!-- Leaderboard Table -->
        <div class="card">
            <h3 class="section-title">📊 Model Leaderboard & Performance Metrics</h3>
            <table id="leaderboardTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">Rank</th>
                        <th onclick="sortTable(1)">Model</th>
                        <th onclick="sortTable(2, true)">R² Score ▾</th>
                        <th onclick="sortTable(3, false)">RMSE</th>
                        <th onclick="sortTable(4, false)">MAE</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows_html)}
                </tbody>
            </table>
        </div>

        <!-- Comparison Bar Charts -->
        <div class="card">
            <h3 class="section-title">📈 Benchmark Metric Comparisons</h3>
            <div class="charts-grid">
                <div class="chart-card">
                    <h4>R² Score Comparison</h4>
                    {f'<img src="{r2_b64}" alt="R2 Comparison" />' if r2_b64 else '<p class="no-img">R2 Chart not available</p>'}
                </div>
                <div class="chart-card">
                    <h4>RMSE Comparison</h4>
                    {f'<img src="{rmse_b64}" alt="RMSE Comparison" />' if rmse_b64 else '<p class="no-img">RMSE Chart not available</p>'}
                </div>
            </div>
        </div>

        <!-- Diagnostic Plots with Tabs -->
        <div class="card">
            <h3 class="section-title">🔍 Model Diagnostics & Residual Analysis</h3>
            <div class="tabs-header">
                {"".join(tab_buttons_html)}
            </div>
            {"".join(tab_contents_html)}
        </div>

        <!-- Footer -->
        <footer class="report-footer">
            Generated automatically by <strong>Regression Model Revolution Framework</strong> • Turn {turn}
        </footer>
    </div>

    <script>
        function switchTab(tabId, btn) {{
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            const target = document.getElementById(tabId);
            if (target) target.classList.add('active');
            btn.classList.add('active');
        }}

        function sortTable(n, isNumeric = true) {{
            const table = document.getElementById("leaderboardTable");
            let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            switching = true;
            dir = "asc";
            while (switching) {{
                switching = false;
                rows = table.rows;
                for (i = 1; i < (rows.length - 1); i++) {{
                    shouldSwitch = false;
                    x = rows[i].getElementsByTagName("TD")[n];
                    y = rows[i + 1].getElementsByTagName("TD")[n];
                    let xVal = x.innerText.replace(/[^0-9.-]+/g,"");
                    let yVal = y.innerText.replace(/[^0-9.-]+/g,"");
                    
                    if (isNumeric && !isNaN(parseFloat(xVal)) && !isNaN(parseFloat(yVal))) {{
                        if (dir == "asc" ? (parseFloat(xVal) > parseFloat(yVal)) : (parseFloat(xVal) < parseFloat(yVal))) {{
                            shouldSwitch = true;
                            break;
                        }}
                    }} else {{
                        if (dir == "asc" ? (x.innerText.toLowerCase() > y.innerText.toLowerCase()) : (x.innerText.toLowerCase() < y.innerText.toLowerCase())) {{
                            shouldSwitch = true;
                            break;
                        }}
                    }}
                }}
                if (shouldSwitch) {{
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount ++;
                }} else {{
                    if (switchcount == 0 && dir == "asc") {{
                        dir = "desc";
                        switching = true;
                    }}
                }}
            }}
        }}
    </script>
</body>
</html>
"""
        filename = f"turn_{turn}_report.html"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Saved interactive HTML report to {filepath}")
        return filepath

    def save_markdown_summary(
        self, 
        metrics: Dict[str, Dict[str, float]], 
        turn: int = 1, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a clean, shareable GitHub Flavored Markdown summary report.
        Perfect for pasting into Notion, Slack, GitHub Issues/PRs, or documentation.
        
        Returns:
            str: Path to the saved Markdown summary report
        """
        metadata = metadata or {}
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sorted_models = sorted(
            metrics.keys(), 
            key=lambda m: metrics[m].get("R2", -float('inf')), 
            reverse=True
        ) if metrics else []
        
        best_model = sorted_models[0] if sorted_models else "N/A"
        best_metrics = metrics.get(best_model, {}) if best_model in metrics else {}
        best_r2 = best_metrics.get("R2", 0.0)
        best_rmse = best_metrics.get("RMSE", 0.0)
        best_mae = best_metrics.get("MAE", 0.0)

        target_col = metadata.get("target_column", "Target_Y")
        split_method = metadata.get("split_method", "train_test_split")
        train_samples = metadata.get("train_samples", "N/A")
        test_samples = metadata.get("test_samples", "N/A")

        lines = [
            f"# 🚀 AutoML Regression Benchmark Summary (Turn {turn})",
            "",
            f"> **Generated at**: `{timestamp_str}`  ",
            f"> **🏆 Champion Model**: **{best_model}** ($R^2$: `{best_r2:.4f}`, RMSE: `{best_rmse:.4f}`, MAE: `{best_mae:.4f}`)",
            "",
            "## 📋 Experiment Configuration",
            f"- **Target Variable**: `{target_col}`",
            f"- **Split Strategy**: `{split_method}`",
            f"- **Dataset Split**: `{train_samples}` train samples / `{test_samples}` test samples",
            f"- **Evaluated Models**: {len(metrics)} models",
            "",
            "## 📊 Model Leaderboard",
            "| Rank | Model Name | $R^2$ Score (↑) | RMSE (↓) | MAE (↓) | Status |",
            "|:---:|:---|:---:|:---:|:---:|:---:|"
        ]

        for rank, m_name in enumerate(sorted_models, start=1):
            m_scores = metrics[m_name]
            rank_str = f"🥇 **1st**" if rank == 1 else (f"🥈 2nd" if rank == 2 else (f"🥉 3rd" if rank == 3 else f"#{rank}"))
            status = "🏆 **Champion**" if m_name == best_model else "Completed"
            lines.append(
                f"| {rank_str} | **{m_name}** | `{m_scores.get('R2', 0.0):.4f}` | `{m_scores.get('RMSE', 0.0):.4f}` | `{m_scores.get('MAE', 0.0):.4f}` | {status} |"
            )

        lines.extend([
            "",
            "## 📁 Generated Output Artifacts",
            f"- **Interactive HTML Dashboard**: [`turn_{turn}_report.html`](turn_{turn}_report.html)",
            f"- **Structured JSON Report**: [`turn_{turn}_report.json`](turn_{turn}_report.json)",
            f"- **Model Comparison Chart (R²)**: [`turn_{turn}_model_comparison_r2.png`](turn_{turn}_model_comparison_r2.png)",
            f"- **Model Comparison Chart (RMSE)**: [`turn_{turn}_model_comparison_rmse.png`](turn_{turn}_model_comparison_rmse.png)",
            "",
            "---",
            "*Report auto-generated by Regression Model Revolution Framework.*"
        ])

        md_content = "\n".join(lines)
        filename = f"turn_{turn}_summary.md"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Saved Markdown summary to {filepath}")
        return filepath

    def plot_shap_explainability(self, model_wrap, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str, turn: int = 1, max_samples: int = 100) -> Dict[str, str]:
        """
        Computes SHAP values and saves Beeswarm and Bar Plots for the given model wrapper.
        Includes safety shielding and fallback strategies.
        
        Returns:
            Dict[str, str]: Dictionary containing file paths to generated plots
        """
        try:
            import shap
        except ImportError:
            logger.warning("⚠️ 'shap' library is not installed. Skipping SHAP analysis.")
            return {}

        logger.info(f"🧠 Computing SHAP values for model: {model_name}...")
        
        # Prepare datasets: downsample for performance
        if len(X_train) > max_samples:
            X_background = shap.sample(X_train, max_samples, random_state=42)
        else:
            X_background = X_train

        X_explain = X_test
        if len(X_test) > max_samples:
            X_explain = shap.sample(X_test, max_samples, random_state=42)

        explainer = None
        shap_values = None

        # Build Explainer based on model class/type
        try:
            from automl_framework.model.model_factory import ModelType
            try:
                model_type = ModelType.from_str(model_name)
            except ValueError:
                model_type = None

            # TreeExplainer is extremely fast and works directly on tree ensembles
            if model_type in [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.RANDOM_FOREST]:
                try:
                    explainer = shap.TreeExplainer(model_wrap.model)
                    shap_values = explainer(X_explain)
                except Exception as e:
                    logger.debug(f"Failed to initialize TreeExplainer for {model_name}: {e}. Falling back to default Explainer.")
                    explainer = None

            # Fallback model-agnostic Permutation/Kernel Explainer
            if explainer is None:
                explainer = shap.Explainer(model_wrap.predict, X_background)
                shap_values = explainer(X_explain)

        except Exception as e:
            logger.error(f"Failed to initialize explainer or calculate SHAP values for {model_name}: {e}", exc_info=True)
            return {}

        paths = {}

        # 1. Beeswarm / Summary Plot
        try:
            plt.figure(figsize=(8, 5))
            
            try:
                if hasattr(shap_values, "values"):
                    shap.plots.beeswarm(shap_values, show=False)
                else:
                    shap.summary_plot(shap_values, X_explain, show=False)
            except Exception:
                shap.summary_plot(shap_values, X_explain, show=False)

            fig = plt.gcf()
            fig.patch.set_facecolor('#0d1117')
            ax = plt.gca()
            ax.set_facecolor('#161b22')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.tick_params(colors='#8b949e')
            plt.title(f"{model_name}: SHAP Summary (Turn {turn})", color='#ffffff', pad=15)
            plt.tight_layout()

            summary_filename = f"turn_{turn}_{model_name}_shap_summary.png"
            summary_path = os.path.join(self.output_dir, summary_filename)
            plt.savefig(summary_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
            plt.close()
            paths["summary_plot"] = summary_path
            logger.info(f"Saved SHAP Beeswarm plot to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to generate SHAP Beeswarm plot for {model_name}: {e}", exc_info=True)
            plt.close()

        # 2. Bar Plot (Feature Importance)
        try:
            plt.figure(figsize=(8, 5))
            
            try:
                if hasattr(shap_values, "values"):
                    shap.plots.bar(shap_values, show=False)
                else:
                    shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
            except Exception:
                shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)

            fig = plt.gcf()
            fig.patch.set_facecolor('#0d1117')
            ax = plt.gca()
            ax.set_facecolor('#161b22')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.tick_params(colors='#8b949e')
            plt.title(f"{model_name}: Feature Importance (Turn {turn})", color='#ffffff', pad=15)
            plt.tight_layout()

            bar_filename = f"turn_{turn}_{model_name}_shap_bar.png"
            bar_path = os.path.join(self.output_dir, bar_filename)
            plt.savefig(bar_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
            plt.close()
            paths["bar_plot"] = bar_path
            logger.info(f"Saved SHAP Bar plot to {bar_path}")
        except Exception as e:
            logger.error(f"Failed to generate SHAP Bar plot for {model_name}: {e}", exc_info=True)
            plt.close()

        return paths

    def plot_learning_curve(self, loss_history: list, model_name: str, turn: int = 1) -> str:
        """
        Plots the training loss curve for iterative models.
        
        Returns:
            str: Path to the saved visualization
        """
        if not loss_history:
            return ""
            
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, len(loss_history) + 1), loss_history, color=self.palette[0], lw=2, label="Train Loss")
        
        ax.set_title(f"{model_name}: Learning Curve (Turn {turn})", color='#ffffff', pad=15)
        ax.set_xlabel("Epoch / Iteration")
        ax.set_ylabel("Loss / Error")
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        
        plt.tight_layout()
        
        filename = f"turn_{turn}_{model_name}_learning_curve.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        plt.close()
        logger.info(f"Saved Learning Curve plot to {filepath}")
        return filepath

    def save_markdown_report(
        self,
        metrics: Dict[str, Dict[str, float]],
        turn: int = 1,
        dataset_info: Optional[Dict[str, Any]] = None,
        shap_reports: Optional[Dict[str, Dict[str, str]]] = None,
        learning_curves: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generates a professional and comprehensive Markdown report detailing the benchmark execution.
        
        Returns:
            str: Path to the saved report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sort models by R2 score
        sorted_models = sorted(metrics.items(), key=lambda x: x[1].get("R2", -999), reverse=True)
        best_model = sorted_models[0][0] if sorted_models else None
        
        lines = []
        lines.append(f"# 📊 AutoML Tabular Regression Benchmark Report (Turn {turn})")
        lines.append(f"**Report Generated At**: `{timestamp}`")
        
        if dataset_info:
            lines.append("\n## 📁 Dataset & Execution Metadata")
            lines.append(f"- **Target Column**: `{dataset_info.get('target_column')}`")
            lines.append(f"- **Features Count**: `{dataset_info.get('num_features')}`")
            lines.append(f"- **Training Set Size**: `{dataset_info.get('train_size')} samples`")
            lines.append(f"- **Testing Set Size**: `{dataset_info.get('test_size')} samples`")
            lines.append(f"- **Validation Strategy**: `{dataset_info.get('split_method', 'Holdout')}`")
            
        lines.append("\n## 🏆 Model Leaderboard")
        lines.append("| Rank | Model | RMSE | MAE | R² Score | Status |")
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: |")
        
        for idx, (model_name, scores) in enumerate(sorted_models):
            rank = idx + 1
            rmse = f"{scores.get('RMSE', 0.0):.4f}"
            mae = f"{scores.get('MAE', 0.0):.4f}"
            r2 = f"{scores.get('R2', 0.0):.4f}"
            status = "🥇 Champion" if model_name == best_model else "Active"
            lines.append(f"| {rank} | **{model_name}** | {rmse} | {mae} | {r2} | {status} |")
            
        lines.append("\n## 🔍 Detailed Model Diagnostics & Explainability")
        
        for model_name, scores in sorted_models:
            lines.append(f"\n### 🤖 {model_name}")
            lines.append(f"- **RMSE**: `{scores.get('RMSE', 0.0):.6f}`")
            lines.append(f"- **MAE**: `{scores.get('MAE', 0.0):.6f}`")
            lines.append(f"- **R² Score**: `{scores.get('R2', 0.0):.6f}`")
            
            lines.append("- **Diagnostic Plots Available**:")
            pred_vs_act_img = f"turn_{turn}_{model_name}_actual_vs_pred.png"
            residuals_img = f"turn_{turn}_{model_name}_residuals.png"
            lines.append(f"  - Actual vs Predicted: [`{pred_vs_act_img}`](file://{os.path.abspath(os.path.join(self.output_dir, pred_vs_act_img))})")
            lines.append(f"  - Residuals Plot: [`{residuals_img}`](file://{os.path.abspath(os.path.join(self.output_dir, residuals_img))})")
            
            # Add learning curve if available
            if learning_curves and model_name in learning_curves:
                curve_filename = os.path.basename(learning_curves[model_name])
                lines.append(f"  - Learning Curve (Loss History): [`{curve_filename}`](file://{os.path.abspath(learning_curves[model_name])})")
                
            # Add SHAP details if available
            if shap_reports and model_name in shap_reports:
                lines.append("- **Model Explainability (SHAP)**:")
                summary_img = os.path.basename(shap_reports[model_name].get("summary_plot", ""))
                bar_img = os.path.basename(shap_reports[model_name].get("bar_plot", ""))
                if summary_img:
                    lines.append(f"  - Beeswarm Summary Plot: [`{summary_img}`](file://{os.path.abspath(shap_reports[model_name]['summary_plot'])})")
                if bar_img:
                    lines.append(f"  - Feature Importance (Bar): [`{bar_img}`](file://{os.path.abspath(shap_reports[model_name]['bar_plot'])})")
                    
        lines.append("\n## 💡 Analytical Insights & System Recommendations")
        if best_model:
            lines.append(f"1. **Champion Selected**: `{best_model}` achieved the highest generalization performance with an R² Score of `{metrics[best_model].get('R2', 0.0):.4f}`.")
            if "MLP" in metrics and "Transformer" in metrics:
                lines.append("2. **Model Comparison**: Deep learning models were benchmarked alongside traditional gradient-boosted trees to verify representation capability.")
            lines.append("3. **Actionable Step**: Deploy the champion model wrapper using serialization tools for inference or API routing.")
            
        markdown_content = "\n".join(lines)
        
        filename = f"turn_{turn}_report.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        logger.info(f"Saved professional Markdown report to {filepath}")
        return filepath
