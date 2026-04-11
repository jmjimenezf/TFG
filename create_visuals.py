# Script to create images from the results of the experiments *stats.json.
# The figures and tables will be saved in the results folder with the name of the experiment and the model
# Generate also truth tables for each model and save them in the results folder with the name of the experiment and the model.
# It must have the style of APA 7th edition.

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# APA 7th edition style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (8.5, 6)
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5

#experiment_folder = "results/20260408_230658_runSA2Test"
experiment_folder = "results/20260408_233711_runEA1Test"


def load_stats_files(experiment_folder):
    """Load all *stats.json files from the experiment folder."""
    stats_files = {}
    for file in Path(experiment_folder).glob("*_stats.json"):
        with open(file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            # Extract model name from filename: 20260408_results_ExpName-model:version_stats.json
            filename = file.stem.replace('_stats', '')  # Remove _stats suffix
            parts = filename.split('_')[2:]  # Skip date and 'results'
            model_full = '_'.join(parts)  # Rejoin the rest
            # Extract just the model name (before the colon)
            model_name = model_full.split(':')[0].split('-')[-1]  # Get last part after splitting by colon and dash
            stats['model'] = model_name  # Update the model name in stats to shortened version
            stats_files[model_name] = stats
    return stats_files


def detect_experiment_type(stats_dict):
    """Detect if experiment is SA (Sentiment Analysis) or EA (Emotion Analysis)."""
    if not stats_dict:
        return None
    first_stats = next(iter(stats_dict.values()))
    if 'true_positives' in first_stats:
        return 'SA'  # Sentiment Analysis
    elif 'per_emotion_stats' in first_stats:
        return 'EA'  # Emotion Analysis
    return None


def create_confusion_matrix_plot(tp, tn, fp, fn, model_name, output_path):
    """Create confusion matrix visualization for SA experiments."""
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Create heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                cbar=False, ax=ax, annot_kws={'size': 12, 'weight': 'bold'},
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_metrics_table(stats_dict, output_path):
    """Create table with performance metrics for all models."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Prepare data
    rows = []
    for model_name in sorted(stats_dict.keys()):
        stats = stats_dict[model_name]
        if 'true_positives' in stats:  # SA experiment
            rows.append([
                model_name,
                f"{stats['accuracy']:.4f}",
                f"{stats['precision']:.4f}",
                f"{stats['recall']:.4f}",
                f"{stats['f1_score']:.4f}"
            ])
    
    # Create table
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center',
                     loc='center', colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E1F2')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Performance Metrics by Model', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_confusion_values_table(tp, tn, fp, fn, model_name, output_path):
    """Create table showing TP, TN, FP, FN values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    rows = [
        ['True Positives', f'{tp}'],
        ['True Negatives', f'{tn}'],
        ['False Positives', f'{fp}'],
        ['False Negatives', f'{fn}'],
        ['Total Samples', f'{tp + tn + fp + fn}']
    ]
    
    table = ax.table(cellText=rows, colLabels=['Metric', 'Count'],
                     cellLoc='center', loc='center', colWidths=[0.4, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E1F2')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title(f'Confusion Values - {model_name}', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_bar_chart(stats_dict, output_path):
    """Create bar chart comparing models across metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    models = sorted(stats_dict.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [stats_dict[model].get(metric, 0) for model in models]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_emotion_metrics_table(stats_dict, output_path):
    """Create table for EA (Emotion Analysis) multi-class metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    rows = []
    for model_name in sorted(stats_dict.keys()):
        stats = stats_dict[model_name]
        if 'overall_accuracy' in stats:
            rows.append([
                model_name,
                f"{stats['overall_accuracy']:.4f}",
                f"{stats['macro_avg']['precision']:.4f}",
                f"{stats['macro_avg']['recall']:.4f}",
                f"{stats['macro_avg']['f1-score']:.4f}",
                f"{stats['weighted_avg']['precision']:.4f}",
                f"{stats['weighted_avg']['f1-score']:.4f}"
            ])
    
    columns = ['Model', 'Accuracy', 'Macro-P', 'Macro-R', 'Macro-F1', 'Weighted-P', 'Weighted-F1']
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center',
                     loc='center', colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E1F2')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Emotion Analysis - Performance Metrics', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_per_emotion_table(model_name, per_emotion_stats, output_path):
    """Create table showing per-emotion metrics for a single model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    rows = []
    for emotion in sorted(per_emotion_stats.keys()):
        stats = per_emotion_stats[emotion]
        rows.append([
            emotion.capitalize(),
            f"{stats['precision']:.4f}",
            f"{stats['recall']:.4f}",
            f"{stats['f1-score']:.4f}",
            f"{stats['support']}"
        ])
    
    columns = ['Emotion', 'Precision', 'Recall', 'F1-Score', 'Support']
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center',
                     loc='center', colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E1F2')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title(f'Per-Emotion Metrics - {model_name}', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_emotion_distribution_chart(stats_dict, output_path):
    """Create bar chart showing dataset distribution per emotion (support counts)."""
    # Collect all emotions and their support values
    emotion_support = {}
    for model_name, stats in stats_dict.items():
        if 'per_emotion_stats' in stats:
            for emotion, metrics in stats['per_emotion_stats'].items():
                if emotion not in emotion_support:
                    emotion_support[emotion] = metrics['support']
            break  # Support is the same across all models for the same dataset
    
    if not emotion_support:
        return
    
    # Sort emotions
    emotions = sorted(emotion_support.keys())
    support_values = [emotion_support[e] for e in emotions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(emotions, support_values, color='#4472C4', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Emotion', fontweight='bold', fontsize=11)
    ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=11)
    ax.set_title('Dataset Distribution - Samples per Emotion', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_emotion_performance_heatmap(stats_dict, output_path, metric='f1-score'):
    """Create heatmap showing performance metric (F1, Precision, Recall) across models and emotions."""
    # Collect data for heatmap
    models = sorted(stats_dict.keys())
    emotions = set()
    
    # Gather all emotions
    for stats in stats_dict.values():
        if 'per_emotion_stats' in stats:
            emotions.update(stats['per_emotion_stats'].keys())
    
    emotions = sorted(emotions)
    
    # Create matrix
    data_matrix = np.zeros((len(models), len(emotions)))
    
    for i, model_name in enumerate(models):
        stats = stats_dict[model_name]
        if 'per_emotion_stats' in stats:
            for j, emotion in enumerate(emotions):
                if emotion in stats['per_emotion_stats']:
                    data_matrix[i, j] = stats['per_emotion_stats'][emotion].get(metric, 0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=emotions, yticklabels=models, 
                cbar_kws={'label': metric.title()}, ax=ax,
                annot_kws={'size': 9, 'weight': 'bold'})
    
    ax.set_xlabel('Emotion', fontweight='bold', fontsize=11)
    ax.set_ylabel('Model', fontweight='bold', fontsize=11)
    ax.set_title(f'Per-Emotion {metric.title()} - All Models', fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_model_emotion_confusion_heatmap(model_name, per_emotion_stats, output_path):
    """Create heatmap showing precision, recall, F1 for each emotion in a single model."""
    emotions = sorted(per_emotion_stats.keys())
    metrics = ['precision', 'recall', 'f1-score']
    
    # Create matrix (emotions x metrics)
    data_matrix = np.zeros((len(emotions), len(metrics)))
    
    for i, emotion in enumerate(emotions):
        for j, metric in enumerate(metrics):
            data_matrix[i, j] = per_emotion_stats[emotion].get(metric, 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=[m.replace('-', ' ').title() for m in metrics],
                yticklabels=emotions,
                cbar_kws={'label': 'Score'}, ax=ax,
                annot_kws={'size': 10, 'weight': 'bold'},
                vmin=0, vmax=1)
    
    ax.set_xlabel('Metric', fontweight='bold', fontsize=11)
    ax.set_ylabel('Emotion', fontweight='bold', fontsize=11)
    ax.set_title(f'Performance Metrics per Emotion - {model_name}', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_visualizations(experiment_folder):
    """Main function to generate all visualizations."""
    print(f"Processing experiment folder: {experiment_folder}")
    
    # Load stats files
    stats_dict = load_stats_files(experiment_folder)
    if not stats_dict:
        print(f"No stats files found in {experiment_folder}")
        return
    
    print(f"Found {len(stats_dict)} models")
    
    # Detect experiment type
    exp_type = detect_experiment_type(stats_dict)
    print(f"Experiment type: {exp_type}")
    
    # Create output directory for figures
    output_dir = Path(experiment_folder) / "visuals"
    output_dir.mkdir(exist_ok=True)
    
    if exp_type == 'SA':
        # Create confusion matrices and performance tables for SA
        for model_name, stats in stats_dict.items():
            print(f"Processing model: {model_name}")
            
            # Confusion matrix plot
            cm_output = output_dir / f"{Path(experiment_folder).name}_{model_name}_confusion_matrix.png"
            create_confusion_matrix_plot(
                stats['true_positives'],
                stats['true_negatives'],
                stats['false_positives'],
                stats['false_negatives'],
                model_name,
                cm_output
            )
            print(f"  ✓ Saved confusion matrix to {cm_output}")
            
            # Confusion values table
            cv_output = output_dir / f"{Path(experiment_folder).name}_{model_name}_confusion_values.png"
            create_confusion_values_table(
                stats['true_positives'],
                stats['true_negatives'],
                stats['false_positives'],
                stats['false_negatives'],
                model_name,
                cv_output
            )
            print(f"  ✓ Saved confusion values to {cv_output}")
        
        # Overall performance table
        perf_output = output_dir / f"{Path(experiment_folder).name}_performance_metrics.png"
        create_performance_metrics_table(stats_dict, perf_output)
        print(f"✓ Saved performance metrics table to {perf_output}")
        
        # Comparison bar chart
        bar_output = output_dir / f"{Path(experiment_folder).name}_comparison_chart.png"
        create_comparison_bar_chart(stats_dict, bar_output)
        print(f"✓ Saved comparison chart to {bar_output}")
    
    elif exp_type == 'EA':
        # Create emotion analysis visualizations
        # Overall emotions metrics table
        emotion_output = output_dir / f"{Path(experiment_folder).name}_emotion_metrics.png"
        create_emotion_metrics_table(stats_dict, emotion_output)
        print(f"✓ Saved emotion metrics table to {emotion_output}")
        
        # Per-emotion tables for each model
        for model_name, stats in stats_dict.items():
            print(f"Processing model: {model_name}")
            if 'per_emotion_stats' in stats:
                per_emotion_output = output_dir / f"{Path(experiment_folder).name}_{model_name}_per_emotion.png"
                create_per_emotion_table(model_name, stats['per_emotion_stats'], per_emotion_output)
                print(f"  ✓ Saved per-emotion metrics to {per_emotion_output}")
        
        # Dataset distribution chart
        dist_output = output_dir / f"{Path(experiment_folder).name}_emotion_distribution.png"
        create_emotion_distribution_chart(stats_dict, dist_output)
        print(f"✓ Saved emotion distribution chart to {dist_output}")
        
        # Per-emotion performance heatmap (F1-scores across models and emotions)
        heatmap_f1_output = output_dir / f"{Path(experiment_folder).name}_emotion_f1_heatmap.png"
        create_emotion_performance_heatmap(stats_dict, heatmap_f1_output, 'f1-score')
        print(f"✓ Saved F1-score heatmap to {heatmap_f1_output}")
        
        # Per-model confusion heatmaps (precision, recall, F1 per emotion)
        for model_name, stats in stats_dict.items():
            if 'per_emotion_stats' in stats:
                confusion_heatmap_output = output_dir / f"{Path(experiment_folder).name}_{model_name}_confusion_heatmap.png"
                create_model_emotion_confusion_heatmap(model_name, stats['per_emotion_stats'], confusion_heatmap_output)
                print(f"  ✓ Saved confusion heatmap for {model_name}")
    
    print(f"\n✓ All visualizations completed for {experiment_folder}")


# Run for the specified experiment folder
if __name__ == "__main__":
    generate_visualizations(experiment_folder)
