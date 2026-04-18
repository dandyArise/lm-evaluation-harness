import argparse
import json
import logging
import requests
import sys
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from datetime import datetime
from lm_eval import evaluator, tasks
from lm_eval.models.lm_studio import LMStudioLM

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def fetch_models(base_url, api_key):
    """Fetch list of models from LM Studio."""
    url = f"{base_url}/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if "models" in data:
            # Filter for LLMs if 'type' is present, otherwise take all keys
            return [m["key"] for m in data["models"] if m.get("type") == "llm" or "key" in m]
        elif "data" in data:
            return [m["id"] for m in data["data"]]
        return []
    except Exception as e:
        logger.error(f"Failed to fetch models from {url}: {e}")
        return []

def run_benchmark(model_id, base_url, api_key, task_list, num_fewshot, limit):
    """Run benchmark for a single model."""
    logger.info(f"--- Starting Benchmark for Model: {model_id} ---")
    
    # Initialize LM instance
    lm = LMStudioLM(base_url=base_url, api_key=api_key, model=model_id)
    
    try:
        # Load model explicitly
        logger.info(f"Loading model {model_id}...")
        lm.load_model(model_id)
        
        # Run evaluation
        logger.info(f"Running evaluation on tasks: {task_list}")
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=num_fewshot,
            limit=limit,
        )
        
        # Unload model
        logger.info(f"Unloading model {model_id}...")
        lm.unload_model(model_id)
        
        # Return results and collected latencies
        return results["results"], lm.latencies
    except Exception as e:
        logger.error(f"Error benchmarking {model_id}: {e}")
        return None, []

def generate_leaderboard(all_results, output_file):
    """Generate a Markdown leaderboard."""
    if not all_results:
        logger.warning("No results to generate leaderboard.")
        return

    # Extract all metrics found across all models/tasks
    all_metrics = set()
    for model_res in all_results.values():
        for task_res in model_res.values():
            for metric in task_res.keys():
                if metric != "alias":
                    all_metrics.add(metric)
    
    all_metrics = sorted(list(all_metrics))
    
    # Get all tasks across all models
    all_tasks = set()
    for model_res in all_results.values():
        if model_res:
            all_tasks.update(model_res.keys())
    
    if not all_tasks:
        logger.warning("No tasks found in results.")
        return

    with open(output_file, "w") as f:
        f.write("# LM Studio Benchmark Leaderboard\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for task in sorted(list(all_tasks)):
            f.write(f"## Task: {task}\n\n")
            header = "| Model | " + " | ".join(all_metrics) + " |\n"
            separator = "| :--- | " + " | ".join([":---:"] * len(all_metrics)) + " |\n"
            f.write(header)
            f.write(separator)
            
            for model_id, results in all_results.items():
                if task in results:
                    row = f"| {model_id} | "
                    metrics_vals = []
                    for metric in all_metrics:
                        val = results[task].get(metric, "N/A")
                        if isinstance(val, float):
                            metrics_vals.append(f"{val:.4f}")
                        else:
                            metrics_vals.append(str(val))
                    row += " | ".join(metrics_vals) + " |\n"
                    f.write(row)
            f.write("\n")

    logger.info(f"Leaderboard saved to {output_file}")

def generate_charts(all_results, all_latencies, output_prefix):
    """Generate bar charts for results and latencies."""
    if not all_results:
        return []

    charts = []
    # 1. Accuracy per Task
    # For simplicity, we'll take the first metric that looks like an accuracy/score
    data = []
    for model_id, model_res in all_results.items():
        for task, res in model_res.items():
            # Find a primary metric (accuracy, exact_match, etc.)
            metric = next((m for m in res if "acc" in m or "exact" in m or "f1" in m), None)
            if metric:
                data.append({"Model": model_id, "Task": task, "Score": res[metric]})

    if data:
        df = pd.DataFrame(data)
        for task in df["Task"].unique():
            df_task = df[df["Task"] == task]
            plt.figure(figsize=(10, 6))
            plt.bar(df_task["Model"], df_task["Score"], color="skyblue")
            plt.title(f"Performance on {task}")
            plt.ylabel("Score")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            chart_file = f"{output_prefix}_{task}_performance.png"
            plt.savefig(chart_file)
            plt.close()
            charts.append(chart_file)

    # 2. Average Latency per Model
    latency_data = []
    for model_id, shifts in all_latencies.items():
        if shifts:
            avg_latency = sum(shifts) / len(shifts)
            latency_data.append({"Model": model_id, "Avg Latency (s)": avg_latency})

    if latency_data:
        df_lat = pd.DataFrame(latency_data)
        plt.figure(figsize=(10, 6))
        plt.bar(df_lat["Model"], df_lat["Avg Latency (s)"], color="salmon")
        plt.title("Average Response Latency")
        plt.ylabel("Seconds")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        chart_file = f"{output_prefix}_latency.png"
        plt.savefig(chart_file)
        plt.close()
        charts.append(chart_file)

    return charts

def generate_pdf_report(all_results, all_latencies, charts, output_file):
    """Generate a PDF report with maximum compatibility."""
    # Use a unique name if it already exists or just logging
    logger.info(f"Producing PDF: {output_file}")
    
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # 1. Title Page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(0, 50, 100)
        pdf.cell(0, 20, text="LM Studio Benchmark Report", align="C", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(20)

        if not all_results:
            pdf.set_font("Helvetica", "I", 14)
            pdf.cell(0, 10, text="No benchmark results were found.", new_x="LMARGIN", new_y="NEXT")
            pdf.output(output_file)
            return

        # 2. Results Table
        all_tasks = set()
        for res in all_results.values():
            if res: all_tasks.update(res.keys())
            
        for task in sorted(list(all_tasks)):
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, text=f"Task Results: {task}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)
            
            # Header
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(90, 10, text="Model Name", border=1, fill=True)
            pdf.cell(60, 10, text="Metric", border=1, fill=True)
            pdf.cell(40, 10, text="Score", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
            
            # Rows
            pdf.set_font("Helvetica", "", 9)
            for mid, mres in all_results.items():
                if task in mres:
                    short_id = mid.split('/')[-1] if '/' in mid else mid
                    short_id = short_id[:45]
                    
                    # Safe metric search
                    metrics = mres[task]
                    best_m = "score"
                    for candidate in ["acc", "exact", "f1", "score", "em"]:
                        found = next((k for k in metrics if candidate in k.lower()), None)
                        if found:
                            best_m = found
                            break
                    
                    val = metrics.get(best_m, 0.0)
                    val_str = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                    
                    # Print row with safe strings
                    pdf.cell(90, 10, text=short_id.encode('latin-1', 'replace').decode('latin-1'), border=1)
                    pdf.cell(60, 10, text=best_m.encode('latin-1', 'replace').decode('latin-1'), border=1)
                    pdf.cell(40, 10, text=val_str, border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(10)

        # 3. Latency Page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, text="Latency Performance", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(120, 10, text="Model", border=1, fill=True)
        pdf.cell(70, 10, text="Avg Response Latency", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 10)
        for mid, lats in all_latencies.items():
            avg = sum(lats)/len(lats) if lats else 0.0
            display_name = mid.split('/')[-1] if '/' in mid else mid
            pdf.cell(120, 10, text=display_name[:60].encode('latin-1', 'replace').decode('latin-1'), border=1)
            pdf.cell(70, 10, text=f"{avg:.3f} seconds", border=1, new_x="LMARGIN", new_y="NEXT")

        # 4. Charts
        for chart in charts:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, text=f"Chart: {chart}", new_x="LMARGIN", new_y="NEXT")
            try:
                pdf.image(chart, x=10, y=30, w=190)
            except:
                pdf.cell(0, 10, text="[Chart could not be rendered]", new_x="LMARGIN", new_y="NEXT")

        pdf.output(output_file)
        logger.info(f"Done outputting PDF to {output_file}")
    except Exception as e:
        logger.error(f"FATAL ERROR during PDF generation: {e}")
        # Final attempt: write a text file if PDF fails
        with open(output_file + ".error.txt", "w") as ef:
            ef.write(f"PDF Gen failed: {e}\nResults: {all_results}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark all LM Studio models.")
    parser.add_argument("--base_url", default="http://localhost:1234", help="LM Studio base URL")
    parser.add_argument("--api_key", default="VITE_LOCAL_KEY=sk-lm-3gwCCdX2:WbSQQH2bebhEvhXM79vs", help="LM Studio API key")
    parser.add_argument("--tasks", default="gsm8k", help="Comma-separated list of tasks")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per task")
    parser.add_argument("--output", default="leaderboard.md", help="Output file for leaderboard")
    parser.add_argument("--model", default=None, help="Specific model key to benchmark (optional)")
    parser.add_argument("--export_pdf", default=None, help="Export a PDF report to this file")
    
    args = parser.parse_args()
    task_list = args.tasks.split(",")
    
    models = fetch_models(args.base_url, args.api_key)
    if not models:
        logger.error("No models found in LM Studio. Is it running?")
        sys.exit(1)
        
    if args.model:
        if args.model in models:
            models = [args.model]
        else:
            logger.error(f"Model {args.model} not found in available models: {models}")
            sys.exit(1)

    logger.info(f"Selected {len(models)} models: {models}")
    
    all_results = {}
    all_latencies = {}
    for model_id in models:
        res, latencies = run_benchmark(model_id, args.base_url, args.api_key, task_list, args.num_fewshot, args.limit)
        if res:
            all_results[model_id] = res
            all_latencies[model_id] = latencies
            
    generate_leaderboard(all_results, args.output)

    if args.export_pdf:
        charts = generate_charts(all_results, all_latencies, "benchmark_chart")
        generate_pdf_report(all_results, all_latencies, charts, args.export_pdf)

if __name__ == "__main__":
    main()
