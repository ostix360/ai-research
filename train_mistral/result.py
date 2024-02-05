import glob
import json
import os
import matplotlib.pyplot as plt


def insert_newlines(string, every=64):
    lines = []
    for i in range(0, len(string), every):
        lines.append(string[i:i + every])
    return '\n'.join(lines)


def get_model_result(model_carac):
    # Get list of all directories that start with 'checkpoint-'
    model_carac_escaped = model_carac.replace("[", "[[").replace("]", "]]")
    model_carac_escaped = model_carac_escaped.replace("[[", "[[]").replace("]]", "[]]")
    checkpoint_dirs = glob.glob(f"./outputs-{model_carac_escaped}/checkpoint-*/")

    result = None
    for dir in checkpoint_dirs:
        try:
            with open(os.path.join(dir, "trainer_state.json"), "r") as f:
                result = json.load(f)
            # If file is successfully opened and loaded, break the loop
            break
        except FileNotFoundError:
            continue
    return result


def format_result(result):
    final_result = {}
    step_results = []
    loss_results = []
    for i in range(len(result["log_history"]) - 1):
        if "loss" in result["log_history"][i]:
            step_results.append(result["log_history"][i]["epoch"])
            loss_results.append(result["log_history"][i]["loss"])
    if "eval_loss" in result["log_history"][-1]:
        eval_loss = result["log_history"][-1]["eval_loss"]
    else:
        eval_loss = result["log_history"][-2]["eval_loss"]
    final_result["step_results"] = step_results
    final_result["loss_results"] = loss_results
    final_result["eval_loss"] = eval_loss
    return final_result


def plot_result(result, model_carac):
    plt.plot(result["step_results"], result["loss_results"])
    plt.title(f"Loss evolution for {model_carac}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def plot_results(model_caracs):
    for model_carac in model_caracs:
        result = get_model_result(model_carac)
        result = format_result(result)
        plot_result(result, model_carac)


def plot_all_results_in_one(model_caracs):
    fig, ax = plt.subplots(figsize=(18, 10))
    for model_carac in model_caracs:
        result = get_model_result(model_carac)
        result = format_result(result)
        label = model_carac.lstrip('_') if model_carac else None
        label = insert_newlines(label, 17)
        ax.plot(result["step_results"], result["loss_results"], label=label)
    ax.set_title(f"Training Loss evolution exp {exp}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(bbox_to_anchor=(1.0001, 1), loc='upper left')
    plt.show()

    # Plot eval loss on a histogram
    eval_losses = []
    for model_carac in model_caracs:
        result = get_model_result(model_carac)
        result = format_result(result)
        eval_losses.append(result["eval_loss"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(model_caracs, eval_losses)
    # add more space for the labels models_caracs
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=55)
    labels = [insert_newlines(label, 25) for label in model_caracs]
    ax.set_xticklabels(labels)

    min_loss = min(eval_losses)
    max_loss = max(eval_losses)
    ax.set_ylim([min_loss - 0.1 * (max_loss - min_loss), max_loss + 0.1 * (max_loss - min_loss)])

    ax.set_title(f"Eval losses Exp {exp}")
    ax.set_xlabel("Models")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()


exp = "4"


def get_caracs():
    caracs = []
    for file in os.listdir("./"):
        if file.startswith("outputs-"):
            if "hug" not in file or "lora" not in file or "ultra" not in file or "lora2" in file:
                continue
            caracs.append(file.title().lower().replace("outputs-", ""))
    return caracs


def plot_all_results():
    caracs = get_caracs()
    plot_results(caracs)


if __name__ == "__main__":
    caracs = get_caracs()
    plot_all_results_in_one(caracs)
