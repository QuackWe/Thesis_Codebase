from model import MultitaskBERTModel
import torch
from train import Config
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import defaultdict
from sys import argv

log = argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# Load the saved dataset
pytorch_dataset = torch.load('datasets/' + log + '/pytorch_dataset_consistent.pt')
pretrained_weights = f"datasets/{log}/mam_pretrained_model"

config = Config(pytorch_dataset)

# Load the model
model = MultitaskBERTModel(config, pretrained_weights=pretrained_weights).to(device)
# Load state_dict with strict=False to handle potential mismatches
checkpoint_path = f'datasets/{log}/multitask_bert_model.pth'
saved_state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(saved_state, strict=False)
# Load the saved prompts
model.prompted_bert.e_prompt.load_prompts(f'datasets/{log}')
model.eval()  # Set to evaluation mode


def evaluate_example(model, dataset, index, device):
    """
    Evaluate a single example and show predictions vs. real labels.
    Args:
        model: Trained multitask model.
        dataset: PyTorch Dataset object (TraceDataset).
        index: Index of the example to evaluate.
        device: Device (CPU or GPU).
    """
    model.eval()

    # Get the sample from the dataset
    example = dataset[index]
    trace = example['trace'].unsqueeze(0).to(device)  # Add batch dimension
    times = example['times'].unsqueeze(0).to(device)  # shape [1, seq_len]
    mask = example['mask'].unsqueeze(0).to(device)  # Add batch dimension
    customer_type = example['customer_type'].unsqueeze(0).to(device)  # Add batch dimension

    # Ensure all inputs have valid shapes
    assert trace.size(1) > 0, "Trace input is empty"
    assert mask.size(1) == trace.size(1), "Mask does not match trace length"
    assert customer_type.size(0) == trace.size(0), "Customer type batch size mismatch"

    real_next_activity = example['next_activity'].tolist()  # True labels for next activity
    real_outcome = example['outcome'].item()  # True label for final outcome

    # Make predictions
    with torch.no_grad():
        next_activity_logits, outcome_logits = model(trace, mask, times, customer_type)
        predicted_next_activity = torch.argmax(next_activity_logits, dim=2).tolist()[0]  # Sequence-level predictions
        predicted_outcome = torch.argmax(outcome_logits, dim=1).item()  # Single-label prediction

    # Print results
    print("Trace:", trace.cpu().tolist())
    print("Attention Mask:", mask.cpu().tolist())
    print("Customer Type:", customer_type.cpu().tolist())
    print(f"| Real Next Activity: \n{real_next_activity} \n| Predicted: \n{predicted_next_activity}")
    print(f"Real Outcome: {real_outcome} | Predicted: {predicted_outcome}")


def analyze_concept_mapping(model):
    """Analyze the mapping between customer types and concept IDs"""
    if hasattr(model.prompted_bert, 'e_prompt'):
        concept_mapping = model.prompted_bert.e_prompt.concept_mapping
        print("\n=== Concept Mapping Analysis ===")
        print("Customer Type -> Concept ID mapping:")
        for customer_type, concept_id in sorted(concept_mapping.items()):
            print(f"Customer Type {customer_type} -> Concept {concept_id}")


def visualize_prompts(model, num_examples=2):
    """Visualize sample prompts for different concepts"""
    print("\n=== Prompt Examples ===")

    # G-Prompt example
    if hasattr(model.prompted_bert, 'g_prompt'):
        g_key, g_value = model.prompted_bert.g_prompt.get_g_prompt(batch_size=1)
        print("\nG-Prompt (Global) Example:")
        print(f"Shape: {g_key.shape}")
        print(f"First few values: {g_key[0, 0, 0, :3, :3]}")  # First 3x3 values

    # E-Prompt examples
    if hasattr(model.prompted_bert, 'e_prompt'):
        print("\nE-Prompt (Concept-Specific) Examples:")
        concept_mapping = model.prompted_bert.e_prompt.concept_mapping
        device = next(model.parameters()).device  # Get model's device

        for i, (customer_type, concept_id) in enumerate(sorted(concept_mapping.items())):
            if i >= num_examples:
                break

            # Create a batch tensor with single customer type
            ctype_tensor = torch.tensor([float(customer_type)], device=device)
            e_prompt = model.prompted_bert.e_prompt.get_e_prompt(ctype_tensor)

            print(f"\nCustomer Type {customer_type} (Concept {concept_id}):")
            print(f"Shape: {e_prompt.shape}")
            print(f"First few values: {e_prompt[0, 0, 0, :3, :3]}")  # First 3x3 values


def evaluate_per_concept(model, dataset, device):
    """Evaluate model performance separately for each customer type/concept."""
    model.eval()
    concept_metrics = defaultdict(lambda: {
        'next_activity': {'true': [], 'pred': []},
        'outcome': {'true': [], 'pred': []}
    })

    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            trace = example['trace'].unsqueeze(0).to(device)
            mask = example['mask'].unsqueeze(0).to(device)
            times = example['times'].unsqueeze(0).to(device)
            customer_type = example['customer_type'].unsqueeze(0).to(device)

            # Get predictions
            next_activity_logits, outcome_logits = model(trace, mask, times, customer_type)

            # Store predictions by concept
            c_type = customer_type.item()
            concept_metrics[c_type]['next_activity']['true'].extend(example['next_activity'].tolist())
            concept_metrics[c_type]['next_activity']['pred'].extend(
                torch.argmax(next_activity_logits, dim=2)[0].cpu().tolist()
            )
            concept_metrics[c_type]['outcome']['true'].append(example['outcome'].item())
            concept_metrics[c_type]['outcome']['pred'].append(
                torch.argmax(outcome_logits, dim=1).item()
            )

    # Compute metrics per concept
    results = {}
    for c_type, metrics in concept_metrics.items():
        results[c_type] = {
            'next_activity': {
                'f1': f1_score(metrics['next_activity']['true'],
                               metrics['next_activity']['pred'], average='macro'),
                'accuracy': accuracy_score(metrics['next_activity']['true'],
                                           metrics['next_activity']['pred'])
            },
            'outcome': {
                'f1': f1_score(metrics['outcome']['true'],
                               metrics['outcome']['pred'], average='macro'),
                'accuracy': accuracy_score(metrics['outcome']['true'],
                                           metrics['outcome']['pred'])
            }
        }

    return results


def analyze_concept_drift(model, dataset, device):
    """Analyze how the model adapts to different customer types."""
    # Get E-Prompt mapping
    concept_mapping = model.prompted_bert.e_prompt.concept_mapping
    print("\n=== Concept Drift Analysis ===")
    print(f"Number of discovered concepts: {len(concept_mapping)}")
    print("Customer Type to Concept ID mapping:", concept_mapping)

    # Evaluate per concept
    concept_results = evaluate_per_concept(model, dataset, device)

    print("\n=== Performance by Customer Type ===")
    for c_type, metrics in concept_results.items():
        print(f"\nCustomer Type {c_type}:")
        print("Next Activity Prediction:")
        print(f"  F1 Score: {metrics['next_activity']['f1']:.4f}")
        print(f"  Accuracy: {metrics['next_activity']['accuracy']:.4f}")
        print("Outcome Prediction:")
        print(f"  F1 Score: {metrics['outcome']['f1']:.4f}")
        print(f"  Accuracy: {metrics['outcome']['accuracy']:.4f}")


def evaluate_model(model, dataset, device):
    """
    Evaluate the model on the entire dataset and compute metrics.
    Args:
        model: Trained multitask model.
        dataset: PyTorch Dataset object (TraceDataset).
        device: Device (CPU or GPU).
    """
    model.eval()

    all_true_next_activity = []
    all_pred_next_activity = []

    all_true_outcome = []
    all_pred_outcome = []

    prefix_length_metrics_next_activity = defaultdict(lambda: {"true": [], "pred": []})
    prefix_length_metrics_outcome = defaultdict(lambda: {"true": [], "pred": []})

    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            trace = example['trace'].unsqueeze(0).to(device)  # Add batch dimension
            mask = example['mask'].unsqueeze(0).to(device)  # Add batch dimension
            times = example['times'].unsqueeze(0).to(device)
            customer_type = example['customer_type'].unsqueeze(0).to(device)  # Add batch dimension
            real_next_activity = example['next_activity'].tolist()
            real_outcome = example['outcome'].item()

            next_activity_logits, outcome_logits = model(trace, mask, times, customer_type)
            predicted_next_activity = torch.argmax(next_activity_logits, dim=2).tolist()[
                0]  # Sequence-level predictions
            predicted_outcome = torch.argmax(outcome_logits, dim=1).item()  # Single-label prediction

            # Collect results for next activity prediction (sequence-level comparison)
            all_true_next_activity.extend(real_next_activity)
            all_pred_next_activity.extend(predicted_next_activity[:len(real_next_activity)])  # Match sequence lengths

            # Collect results for final outcome prediction (single-label comparison)
            all_true_outcome.append(real_outcome)
            all_pred_outcome.append(predicted_outcome)

            # Collect prefix-length-specific metrics
            prefix_length = mask.sum().item()  # Count of non-padded tokens
            prefix_length_metrics_next_activity[prefix_length]["true"].extend(real_next_activity)
            prefix_length_metrics_next_activity[prefix_length]["pred"].extend(
                predicted_next_activity[:len(real_next_activity)])
            prefix_length_metrics_outcome[prefix_length]["true"].append(real_outcome)
            prefix_length_metrics_outcome[prefix_length]["pred"].append(predicted_outcome)

    # Compute overall metrics for next activity prediction
    overall_f1_next_activity = f1_score(all_true_next_activity, all_pred_next_activity, average="macro")
    overall_accuracy_next_activity = accuracy_score(all_true_next_activity, all_pred_next_activity)
    overall_precision_next_activity = precision_score(all_true_next_activity, all_pred_next_activity, average="macro")
    overall_recall_next_activity = recall_score(all_true_next_activity, all_pred_next_activity, average="macro")

    # Compute metrics for outcome prediction
    overall_f1_outcome = f1_score(all_true_outcome, all_pred_outcome, average="macro")
    overall_accuracy_outcome = accuracy_score(all_true_outcome, all_pred_outcome)
    overall_precision_outcome = precision_score(all_true_outcome, all_pred_outcome, average="macro")
    overall_recall_outcome = recall_score(all_true_outcome, all_pred_outcome, average="macro")

    # Display overall results
    print("\n=== Overall Metrics ===")
    print("Next Activity Prediction:")
    print(f"  F1 Score: {overall_f1_next_activity:.4f}")
    print(f"  Accuracy: {overall_accuracy_next_activity:.4f}")
    print(f"  Precision: {overall_precision_next_activity:.4f}")
    print(f"  Recall: {overall_recall_next_activity:.4f}")

    print("\nOutcome Prediction:")
    print(f"  F1 Score: {overall_f1_outcome:.4f}")
    print(f"  Accuracy: {overall_accuracy_outcome:.4f}")
    print(f"  Precision: {overall_precision_outcome:.4f}")
    print(f"  Recall: {overall_recall_outcome:.4f}")

    # Display prefix-length-specific metrics for next activity
    print("\n=== Metrics by Prefix Length (Next Activity Prediction) ===")
    for prefix_length, metrics in sorted(prefix_length_metrics_next_activity.items()):
        true_values = metrics["true"]
        pred_values = metrics["pred"]

        f1 = f1_score(true_values, pred_values, average="macro")
        accuracy = accuracy_score(true_values, pred_values)
        precision = precision_score(true_values, pred_values, average="macro", zero_division=0)
        recall = recall_score(true_values, pred_values, average="macro", zero_division=0)

        print(f"Prefix Length {prefix_length}:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

    # Display prefix-length-specific metrics for final outcome
    print("\n=== Metrics by Prefix Length (Final Outcome Prediction) ===")
    for prefix_length, metrics in sorted(prefix_length_metrics_outcome.items()):
        true_values = metrics["true"]
        pred_values = metrics["pred"]

        f1 = f1_score(true_values, pred_values, average="macro")
        accuracy = accuracy_score(true_values, pred_values)
        precision = precision_score(true_values, pred_values, average="macro", zero_division=0)
        recall = recall_score(true_values, pred_values, average="macro", zero_division=0)

        print(f"Prefix Length {prefix_length}:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")


# Evaluate a single example
evaluate_example(model, pytorch_dataset, index=5, device=device)

# Concept mapping example analysis
analyze_concept_mapping(model)

# Prompt visualization
visualize_prompts(model)

# Concept drift analysis
analyze_concept_drift(model, pytorch_dataset, device)

# Perform comprehensive evaluation
evaluate_model(model, pytorch_dataset, device=device)
