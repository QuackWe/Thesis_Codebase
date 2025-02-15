from model import MultitaskBERTModel
import torch
from train import Config
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import defaultdict
from sys import argv
import numpy as np
from preprocess import MemoryMappedDataset
from FeatureEngineering import add_all_features, define_features

log = argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# Load the saved dataset
dataset_info = torch.load(f'datasets/{log}/pytorch_dataset_consistent.pt')
if isinstance(dataset_info, dict) and 'dataset_dir' in dataset_info:
    # Handle memory-mapped dataset
    pytorch_dataset = MemoryMappedDataset(dataset_info['dataset_dir'])
else:
    # Regular dataset
    pytorch_dataset = dataset_info
pretrained_weights = f"datasets/{log}/mam_pretrained_model"

# Load mappings
mappings = torch.load(f'datasets/{log}/mappings_consistent.pt')

# Add loop features to dataset
loop_activities_by_outcome, time_sensitive_transitions = define_features(log)
add_all_features(pytorch_dataset, log, mappings, loop_activities_by_outcome, time_sensitive_transitions)

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
    print('\n=== Evaluate Example ===')
    model.eval()
    example = dataset[index]
    # Handle missing loop features
    loop_features = example.get('loop_features',
                                torch.zeros(model.config.total_feature_dim))
    trace = example['trace'].unsqueeze(0).to(device)
    times = example['times'].unsqueeze(0).to(device)
    mask = example['mask'].unsqueeze(0).to(device)
    customer_type = example['customer_type'].unsqueeze(0).to(device)
    loop_features = loop_features.unsqueeze(0).to(device)
    # Ensure correct feature dimension
    expected_feat_dim = model.config.total_feature_dim
    if loop_features.shape[1] != expected_feat_dim:
        if loop_features.shape[1] < expected_feat_dim:
            padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
            loop_features = torch.cat([loop_features, padding], dim=1)
        else:
            loop_features = loop_features[:, :expected_feat_dim]

    real_next_activity = example['next_activity'].tolist()
    real_outcome = example['outcome'].item()

    with torch.no_grad():
        next_activity_logits, outcome_logits = model(
            trace, mask, times, loop_features, customer_type
        )
        predicted_next_activity = torch.argmax(next_activity_logits, dim=2).tolist()[0]
        predicted_outcome = torch.argmax(outcome_logits, dim=1).item()

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
            loop_features = example.get('loop_features', torch.zeros(model.config.total_feature_dim)).unsqueeze(0).to(
                device)
            # Ensure correct feature dimension
            expected_feat_dim = model.config.total_feature_dim
            if loop_features.shape[1] != expected_feat_dim:
                if loop_features.shape[1] < expected_feat_dim:
                    padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                    loop_features = torch.cat([loop_features, padding], dim=1)
                else:
                    loop_features = loop_features[:, :expected_feat_dim]

            # Get predictions
            next_activity_logits, outcome_logits = model(trace, mask, times, loop_features, customer_type)

            c_type = customer_type.item()
            valid_mask = mask[0].bool()

            concept_metrics[c_type]['next_activity']['true'].extend(
                example['next_activity'].to(device)[valid_mask].cpu().tolist()
            )
            concept_metrics[c_type]['next_activity']['pred'].extend(
                torch.argmax(next_activity_logits[0, valid_mask], dim=1).cpu().tolist()
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
                'f1': f1_score(
                    metrics['next_activity']['true'],
                    metrics['next_activity']['pred'],
                    average='macro'
                ),
                'accuracy': accuracy_score(
                    metrics['next_activity']['true'],
                    metrics['next_activity']['pred']
                )
            },
            'outcome': {
                'f1': f1_score(
                    metrics['outcome']['true'],
                    metrics['outcome']['pred'],
                    average='macro'
                ),
                'accuracy': accuracy_score(
                    metrics['outcome']['true'],
                    metrics['outcome']['pred']
                )
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

    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            trace = example['trace'].unsqueeze(0).to(device)
            mask = example['mask'].unsqueeze(0).to(device)
            times = example['times'].unsqueeze(0).to(device)
            customer_type = example['customer_type'].unsqueeze(0).to(device)
            loop_features = example.get('loop_features', torch.zeros(model.config.total_feature_dim)).unsqueeze(0).to(
                device)
            # Ensure correct feature dimension
            expected_feat_dim = model.config.total_feature_dim
            if loop_features.shape[1] != expected_feat_dim:
                if loop_features.shape[1] < expected_feat_dim:
                    padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                    loop_features = torch.cat([loop_features, padding], dim=1)
                else:
                    loop_features = loop_features[:, :expected_feat_dim]

            # Get predictions
            next_activity_logits, outcome_logits = model(trace, mask, times, loop_features, customer_type)

            # Get actual sequence length from model output
            seq_len = next_activity_logits.shape[1]  # [batch_size, seq_len, num_activities]

            # Truncate masks and labels to match model output
            valid_positions = mask[0, :seq_len].bool().cpu()
            next_act_true = example['next_activity'][:seq_len].cpu()

            # Get predictions and apply mask
            pred_activities = torch.argmax(next_activity_logits[0], dim=1).cpu()[valid_positions].tolist()
            true_activities = next_act_true[valid_positions].tolist()

            all_true_next_activity.extend(true_activities)
            all_pred_next_activity.extend(pred_activities)
            all_true_outcome.append(example['outcome'].item())
            all_pred_outcome.append(torch.argmax(outcome_logits, dim=1).cpu().item())

    # Compute metrics with proper handling of edge cases
    next_activity_metrics = {
        'f1': f1_score(all_true_next_activity, all_pred_next_activity, average='macro', zero_division=0),
        'accuracy': accuracy_score(all_true_next_activity, all_pred_next_activity),
        'precision': precision_score(all_true_next_activity, all_pred_next_activity, average='macro', zero_division=0),
        'recall': recall_score(all_true_next_activity, all_pred_next_activity, average='macro', zero_division=0)
    }

    outcome_metrics = {
        'f1': f1_score(all_true_outcome, all_pred_outcome, average='macro', zero_division=0),
        'accuracy': accuracy_score(all_true_outcome, all_pred_outcome),
        'precision': precision_score(all_true_outcome, all_pred_outcome, average='macro', zero_division=0),
        'recall': recall_score(all_true_outcome, all_pred_outcome, average='macro', zero_division=0)
    }

    print("\n=== Overall Metrics ===")
    print("Next Activity Prediction:")
    for metric, value in next_activity_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

    print("\nOutcome Prediction:")
    for metric, value in outcome_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")


def analyze_feature_importance(model, dataset, device):
    print("\n=== Feature Importance Analysis ===")

    model.eval()
    outcome_results = defaultdict(lambda: {
        'original': [],
        'no_features': []
    })

    # Get expected feature dimension from model config
    expected_feat_dim = model.config.total_feature_dim

    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            trace = example['trace'].unsqueeze(0).to(device)
            mask = example['mask'].unsqueeze(0).to(device)
            times = example['times'].unsqueeze(0).to(device)
            customer_type = example['customer_type'].unsqueeze(0).to(device)
            loop_features = example['loop_features'].unsqueeze(0).to(device)

            # Ensure loop_features has correct shape
            if loop_features.shape[1] != expected_feat_dim:
                if loop_features.shape[1] < expected_feat_dim:
                    padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                    loop_features = torch.cat([loop_features, padding], dim=1)
                else:
                    loop_features = loop_features[:, :expected_feat_dim]

            # Original predictions
            next_act_logits, outcome_logits = model(
                trace, mask, times, loop_features, customer_type
            )

            # Zero feature predictions
            zero_features = torch.zeros_like(loop_features)
            next_act_logits_p, outcome_logits_p = model(
                trace, mask, times, zero_features, customer_type
            )

            # Only store outcome predictions
            outcome_results[i]['original'] = torch.softmax(outcome_logits, dim=1)
            outcome_results[i]['no_features'] = torch.softmax(outcome_logits_p, dim=1)

    # Compute impact metrics only for outcome
    outcome_impact = torch.mean(torch.abs(
        torch.cat([r['original'] for r in outcome_results.values()]) -
        torch.cat([r['no_features'] for r in outcome_results.values()])
    )).item()

    print(f"Feature Impact on Outcome Prediction: {outcome_impact:.4f}")


def analyze_temporal_patterns(model, dataset, device):
    print("\n=== Temporal Pattern Analysis ===")
    model.eval()
    temporal_metrics = defaultdict(lambda: {
        'success_times': [],
        'failure_times': [],
        'predictions': {'correct': 0, 'total': 0}
    })

    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            trace = example['trace'].unsqueeze(0).to(device)
            mask = example['mask'].unsqueeze(0).to(device)
            times = example['times'].unsqueeze(0).to(device)
            customer_type = example['customer_type'].unsqueeze(0).to(device)
            loop_features = example['loop_features'].unsqueeze(0).to(device)

            # Ensure correct feature dimension
            expected_feat_dim = model.config.total_feature_dim
            if loop_features.shape[1] != expected_feat_dim:
                if loop_features.shape[1] < expected_feat_dim:
                    padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                    loop_features = torch.cat([loop_features, padding], dim=1)
                else:
                    loop_features = loop_features[:, :expected_feat_dim]

            # Get predictions
            _, outcome_logits = model(trace, mask, times, loop_features, customer_type)
            predicted_outcome = torch.argmax(outcome_logits, dim=1).item()
            actual_outcome = example['outcome'].item()

            # Analyze transition times
            valid_mask = mask[0].bool()
            trace_times = times[0][valid_mask].cpu().tolist()

            # Calculate time differences between consecutive events
            time_diffs = [abs(trace_times[i + 1] - trace_times[i])
                          for i in range(len(trace_times) - 1)]

            for time_diff in time_diffs:
                if actual_outcome == 1:  # Success case
                    temporal_metrics['all']['success_times'].append(time_diff)
                else:  # Failure case
                    temporal_metrics['all']['failure_times'].append(time_diff)

    # Print analysis results
    print("\nTransition Time Analysis:")
    success_avg = np.mean(temporal_metrics['all']['success_times']) if temporal_metrics['all']['success_times'] else 0
    failure_avg = np.mean(temporal_metrics['all']['failure_times']) if temporal_metrics['all']['failure_times'] else 0

    print(f"Average transition time in successful cases: {success_avg:.2f}")
    print(f"Average transition time in failure cases: {failure_avg:.2f}")
    print(f"Time difference ratio (failure/success): {(failure_avg / success_avg if success_avg else 0):.2f}")


def evaluate_per_class_metrics(model, dataset, device):
    """Evaluate detailed per-class metrics"""
    model.eval()
    class_metrics = defaultdict(lambda: {'true': [], 'pred': []})
    outcome_metrics = defaultdict(lambda: {'true': [], 'pred': []})

    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            trace = example['trace'].unsqueeze(0).to(device)
            mask = example['mask'].unsqueeze(0).to(device)
            times = example['times'].unsqueeze(0).to(device)
            customer_type = example['customer_type'].unsqueeze(0).to(device)
            loop_features = example['loop_features'].unsqueeze(0).to(device)

            # Ensure correct feature dimension
            expected_feat_dim = model.config.total_feature_dim
            if loop_features.shape[1] != expected_feat_dim:
                if loop_features.shape[1] < expected_feat_dim:
                    padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                    loop_features = torch.cat([loop_features, padding], dim=1)
                else:
                    loop_features = loop_features[:, :expected_feat_dim]

            next_act_logits, outcome_logits = model(
                trace, mask, times, loop_features, customer_type
            )

            # Move tensors to CPU for sklearn metrics
            next_act_preds = torch.argmax(next_act_logits, dim=2)[0].cpu()
            next_act_true = example['next_activity'].cpu()
            outcome_pred = torch.argmax(outcome_logits, dim=1).cpu().item()
            outcome_true = example['outcome'].cpu().item()

            # Store predictions
            valid_positions = mask[0].cpu().bool()
            for true, pred in zip(next_act_true[valid_positions],
                                  next_act_preds[valid_positions]):
                true_val = true.item()
                pred_val = pred.item()
                class_metrics[true_val]['true'].append(true_val)
                class_metrics[true_val]['pred'].append(pred_val)

            outcome_metrics[outcome_true]['true'].append(outcome_true)
            outcome_metrics[outcome_true]['pred'].append(outcome_pred)

    # Compute per-class metrics
    print("\n=== Per-Class Metrics ===")

    print("\nNext Activity Prediction:")
    for class_idx in sorted(class_metrics.keys()):
        true = class_metrics[class_idx]['true']
        pred = class_metrics[class_idx]['pred']
        if len(true) > 0:
            f1 = f1_score(true, pred, average='macro', zero_division=0)
            precision = precision_score(true, pred, average='macro', zero_division=0)
            recall = recall_score(true, pred, average='macro', zero_division=0)
            print(f"\nClass {class_idx}:")
            print(f"  Samples: {len(true)}")
            print(f"  F1: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")

    print("\nOutcome Prediction:")
    for class_idx in sorted(outcome_metrics.keys()):
        true = outcome_metrics[class_idx]['true']
        pred = outcome_metrics[class_idx]['pred']
        if len(true) > 0:
            f1 = f1_score(true, pred, average='macro', zero_division=0)
            precision = precision_score(true, pred, average='macro', zero_division=0)
            recall = recall_score(true, pred, average='macro', zero_division=0)
            print(f"\nClass {class_idx}:")
            print(f"  Samples: {len(true)}")
            print(f"  F1: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")


def analyze_feature_positions(dataset, n_buckets=10):
    """
    Analyze where engineered features appear in traces and create n most frequent buckets

    Args:
        dataset: TraceDataset instance
        n_buckets: Number of buckets to create based on most frequent positions
    """
    feature_positions = defaultdict(list)
    position_counts = defaultdict(int)

    for i in range(len(dataset)):
        example = dataset[i]
        valid_positions = example['mask'].bool()
        trace_length = valid_positions.sum().item()

        # Check loop features
        loop_features = example['loop_features']
        print('Loop features: ', loop_features)
        for j, is_active in enumerate(loop_features):
            if is_active:
                position_counts[j] += 1
                feature_positions['loop_features'].append(j)

        # Check time transitions
        # times = example['times'][valid_positions]
        # time_diffs = torch.diff(times)
        # significant_transitions = (time_diffs > time_diffs.mean() + time_diffs.std()).nonzero()
        # print('TIme diffs and significant:', time_diffs, significant_transitions)
        # if len(significant_transitions) > 0:
        #     for pos in significant_transitions.flatten():
        #         pos_val = pos.item() + 1
        #         position_counts[pos_val] += 1
        #         feature_positions['time_transitions'].append(pos_val)

    # Get n most frequent positions
    sorted_positions = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)
    top_positions = sorted([pos for pos, count in sorted_positions[:n_buckets]])

    # Create buckets based on top positions
    buckets = {}
    if top_positions:
        # First bucket: start to first frequent position
        buckets[f"start-{top_positions[0]}"] = (1, top_positions[0])

        # Middle buckets
        for i in range(len(top_positions) - 1):
            bucket_name = f"{top_positions[i] + 1}-{top_positions[i + 1]}"
            buckets[bucket_name] = (top_positions[i] + 1, top_positions[i + 1])

        # Last bucket: last frequent position to end
        buckets[f"{top_positions[-1] + 1}+"] = (top_positions[-1] + 1, None)

    # Print feature position statistics
    print("\n=== Feature Position Analysis ===")
    print(f"Top {n_buckets} most frequent feature positions:")
    for pos, count in sorted_positions[:n_buckets]:
        print(f"Position {pos}: {count} occurrences")

    return buckets


def analyze_loop_positions(dataset, activity_mappings, loop_activities_by_outcome, n_most_frequent=5):
    """
    Analyze where loops occur in traces using the activity mappings
    """
    # Create id_to_activity mapping
    id_to_activity = {v: k for k, v in activity_mappings['activity_to_id'].items()}
    loop_position_counts = defaultdict(int)

    for i in range(len(dataset)):
        example = dataset[i]
        trace = example['trace']
        trace_list = trace[trace != 0].tolist()

        # Convert IDs to activity names
        trace_activities = [id_to_activity[x] for x in trace_list]

        # Check for consecutive appearances of monitored activities
        for j in range(len(trace_activities) - 1):
            current_activity = trace_activities[j]
            next_activity = trace_activities[j + 1]

            # Only count position if it's a monitored activity
            for outcome, activities in loop_activities_by_outcome.items():
                if current_activity in activities and current_activity == next_activity:
                    loop_position_counts[j + 1] += 1  # Add 1 to avoid 0-based indexing

    # Get N most frequent positions
    sorted_positions = sorted(loop_position_counts.items(),
                              key=lambda x: x[1],
                              reverse=True)[:n_most_frequent]
    frequent_positions = sorted([pos for pos, count in sorted_positions])

    # Create buckets based on frequent loop positions
    buckets = {}
    if frequent_positions:
        # First bucket: start to first frequent loop
        buckets[f"start-{frequent_positions[0]}"] = (1, frequent_positions[0])

        # Middle buckets between frequent loops
        for i in range(len(frequent_positions) - 1):
            bucket_name = f"{frequent_positions[i] + 1}-{frequent_positions[i + 1]}"
            buckets[bucket_name] = (frequent_positions[i] + 1, frequent_positions[i + 1])

        # Last bucket: last frequent loop to end
        buckets[f"{frequent_positions[-1] + 1}+"] = (frequent_positions[-1] + 1, None)

    # Print statistics about loop positions
    print("\n=== Loop Position Analysis ===")
    print("Most frequent loop positions:")
    for pos, count in sorted_positions:
        print(f"Position {pos}: {count} occurrences")

    return buckets


def evaluate_with_buckets(model, dataset, device):
    """Evaluate outcome and next activity prediction at feature-based intervals"""
    buckets = analyze_loop_positions(pytorch_dataset, mappings, loop_activities_by_outcome)

    bucket_metrics = {name: {
        'outcome': {'true': [], 'pred': []},
        'next_activity': {'true': [], 'pred': []}
    } for name in buckets}

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            trace_length = example['mask'].sum().item()

            for bucket_name, (min_len, max_len) in buckets.items():
                if min_len <= trace_length and (max_len is None or trace_length <= max_len):
                    slice_len = max_len if max_len is not None else trace_length

                    # Move all tensors to device
                    trace = example['trace'][:slice_len].unsqueeze(0).to(device)
                    mask = example['mask'][:slice_len].unsqueeze(0).to(device)
                    times = example['times'][:slice_len].unsqueeze(0).to(device)
                    loop_features = example['loop_features'].unsqueeze(0).to(device)
                    customer_type = example['customer_type'].unsqueeze(0).to(device)

                    # Ensure correct feature dimension
                    expected_feat_dim = model.config.total_feature_dim
                    if loop_features.shape[1] != expected_feat_dim:
                        if loop_features.shape[1] < expected_feat_dim:
                            padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                            loop_features = torch.cat([loop_features, padding], dim=1)
                        else:
                            loop_features = loop_features[:, :expected_feat_dim]

                    next_activity_logits, outcome_logits = model(trace, mask, times, loop_features, customer_type)

                    # Store predictions
                    valid_positions = mask[0].bool()
                    true_activities = example['next_activity'][:slice_len].to(device)[valid_positions].cpu().tolist()
                    pred_activities = torch.argmax(next_activity_logits[0], dim=1)[valid_positions].cpu().tolist()

                    bucket_metrics[bucket_name]['next_activity']['true'].extend(true_activities)
                    bucket_metrics[bucket_name]['next_activity']['pred'].extend(pred_activities)
                    bucket_metrics[bucket_name]['outcome']['true'].append(example['outcome'].item())
                    bucket_metrics[bucket_name]['outcome']['pred'].append(
                        torch.argmax(outcome_logits, dim=1).cpu().item())
                    break

    print("\n=== Prediction Performance by Feature-Based Intervals ===")
    for bucket_name, metrics in bucket_metrics.items():
        if len(metrics['outcome']['true']) > 0:
            outcome_f1 = f1_score(metrics['outcome']['true'], metrics['outcome']['pred'],
                                  average='macro', zero_division=0)
            outcome_acc = accuracy_score(metrics['outcome']['true'], metrics['outcome']['pred'])

            next_act_f1 = f1_score(metrics['next_activity']['true'], metrics['next_activity']['pred'],
                                   average='macro', zero_division=0)
            next_act_acc = accuracy_score(metrics['next_activity']['true'], metrics['next_activity']['pred'])

            print(f"\nInterval {bucket_name}:")
            print(f"  Samples: {len(metrics['outcome']['true'])}")
            print(f"  Outcome F1 Score: {outcome_f1:.4f}")
            print(f"  Outcome Accuracy: {outcome_acc:.4f}")
            print(f"  Next Activity F1 Score: {next_act_f1:.4f}")
            print(f"  Next Activity Accuracy: {next_act_acc:.4f}")


def analyze_feature_positions_for_bucketing(dataset, activity_mappings, loop_activities_by_outcome,
                                            time_sensitive_transitions):
    """Analyze feature positions for bucketing using existing detection logic"""
    feature_positions = defaultdict(list)

    # Use existing id_to_activity mapping
    id_to_activity = {v: k for k, v in activity_mappings['activity_to_id'].items()}

    for i in range(len(dataset)):
        trace = dataset.traces[i]
        times = dataset.times[i]
        trace_list = trace[trace != 0].tolist()
        trace_activities = [id_to_activity[x] for x in trace_list]

        # Check for loops using existing logic
        for j in range(len(trace_activities) - 1):
            if trace_activities[j] == trace_activities[j + 1]:
                for outcome, activities in loop_activities_by_outcome.items():
                    if trace_activities[j] in activities:
                        print(f"Loop detected at position {j + 1} for activity {trace_activities[j]}")
                        feature_positions['loops'].append(j + 1)

        # Check for time-sensitive transitions
        valid_mask = trace != 0
        time_list = times[valid_mask].tolist()
        for j in range(len(trace_activities) - 1):
            current_act = trace_activities[j]
            next_act = trace_activities[j + 1]
            for trans in time_sensitive_transitions:
                if current_act == trans['act1'] and next_act == trans['act2']:
                    time_diff = abs(time_list[j + 1] - time_list[j])
                    threshold = (trans['success_threshold'] + trans['failure_threshold']) / 2
                    print(f"Time transition detected at position {j + 1} between {current_act} -> {next_act}")
                    print(f"Time difference: {time_diff:.2f}, Threshold: {threshold:.2f}")
                    feature_positions['time_transitions'].append(j + 1)

    # Get unique positions where features occur
    all_positions = sorted(set(feature_positions['loops'] + feature_positions['time_transitions']))

    # Create buckets based on feature positions
    buckets = {}
    if all_positions:
        # First bucket: start to first feature
        buckets[f"short (1-{all_positions[0]})"] = (1, all_positions[0])

        # Middle bucket(s)
        mid_point = all_positions[len(all_positions) // 2]
        buckets[f"medium ({all_positions[0] + 1}-{mid_point})"] = (all_positions[0] + 1, mid_point)

        # Last bucket: mid point to end
        buckets[f"long ({mid_point + 1}+)"] = (mid_point + 1, None)

    return buckets


def evaluate_model_by_buckets(model, dataset, device):
    """Evaluate model performance separately for each trace length bucket"""
    buckets = analyze_feature_positions_for_bucketing(
        dataset,
        mappings,
        loop_activities_by_outcome,
        time_sensitive_transitions
    )

    print("\n=== Performance by Trace Length Buckets ===")
    for bucket_name, (min_len, max_len) in buckets.items():
        print(f"\nEvaluating {bucket_name} traces:")

        bucket_metrics = {
            'next_activity': {'true': [], 'pred': []},
            'outcome': {'true': [], 'pred': []}
        }

        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                example = dataset[i]
                trace_length = (example['trace'] != 0).sum().item()

                if min_len <= trace_length and (max_len is None or trace_length <= max_len):
                    # Move all tensors to device
                    trace = example['trace'].unsqueeze(0).to(device)
                    mask = example['mask'].unsqueeze(0).to(device)
                    times = example['times'].unsqueeze(0).to(device)
                    customer_type = example['customer_type'].unsqueeze(0).to(device)
                    loop_features = example['loop_features'].unsqueeze(0).to(device)
                    next_activity = example['next_activity'].to(device)

                    # Ensure correct feature dimension
                    expected_feat_dim = model.config.total_feature_dim
                    if loop_features.shape[1] != expected_feat_dim:
                        if loop_features.shape[1] < expected_feat_dim:
                            padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                            loop_features = torch.cat([loop_features, padding], dim=1)
                        else:
                            loop_features = loop_features[:, :expected_feat_dim]

                    next_activity_logits, outcome_logits = model(
                        trace, mask, times, loop_features, customer_type
                    )

                    # Move everything to same device before indexing
                    valid_positions = mask[0].bool()
                    true_activities = next_activity[valid_positions].cpu().tolist()
                    pred_activities = torch.argmax(next_activity_logits[0], dim=1)[valid_positions].cpu().tolist()

                    bucket_metrics['next_activity']['true'].extend(true_activities)
                    bucket_metrics['next_activity']['pred'].extend(pred_activities)
                    bucket_metrics['outcome']['true'].append(example['outcome'].item())
                    bucket_metrics['outcome']['pred'].append(torch.argmax(outcome_logits, dim=1).cpu().item())

        # Compute metrics for this bucket
        if bucket_metrics['outcome']['true']:
            next_act_f1 = f1_score(
                bucket_metrics['next_activity']['true'],
                bucket_metrics['next_activity']['pred'],
                average='macro',
                zero_division=0
            )
            next_act_acc = accuracy_score(
                bucket_metrics['next_activity']['true'],
                bucket_metrics['next_activity']['pred']
            )
            outcome_f1 = f1_score(
                bucket_metrics['outcome']['true'],
                bucket_metrics['outcome']['pred'],
                average='macro',
                zero_division=0
            )
            outcome_acc = accuracy_score(
                bucket_metrics['outcome']['true'],
                bucket_metrics['outcome']['pred']
            )

            print(f"Samples in bucket: {len(bucket_metrics['outcome']['true'])}")
            print(f"Next Activity - F1: {next_act_f1:.4f}, Accuracy: {next_act_acc:.4f}")
            print(f"Outcome - F1: {outcome_f1:.4f}, Accuracy: {outcome_acc:.4f}")


import numpy as np


def analyze_prediction_errors(model, dataset, device, top_k=100):
    """Analyze and print most common prediction errors"""
    model.eval()
    error_counter = defaultdict(lambda: {'count': 0, 'wrong_predictions': defaultdict(int)})

    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]
            if 'loop_features' not in example:
                example['loop_features'] = torch.zeros(model.config.total_feature_dim)
            trace = example['trace'].unsqueeze(0).to(device)
            mask = example['mask'].unsqueeze(0).to(device)
            times = example['times'].unsqueeze(0).to(device)
            customer_type = example['customer_type'].unsqueeze(0).to(device)
            loop_features = example.get('loop_features', torch.zeros(1, model.config.total_feature_dim)).unsqueeze(
                0).to(device)
            # Ensure correct feature dimension
            expected_feat_dim = model.config.total_feature_dim
            if loop_features.shape[1] != expected_feat_dim:
                if loop_features.shape[1] < expected_feat_dim:
                    padding = torch.zeros(1, expected_feat_dim - loop_features.shape[1], device=device)
                    loop_features = torch.cat([loop_features, padding], dim=1)
                else:
                    loop_features = loop_features[:, :expected_feat_dim]

            # Get predictions
            next_act_logits, _ = model(trace, mask, times, loop_features, customer_type)
            seq_len = next_act_logits.shape[1]  # Get model's output length

            # Truncate masks and labels to model's actual output length
            valid_positions = mask[0, :seq_len].bool().cpu()
            next_act_preds = torch.argmax(next_act_logits, dim=2)[0, :seq_len].cpu()
            next_act_true = example['next_activity'][:seq_len].cpu()

            # Analyze errors
            for true, pred in zip(next_act_true[valid_positions], next_act_preds[valid_positions]):
                true_val = true.item()
                pred_val = pred.item()
                if true_val != pred_val:
                    error_counter[true_val]['count'] += 1
                    error_counter[true_val]['wrong_predictions'][pred_val] += 1

    # Sort activities by error frequency
    sorted_errors = sorted(error_counter.items(),
                           key=lambda x: x[1]['count'],
                           reverse=True)[:top_k]

    print("\n## Most Frequently Mispredicted Activities")
    print("\nActivity | Error Count | Top Wrong Predictions")
    print("-" * 60)

    for act_idx, stats in sorted_errors:
        # Get top 3 wrong predictions
        top_wrong = sorted(stats['wrong_predictions'].items(),
                           key=lambda x: x[1],
                           reverse=True)[:3]
        wrong_str = ", ".join([f"Activity_{pred}({cnt})"
                               for pred, cnt in top_wrong])

        print(f"Activity_{act_idx} | {stats['count']} | {wrong_str}")


# Add this to your evaluation pipeline
analyze_prediction_errors(model, pytorch_dataset, device)

# evaluate_with_buckets(model, pytorch_dataset, device)

# Add both analyses
# analyze_feature_importance(model, pytorch_dataset, device)
# analyze_temporal_patterns(model, pytorch_dataset, device)

# # Evaluate a single example
evaluate_example(model, pytorch_dataset, index=5, device=device)

# # Concept mapping example analysis
# analyze_concept_mapping(model)

# # # Prompt visualization
# # visualize_prompts(model)


# # Concept drift analysis
# analyze_concept_drift(model, pytorch_dataset, device)

# # Per class evaluation
# evaluate_per_class_metrics(model, pytorch_dataset, device=device)

# # Perform comprehensive evaluation
evaluate_model(model, pytorch_dataset, device=device)

# Evaluate model performance by buckets
evaluate_model_by_buckets(model, pytorch_dataset, device)
