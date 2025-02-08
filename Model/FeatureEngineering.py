import torch
from collections import defaultdict

# Define activities to monitor for each outcome
loop_activities_by_outcome = {
    0: ['Funnel_Offerte uitbrengen', 'Mailing_Controle rentekorting hyp.'],
    # Activities that loop in unsuccessful cases
    1: ['Funnel_Huismap vullen']  # Activities that loop in successful cases
}

# Define time-sensitive transitions and their thresholds
time_sensitive_transitions = [
    {
        'act1': 'Funnel_Hebben afspraak',
        'act2': 'Funnel_Offerte aanvraag',
        'success_threshold': 222.0,
        'failure_threshold': 2359.5
    },
    {
        'act1': 'Funnel_Hebben afspraak',
        'act2': 'Funnel_Plannen afspraak',
        'success_threshold': 312.8,
        'failure_threshold': 3060.7
    },
    {
        'act1': 'Regelen_Bouwdepot',
        'act2': 'Regelen_Bouwdepot',
        'success_threshold': 10.8,
        'failure_threshold': 203.5
    },
    {
        'act1': 'Funnel_Huismap vullen',
        'act2': 'Funnel_Hebben afspraak',
        'success_threshold': 51.1,
        'failure_threshold': 115.8
    },
    {
        'act1': 'Huismap overig_Huismap voorbereiding',
        'act2': 'Funnel_Hebben afspraak',
        'success_threshold': 68.6,
        'failure_threshold': 138.6
    },
    {
        'act1': 'Funnel_Hebben afspraak',
        'act2': 'Contact_Aankoop/verkoop',
        'success_threshold': 44.6,
        'failure_threshold': 118.1
    },
    {
        'act1': 'Verkoop_OTD',
        'act2': 'Funnel_Offerte aanvraag',
        'success_threshold': 151.4,
        'failure_threshold': 273.4
    },
    {
        'act1': 'Funnel_Plannen afspraak',
        'act2': 'Funnel_Offerte acceptatie',
        'success_threshold': 31.9,
        'failure_threshold': 116.3
    },
    {
        'act1': 'Funnel_Offerte acceptatie',
        'act2': 'Contact_Overig',
        'success_threshold': 64.3,
        'failure_threshold': 730.6
    },
    {
        'act1': 'Online_OriÃ«ntatie',
        'act2': 'Funnel_Offerte aanvraag',
        'success_threshold': 1303.1,
        'failure_threshold': 6843.4
    },
    {
        'act1': 'Systeembrief_Informatie',
        'act2': 'Regelen_Bouwdepot',
        'success_threshold': 243.0,
        'failure_threshold': 399.6
    },
    {
        'act1': 'Funnel_Offerte acceptatie',
        'act2': 'Funnel_Offerte uitbrengen',
        'success_threshold': 168.0,
        'failure_threshold': 228.0
    },
    {
        'act1': 'Funnel_Offerte acceptatie',
        'act2': 'Productgroepmutatie_Uitstroom',
        'success_threshold': 864.0,
        'failure_threshold': 1164.0
    },
    {
        'act1': 'Funnel_Offerte acceptatie',
        'act2': 'Productgroepmutatie_Instroom',
        'success_threshold': 72.0,
        'failure_threshold': 768.0
    }
]


def add_loop_features(pytorch_dataset, activity_mappings, loop_activities_by_outcome, print_statistics=False):
    """
    Add loop detection features using existing activity mappings

    Args:
        pytorch_dataset: TraceDataset instance
        activity_mappings: dict from preprocessing containing activity_to_id mapping
        loop_activities_by_outcome: dict mapping outcomes to activities to monitor
            e.g., {0: ['activity_A', 'activity_B'], 1: ['activity_C']}
    """
    num_traces = len(pytorch_dataset.traces)

    # Reverse the activity mapping for easier lookup
    id_to_activity = {v: k for k, v in activity_mappings['activity_to_id'].items()}

    # Initialize features tensor
    feature_names = []
    for outcome, activities in loop_activities_by_outcome.items():
        for activity in activities:
            feature_names.append(f"loop_{activity}_outcome_{outcome}")

    loop_features = torch.zeros((num_traces, len(feature_names)), dtype=torch.float)

    # Detect loops for each trace
    for i in range(num_traces):
        trace = pytorch_dataset.traces[i]
        trace_list = trace[trace != 0].tolist()  # yields a 1D list of non-zero IDs

        # Convert IDs to activity names
        trace_activities = [id_to_activity[x] for x in trace_list]

        # Check for loops
        for idx, (outcome, activities) in enumerate(loop_activities_by_outcome.items()):
            for activity in activities:
                # Check for consecutive appearances
                for j in range(len(trace_activities) - 1):
                    if (trace_activities[j] == activity and
                            trace_activities[j + 1] == activity):
                        feature_idx = feature_names.index(f"loop_{activity}_outcome_{outcome}")
                        loop_features[i, feature_idx] = 1
                        break

    # Add features to dataset
    pytorch_dataset.loop_features = loop_features

    # print_loop_statistics
    total_traces = len(pytorch_dataset.traces)
    loop_counts = {feature: pytorch_dataset.loop_features[:, i].sum().item()
                   for i, feature in enumerate(feature_names)}

    if print_statistics:
        print("\n=== Loop Statistics ===")
        for feature, count in loop_counts.items():
            percentage = (count / total_traces) * 100
            print(f"{feature}: {count} traces ({percentage:.2f}%)")

    # Modify __getitem__
    original_getitem = pytorch_dataset.__getitem__

    def new_getitem(self, idx):
        item = original_getitem(idx)
        item['loop_features'] = self.loop_features[idx]
        return item

    pytorch_dataset.__getitem__ = new_getitem.__get__(pytorch_dataset)


def add_temporal_features(pytorch_dataset, activity_mappings, time_sensitive_transitions, print_statistics=False):
    """
    Add temporal transition features using existing activity mappings and transition-specific thresholds

    Args:
        pytorch_dataset: TraceDataset instance
        activity_mappings: dict containing activity_to_id mapping
        time_sensitive_transitions: list of dicts containing:
            [
                {
                    'act1': 'activity_name_1',
                    'act2': 'activity_name_2',
                    'success_threshold': float,  # Average time for successful cases
                    'failure_threshold': float   # Average time for unsuccessful cases
                },
                ...
            ]
    """
    num_traces = len(pytorch_dataset.traces)
    id_to_activity = {v: k for k, v in activity_mappings['activity_to_id'].items()}

    # Initialize features tensor - 2 features per transition (quick and slow)
    feature_names = []
    for trans in time_sensitive_transitions:
        feature_names.append(f"quick_transition_{trans['act1']}_{trans['act2']}")
        feature_names.append(f"slow_transition_{trans['act1']}_{trans['act2']}")

    temporal_features = torch.zeros((num_traces, len(feature_names)), dtype=torch.float)

    # Detect temporal patterns for each trace
    for i in range(num_traces):
        trace = pytorch_dataset.traces[i]
        times = pytorch_dataset.times[i]

        valid_mask = trace != 0
        trace_list = trace[valid_mask].tolist()
        time_list = times[valid_mask].tolist()
        trace_activities = [id_to_activity[x] for x in trace_list]

        # Check each transition pattern
        for idx, trans in enumerate(time_sensitive_transitions):
            feat_idx_quick = idx * 2
            feat_idx_slow = idx * 2 + 1

            # Look for consecutive appearances
            for j in range(len(trace_activities) - 1):
                if (trace_activities[j] == trans['act1'] and
                        trace_activities[j + 1] == trans['act2']):
                    time_diff = abs(time_list[j + 1] - time_list[j])

                    # Use midpoint between success and failure thresholds
                    threshold = (trans['success_threshold'] + trans['failure_threshold']) / 2

                    if time_diff <= threshold:
                        temporal_features[i, feat_idx_quick] = 1
                    else:
                        temporal_features[i, feat_idx_slow] = 1
                    break

    # Print statistics
    if print_statistics:
        total_traces = len(pytorch_dataset.traces)
        print("\n=== Temporal Pattern Statistics ===")
        for idx, name in enumerate(feature_names):
            count = temporal_features[:, idx].sum().item()
            percentage = (count / total_traces) * 100
            print(f"{name}: {count} traces ({percentage:.2f}%)")

    return temporal_features


def add_all_features(pytorch_dataset, activity_mappings, loop_activities_by_outcome,
                     time_sensitive_transitions, print_statistics=False):
    """Combined function to add both loop and temporal features"""

    # Add loop features
    add_loop_features(pytorch_dataset, activity_mappings, loop_activities_by_outcome)

    # Add temporal features
    temporal_features = add_temporal_features(pytorch_dataset, activity_mappings,
                                              time_sensitive_transitions)

    # Combine features
    pytorch_dataset.loop_features = torch.cat([
        pytorch_dataset.loop_features,
        temporal_features
    ], dim=1)

    # Print statistics for temporal features
    if print_statistics:
        total_traces = len(pytorch_dataset.traces)
        temporal_counts = {
            f"quick_transition_{trans['act1']}_{trans['act2']}": temporal_features[:, i].sum().item()
            for i, trans in enumerate(time_sensitive_transitions)
        }

        print("\n=== Temporal Pattern Statistics ===")
        for feature, count in temporal_counts.items():
            percentage = (count / total_traces) * 100
            print(f"{feature}: {count} traces ({percentage:.2f}%)")


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
                        # print(f"Loop detected at position {j+1} for activity {trace_activities[j]}")
                        feature_positions['loops'].append(j + 1)

        # Check for time-sensitive transitions using existing logic
        valid_mask = trace != 0
        time_list = times[valid_mask].tolist()
        for j in range(len(trace_activities) - 1):
            current_act = trace_activities[j]
            next_act = trace_activities[j + 1]
            for trans in time_sensitive_transitions:
                if current_act == trans['act1'] and next_act == trans['act2']:
                    time_diff = abs(time_list[j + 1] - time_list[j])
                    threshold = (trans['success_threshold'] + trans['failure_threshold']) / 2
                    # print(f"Time transition detected at position {j+1} between {current_act} -> {next_act}")
                    # print(f"Time difference: {time_diff:.2f}, Threshold: {threshold:.2f}")
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
