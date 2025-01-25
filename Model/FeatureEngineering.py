import torch 

def add_loop_features(pytorch_dataset, activity_mappings, loop_activities_by_outcome):
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
        trace_list = trace[trace != 0].tolist()       # yields a 1D list of non-zero IDs        
        
        # Convert IDs to activity names
        trace_activities = [id_to_activity[x] for x in trace_list]
        
        # Check for loops
        for idx, (outcome, activities) in enumerate(loop_activities_by_outcome.items()):
            for activity in activities:
                # Check for consecutive appearances
                for j in range(len(trace_activities) - 1):
                    if (trace_activities[j] == activity and 
                        trace_activities[j+1] == activity):
                        feature_idx = feature_names.index(f"loop_{activity}_outcome_{outcome}")
                        loop_features[i, feature_idx] = 1
                        break
    
    # Add features to dataset
    pytorch_dataset.loop_features = loop_features

    # print_loop_statistics
    total_traces = len(pytorch_dataset.traces)
    loop_counts = {feature: pytorch_dataset.loop_features[:, i].sum().item() 
                  for i, feature in enumerate(feature_names)}
    
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

