def filter_entity_map(entity_map: dict) -> dict:
    """
    Given a dict of {entity_id: state_dict, ...}, returns a new dict
    where each state dict has noisy keys removed:
    'context', 'last_reported', 'last_updated', 'last_changed'
    """
    if not isinstance(entity_map, dict):
        print("[filter_entity_map] Input is not a dict:", repr(entity_map))
        return entity_map

    keys_to_drop = {"context", "last_reported", "last_updated", "last_changed"}
    filtered_map = {}
    for eid, state in entity_map.items():
        if not isinstance(state, dict):
            print(f"[filter_entity_map] State for {eid} is not a dict: {repr(state)}")
            filtered_map[eid] = state
            continue

        filtered = {k: v for k, v in state.items() if k not in keys_to_drop}
        filtered_map[eid] = filtered

    return filtered_map
