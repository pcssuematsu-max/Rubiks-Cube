"""Search and training result container classes."""


class data:
    """Search2用の学習データを保持する。"""

    def __init__(
        self,
        scramble,
        moves,
        rewards,
        perfect_key = None,
        top_group = None,
        source_ai_index = None,
        source_search_mode = 'search2',
        source_search2_value_loss_type = None,
        steps_to_goal = None,
    ):
        self.scramble = scramble
        self.moves = moves
        self.rewards = rewards
        self.perfect_key = perfect_key
        self.top_group = top_group
        self.source_ai_index = source_ai_index
        self.source_search_mode = source_search_mode
        self.source_search2_value_loss_type = source_search2_value_loss_type
        if steps_to_goal is None:
            steps_to_goal = tuple(range(len(moves),-1,-1))
        self.steps_to_goal = tuple(steps_to_goal)
        self.succeeded = False


class data_search3:
    """Search3用の学習データを保持する。"""

    def __init__(
        self,
        scramble,
        moves,
        rewards,
        root_value,
        value_trace,
        best_value,
        stats,
        policy_target=None,
        search_mode='search3',
        sample_weight=1.0,
        value_targets=None,
        root_value_raw=None,
        value_trace_raw=None,
        best_value_raw=None,
        perfect_key=None,
        top_group=None,
        end_reason=None,
        source_succeeded=False,
        solve_succeeded=False,
    ):
        self.scramble = scramble
        self.moves = moves
        self.rewards = rewards
        self.root_value = root_value
        self.value_trace = value_trace
        self.best_value = best_value
        self.root_value_raw = root_value if root_value_raw is None else root_value_raw
        self.value_trace_raw = value_trace if value_trace_raw is None else value_trace_raw
        self.best_value_raw = best_value if best_value_raw is None else best_value_raw
        self.stats = stats
        self.policy_target = policy_target
        self.search_mode = search_mode
        self.search_depth3 = 100
        self.sample_weight = sample_weight
        self.value_targets = value_targets
        self.perfect_key = perfect_key
        self.top_group = top_group
        self.end_reason = end_reason
        self.source_succeeded = bool(source_succeeded)
        self.solve_succeeded = bool(solve_succeeded)
        self.succeeded = self.source_succeeded


class SearchResult:
    """Search2/Search3の返り値を共通形式で保持する。"""

    def __init__(
        self,
        succeeded,
        moves,
        root_value,
        value_trace,
        best_value,
        stats,
        policy_target=None,
        search_mode='search2',
        end_reason='budget',
        attempt_results=None,
        attempt_index=None,
        root_value_raw=None,
        value_trace_raw=None,
        best_value_raw=None,
    ):
        self.succeeded = succeeded
        self.moves = tuple(moves)
        self.root_value = root_value
        self.value_trace = list(value_trace)
        self.best_value = best_value
        self.root_value_raw = root_value if root_value_raw is None else root_value_raw
        self.value_trace_raw = list(value_trace) if value_trace_raw is None else list(value_trace_raw)
        self.best_value_raw = best_value if best_value_raw is None else best_value_raw
        self.stats = stats
        self.policy_target = policy_target
        self.search_mode = search_mode
        self.end_reason = end_reason
        self.attempt_results = [] if attempt_results is None else list(attempt_results)
        self.attempt_index = attempt_index

    def __getitem__(self, index):
        legacy_values = (
            self.succeeded,
            self.moves,
            self.root_value,
            self.value_trace,
            self.best_value,
            self.stats,
        )
        return legacy_values[index]
