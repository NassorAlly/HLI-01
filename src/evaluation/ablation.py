from copy import deepcopy
from typing import Any, Dict, List


class AblationStudy:
    """
    Build controlled configuration variants for HLI-01 ablation studies.
    """

    @staticmethod
    def generate_variants(
        base_config: Dict[str, Any],
        parameter_path: List[str],
        values: List[Any],
    ) -> List[Dict[str, Any]]:
        if not parameter_path:
            raise ValueError("parameter_path cannot be empty.")

        if not values:
            raise ValueError("values cannot be empty.")

        variants = []

        for value in values:
            variant = deepcopy(base_config)

            target = variant

            for key in parameter_path[:-1]:
                if key not in target:
                    raise KeyError(
                        f"Configuration key '{key}' not found."
                    )

                target = target[key]

            final_key = parameter_path[-1]

            if final_key not in target:
                raise KeyError(
                    f"Configuration key '{final_key}' not found."
                )

            target[final_key] = value

            variants.append(variant)

        return variants