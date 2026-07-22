# liquid_level_analyzer.py
import argparse
import json
import numpy as np
from pathlib import Path

# All glassware class names
GLASSWARE_CLASSES = {
    "beaker", "burette", "centrifuge_tube", "colorimetric_tube", "conical_flask",
    "dropper_bottle", "measuring_cup", "measuring_cylinder", "narrow_mouth_bottle",
    "pear_shaped_separatory_funnel", "powder_funnel_with_joint", "round_bottle_flask",
    "screw_top_glass_bottle", "test_tube", "three_mouth_flask", "two_necked_flask",
    "upper_nozzle_filtering_flask", "volumetric_flask", "wide_mouth_bottle"
}


def compute_liquid_ratio(container_points: np.ndarray, interface_points: np.ndarray) -> float:
    """
    Calculate liquid height ratio (0 ~ 1) inside container.
    Image coordinate: y increases downward.
    """
    container_points = np.array(container_points)
    interface_points = np.array(interface_points)

    y_min, y_max = container_points[:, 1].min(), container_points[:, 1].max()
    container_height = y_max - y_min
    if container_height < 1e-6:
        return 0.0

    y_interface = interface_points[:, 1].mean()
    ratio = (y_max - y_interface) / container_height
    return float(np.clip(ratio, 0.0, 1.0))


def process_single_json(json_path: Path, output_dir: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        detection_objects = json.load(f)

    target_glassware = None
    target_interface = None
    liquid_ratio = None
    risk_status = "normal"

    for obj in detection_objects:
        cls_name = obj["class"]
        if cls_name in GLASSWARE_CLASSES:
            target_glassware = obj
        if cls_name == "gas_liquid_interface":
            target_interface = obj

    if target_glassware is not None and target_interface is not None:
        liquid_ratio = compute_liquid_ratio(target_glassware["mask"], target_interface["mask"])
        if liquid_ratio > 0.8:
            risk_status = "overflow_risk"
        elif liquid_ratio < 0.1:
            risk_status = "too_low"

    summary_result = {
        "glassware_class": target_glassware["class"] if target_glassware else None,
        "interface_class": target_interface["class"] if target_interface else None,
        "liquid_level_ratio": round(liquid_ratio, 2) if liquid_ratio is not None else None,
        "risk": risk_status
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = json_path.stem

    # Save summary json
    json_out_path = output_dir / f"{stem}_summary.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)

    # Build LLM prompt
    if target_glassware and target_interface:
        prompt_text = f"This is a {summary_result['glassware_class']} with {summary_result['interface_class']}. "
        prompt_text += f"The liquid level occupies approximately {int(liquid_ratio * 100)}% of the container height. "
    else:
        prompt_text = "No valid glassware and gas-liquid interface detected. "
    prompt_text += "Does this scene have potential laboratory safety risks?"

    prompt_out_path = output_dir / f"{stem}_prompt.txt"
    with open(prompt_out_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    print(f"[Finished] {json_path.name} -> saved to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze liquid level ratio inside glassware from YOLO segmentation JSON output, generate risk summary and LLM prompt."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path of single input json file OR directory containing multiple json files")
    parser.add_argument("--output-dir", type=str, default="outputs/liquid_analysis",
                        help="Directory to save summary json and prompt txt")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if input_path.is_file() and input_path.suffix.lower() == ".json":
        process_single_json(input_path, output_dir)
    elif input_path.is_dir():
        json_list = list(input_path.glob("*.json"))
        if len(json_list) == 0:
            print("No json files found in input directory.")
            return
        for json_file in json_list:
            process_single_json(json_file, output_dir)
    else:
        print("Invalid input path. Must be a .json file or a folder with json files.")


if __name__ == "__main__":
    main()
