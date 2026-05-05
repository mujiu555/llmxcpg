"""
Complete Process:
1. Curate dataset of code samples (vulnerable and safe) with the following properties code, cwe, and line number
2. Generate queries using LLM or manually (for now)
3. Execute the Joern queries and extract the paths (node_id and line_number)
4. Merge the extracted paths into a small and concise set of paths that comprehensivly captures the vulnerabilities within the code
5. Convert the contexts to well formatted source code
6. Evaluate and label each context individually using an LLM
"""

import os
import json
from Components.enhancer import analyze_c_code, get_context, save_context


def construct_code_snippet():
    """Executes the queries acquired from the 'Generate Queries' step and extract the paths in the following json format:
    [
        {
            "file_name": name of the file,
            "queries": [
                {
                    "id": node id,
                    "line_number": the line number where this node exists in the source code
                },
                ...
            ]
        },
        ...
    ]
    Then merge the extracted paths into more concise paths, finally we convert each of these result paths into source code and we save each in the folder 'EnhancedCode'
    """

    # Ensure EnhancedCode directory exists
    folder_path = "ReposVulSelfContainedTest"
    os.makedirs(folder_path, exist_ok=True)

    # Read the extracted paths
    with open("reposvul_test_paths_10.json", "r") as f:
        dataset = json.load(f)

    enhanced_results = []

    for item in dataset:
        source_code = item["code"]
        # paths = item["all_paths"]
        paths = item["all_paths"]
        if len(paths) == 0:
            continue

        # Merge paths
        merged_paths_list = merge(paths)
        print("merged paths list:", merged_paths_list)

        # Process each merged path
        for merged_path in merged_paths_list:
            print("merged path", merged_path)

            # Analyze code blocks
            blocks = analyze_c_code(source_code)

            # Extract line numbers from the path
            path_line_numbers = list(
                filter(
                    (lambda line: line != None),
                    [node.get("line_number") for node in merged_path],
                )
            )

            print("path line numbers", path_line_numbers)
            # Get context lines
            context_lines = get_context(path_line_numbers, blocks)

            # Save enhanced context
            output_file = os.path.join(
                folder_path, f"{os.path.basename(item['file_name'])}_enhanced.c"
            )
            enhanced_code = save_context(list(context_lines), source_code, output_file)

            enhanced_results.append(
                {
                    "file_name": item["file_name"],
                    "path_file": output_file,
                    "self_contained_code": enhanced_code,
                    "context_lines": list(context_lines),
                    "cwe": item["cwe"],
                    "label": item["label"],
                }
            )

    # Save enhanced results to a JSON file
    with open("reposvul_test_self_contained_concise.json", "w") as f:
        json.dump(enhanced_results, f, indent=4)


def merge(paths: list[list]) -> list:
    """Returns contained code snippets based on each of the received paths.
    Args:
        paths: The list of flows returned by Joern
    """
    contained_paths = []

    for path in paths:
        merged_line_numbers = set()
        contained_path = []
        for node in path:
            if node.get("line_number") in merged_line_numbers:
                continue

            merged_line_numbers.add(node.get("line_number"))
            contained_path.append(node)

        contained_paths.append(contained_path)

    return contained_paths
